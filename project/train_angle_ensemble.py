"""Build sharded angle graphs and train a five-seed AngleGNN ensemble."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
from pymatgen.core import Structure
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

from graph_data import GraphConfig, collate_graphs, read_csv, structure_to_graph
from models import AngleGNN


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "dataset/processed"
CACHE_DIR = PROCESSED / "angle_graph_shards"
OUTPUT_DIR = ROOT / "outputs/angle_ensemble"
SEEDS = (42, 123, 2026, 3407, 7777)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _cache_one_shard(task: tuple[str, list[dict[str, str]], dict[str, dict[str, str]], dict]) -> dict:
    """Parse one raw JSON shard once and write all its graphs atomically."""
    import gzip

    relative, index_rows, rows_by_structure, config_values = task
    source = ROOT / relative
    target = CACHE_DIR / (source.name.removesuffix(".json.gz") + ".pt")
    if target.exists():
        return {"shard": relative, "status": "existing"}
    with gzip.open(source, "rt", encoding="utf-8") as handle:
        records = json.load(handle)
    config = GraphConfig(**config_values)
    graphs: list[dict[str, object] | None] = [None] * len(records)
    nodes = edges = triplets = 0
    for index_row in index_rows:
        offset = int(index_row["offset"])
        record = records[offset]
        if record["id"] != index_row["structure_id"] or record["md5"] != index_row["md5"]:
            raise ValueError(f"index mismatch in {relative} offset {offset}")
        row = rows_by_structure[index_row["structure_id"]]
        graph = structure_to_graph(Structure.from_str(record["cif"], fmt="cif"), row, config)
        graphs[offset] = graph
        nodes += len(graph["z"])
        edges += graph["edge_index"].shape[1]
        triplets += graph["triplet_edge_index"].shape[1]
    if any(graph is None for graph in graphs):
        raise ValueError(f"incomplete graph shard {relative}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(f".tmp.{os.getpid()}")
    torch.save(graphs, temporary)
    temporary.replace(target)
    return {
        "shard": relative, "status": "built", "graphs": len(graphs),
        "nodes": nodes, "edges": edges, "triplets": triplets,
    }


def build_sharded_cache(workers: int, config: GraphConfig) -> None:
    rows = read_csv(PROCESSED / "perovskites.csv")
    rows_by_structure = {row["structure_id"]: row for row in rows}
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(PROCESSED / "structure_index.csv"):
        grouped[row["shard"]].append(row)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tasks = [
        (shard, index_rows, {item["structure_id"]: rows_by_structure[item["structure_id"]] for item in index_rows}, asdict(config))
        for shard, index_rows in grouped.items()
    ]
    started = time.perf_counter()
    built = existing = 0
    totals = defaultdict(int)
    print(f"cache: {len(tasks):,} shards, workers={workers}", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_cache_one_shard, task) for task in tasks]
        for position, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            if result["status"] == "built":
                built += 1
                for key in ("graphs", "nodes", "edges", "triplets"):
                    totals[key] += result[key]
            else:
                existing += 1
            if position % 250 == 0 or position == len(tasks):
                rate = position / (time.perf_counter() - started)
                print(f"cache: {position:,}/{len(tasks):,} shards ({rate:.1f} shard/s)", flush=True)
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(), "config": asdict(config),
        "shards": len(tasks), "built": built, "existing": existing, "built_totals": dict(totals),
    }
    (CACHE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


class CachedGraphDataset(Dataset):
    def __init__(self, rows: Sequence[dict[str, str]], index_by_structure: dict[str, dict[str, str]]) -> None:
        self.rows = list(rows)
        self.locations = []
        for row in self.rows:
            index = index_by_structure[row["structure_id"]]
            source = Path(index["shard"])
            cache = CACHE_DIR / (source.name.removesuffix(".json.gz") + ".pt")
            self.locations.append((str(cache), int(index["offset"])))

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    @lru_cache(maxsize=32)
    def _load(path: str) -> list[dict[str, object]]:
        return torch.load(path, map_location="cpu", weights_only=False)

    def __getitem__(self, item: int) -> dict[str, object]:
        path, offset = self.locations[item]
        return self._load(path)[offset]


class ShardBatchSampler(Sampler[list[int]]):
    """Shuffle shard order and samples while keeping nearby reads in the same cache shard."""
    def __init__(self, dataset: CachedGraphDataset, batch_size: int, seed: int, shuffle: bool) -> None:
        self.batch_size, self.seed, self.shuffle, self.epoch = batch_size, seed, shuffle, 0
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, (path, _) in enumerate(dataset.locations):
            grouped[path].append(index)
        self.groups = list(grouped.values())

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return math.ceil(sum(map(len, self.groups)) / self.batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        groups = [group.copy() for group in self.groups]
        if self.shuffle:
            rng.shuffle(groups)
            for group in groups:
                rng.shuffle(group)
        ordered = [item for group in groups for item in group]
        for start in range(0, len(ordered), self.batch_size):
            yield ordered[start:start + self.batch_size]


class GpuMonitor:
    def __init__(self, output: Path, interval: float = 5.0) -> None:
        self.output, self.interval = output, interval
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text(
            "timestamp,index,name,utilization_gpu_pct,memory_used_mib,memory_total_mib,temperature_c,power_w\n",
            encoding="utf-8",
        )
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        query = "index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
        while not self.stop_event.is_set():
            try:
                result = subprocess.run(
                    ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5, check=True,
                )
                timestamp = datetime.now().astimezone().isoformat()
                with self.output.open("a", encoding="utf-8") as handle:
                    for line in result.stdout.strip().splitlines():
                        handle.write(f"{timestamp},{line}\n")
            except Exception as error:
                print(f"gpu monitor warning: {error}", file=sys.stderr, flush=True)
            self.stop_event.wait(self.interval)

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=self.interval + 2)


def move_batch(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def metrics(target: np.ndarray, prediction: np.ndarray, threshold: float = 0.05) -> dict[str, float]:
    result = {
        "mae": float(mean_absolute_error(target, prediction)),
        "rmse": float(mean_squared_error(target, prediction) ** 0.5),
        "r2": float(r2_score(target, prediction)),
    }
    labels = target <= threshold
    result["pr_auc"] = float(average_precision_score(labels, -prediction))
    return result


@torch.inference_mode()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device, mean: float, std: float,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, list[str]]:
    model.eval()
    predictions, targets, material_ids = [], [], []
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            normalized = model(batch)
        predictions.append((normalized.float().cpu() * std + mean).numpy())
        targets.append(batch["target"].float().cpu().numpy())
        material_ids.extend(batch["material_id"])
    target = np.concatenate(targets)
    prediction = np.concatenate(predictions)
    return metrics(target, prediction), target, prediction, material_ids


def make_loader(
    dataset: CachedGraphDataset, batch_size: int, seed: int, shuffle: bool, workers: int,
) -> tuple[DataLoader, ShardBatchSampler]:
    sampler = ShardBatchSampler(dataset, batch_size, seed, shuffle)
    loader = DataLoader(
        dataset, batch_sampler=sampler, collate_fn=collate_graphs,
        num_workers=workers, pin_memory=True, persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
    )
    return loader, sampler


def train_seed(
    seed: int, train_set: CachedGraphDataset, validation_set: CachedGraphDataset,
    test_set: CachedGraphDataset, args: argparse.Namespace, device: torch.device,
) -> Path:
    seed_everything(seed)
    seed_dir = args.output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = seed_dir / "best_model.pt"
    history_path = seed_dir / "history.csv"
    targets = np.asarray([float(row["decomposition_energy_per_atom"]) for row in train_set.rows])
    target_mean, target_std = float(targets.mean()), float(targets.std())
    train_loader, train_sampler = make_loader(train_set, args.batch_size, seed, True, args.workers)
    validation_loader, _ = make_loader(validation_set, args.eval_batch_size, seed, False, args.workers)
    test_loader, _ = make_loader(test_set, args.eval_batch_size, seed, False, args.workers)
    model_config = {
        "hidden_dim": args.hidden_dim, "layers": args.layers, "cutoff": args.cutoff,
        "radial_dim": args.radial_dim, "angle_dim": args.angle_dim, "angle_scope": "typed",
    }
    model = AngleGNN(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
        fused=device.type == "cuda",
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(2, args.patience // 3), min_lr=1e-6,
    )
    criterion = nn.SmoothL1Loss(beta=0.1)
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp == "fp16")
    fields = ["epoch", "train_loss", "validation_mae", "validation_rmse", "validation_r2", "validation_pr_auc", "lr", "seconds", "peak_allocated_mib", "peak_reserved_mib"]
    history_path.write_text(",".join(fields) + "\n", encoding="utf-8")
    best_mae, bad_epochs = math.inf, 0
    print(f"seed={seed}: parameters={sum(p.numel() for p in model.parameters()):,}", flush=True)
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        train_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = total_graphs = 0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for step, batch in enumerate(train_loader, start=1):
            batch = move_batch(batch, device)
            normalized_target = (batch["target"] - target_mean) / target_std
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                prediction = model(batch)
                loss = criterion(prediction, normalized_target) / args.accumulation_steps
            scaler.scale(loss).backward()
            if step % args.accumulation_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            graphs = len(batch["target"])
            total_loss += float(loss.detach()) * args.accumulation_steps * graphs
            total_graphs += graphs
        validation, _, _, _ = evaluate(model, validation_loader, device, target_mean, target_std)
        scheduler.step(validation["mae"])
        elapsed = time.perf_counter() - started
        allocated = torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0
        reserved = torch.cuda.max_memory_reserved() / 1024**2 if device.type == "cuda" else 0
        row = {
            "epoch": epoch, "train_loss": total_loss / total_graphs,
            **{f"validation_{key}": value for key, value in validation.items()},
            "lr": optimizer.param_groups[0]["lr"], "seconds": elapsed,
            "peak_allocated_mib": allocated, "peak_reserved_mib": reserved,
        }
        with history_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writerow(row)
        print(
            f"seed={seed} epoch={epoch:03d} loss={row['train_loss']:.5f} "
            f"val_mae={validation['mae']:.6f} val_rmse={validation['rmse']:.6f} "
            f"time={elapsed:.1f}s peak={allocated:.0f}/{reserved:.0f}MiB",
            flush=True,
        )
        if validation["mae"] < best_mae - args.min_delta:
            best_mae, bad_epochs = validation["mae"], 0
            torch.save({
                "model_state_dict": model.state_dict(), "model_config": model_config,
                "seed": seed, "epoch": epoch, "validation_metrics": validation,
                "target_mean": target_mean, "target_std": target_std,
                "graph_config": asdict(GraphConfig(args.cutoff, args.max_neighbors, args.max_angle_neighbors)),
            }, checkpoint)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"seed={seed}: early stop at epoch {epoch}; best MAE={best_mae:.6f}", flush=True)
                break
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved["model_state_dict"])
    test_metrics, target, prediction, ids = evaluate(model, test_loader, device, target_mean, target_std)
    saved["test_metrics"] = test_metrics
    torch.save(saved, checkpoint)
    with (seed_dir / "test_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("material_id", "target", "prediction", "error"))
        writer.writerows((mid, y, p, p - y) for mid, y, p in zip(ids, target, prediction))
    print(f"seed={seed}: test={json.dumps(test_metrics)}", flush=True)
    del model, optimizer, train_loader, validation_loader, test_loader
    torch.cuda.empty_cache()
    return checkpoint


def ensemble_summary(checkpoints: Sequence[Path], output_dir: Path) -> None:
    rows = []
    predictions = []
    ids = target = None
    for path in checkpoints:
        saved = torch.load(path, map_location="cpu", weights_only=False)
        resolved = path.resolve()
        rows.append({"seed": saved["seed"], "best_epoch": saved["epoch"], **saved["validation_metrics"], **{f"test_{key}": value for key, value in saved["test_metrics"].items()}, "checkpoint": str(resolved.relative_to(ROOT))})
        with (path.parent / "test_predictions.csv").open(encoding="utf-8", newline="") as handle:
            data = list(csv.DictReader(handle))
        current_ids = [row["material_id"] for row in data]
        current_target = np.asarray([float(row["target"]) for row in data])
        if ids is None:
            ids, target = current_ids, current_target
        elif ids != current_ids:
            raise ValueError("test prediction order differs between seeds")
        predictions.append(np.asarray([float(row["prediction"]) for row in data]))
    with (output_dir / "model_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    matrix = np.stack(predictions)
    mean, uncertainty = matrix.mean(axis=0), matrix.std(axis=0, ddof=1)
    with (output_dir / "ensemble_test_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("material_id", "target", "ensemble_mean", "ensemble_std", "error"))
        writer.writerows((mid, y, p, u, p - y) for mid, y, p, u in zip(ids, target, mean, uncertainty))
    summary = {"seeds": [row["seed"] for row in rows], "ensemble_test_metrics": metrics(target, mean), "mean_predictive_std": float(uncertainty.mean()), "checkpoints": [row["checkpoint"] for row in rows]}
    (output_dir / "ensemble_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--skip-cache", action="store_true")
    parser.add_argument("--cache-workers", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--accumulation-steps", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--radial-dim", type=int, default=32)
    parser.add_argument("--angle-dim", type=int, default=16)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--max-neighbors", type=int, default=16)
    parser.add_argument("--max-angle-neighbors", type=int, default=8)
    parser.add_argument("--amp", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    config = GraphConfig(args.cutoff, args.max_neighbors, args.max_angle_neighbors)
    if not args.skip_cache:
        build_sharded_cache(args.cache_workers, config)
    if args.cache_only:
        return
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required for the requested training")
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_config.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n", encoding="utf-8")
    all_rows = read_csv(PROCESSED / "perovskites.csv")
    by_id = {row["material_id"]: row for row in all_rows}
    split = {row["material_id"]: row["split"] for row in read_csv(PROCESSED / "random_split.csv")}
    index = {row["structure_id"]: row for row in read_csv(PROCESSED / "structure_index.csv")}
    datasets = {
        name: CachedGraphDataset([by_id[mid] for mid, label in split.items() if label == name], index)
        for name in ("train", "validation", "test")
    }
    device = torch.device("cuda")
    monitor = GpuMonitor(args.output_dir / "gpu_monitor.csv")
    monitor.start()
    try:
        checkpoints = [train_seed(seed, datasets["train"], datasets["validation"], datasets["test"], args, device) for seed in args.seeds]
        ensemble_summary(checkpoints, args.output_dir)
    finally:
        monitor.stop()


if __name__ == "__main__":
    main()
