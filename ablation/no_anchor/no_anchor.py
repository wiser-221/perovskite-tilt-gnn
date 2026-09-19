"""Run the five-seed no-anchor ablation using the frozen split/configuration."""
from __future__ import annotations

import csv
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "transfer_learning"), str(ROOT / "project")]
import train_delta as full
import train_formation as common

OUT = Path(__file__).resolve().parent
CHECKPOINT = OUT / "no_anchor_models.pt"
LOG = OUT / "no_anchor_epochs.csv"


class NoAnchor(full.DualAngleGNN):
    """Full architecture with anchor_head removed; prediction is base + delta."""
    def __init__(self, saved):
        super().__init__(saved)
        self.anchor_head = None

    def components(self, x, r):
        hx, hr = self.encoder.encode(x), self.encoder.encode(r)
        z = torch.cat((hx - hr, abs(hx - hr), hx * hr), 1)
        zero = torch.zeros_like(z)
        delta = (self.delta_head(z) - self.delta_head(zero)).squeeze(1)
        return self.base(r), torch.zeros_like(delta), delta


def train_seed(saved, train, val, seed, device, cfg):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = NoAnchor(saved).to(device)
    encoder = [p for p in model.encoder.parameters() if p.requires_grad]
    heads = list(model.delta_head.parameters())
    opt = torch.optim.AdamW([
        {"params": encoder, "lr": cfg["dual_encoder_lr"]},
        {"params": heads, "lr": cfg["dual_head_lr"]},
    ], weight_decay=1e-5)
    dl = DataLoader(train, batch_size=cfg["dual_batch_size"], shuffle=True,
                    collate_fn=full.collate,
                    generator=torch.Generator().manual_seed(seed), num_workers=0)
    mean, std = float(saved["target_mean"]), float(saved["target_std"])
    best, best_score, bad = None, 1e9, 0
    for epoch in range(1, cfg["dual_epochs"] + 1):
        started = time.perf_counter()
        model.train()
        losses = []
        for x, r, d, _, _ in dl:
            x, r = full.move(x, device), full.move(r, device)
            d = d.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, _, delta = model.components(x, r)
                loss = nn.functional.smooth_l1_loss(delta, d / std, beta=.03)
            loss.backward()
            nn.utils.clip_grad_norm_(encoder + heads, 5.)
            opt.step()
            losses.append(float(loss.detach()))
        metric = full.evaluate(model, val, device, mean, std, cfg["dual_batch_size"])
        score = metric["delta_mae"] + .25 * metric["absolute_mae"]
        seconds = time.perf_counter() - started
        row = {"seed": seed, "epoch": epoch, "loss": float(np.mean(losses)),
               "delta_mae": metric["delta_mae"], "absolute_mae": metric["absolute_mae"],
               "seconds": seconds,
               "peak_gpu_mib": torch.cuda.max_memory_allocated() / 2**20}
        with LOG.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(row)
        print(json.dumps(row), flush=True)
        if score < best_score - 1e-5:
            best_score, bad = score, 0
            best_epoch = epoch
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= cfg["dual_patience"]:
            break
    result = {**saved, "architecture": "dual_cubic_delta_no_anchor_v1",
              "model_state_dict": best, "best_epoch": best_epoch,
              "selection_score": best_score,
              "label_source": "dataset_DFT_anchor_plus_CHGNet_delta",
              "ablation": "anchor_head_removed"}
    del model
    torch.cuda.empty_cache()
    return result


@torch.inference_mode()
def evaluate_reference(model, dataset, device, mean, std, batch_size):
    model.eval()
    errors, corrections = [], []
    dl = DataLoader(dataset, batch_size=batch_size, collate_fn=full.collate)
    for x, r, _, _, y_ref in dl:
        x, r = full.move(x, device), full.move(r, device)
        base, anchor, _ = model.components(x, r)
        reference_pred = ((base + anchor).float() * std + mean).cpu().numpy()
        errors.extend(np.abs(reference_pred - y_ref.numpy()).tolist())
        corrections.extend((anchor.float() * std).cpu().numpy().tolist())
    return float(np.mean(errors)), np.asarray(corrections)


def main():
    device = common.setup()
    cfg = common.config()
    data = common.load_data()
    proxy = __import__("label_and_active").proxy_labels(data)
    with sqlite3.connect(common.DB) as db:
        raw = dict(db.execute("select id,energy from labels"))
    from collections import defaultdict
    by_comp = defaultdict(list)
    for item in data["candidates"]:
        by_comp[item["composition"]].append(item)
    refs = {}
    for composition, items in by_comp.items():
        exact = [x for x in items if x["pattern"] == "cubic"]
        generated = [x for x in items if x["pattern"] != "parent"]
        refs[composition] = exact[0] if exact else min(
            generated, key=lambda x: (x["amplitude_deg"], x["candidate_id"]))
    train_items = [x for x in data["candidates"]
                   if x["split"] == "train" and x["pattern"] not in ("parent", "cubic")]
    val_items = [x for x in data["candidates"] if x["split"] == "validation"]
    test_items = [x for x in data["candidates"] if x["split"] == "test"]
    train = full.PairSet(train_items, refs, proxy, raw)
    val = full.PairSet(val_items, refs, proxy, raw)
    test = full.PairSet(test_items, refs, proxy, raw)
    base = torch.load(ROOT / "transfer_learning/base_models.pt", map_location="cpu",
                      weights_only=False)
    bundle = torch.load(CHECKPOINT, map_location="cpu", weights_only=False) if CHECKPOINT.exists() else {}
    for seed in cfg["seeds"]:
        if seed not in bundle:
            bundle[seed] = train_seed(base[seed], train, val, seed, device, cfg)
            common.save_atomic(bundle, CHECKPOINT)
    predictions, deltas, per_seed_metrics = [], [], {}
    no_anchor_ref_mae, full_ref_mae, anchor_values = [], [], []
    full_bundle = torch.load(ROOT / "transfer_learning/final_models.pt",
                             map_location="cpu", weights_only=False)
    for seed in cfg["seeds"]:
        model = NoAnchor(base[seed]).to(device)
        model.load_state_dict(bundle[seed]["model_state_dict"])
        metric = full.evaluate(model, test, device, base[seed]["target_mean"],
                               base[seed]["target_std"], cfg["dual_batch_size"])
        predictions.append(metric["pred"])
        deltas.append(metric["delta_pred"])
        per_seed_metrics[str(seed)] = {
            "absolute_mae_eV_atom": metric["absolute_mae"],
            "delta_mae_eV_atom": metric["delta_mae"],
        }
        reference_mae, correction = evaluate_reference(
            model, test, device, base[seed]["target_mean"], base[seed]["target_std"],
            cfg["dual_batch_size"])
        no_anchor_ref_mae.append(reference_mae)
        del model
        full_model = full.DualAngleGNN(base[seed]).to(device)
        full_model.load_state_dict(full_bundle[seed]["model_state_dict"])
        full_reference_mae, full_correction = evaluate_reference(
            full_model, test, device, base[seed]["target_mean"], base[seed]["target_std"],
            cfg["dual_batch_size"])
        full_ref_mae.append(full_reference_mae)
        anchor_values.extend(full_correction.tolist())
        del full_model
    truth, delta_truth = metric["truth"], metric["delta_true"]
    pred, delta_pred = np.asarray(predictions), np.asarray(deltas)
    result = {
        "experiment": "no_anchor_head",
        "seeds": cfg["seeds"],
        "train_pairs": len(train), "validation_pairs": len(val), "test_pairs": len(test),
        "ensemble_test_absolute_mae_eV_atom": float(abs(pred.mean(0) - truth).mean()),
        "ensemble_test_delta_mae_eV_atom": float(abs(delta_pred.mean(0) - delta_truth).mean()),
        "no_anchor_reference_energy_mae_eV_atom_mean_over_seeds": float(np.mean(no_anchor_ref_mae)),
        "full_reference_energy_mae_eV_atom_mean_over_seeds": float(np.mean(full_ref_mae)),
        "full_anchor_correction_eV_atom_mean": float(np.mean(anchor_values)),
        "full_anchor_correction_eV_atom_std": float(np.std(anchor_values)),
        "per_seed_test_metrics": per_seed_metrics,
        "per_seed_best_epoch": {str(s): bundle[s]["best_epoch"] for s in cfg["seeds"]},
        "parameters_total_no_anchor": sum(p.numel() for p in NoAnchor(base[cfg["seeds"][0]]).parameters()),
        "trainable_parameters_no_anchor": sum(p.numel() for p in NoAnchor(base[cfg["seeds"][0]]).parameters() if p.requires_grad),
    }
    (OUT / "no_anchor_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
