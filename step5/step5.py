"""Step 5: infer Step-4 CIF candidates with the five trained AngleGNN seeds."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "step4/candidate_manifest.csv"
METADATA = ROOT / "step3/step3_results.csv"
CHECKPOINTS = sorted((ROOT / "outputs/angle_ensemble").glob("seed_*/best_model.pt"))
OUT = Path(__file__).resolve().parent

import sys
sys.path.insert(0, str(ROOT / "project"))
from graph_data import GraphConfig, collate_graphs, structure_to_graph  # noqa: E402
from models import AngleGNN  # noqa: E402
from pymatgen.core import Structure  # noqa: E402


def load_csv(path: Path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def load_ensemble(device):
    if len(CHECKPOINTS) != 5:
        raise RuntimeError(f"expected five checkpoints, found {len(CHECKPOINTS)}")
    models, config = [], None
    for path in CHECKPOINTS:
        saved = torch.load(path, map_location=device, weights_only=False)
        current = GraphConfig(**saved["graph_config"])
        if config is None:
            config = current
        elif current != config:
            raise RuntimeError("ensemble checkpoints have different graph configurations")
        model = AngleGNN(**saved["model_config"]).to(device).eval()
        model.load_state_dict(saved["model_state_dict"])
        models.append((int(saved["seed"]), model, float(saved["target_mean"]), float(saved["target_std"])))
    return models, config


def infer(rows, metadata, models, config, device, batch_size=64):
    result = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            part = rows[start:start + batch_size]
            graphs = []
            for item in part:
                structure = Structure.from_file(ROOT / item["structure_path"])
                row = metadata[item["material_id"]]
                graph = structure_to_graph(structure, row, config)
                graphs.append(graph)
            batch = collate_graphs(graphs)
            tensors = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            predictions = []
            for _, model, mean, std in models:
                normalized = model(tensors)
                predictions.append((normalized.float().cpu() * std + mean).numpy())
            for index, item in enumerate(part):
                values = [float(prediction[index]) for prediction in predictions]
                enriched = dict(item)
                for (seed, _, _, _), value in zip(models, values):
                    enriched[f"prediction_seed_{seed}"] = value
                enriched["ensemble_mean"] = sum(values) / len(values)
                enriched["ensemble_std"] = float(torch.tensor(values).std(unbiased=True))
                enriched["parent_decomposition_energy_per_atom"] = float(
                    metadata[item["material_id"]]["decomposition_energy_per_atom"]
                )
                result.append(enriched)
            print(f"predicted {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    return result


def choose_dft(rows):
    """For each development composition select parent, top-1, uncertain and random CIF."""
    selected = []
    for material_id in sorted({row["material_id"] for row in rows}):
        group = [row for row in rows if row["material_id"] == material_id]
        if group[0]["research_role"] != "development":
            continue
        by_id = {row["candidate_id"]: row for row in group}
        parent = next(row for row in group if row["tilt_pattern"] == "original")
        available = [row for row in group if row["candidate_id"] != parent["candidate_id"]]
        top = min(available, key=lambda row: (row["ensemble_mean"], row["candidate_id"]))
        uncertain = max(
            (row for row in available if row["candidate_id"] not in {top["candidate_id"]}),
            key=lambda row: (row["ensemble_std"], row["candidate_id"]),
        )
        remaining = [row for row in available if row["candidate_id"] not in {top["candidate_id"], uncertain["candidate_id"]}]
        random_row = min(remaining, key=lambda row: hashlib.sha256(row["candidate_id"].encode()).hexdigest())
        for role, row in (("parent_reference", parent), ("gnn_lowest_energy", top),
                          ("gnn_highest_uncertainty", uncertain), ("random_control", random_row)):
            item = dict(row); item["dft_selection_role"] = role; selected.append(item)
    return selected


def write(path, rows):
    fields = ["candidate_id", "material_id", "composition", "parent_structure_id", "research_role",
              "selection_category", "dft_selection_role", "tilt_pattern", "nominal_tilt_deg",
              "A_site_ordering", "B_site_ordering", "space_group", "structure_path",
              "ensemble_mean", "ensemble_std", "parent_decomposition_energy_per_atom",
              "actual_bob_mean_deg", "actual_bob_std_deg", "bo_length_mean_ang", "bo_length_std_ang",
              "minimum_distance_ang"]
    seed_fields = sorted({key for row in rows for key in row if key.startswith("prediction_seed_")})
    fields += seed_fields
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidates = load_csv(MANIFEST)
    metadata = {row["material_id"]: row for row in load_csv(METADATA)}
    models, config = load_ensemble(device)
    predicted = infer(candidates, metadata, models, config, device)
    write(OUT / "candidate_predictions.csv", predicted)
    dft = choose_dft(predicted)
    write(OUT / "round1_dft_selection.csv", dft)
    print(f"device={device}; candidates={len(predicted)}; dft_selection={len(dft)}")


if __name__ == "__main__":
    main()
