"""Step 3: select 30 polymorph-study compositions from ensemble test results."""

import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "dataset/processed/perovskites.csv"
PRED = ROOT / "outputs/angle_ensemble/ensemble_test_predictions.csv"
OUT = Path(__file__).with_name("step3_results.csv")
SITES = ("a1", "a2", "b1", "b2")
MAGNETIC_B = {"V", "Cr", "Mn", "Fe", "Co", "Ni"}


def load(path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def elements(row):
    return {row[f"{site}_element"] for site in SITES}


def pair(row, site):
    return tuple(sorted((row[f"{site}1_element"], row[f"{site}2_element"])))


def magnetic_proxy(row):
    return "magnetic_B_proxy" if set(pair(row, "b")) & MAGNETIC_B else "other_B"


def choose(pool, selected, high_uncertainty):
    """Select six while rewarding chemistry and tolerance-factor diversity."""
    ranked = sorted(pool, key=lambda x: float(x["ensemble_std"]), reverse=high_uncertainty)
    rank = {x["material_id"]: 1 - i / max(len(ranked) - 1, 1) for i, x in enumerate(ranked)}
    used_e = set().union(*(elements(x) for x in selected)) if selected else set()
    used_a = {pair(x, "a") for x in selected}
    used_b = {pair(x, "b") for x in selected}
    used_t = {int(float(x["goldschmidt_t"]) / 0.05) for x in selected}
    used_m = {magnetic_proxy(x) for x in selected}
    chosen = []
    while len(chosen) < 6:
        def score(x):
            diversity = 0.04 * len(elements(x) - used_e)
            diversity += 0.08 * (pair(x, "a") not in used_a)
            diversity += 0.08 * (pair(x, "b") not in used_b)
            diversity += 0.05 * (int(float(x["goldschmidt_t"]) / 0.05) not in used_t)
            diversity += 0.03 * (magnetic_proxy(x) not in used_m)
            return rank[x["material_id"]] + diversity, x["material_id"]

        winner = max(ranked, key=score)
        ranked.remove(winner)
        chosen.append(winner)
        used_e.update(elements(winner))
        used_a.add(pair(winner, "a"))
        used_b.add(pair(winner, "b"))
        used_t.add(int(float(winner["goldschmidt_t"]) / 0.05))
        used_m.add(magnetic_proxy(winner))
    return chosen


def main():
    metadata = {x["material_id"]: x for x in load(META)}
    predictions = load(PRED)
    if len(predictions) != 5971:
        raise RuntimeError(f"expected 5,971 test predictions, found {len(predictions):,}")
    rows = []
    for prediction in predictions:
        row = {**metadata[prediction["material_id"]], **prediction}
        row["decomposition_energy_per_atom"] = prediction["target"]
        row.update(selected="False", selection_category="", research_role="")
        row["magnetic_proxy"] = magnetic_proxy(row)
        rows.append(row)

    categories = (
        ("high_uncertainty", lambda e: True, True),
        ("near_stable_0.05_0.10", lambda e: 0.05 < e <= 0.10, True),
        ("higher_energy_0.10_0.15", lambda e: 0.10 < e <= 0.15, True),
        ("stable_positive_control_le_0.03", lambda e: e <= 0.03, False),
        ("unstable_negative_control_gt_0.20", lambda e: e > 0.20, False),
    )
    selected, used = [], set()
    for category, eligible, high_uncertainty in categories:
        pool = [x for x in rows if x["material_id"] not in used and eligible(float(x["target"]))]
        group = choose(pool, selected, high_uncertainty)
        for row in group:
            row["selected"], row["selection_category"] = "True", category
        selected.extend(group)
        used.update(x["material_id"] for x in group)

    # Deterministically freeze two blind and four development compositions per category.
    for category, _, _ in categories:
        group = [x for x in selected if x["selection_category"] == category]
        group.sort(key=lambda x: hashlib.sha256(x["material_id"].encode()).hexdigest())
        for i, row in enumerate(group):
            row["research_role"] = "blind_test" if i < 2 else "development"

    ordered = sorted(rows, key=lambda x: (x["selected"] != "True", x["research_role"],
                                           x["selection_category"], x["material_id"]))
    fields = [
        "selected", "research_role", "selection_category", "material_id", "structure_id",
        "identifier", "formula", "a1_element", "a2_element", "b1_element", "b2_element",
        "a1_oxidation_state", "a2_oxidation_state", "b1_oxidation_state", "b2_oxidation_state",
        "decomposition_energy_per_atom", "ensemble_mean", "ensemble_std", "error",
        "goldschmidt_t", "bartel_tau", "site_group", "magnetic_proxy", "data_source",
    ]
    with OUT.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ordered)
    print(f"input={len(rows):,}; selected=30; development=20; blind_test=10")
    print(f"output={OUT}")


if __name__ == "__main__":
    main()
