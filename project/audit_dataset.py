"""Audit the frozen perovskite dataset used by every project experiment."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "dataset/processed"
STRUCTURES = ROOT / "dataset/raw/structures"
DATA = PROCESSED / "perovskites.csv"
INDEX = PROCESSED / "structure_index.csv"
SPLITS = {
    "random": PROCESSED / "random_split.csv",
    "ood_a_pair": PROCESSED / "ood_a_pair_split.csv",
}
AUDIT_OUTPUT = PROCESSED / "data_audit.csv"
MANIFEST_OUTPUT = PROCESSED / "dataset_manifest.json"
EXPECTED_ROWS = 59_708
VALID_SPLITS = {"train", "validation", "test"}
NUMERIC_FIELDS = (
    "a1_oxidation_state",
    "a2_oxidation_state",
    "b1_oxidation_state",
    "b2_oxidation_state",
    "goldschmidt_t",
    "bartel_tau",
    "formation_energy_per_atom",
    "decomposition_energy_per_atom",
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add(
    results: list[dict[str, str]], category: str, metric: str,
    value: object, expected: object = "", passed: bool | None = None,
) -> None:
    results.append({
        "category": category,
        "metric": metric,
        "value": str(value),
        "expected": str(expected),
        "status": "INFO" if passed is None else ("PASS" if passed else "FAIL"),
    })


def composition_type(row: dict[str, str]) -> str:
    same_a = row["a1_element"] == row["a2_element"]
    same_b = row["b1_element"] == row["b2_element"]
    if same_a and same_b:
        return "ABO3"
    if same_a:
        return "A2BB'O6"
    if same_b:
        return "AA'B2O6"
    return "AA'BB'O6"


def audit() -> tuple[list[dict[str, str]], dict[str, object]]:
    results: list[dict[str, str]] = []
    data = read_rows(DATA)
    index = read_rows(INDEX)
    material_ids = [row["material_id"] for row in data]
    structure_ids = [row["structure_id"] for row in data]

    add(results, "dataset", "rows", len(data), EXPECTED_ROWS, len(data) == EXPECTED_ROWS)
    add(results, "dataset", "duplicate_material_id", len(data) - len(set(material_ids)), 0,
        len(data) == len(set(material_ids)))
    add(results, "dataset", "duplicate_structure_id", len(data) - len(set(structure_ids)), 0,
        len(data) == len(set(structure_ids)))
    blank_required = sum(
        not row.get(field, "").strip()
        for row in data for field in ("material_id", "structure_id", "formula", *NUMERIC_FIELDS)
    )
    add(results, "dataset", "blank_required_values", blank_required, 0, blank_required == 0)

    invalid_numeric = 0
    numeric_values: dict[str, list[float]] = defaultdict(list)
    for row in data:
        for field in NUMERIC_FIELDS:
            try:
                value = float(row[field])
                if not math.isfinite(value):
                    raise ValueError
                numeric_values[field].append(value)
            except (TypeError, ValueError):
                invalid_numeric += 1
    add(results, "dataset", "invalid_numeric_values", invalid_numeric, 0, invalid_numeric == 0)
    label_mismatch = sum(
        (row["near_stable"].lower() == "true")
        != (float(row["decomposition_energy_per_atom"]) <= 0.05)
        for row in data
    )
    non_perovskites = sum(row["is_perovskite"].lower() != "true" for row in data)
    add(results, "dataset", "near_stable_label_mismatch", label_mismatch, 0, label_mismatch == 0)
    add(results, "dataset", "non_perovskite_rows", non_perovskites, 0, non_perovskites == 0)
    for field, values in numeric_values.items():
        add(results, "range", f"{field}_min", min(values))
        add(results, "range", f"{field}_max", max(values))

    elements = sorted({row[f"{site}_element"] for row in data for site in ("a1", "a2", "b1", "b2")})
    type_counts = Counter(composition_type(row) for row in data)
    add(results, "coverage", "unique_cations", len(elements), 39, len(elements) == 39)
    add(results, "coverage", "cation_symbols", ",".join(elements))
    for name, count in sorted(type_counts.items()):
        add(results, "coverage", f"composition_type_{name}", count)

    index_ids = [row["structure_id"] for row in index]
    structure_hashes = [row["md5"] for row in index]
    add(results, "structure_index", "rows", len(index), EXPECTED_ROWS, len(index) == EXPECTED_ROWS)
    add(results, "structure_index", "duplicate_structure_id", len(index) - len(set(index_ids)), 0,
        len(index) == len(set(index_ids)))
    add(results, "structure_index", "duplicate_structure_md5",
        len(index) - len(set(structure_hashes)), 0,
        len(index) == len(set(structure_hashes)))
    missing_index = set(structure_ids) - set(index_ids)
    extra_index = set(index_ids) - set(structure_ids)
    add(results, "structure_index", "missing_dataset_structures", len(missing_index), 0,
        not missing_index)
    add(results, "structure_index", "extra_structures", len(extra_index), 0, not extra_index)

    indexed_by_shard: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in index:
        indexed_by_shard[row["shard"]].append(row)
    shard_files = sorted(STRUCTURES.glob("*.json.gz"))
    indexed_paths = {ROOT / path for path in indexed_by_shard}
    missing_shards = indexed_paths - set(shard_files)
    unindexed_shards = set(shard_files) - indexed_paths
    add(results, "structures", "shard_files", len(shard_files), len(indexed_by_shard),
        len(shard_files) == len(indexed_by_shard))
    add(results, "structures", "missing_shards", len(missing_shards), 0, not missing_shards)
    add(results, "structures", "unindexed_shards", len(unindexed_shards), 0, not unindexed_shards)

    invalid_gzip = invalid_offset = identity_mismatch = 0
    structure_records = 0
    for relative, rows in indexed_by_shard.items():
        path = ROOT / relative
        if not path.is_file():
            continue
        try:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                records = json.load(handle)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            invalid_gzip += 1
            continue
        structure_records += len(records)
        for row in rows:
            offset = int(row["offset"])
            if offset < 0 or offset >= len(records):
                invalid_offset += 1
                continue
            record = records[offset]
            if record.get("id") != row["structure_id"] or record.get("md5") != row["md5"]:
                identity_mismatch += 1
    add(results, "structures", "records", structure_records, EXPECTED_ROWS,
        structure_records == EXPECTED_ROWS)
    add(results, "structures", "invalid_gzip", invalid_gzip, 0, invalid_gzip == 0)
    add(results, "structures", "invalid_offsets", invalid_offset, 0, invalid_offset == 0)
    add(results, "structures", "id_or_md5_mismatch", identity_mismatch, 0,
        identity_mismatch == 0)

    data_id_set = set(material_ids)
    for split_name, path in SPLITS.items():
        rows = read_rows(path)
        ids = [row["material_id"] for row in rows]
        labels = Counter(row["split"] for row in rows)
        add(results, split_name, "rows", len(rows), EXPECTED_ROWS, len(rows) == EXPECTED_ROWS)
        add(results, split_name, "duplicate_material_id", len(rows) - len(set(ids)), 0,
            len(rows) == len(set(ids)))
        add(results, split_name, "missing_material_id", len(data_id_set - set(ids)), 0,
            set(ids) == data_id_set)
        add(results, split_name, "extra_material_id", len(set(ids) - data_id_set), 0,
            set(ids) == data_id_set)
        add(results, split_name, "invalid_split_labels", len(set(labels) - VALID_SPLITS), 0,
            not (set(labels) - VALID_SPLITS))
        for label in sorted(VALID_SPLITS):
            add(results, split_name, f"{label}_rows", labels[label])

    by_id = {row["material_id"]: row for row in data}
    ood_rows = read_rows(SPLITS["ood_a_pair"])
    pairs: dict[str, set[str]] = defaultdict(set)
    for row in ood_rows:
        material = by_id[row["material_id"]]
        pair = "-".join(sorted((material["a1_element"], material["a2_element"])))
        pairs[row["split"]].add(pair)
    overlap = (
        (pairs["train"] & pairs["validation"])
        | (pairs["train"] & pairs["test"])
        | (pairs["validation"] & pairs["test"])
    )
    add(results, "ood_a_pair", "a_pair_overlap", len(overlap), 0, not overlap)

    files = [DATA, INDEX, *SPLITS.values(), *shard_files]
    manifest = {
        "dataset": "MPContribs Multinary_Oxides perovskites",
        "expected_rows": EXPECTED_ROWS,
        "material_rows": len(data),
        "structure_records": structure_records,
        "structure_shards": len(shard_files),
        "unique_cations": elements,
        "composition_type_counts": dict(sorted(type_counts.items())),
        "files": {
            str(path.relative_to(ROOT)): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in files
        },
    }
    return results, manifest


def main() -> None:
    results, manifest = audit()
    with AUDIT_OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("category", "metric", "value", "expected", "status"))
        writer.writeheader()
        writer.writerows(results)
    with MANIFEST_OUTPUT.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    failures = [row for row in results if row["status"] == "FAIL"]
    print(f"checks={len(results)} failures={len(failures)}")
    print(f"wrote {AUDIT_OUTPUT.relative_to(ROOT)}")
    print(f"wrote {MANIFEST_OUTPUT.relative_to(ROOT)}")
    if failures:
        raise SystemExit("dataset audit failed")


if __name__ == "__main__":
    main()
