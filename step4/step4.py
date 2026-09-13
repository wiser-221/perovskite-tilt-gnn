"""Generate Glazer-tilt and cation-ordering candidates without PySPuDS."""

import csv, gzip, json, math, warnings, hashlib, sys
from collections import defaultdict
import torch
from pathlib import Path
import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Composition, Lattice, Structure
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
SELECTION = ROOT / "step3/step3_results.csv"
INDEX = ROOT / "dataset/processed/structure_index.csv"
ORDERINGS = ("rocksalt", "layered", "columnar")


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parent_structure(row, index):
    item = index[row["structure_id"]]
    with gzip.open(ROOT / item["shard"], "rt", encoding="utf-8") as f:
        record = json.load(f)[int(item["offset"])]
    if record["id"] != row["structure_id"] or record["md5"] != item["md5"]:
        raise RuntimeError(f"structure-index mismatch: {row['structure_id']}")
    structure = Structure.from_str(record["cif"], fmt="cif")
    # Oxidation states remain available in Step-3 metadata; remove them from
    # CIF species so source and generated files use one consistent convention.
    structure.remove_oxidation_states()
    return structure


def rotation_matrix(rx, ry, rz):
    x, y, z = np.deg2rad((rx, ry, rz))
    rxm = np.array(((1, 0, 0), (0, np.cos(x), -np.sin(x)), (0, np.sin(x), np.cos(x))))
    rym = np.array(((np.cos(y), 0, np.sin(y)), (0, 1, 0), (-np.sin(y), 0, np.cos(y))))
    rzm = np.array(((np.cos(z), -np.sin(z), 0), (np.sin(z), np.cos(z), 0), (0, 0, 1)))
    return rzm @ rym @ rxm


def signed_rotation(grid, pattern, angle):
    i, j, k = grid
    anti = (-1) ** (i + j + k)
    in_z = (-1) ** (i + j)
    rules = {
        "a0a0a0": (0, 0, 0),
        "a0a0c-": (0, 0, angle * anti),
        "a-a-a-": (angle * anti,) * 3,
        "a-a-c+": (angle * anti, angle * anti, angle * in_z),
        "a0a0c+": (0, 0, angle * in_z),
    }
    return rules[pattern]


def order_index(grid, ordering):
    i, j, k = grid
    return {"rocksalt": (i + j + k) % 2, "layered": k % 2,
            "columnar": (i + j) % 2}[ordering]


def site_species(row, site, grid, ordering):
    values = (row[f"{site}1_element"], row[f"{site}2_element"])
    return values[order_index(grid, ordering)] if values[0] != values[1] else values[0]


def build(row, a, pattern, angle, a_order, b_order):
    """Build a 40-atom 2x2x2 corner-sharing perovskite supercell."""
    lattice = Lattice.cubic(2 * a)
    grids = [(i, j, k) for i in range(2) for j in range(2) for k in range(2)]
    species, coords = [], []
    for grid in grids:
        species.append(site_species(row, "b", grid, b_order)); coords.append(a * np.array(grid))
    for grid in grids:
        species.append(site_species(row, "a", grid, a_order)); coords.append(a * (np.array(grid) + .5))
    for grid in grids:
        left = np.array(grid, dtype=float)
        left_r = rotation_matrix(*signed_rotation(grid, pattern, angle))
        for axis, direction in enumerate(np.eye(3)):
            right_grid = list(grid); right_grid[axis] += 1; right_grid = tuple(right_grid)
            right = np.array(right_grid, dtype=float)
            right_r = rotation_matrix(*signed_rotation(right_grid, pattern, angle))
            p1 = a * left + left_r @ (.5 * a * direction)
            p2 = a * right + right_r @ (-.5 * a * direction)
            species.append("O"); coords.append((p1 + p2) / 2)
    structure = Structure(lattice, species, coords, coords_are_cartesian=True, to_unit_cell=True)
    if structure.composition.element_composition.reduced_composition != Composition(row["formula"]).reduced_composition:
        raise RuntimeError(f"composition mismatch: {row['formula']} != {structure.composition}")
    return structure


def descriptors(structure, row, a):
    b_elements = {row["b1_element"], row["b2_element"]}
    lengths, angles, coordination = [], [], []
    for site in structure:
        if site.specie.symbol in b_elements:
            near = [n for n in structure.get_neighbors(site, .72 * a) if n.specie.symbol == "O"]
            coordination.append(len(near)); lengths.extend(float(n.nn_distance) for n in near)
        elif site.specie.symbol == "O":
            near = [n for n in structure.get_neighbors(site, .72 * a)
                    if n.specie.symbol in b_elements]
            near.sort(key=lambda n: n.nn_distance)
            if len(near) >= 2:
                v1, v2 = near[0].coords - site.coords, near[1].coords - site.coords
                c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                angles.append(math.degrees(math.acos(float(np.clip(c, -1, 1)))))
    if not lengths or not angles or min(coordination) != 6 or max(coordination) != 6:
        raise RuntimeError(f"invalid BO6 network; coordination={coordination}")
    distances = structure.distance_matrix.copy(); np.fill_diagonal(distances, np.inf)
    return {"actual_bob_mean_deg": np.mean(angles), "actual_bob_std_deg": np.std(angles),
            "bo_length_mean_ang": np.mean(lengths), "bo_length_std_ang": np.std(lengths),
            "minimum_distance_ang": distances.min()}



def main():
    sys.path.insert(0, str(ROOT / "project"))
    from graph_data import GraphConfig, structure_to_graph, geometry_descriptors
    rows = read_csv(ROOT / "dataset/processed/perovskites.csv")
    prior = {r["material_id"]: r["split"] for r in read_csv(ROOT / "dataset/processed/random_split.csv")}
    groups = defaultdict(list)
    for row in rows:
        if not math.isfinite(float(row["formation_energy_per_atom"])):
            raise ValueError("invalid formation label")
        groups[Composition(row["formula"]).reduced_formula].append(row)
    # Validation/test never include a composition seen by the pretrained weights.
    splits = {}
    for formula, members in groups.items():
        labels = {prior[r["material_id"]] for r in members}
        splits[formula] = "train" if "train" in labels else ("validation" if "validation" in labels else "test")
    old_used = {Composition(r["formula"]).reduced_formula for r in read_csv(SELECTION) if r["selected"] == "True"}
    eligible = [f for f in groups if splits[f] == "test" and f not in old_used]
    eligible.sort(key=lambda f: hashlib.sha256(("formation-v1:" + f).encode()).hexdigest())
    selected = eligible[:100]
    assert len(selected) == 100
    # Seventy new training compositions; fifteen validation, fifteen sealed test.
    for i, formula in enumerate(selected):
        splits[formula] = "train" if i < 70 else ("validation" if i < 85 else "test")
    index = {x["structure_id"]: x for x in read_csv(INDEX)}
    cfg = GraphConfig()
    matcher = StructureMatcher(ltol=.001, stol=.002, angle_tol=.1,
                               primitive_cell=False, scale=False, attempt_supercell=False)
    candidates, rejected = [], []
    for position, formula in enumerate(selected, 1):
        row = sorted(groups[formula], key=lambda r:r["material_id"])[0]
        parent = parent_structure(row, index)
        a = (parent.volume / (sum(s.specie.symbol == "O" for s in parent) / 3)) ** (1/3)
        variants = [("parent", 0., parent), ("cubic", 0., build(row, a, "a0a0a0", 0., "rocksalt", "rocksalt"))]
        variants += [(pattern, float(angle), build(row, a, pattern, angle, "rocksalt", "rocksalt"))
                     for pattern in ("a0a0c-", "a-a-a-") for angle in range(2, 15, 2)]
        accepted = []
        for pattern, amplitude, structure in variants:
            try:
                desc = descriptors(structure, row, a)
                if desc["minimum_distance_ang"] < 1.2:
                    raise ValueError("atomic overlap")
                if any(matcher.fit(structure, previous) for previous in accepted):
                    continue
                graph = structure_to_graph(structure, row, cfg)
                graph["target"] = torch.tensor(float(row["formation_energy_per_atom"]))
                accepted.append(structure)
                geom = geometry_descriptors(graph)
                cid = row["material_id"] + "_" + pattern + "_" + str(int(amplitude))
                # Angle task: interpolate held-out distortions of compositions that are
                # present in the query pool. Original formation rows retain the stricter
                # composition-disjoint split below.
                angle_split = ("validation" if amplitude == 8 and pattern not in ("parent","cubic")
                               else "test" if amplitude in (4,12) and pattern not in ("parent","cubic")
                               else "train")
                candidates.append({"candidate_id":cid, "material_id":row["material_id"],
                    "composition":formula, "split":angle_split, "pattern":pattern,
                    "amplitude_deg":amplitude, "parent_formation_ev_atom":float(row["formation_energy_per_atom"]),
                    "generator":"dataset_parent" if pattern=="parent" else "native_shared_oxygen_Glazer",
                    "ordering":"dataset" if pattern=="parent" else "fixed_rocksalt",
                    "structure":structure.as_dict(), "graph":graph, **{k:float(v) for k,v in desc.items()},
                    "obo_mean_deg":geom["obo_angle_deg_mean"], "volume":structure.volume, "natoms":len(structure)})
            except (RuntimeError, ValueError) as exc:
                if pattern == "parent":
                    raise RuntimeError(f"invalid parent {formula}: {exc}") from exc
                rejected.append({"composition":formula,"pattern":pattern,"amplitude":amplitude,"error":str(exc)})
        print(f"{position}/100 {formula}: {len(accepted)} structures", flush=True)
    parents = {c["material_id"] for c in candidates if c["pattern"]=="parent"}
    assert len(parents)==100
    for row in rows:
        row["formation_split"] = splits[Composition(row["formula"]).reduced_formula]
    output = {"schema":1, "target":"formation_energy_per_atom", "graph_config":cfg.__dict__,
              "rows":rows,"candidates":candidates,"rejected":rejected,
              "selection":"100 formerly test-only compositions; angle-stratified holdout; excludes old 30",
              "split_counts":{k:sum(r["formation_split"]==k for r in rows) for k in ("train","validation","test")}}
    tmp = HERE / "data.tmp"
    torch.save(output, tmp); tmp.replace(HERE / "data.pt")
    print("completed", len(candidates), output["split_counts"], flush=True)


if __name__ == "__main__":
    main()
