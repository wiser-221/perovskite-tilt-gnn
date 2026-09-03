"""Generate Glazer-tilt and cation-ordering candidates without PySPuDS."""

import csv, gzip, json, math, warnings
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


def specs(row):
    """Nine non-degenerate angle cells plus one independent ordering control."""
    result = [("a0a0a0", 0., "rocksalt", "rocksalt")]
    result += [(pattern, angle, "rocksalt", "rocksalt")
               for pattern in ("a0a0c-", "a-a-a-", "a-a-c+", "a0a0c+")
               for angle in (6., 12.)]
    # a0a0a0 has no meaningful angle value; use its tenth cell for a
    # separately labelled cation-ordering control instead of a duplicate CIF.
    if row["a1_element"] != row["a2_element"]:
        result.append(("a0a0a0_ordering_control", 0., "layered", "rocksalt"))
    else:
        result.append(("a0a0a0_ordering_control", 0., "rocksalt", "layered"))
    return result


def save(structure, row, number, pattern, angle, a_order, b_order, geometry, folder):
    cid = f"{row['material_id']}_p{number:02d}"
    path = folder / f"{cid}.cif"
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Site labels are not unique.*")
        CifWriter(structure).write_file(path)
    try: sg = SpacegroupAnalyzer(structure, symprec=.1).get_space_group_symbol()
    except Exception: sg = "undetermined"
    return {"candidate_id": cid, "material_id": row["material_id"], "composition": row["formula"],
            "parent_structure_id": row["structure_id"], "research_role": row["research_role"],
            "selection_category": row["selection_category"], "tilt_pattern": pattern,
            "nominal_tilt_deg": angle, "A_site_ordering": a_order, "B_site_ordering": b_order,
            "space_group": sg, "generation_method": "MP parent" if pattern == "original" else "native Glazer generator v1",
            **geometry, "structure_path": str(path.relative_to(ROOT))}


def main():
    rows = [x for x in read_csv(SELECTION) if x["selected"] == "True"]
    if len(rows) != 30: raise RuntimeError(f"expected 30 selected compositions, found {len(rows)}")
    index = {x["structure_id"]: x for x in read_csv(INDEX)}
    folder = HERE / "candidates"; folder.mkdir(exist_ok=True)
    for path in folder.glob("*.cif"): path.unlink()
    # Tight thresholds are intentional: 6° and 12° are distinct hypotheses and
    # must not be merged by the looser defaults commonly used for relaxed cells.
    matcher = StructureMatcher(ltol=.03, stol=.04, angle_tol=1, primitive_cell=False,
                               scale=True, attempt_supercell=False)
    manifest = []
    for position, row in enumerate(rows, 1):
        parent = parent_structure(row, index)
        oxygen = sum(s.specie.symbol == "O" for s in parent)
        a = (parent.volume / (oxygen / 3)) ** (1 / 3)
        local = [save(parent, row, 1, "original", "original", "original", "original",
                      descriptors(parent, row, a), folder)]
        generated = []
        for pattern, angle, a_order, b_order in specs(row):
            if len(local) == 11: break
            build_pattern = "a0a0a0" if pattern.endswith("_ordering_control") else pattern
            structure = build(row, a, build_pattern, angle, a_order, b_order)
            geometry = descriptors(structure, row, a)
            if geometry["minimum_distance_ang"] < 1.2: continue
            if any(matcher.fit(structure, old) for old in generated): continue
            generated.append(structure)
            local.append(save(structure, row, len(local) + 1, pattern, angle,
                              a_order, b_order, geometry, folder))
        if len(local) != 11: raise RuntimeError(f"{row['formula']} produced only {len(local)} candidates")
        manifest.extend(local); print(f"[{position:02d}/30] {row['formula']}: 10 candidates")
    fields = ["candidate_id", "material_id", "composition", "parent_structure_id", "research_role",
              "selection_category", "tilt_pattern", "nominal_tilt_deg", "A_site_ordering",
              "B_site_ordering", "space_group", "generation_method", "actual_bob_mean_deg",
              "actual_bob_std_deg", "bo_length_mean_ang", "bo_length_std_ang",
              "minimum_distance_ang", "structure_path"]
    with (HERE / "candidate_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(manifest)
    print(f"wrote {len(manifest)} candidates")


if __name__ == "__main__": main()
