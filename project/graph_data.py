"""Periodic crystal graphs with distance edges and invariant three-body angles."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from pymatgen.core import Element, Structure
from torch.utils.data import Dataset


ROLE_O, ROLE_A, ROLE_B = 0, 1, 2
TRIPLET_OTHER, TRIPLET_B_O_B, TRIPLET_O_B_O, TRIPLET_A_O_B = 0, 1, 2, 3
SITES = ("a1", "a2", "b1", "b2")


@dataclass(frozen=True)
class GraphConfig:
    cutoff: float = 5.0
    max_neighbors: int = 16
    max_angle_neighbors: int = 8


def site_roles(structure: Structure, row: dict[str, object]) -> torch.Tensor:
    a_elements = {str(row["a1_element"]), str(row["a2_element"])}
    b_elements = {str(row["b1_element"]), str(row["b2_element"])}
    if a_elements & b_elements:
        raise ValueError("A/B element sets overlap; site roles are ambiguous")
    roles = []
    for site in structure:
        symbol = site.specie.symbol
        if symbol == "O":
            roles.append(ROLE_O)
        elif symbol in a_elements:
            roles.append(ROLE_A)
        elif symbol in b_elements:
            roles.append(ROLE_B)
        else:
            raise ValueError(f"unexpected element {symbol} in {row.get('material_id', '')}")
    return torch.tensor(roles, dtype=torch.long)


def composition_features(row: dict[str, object]) -> torch.Tensor:
    values: list[float] = []
    for site in SITES:
        values.extend((
            float(Element(str(row[f"{site}_element"])).Z),
            float(row[f"{site}_oxidation_state"]),
        ))
    values.extend((float(row["goldschmidt_t"]), float(row["bartel_tau"])))
    return torch.tensor(values, dtype=torch.float32)


def _triplet_type(center_role: int, role_j: int, role_k: int) -> int:
    neighbors = {role_j, role_k}
    if center_role == ROLE_O and role_j == ROLE_B and role_k == ROLE_B:
        return TRIPLET_B_O_B
    if center_role == ROLE_B and role_j == ROLE_O and role_k == ROLE_O:
        return TRIPLET_O_B_O
    if center_role == ROLE_O and neighbors == {ROLE_A, ROLE_B}:
        return TRIPLET_A_O_B
    return TRIPLET_OTHER


def structure_to_graph(
    structure: Structure, row: dict[str, object], config: GraphConfig = GraphConfig()
) -> dict[str, object]:
    if not structure.is_ordered:
        raise ValueError("partially occupied structures are not supported")
    center, neighbor, images, distances = structure.get_neighbor_list(config.cutoff)
    kept: list[int] = []
    for atom in range(len(structure)):
        candidates = np.flatnonzero(center == atom)
        if not len(candidates):
            raise ValueError(f"atom {atom} has no neighbor within {config.cutoff} Å")
        local_images = images[candidates]
        order = candidates[np.lexsort((
            local_images[:, 2], local_images[:, 1], local_images[:, 0],
            neighbor[candidates], np.round(distances[candidates], decimals=8),
        ))]
        kept.extend(order[: config.max_neighbors].tolist())
    kept_array = np.asarray(kept, dtype=int)
    center = center[kept_array]
    neighbor = neighbor[kept_array]
    images = images[kept_array]
    distances = distances[kept_array]
    fractional = structure.frac_coords
    vectors = (fractional[neighbor] + images - fractional[center]) @ structure.lattice.matrix

    roles = site_roles(structure, row)
    incoming: dict[int, list[int]] = defaultdict(list)
    for edge, atom in enumerate(center.tolist()):
        incoming[atom].append(edge)
    triplet_edges: list[tuple[int, int]] = []
    triplet_cosines: list[float] = []
    triplet_types: list[int] = []
    for atom, edges in incoming.items():
        angle_edges = sorted(edges, key=lambda edge: distances[edge])[: config.max_angle_neighbors]
        for position, edge_j in enumerate(angle_edges):
            for edge_k in angle_edges[position + 1:]:
                vector_j, vector_k = vectors[edge_j], vectors[edge_k]
                cosine = float(np.dot(vector_j, vector_k) / (distances[edge_j] * distances[edge_k]))
                cosine = float(np.clip(cosine, -1.0, 1.0))
                triplet_edges.append((edge_j, edge_k))
                triplet_cosines.append(cosine)
                triplet_types.append(_triplet_type(
                    int(roles[atom]), int(roles[neighbor[edge_j]]), int(roles[neighbor[edge_k]])
                ))

    target = float(row.get("decomposition_energy_per_atom", 0.0))
    return {
        "z": torch.tensor([site.specie.Z for site in structure], dtype=torch.long),
        "role": roles,
        "edge_index": torch.tensor(np.stack((center, neighbor)), dtype=torch.long),
        "distance": torch.tensor(distances, dtype=torch.float32),
        "edge_vector": torch.tensor(vectors, dtype=torch.float32),
        "triplet_edge_index": torch.tensor(triplet_edges, dtype=torch.long).T.contiguous()
        if triplet_edges else torch.empty((2, 0), dtype=torch.long),
        "triplet_cosine": torch.tensor(triplet_cosines, dtype=torch.float32),
        "triplet_type": torch.tensor(triplet_types, dtype=torch.long),
        "composition_features": composition_features(row),
        "target": torch.tensor(target, dtype=torch.float32),
        "material_id": str(row.get("material_id", "unknown")),
    }


def geometry_descriptors(graph: dict[str, object]) -> dict[str, float]:
    distance = graph["distance"]
    edge_index = graph["edge_index"]
    roles = graph["role"]
    triplet_type = graph["triplet_type"]
    cosine = graph["triplet_cosine"]
    bo_mask = (
        ((roles[edge_index[0]] == ROLE_B) & (roles[edge_index[1]] == ROLE_O))
        | ((roles[edge_index[0]] == ROLE_O) & (roles[edge_index[1]] == ROLE_B))
    )
    bo = distance[bo_mask]
    bob = torch.rad2deg(torch.acos(cosine[triplet_type == TRIPLET_B_O_B].clamp(-1, 1)))
    obo = torch.rad2deg(torch.acos(cosine[triplet_type == TRIPLET_O_B_O].clamp(-1, 1)))

    def stats(values: torch.Tensor, prefix: str) -> dict[str, float]:
        if values.numel() == 0:
            return {f"{prefix}_mean": float("nan"), f"{prefix}_std": float("nan")}
        return {
            f"{prefix}_mean": float(values.mean()),
            f"{prefix}_std": float(values.std(unbiased=False)),
        }

    result = stats(bo, "bo_bond_length")
    result.update(stats(bob, "bob_angle_deg"))
    result.update(stats(obo, "obo_angle_deg"))
    result["bo6_bond_distortion"] = (
        float(torch.mean(torch.abs(bo - bo.mean()) / bo.mean())) if bo.numel() else float("nan")
    )
    result["bo6_angle_distortion"] = (
        float(torch.mean(torch.abs(obo - 90.0))) if obo.numel() else float("nan")
    )
    return result


@lru_cache(maxsize=32)
def load_shard(path: str) -> list[dict[str, object]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def load_indexed_structure(root: Path, index_row: dict[str, object]) -> Structure:
    shard = Path(str(index_row["shard"]))
    path = shard if shard.is_absolute() else root / shard
    record = load_shard(str(path))[int(index_row["offset"])]
    if record["id"] != index_row["structure_id"] or record["md5"] != index_row["md5"]:
        raise ValueError(f"structure index mismatch for {index_row['structure_id']}")
    return Structure.from_str(str(record["cif"]), fmt="cif")


class CrystalGraphDataset(Dataset):
    def __init__(
        self, root: Path, rows: Sequence[dict[str, object]],
        index: dict[str, dict[str, object]], config: GraphConfig = GraphConfig(),
    ) -> None:
        self.root, self.rows, self.index, self.config = root, list(rows), index, config

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, item: int) -> dict[str, object]:
        row = self.rows[item]
        index_row = self.index[str(row["structure_id"])]
        return structure_to_graph(load_indexed_structure(self.root, index_row), row, self.config)


def collate_graphs(graphs: Sequence[dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    node_offset = edge_offset = 0
    node_parts, role_parts, edge_parts, distance_parts, vector_parts = [], [], [], [], []
    triplet_parts, cosine_parts, type_parts, batch_parts = [], [], [], []
    features, targets, material_ids = [], [], []
    for graph_index, graph in enumerate(graphs):
        node_parts.append(graph["z"])
        role_parts.append(graph["role"])
        edge_parts.append(graph["edge_index"] + node_offset)
        distance_parts.append(graph["distance"])
        vector_parts.append(graph["edge_vector"])
        triplet_parts.append(graph["triplet_edge_index"] + edge_offset)
        cosine_parts.append(graph["triplet_cosine"])
        type_parts.append(graph["triplet_type"])
        batch_parts.append(torch.full((len(graph["z"]),), graph_index, dtype=torch.long))
        features.append(graph["composition_features"])
        targets.append(graph["target"])
        material_ids.append(graph["material_id"])
        node_offset += len(graph["z"])
        edge_offset += graph["edge_index"].shape[1]
    result.update({
        "z": torch.cat(node_parts),
        "role": torch.cat(role_parts),
        "edge_index": torch.cat(edge_parts, dim=1),
        "distance": torch.cat(distance_parts),
        "edge_vector": torch.cat(vector_parts),
        "triplet_edge_index": torch.cat(triplet_parts, dim=1),
        "triplet_cosine": torch.cat(cosine_parts),
        "triplet_type": torch.cat(type_parts),
        "batch": torch.cat(batch_parts),
        "composition_features": torch.stack(features),
        "target": torch.stack(targets),
        "material_id": material_ids,
    })
    return result


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_cache(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parents[1]
    rows = read_csv(root / "dataset/processed/perovskites.csv")
    if args.max_samples is not None:
        rows = rows[:args.max_samples]
    index_rows = read_csv(root / "dataset/processed/structure_index.csv")
    index = {row["structure_id"]: row for row in index_rows}
    config = GraphConfig(args.cutoff, args.max_neighbors, args.max_angle_neighbors)
    dataset = CrystalGraphDataset(root, rows, index, config)
    cache: dict[str, dict[str, object]] = {}
    descriptor_rows = []
    for position, graph in enumerate(dataset, start=1):
        material_id = str(graph["material_id"])
        cache[material_id] = graph
        descriptor_rows.append({"material_id": material_id, **geometry_descriptors(graph)})
        if position % 1000 == 0 or position == len(dataset):
            print(f"built {position:,}/{len(dataset):,}")
    args.cache_output.parent.mkdir(parents=True, exist_ok=True)
    args.descriptor_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": vars(config), "graphs": cache}, args.cache_output)
    fields = list(descriptor_rows[0]) if descriptor_rows else ["material_id"]
    with args.descriptor_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(descriptor_rows)
    print(f"wrote {args.cache_output}")
    print(f"wrote {args.descriptor_output}")


def arguments() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build angle graphs and geometry descriptors")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--max-neighbors", type=int, default=16)
    parser.add_argument("--max-angle-neighbors", type=int, default=8)
    parser.add_argument(
        "--cache-output", type=Path,
        default=root / "dataset/processed/graph_cache.pt",
    )
    parser.add_argument(
        "--descriptor-output", type=Path,
        default=root / "dataset/processed/geometry_descriptors.csv",
    )
    return parser.parse_args()


if __name__ == "__main__":
    build_cache(arguments())
