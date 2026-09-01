from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import torch
from pymatgen.core import Lattice, Structure


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "project"))

from graph_data import (  # noqa: E402
    GraphConfig, TRIPLET_A_O_B, TRIPLET_B_O_B, TRIPLET_O_B_O,
    collate_graphs, structure_to_graph,
)
from models import AngleGNN, CompositionBaseline, DistanceGNN  # noqa: E402


ROW = {
    "material_id": "cubic_batio3",
    "a1_element": "Ba", "a2_element": "Ba",
    "b1_element": "Ti", "b2_element": "Ti",
    "a1_oxidation_state": 2, "a2_oxidation_state": 2,
    "b1_oxidation_state": 4, "b2_oxidation_state": 4,
    "goldschmidt_t": 1.06, "bartel_tau": 3.0,
    "decomposition_energy_per_atom": 0.01,
}
CONFIG = GraphConfig(cutoff=3.6, max_neighbors=16, max_angle_neighbors=8)


def cubic_perovskite() -> Structure:
    return Structure(
        Lattice.cubic(4.0), ["Ba", "Ti", "O", "O", "O"],
        [[0.5, 0.5, 0.5], [0, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]],
    )


def graph(structure: Structure | None = None):
    return structure_to_graph(structure or cubic_perovskite(), ROW, CONFIG)


def test_periodic_graph_and_required_triplet_types() -> None:
    item = graph()
    assert item["edge_index"].shape[0] == 2
    assert item["triplet_edge_index"].shape[0] == 2
    assert item["edge_index"].shape[1] == item["distance"].shape[0]
    assert item["triplet_edge_index"].shape[1] == item["triplet_cosine"].shape[0]
    assert {TRIPLET_B_O_B, TRIPLET_O_B_O, TRIPLET_A_O_B} <= set(item["triplet_type"].tolist())
    assert torch.all(item["distance"] > 0)
    assert torch.all(item["triplet_cosine"].abs() <= 1)


def test_rigid_rotation_preserves_invariant_graph_features() -> None:
    original = cubic_perovskite()
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rotated = Structure(
        Lattice(np.asarray(original.lattice.matrix) @ rotation.T),
        original.species, np.asarray(original.cart_coords) @ rotation.T,
        coords_are_cartesian=True,
    )
    first, second = graph(original), graph(rotated)
    assert torch.allclose(first["distance"].sort().values, second["distance"].sort().values, atol=1e-5)
    assert torch.allclose(
        first["triplet_cosine"].sort().values,
        second["triplet_cosine"].sort().values, atol=1e-5,
    )


def test_internal_oxygen_displacement_changes_angles() -> None:
    distorted = cubic_perovskite()
    distorted.translate_sites([2], [0.0, 0.08, 0.0], frac_coords=True, to_unit_cell=True)
    original_cos = graph()["triplet_cosine"].sort().values
    distorted_cos = graph(distorted)["triplet_cosine"].sort().values
    assert original_cos.shape == distorted_cos.shape
    assert not torch.allclose(original_cos, distorted_cos, atol=1e-5)


def test_batch_offsets_are_valid() -> None:
    item = graph()
    batch = collate_graphs([item, copy.deepcopy(item)])
    assert batch["composition_features"].shape == (2, 10)
    assert batch["target"].shape == (2,)
    assert int(batch["edge_index"].max()) < len(batch["z"])
    assert int(batch["triplet_edge_index"].max()) < batch["edge_index"].shape[1]
    assert batch["batch"].tolist().count(0) == len(item["z"])
    assert batch["batch"].tolist().count(1) == len(item["z"])


def test_all_models_forward_and_backward() -> None:
    batch = collate_graphs([graph(), graph()])
    models = (
        CompositionBaseline(hidden_dim=16),
        DistanceGNN(hidden_dim=16, layers=2, radial_dim=8),
        AngleGNN(hidden_dim=16, layers=2, radial_dim=8, angle_dim=6, angle_scope="typed"),
    )
    for model in models:
        prediction = model(batch)
        assert prediction.shape == (2,)
        prediction.square().mean().backward()
        assert any(parameter.grad is not None for parameter in model.parameters())


def test_angle_model_rotation_invariance_and_tilt_sensitivity() -> None:
    torch.manual_seed(7)
    model = AngleGNN(hidden_dim=16, layers=2, radial_dim=8, angle_dim=6).eval()
    original = cubic_perovskite()
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rotated = Structure(
        Lattice(np.asarray(original.lattice.matrix) @ rotation.T), original.species,
        np.asarray(original.cart_coords) @ rotation.T, coords_are_cartesian=True,
    )
    distorted = cubic_perovskite()
    distorted.translate_sites([2], [0.0, 0.08, 0.0], frac_coords=True, to_unit_cell=True)
    with torch.no_grad():
        base = model(collate_graphs([graph(original)]))
        rigid = model(collate_graphs([graph(rotated)]))
        tilted = model(collate_graphs([graph(distorted)]))
    assert torch.allclose(base, rigid, atol=1e-5)
    assert not torch.allclose(base, tilted, atol=1e-6)


def test_angle_ablation_modes_run() -> None:
    batch = collate_graphs([graph()])
    for scope in ("bob", "all", "typed"):
        output = AngleGNN(
            hidden_dim=12, layers=1, radial_dim=6, angle_dim=4, angle_scope=scope
        )(batch)
        assert output.shape == (1,)
