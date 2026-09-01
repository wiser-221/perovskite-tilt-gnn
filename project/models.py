"""Composition, distance-GNN and angle-GNN baselines for fair comparison."""

from __future__ import annotations

import torch
from torch import nn

from graph_data import TRIPLET_B_O_B


def aggregate(values: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    output = values.new_zeros((size, values.shape[-1]))
    output.index_add_(0, index, values)
    counts = torch.bincount(index, minlength=size).clamp_min(1).to(values.dtype).unsqueeze(1)
    return output / counts


class GaussianBasis(nn.Module):
    def __init__(self, start: float, stop: float, count: int) -> None:
        super().__init__()
        centers = torch.linspace(start, stop, count)
        self.register_buffer("centers", centers)
        spacing = (stop - start) / max(count - 1, 1)
        self.gamma = 1.0 / max(spacing, 1e-6) ** 2

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.gamma * (values.unsqueeze(-1) - self.centers) ** 2)


def mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim), nn.SiLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class CompositionBaseline(nn.Module):
    def __init__(self, input_dim: int = 10, hidden_dim: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim), mlp(input_dim, hidden_dim, hidden_dim),
            nn.SiLU(), nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.network(batch["composition_features"]).squeeze(-1)


class DistanceLayer(nn.Module):
    def __init__(self, hidden_dim: int, radial_dim: int) -> None:
        super().__init__()
        self.message = mlp(2 * hidden_dim + radial_dim, hidden_dim, hidden_dim)
        self.update = mlp(2 * hidden_dim, hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self, nodes: torch.Tensor, edge_index: torch.Tensor, radial: torch.Tensor
    ) -> torch.Tensor:
        center, neighbor = edge_index
        messages = self.message(torch.cat((nodes[center], nodes[neighbor], radial), dim=-1))
        received = aggregate(messages, center, len(nodes))
        return self.norm(nodes + self.update(torch.cat((nodes, received), dim=-1)))


class DistanceGNN(nn.Module):
    def __init__(
        self, hidden_dim: int = 128, layers: int = 4,
        cutoff: float = 5.0, radial_dim: int = 32,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(119, hidden_dim, padding_idx=0)
        self.radial = GaussianBasis(0.0, cutoff, radial_dim)
        self.layers = nn.ModuleList(DistanceLayer(hidden_dim, radial_dim) for _ in range(layers))
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        nodes = self.embedding(batch["z"])
        radial = self.radial(batch["distance"])
        for layer in self.layers:
            nodes = layer(nodes, batch["edge_index"], radial)
        crystals = aggregate(nodes, batch["batch"], int(batch["batch"].max()) + 1)
        return self.head(crystals).squeeze(-1)


class AngleLayer(nn.Module):
    def __init__(self, hidden_dim: int, angle_dim: int) -> None:
        super().__init__()
        self.triplet_message = mlp(3 * hidden_dim + angle_dim, hidden_dim, hidden_dim)
        self.edge_update = mlp(2 * hidden_dim, hidden_dim, hidden_dim)
        self.node_message = mlp(hidden_dim, hidden_dim, hidden_dim)
        self.node_update = mlp(2 * hidden_dim, hidden_dim, hidden_dim)
        self.edge_norm = nn.LayerNorm(hidden_dim)
        self.node_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self, nodes: torch.Tensor, edges: torch.Tensor, edge_index: torch.Tensor,
        triplet_index: torch.Tensor, angle_basis: torch.Tensor,
        type_features: torch.Tensor, triplet_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        edge_j = triplet_index[0, triplet_mask]
        edge_k = triplet_index[1, triplet_mask]
        if edge_j.numel():
            messages = self.triplet_message(torch.cat((
                edges[edge_j] + edges[edge_k],
                torch.abs(edges[edge_j] - edges[edge_k]), angle_basis[triplet_mask],
                type_features[triplet_mask],
            ), dim=-1))
            indices = torch.cat((edge_j, edge_k))
            edge_messages = aggregate(torch.cat((messages, messages)), indices, len(edges))
        else:
            edge_messages = torch.zeros_like(edges)
        edges = self.edge_norm(edges + self.edge_update(torch.cat((edges, edge_messages), dim=-1)))
        center = edge_index[0]
        node_messages = aggregate(self.node_message(edges), center, len(nodes))
        nodes = self.node_norm(nodes + self.node_update(torch.cat((nodes, node_messages), dim=-1)))
        return nodes, edges


class AngleGNN(nn.Module):
    """Angle scope: ``bob``, ``all`` or ``typed`` (the full model)."""

    def __init__(
        self, hidden_dim: int = 128, layers: int = 4, cutoff: float = 5.0,
        radial_dim: int = 32, angle_dim: int = 16, angle_scope: str = "typed",
    ) -> None:
        super().__init__()
        if angle_scope not in {"bob", "all", "typed"}:
            raise ValueError("angle_scope must be bob, all or typed")
        self.angle_scope = angle_scope
        self.embedding = nn.Embedding(119, hidden_dim, padding_idx=0)
        self.radial = GaussianBasis(0.0, cutoff, radial_dim)
        self.angular = GaussianBasis(-1.0, 1.0, angle_dim)
        self.triplet_type_embedding = nn.Embedding(4, hidden_dim)
        self.edge_embedding = mlp(2 * hidden_dim + radial_dim, hidden_dim, hidden_dim)
        self.layers = nn.ModuleList(AngleLayer(hidden_dim, angle_dim) for _ in range(layers))
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        nodes = self.embedding(batch["z"])
        center, neighbor = batch["edge_index"]
        radial = self.radial(batch["distance"])
        edges = self.edge_embedding(torch.cat((nodes[center], nodes[neighbor], radial), dim=-1))
        angle_basis = self.angular(batch["triplet_cosine"])
        if self.angle_scope == "typed":
            type_features = self.triplet_type_embedding(batch["triplet_type"])
        else:
            type_features = torch.zeros(
                (len(batch["triplet_type"]), edges.shape[-1]), device=edges.device, dtype=edges.dtype
            )
        mask = (
            batch["triplet_type"] == TRIPLET_B_O_B
            if self.angle_scope == "bob"
            else torch.ones_like(batch["triplet_type"], dtype=torch.bool)
        )
        for layer in self.layers:
            nodes, edges = layer(
                nodes, edges, batch["edge_index"], batch["triplet_edge_index"],
                angle_basis, type_features, mask,
            )
        crystals = aggregate(nodes, batch["batch"], int(batch["batch"].max()) + 1)
        return self.head(crystals).squeeze(-1)
