from __future__ import annotations

import numpy as np
import torch
import torch_geometric

from .common import safe_print_error


class PreNormException(Exception):
    pass


class PreNormLayer(torch.nn.Module):
    """Adapted from ds4dm/learn2branch-ecole, MIT licensed."""

    def __init__(self, n_units: int, shift: bool = True, scale: bool = True):
        super().__init__()
        self.register_buffer("shift", torch.zeros(n_units) if shift else None)
        self.register_buffer("scale", torch.ones(n_units) if scale else None)
        self.n_units = n_units
        self.waiting_updates = False
        self.received_updates = False

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        if self.waiting_updates:
            self.update_stats(input_)
            self.received_updates = True
            raise PreNormException
        if self.shift is not None:
            input_ = input_ + self.shift
        if self.scale is not None:
            input_ = input_ * self.scale
        return input_

    def start_updates(self) -> None:
        self.avg = 0
        self.var = 0
        self.m2 = 0
        self.count = 0
        self.waiting_updates = True
        self.received_updates = False

    def update_stats(self, input_: torch.Tensor) -> None:
        input_ = input_.reshape(-1, self.n_units)
        sample_avg = input_.mean(dim=0)
        sample_var = (input_ - sample_avg).pow(2).mean(dim=0)
        sample_count = np.prod(input_.size()) / self.n_units
        delta = sample_avg - self.avg
        self.m2 = self.var * self.count + sample_var * sample_count + delta**2 * self.count * sample_count / (
            self.count + sample_count
        )
        self.count += sample_count
        self.avg += delta * sample_count / self.count
        self.var = self.m2 / self.count if self.count > 0 else 1

    def stop_updates(self) -> None:
        if self.shift is not None:
            self.shift = -self.avg
        if self.scale is not None:
            self.var[self.var < 1e-8] = 1
            self.scale = 1 / torch.sqrt(self.var)
        del self.avg, self.var, self.m2, self.count
        self.waiting_updates = False


class BipartiteGraphConvolution(torch_geometric.nn.MessagePassing):
    """Directional bipartite convolution from the Learn2Branch PyG reimplementation."""

    def __init__(self, emb_size: int = 64):
        super().__init__(aggr="add")
        self.feature_module_left = torch.nn.Linear(emb_size, emb_size)
        self.feature_module_edge = torch.nn.Linear(1, emb_size, bias=False)
        self.feature_module_right = torch.nn.Linear(emb_size, emb_size, bias=False)
        self.feature_module_final = torch.nn.Sequential(
            PreNormLayer(1, shift=False),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )
        self.post_conv_module = torch.nn.Sequential(PreNormLayer(1, shift=False))
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(2 * emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

    def forward(
        self,
        left_features: torch.Tensor,
        edge_indices: torch.Tensor,
        edge_features: torch.Tensor,
        right_features: torch.Tensor,
    ) -> torch.Tensor:
        output = self.propagate(
            edge_indices,
            size=(left_features.shape[0], right_features.shape[0]),
            node_features=(left_features, right_features),
            edge_features=edge_features,
        )
        return self.output_module(torch.cat([self.post_conv_module(output), right_features], dim=-1))

    def message(self, node_features_i: torch.Tensor, node_features_j: torch.Tensor, edge_features: torch.Tensor) -> torch.Tensor:
        return self.feature_module_final(
            self.feature_module_left(node_features_i)
            + self.feature_module_edge(edge_features)
            + self.feature_module_right(node_features_j)
        )


class Learn2BranchPolicy(torch.nn.Module):
    """Learn2Branch-style GCNN adapted to our PySCIPOpt graph tensors."""

    def __init__(self, var_dim: int = 6, cons_dim: int = 4, edge_dim: int = 1, emb_size: int = 64):
        super().__init__()
        if edge_dim != 1:
            raise ValueError("Learn2BranchPolicy currently expects one scalar edge feature")

        self.cons_embedding = torch.nn.Sequential(
            PreNormLayer(cons_dim),
            torch.nn.Linear(cons_dim, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )
        self.edge_embedding = torch.nn.Sequential(PreNormLayer(edge_dim))
        self.var_embedding = torch.nn.Sequential(
            PreNormLayer(var_dim),
            torch.nn.Linear(var_dim, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )
        self.conv_v_to_c = BipartiteGraphConvolution(emb_size=emb_size)
        self.conv_c_to_v = BipartiteGraphConvolution(emb_size=emb_size)
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )

    def forward(self, data) -> torch.Tensor:
        try:
            constraint_features = self.cons_embedding(data.x_constraints)
            edge_features = self.edge_embedding(data.edge_attr)
            variable_features = self.var_embedding(data.x_variables)
            if data.edge_index.numel() > 0:
                reversed_edge_indices = torch.stack([data.edge_index[1], data.edge_index[0]], dim=0)
                constraint_features = self.conv_v_to_c(
                    variable_features,
                    reversed_edge_indices,
                    edge_features,
                    constraint_features,
                )
                variable_features = self.conv_c_to_v(
                    constraint_features,
                    data.edge_index,
                    edge_features,
                    variable_features,
                )
            return self.output_module(variable_features).squeeze(-1)
        except Exception as exc:
            safe_print_error("Learn2BranchPolicy.forward", exc)
            return torch.zeros(data.x_variables.shape[0], dtype=torch.float32)
