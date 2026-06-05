from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data

from .common import finite_or_zero, safe_print_error, variable_name, variable_obj
from .distance import bucket_variable


def encode_variable_type(var: Any) -> list[float]:
    try:
        cls = bucket_variable(var)
        return {
            "B": [1.0, 0.0, 0.0],
            "I": [0.0, 1.0, 0.0],
            "C": [0.0, 0.0, 1.0],
        }.get(cls, [0.0, 0.0, 1.0])
    except Exception as exc:
        safe_print_error("encoding variable type", exc)
        return [0.0, 0.0, 1.0]


def scale_signed(value: float, cap: float = 1e6) -> float:
    value = finite_or_zero(value)
    value = max(-cap, min(cap, value))
    return float(np.sign(value) * np.log1p(abs(value)))


def model_to_bipartite_graph(model: Any) -> Data | None:
    try:
        variables = model.getVars()
        constraints = model.getConss()
        var_index = {var.name: idx for idx, var in enumerate(variables)}

        variable_features = []
        for var in variables:
            try:
                obj = scale_signed(variable_obj(var))
                lb = scale_signed(var.getLbGlobal())
                ub = scale_signed(var.getUbGlobal())
                variable_features.append([obj, lb, ub] + encode_variable_type(var))
            except Exception as exc:
                safe_print_error(f"variable features for {var}", exc)
                variable_features.append([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

        constraint_features = []
        edges = []
        edge_attr = []
        skipped_constraints = 0

        for cons in constraints:
            try:
                vals = model.getValsLinear(cons)
            except Exception:
                skipped_constraints += 1
                continue

            try:
                constraint_idx = len(constraint_features)
                lhs = scale_signed(model.getLhs(cons))
                rhs = scale_signed(model.getRhs(cons))
                has_lhs = float(model.getLhs(cons) > -model.infinity())
                has_rhs = float(model.getRhs(cons) < model.infinity())
                constraint_features.append([lhs, rhs, has_lhs, has_rhs])

                for raw_name, coef in vals.items():
                    name = variable_name(raw_name)
                    if name in var_index and abs(float(coef)) > 1e-12:
                        edges.append([constraint_idx, var_index[name]])
                        edge_attr.append([scale_signed(coef)])
            except Exception as exc:
                safe_print_error(f"constraint features for {cons}", exc)
                continue

        if not constraint_features:
            constraint_features = [[0.0, 0.0, 0.0, 1.0]]

        graph = Data(
            x_constraints=torch.tensor(constraint_features, dtype=torch.float32),
            x_variables=torch.tensor(variable_features, dtype=torch.float32),
            edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float32) if edge_attr else torch.empty((0, 1), dtype=torch.float32),
            var_names=[var.name for var in variables],
            skipped_constraints=skipped_constraints,
        )
        return graph
    except Exception as exc:
        safe_print_error("model_to_bipartite_graph", exc)
        return None


def milp_to_bipartite_graph(mps_path: str | Path) -> Data | None:
    try:
        import pyscipopt

        model = pyscipopt.Model()
        model.hideOutput(True)
        model.readProblem(str(mps_path))
        return model_to_bipartite_graph(model)
    except Exception as exc:
        safe_print_error(f"milp_to_bipartite_graph({mps_path})", exc)
        return None

