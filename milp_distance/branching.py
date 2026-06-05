from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .common import safe_print_error
from .graph import milp_to_bipartite_graph, model_to_bipartite_graph
from .learn2branch_policy import Learn2BranchPolicy


class BipartiteGNN(nn.Module):
    def __init__(self, var_dim: int = 6, cons_dim: int = 4, edge_dim: int = 1, hidden_dim: int = 64):
        super().__init__()
        self.var_encoder = nn.Sequential(nn.Linear(var_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.cons_encoder = nn.Sequential(nn.Linear(cons_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.edge_encoder = nn.Sequential(nn.Linear(edge_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.var_update = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.cons_update = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.scorer = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, data: Any) -> torch.Tensor:
        try:
            var_state = self.var_encoder(data.x_variables)
            cons_state = self.cons_encoder(data.x_constraints)

            if data.edge_index.numel() == 0:
                return self.scorer(var_state).squeeze(-1)

            cons_idx, var_idx = data.edge_index[0], data.edge_index[1]
            edge_state = self.edge_encoder(data.edge_attr)

            for _ in range(2):
                msg_to_var = cons_state[cons_idx] + edge_state
                agg_var = torch.zeros_like(var_state).index_add(0, var_idx, msg_to_var)
                deg_var = torch.zeros((var_state.shape[0], 1), device=var_state.device)
                deg_var = deg_var.index_add(0, var_idx, torch.ones((var_idx.shape[0], 1), device=var_state.device)).clamp_min(1.0)
                agg_var = agg_var / deg_var
                var_state = self.var_update(torch.cat([var_state, agg_var, var_state * agg_var], dim=-1))

                msg_to_cons = var_state[var_idx] + edge_state
                agg_cons = torch.zeros_like(cons_state).index_add(0, cons_idx, msg_to_cons)
                deg_cons = torch.zeros((cons_state.shape[0], 1), device=cons_state.device)
                deg_cons = deg_cons.index_add(0, cons_idx, torch.ones((cons_idx.shape[0], 1), device=cons_state.device)).clamp_min(1.0)
                agg_cons = agg_cons / deg_cons
                cons_state = self.cons_update(torch.cat([cons_state, agg_cons, cons_state * agg_cons], dim=-1))

            return self.scorer(var_state).squeeze(-1)
        except Exception as exc:
            safe_print_error("BipartiteGNN.forward", exc)
            n_vars = getattr(data, "x_variables", torch.empty((0, 6))).shape[0]
            return torch.zeros(n_vars)


try:
    import pyscipopt

    BranchruleBase = pyscipopt.Branchrule
except Exception:
    BranchruleBase = object


def scip_name_keys(name: str) -> list[str]:
    keys = [name]
    if name.startswith("t_"):
        keys.append(name[2:])
    else:
        keys.append(f"t_{name}")
    return keys


class MLBranchingRule(BranchruleBase):
    def __init__(self, gnn_model: BipartiteGNN):
        super().__init__()
        self.gnn_model = gnn_model
        self.gnn_model.eval()
        self.calls = 0
        self.branches = 0
        self.no_candidates = 0
        self.no_graph = 0
        self.no_match = 0
        self.errors = 0
        self.sample_candidate_names: list[str] = []
        self.sample_graph_names: list[str] = []

    def branchexeclp(self, allowaddcons: bool):
        try:
            import pyscipopt

            self.calls += 1
            result = self.model.getLPBranchCands()
            candidates = result[0] if isinstance(result, tuple) else result
            if not candidates:
                self.no_candidates += 1
                return {"result": pyscipopt.SCIP_RESULT.DIDNOTRUN}

            graph = model_to_bipartite_graph(self.model)
            if graph is None or graph.x_variables.shape[0] == 0:
                self.no_graph += 1
                return {"result": pyscipopt.SCIP_RESULT.DIDNOTRUN}

            with torch.no_grad():
                scores = self.gnn_model(graph)

            name_to_idx = {}
            for idx, name in enumerate(graph.var_names):
                for key in scip_name_keys(str(name)):
                    name_to_idx.setdefault(key, idx)
            best_var = None
            best_score = -float("inf")
            for var in candidates:
                idx = None
                for key in scip_name_keys(getattr(var, "name", str(var))):
                    idx = name_to_idx.get(key)
                    if idx is not None:
                        break
                if idx is None:
                    continue
                score = float(scores[idx].item())
                if score > best_score:
                    best_var = var
                    best_score = score

            if best_var is None:
                self.no_match += 1
                if not self.sample_candidate_names:
                    self.sample_candidate_names = [getattr(var, "name", str(var)) for var in candidates[:5]]
                    self.sample_graph_names = list(graph.var_names[:5])
                return {"result": pyscipopt.SCIP_RESULT.DIDNOTRUN}

            self.model.branchVar(best_var)
            self.branches += 1
            return {"result": pyscipopt.SCIP_RESULT.BRANCHED}
        except Exception as exc:
            self.errors += 1
            safe_print_error("MLBranchingRule.branchexeclp", exc)
            try:
                import pyscipopt

                return {"result": pyscipopt.SCIP_RESULT.DIDNOTRUN}
            except Exception:
                return {"result": None}


class StrongBranchingCollector(BranchruleBase):
    def __init__(self, itlim: int = 100, max_candidates: int = 64, max_samples: int = 1):
        super().__init__()
        self.itlim = itlim
        self.max_candidates = max_candidates
        self.max_samples = max_samples
        self.graph = None
        self.labels = None
        self.label_mask = None
        self.collected = False
        self.used_candidates = 0
        self.errors = 0
        self.examples: list[tuple[Any, torch.Tensor, torch.Tensor]] = []

    def branchexeclp(self, allowaddcons: bool):
        try:
            import pyscipopt

            if len(self.examples) >= self.max_samples:
                self.collected = True
                return {"result": pyscipopt.SCIP_RESULT.DIDNOTRUN}

            result = self.model.getLPBranchCands()
            candidates = result[0] if isinstance(result, tuple) else result
            if not candidates:
                return {"result": pyscipopt.SCIP_RESULT.DIDNOTRUN}

            graph = model_to_bipartite_graph(self.model)
            if graph is None or graph.x_variables.shape[0] == 0:
                return {"result": pyscipopt.SCIP_RESULT.DIDNOTRUN}

            name_to_idx = {}
            for idx, name in enumerate(graph.var_names):
                for key in scip_name_keys(str(name)):
                    name_to_idx.setdefault(key, idx)

            labels = torch.zeros(graph.x_variables.shape[0], dtype=torch.float32)
            label_mask = torch.zeros(graph.x_variables.shape[0], dtype=torch.bool)
            lp_obj = float(self.model.getLPObjVal())
            self.model.startStrongbranch()
            try:
                for var in candidates[: self.max_candidates]:
                    try:
                        idx = None
                        for key in scip_name_keys(getattr(var, "name", str(var))):
                            idx = name_to_idx.get(key)
                            if idx is not None:
                                break
                        if idx is None:
                            continue

                        sb_result = self.model.getVarStrongbranch(var, self.itlim, idempotent=True)
                        down_bound, up_bound = float(sb_result[0]), float(sb_result[1])
                        down_valid, up_valid = bool(sb_result[2]), bool(sb_result[3])
                        down_gain = max(0.0, down_bound - lp_obj) if down_valid else 0.0
                        up_gain = max(0.0, up_bound - lp_obj) if up_valid else 0.0
                        score = float(self.model.getBranchScoreMultiple(var, [down_gain, up_gain]))
                        if np.isfinite(score):
                            labels[idx] = score
                            label_mask[idx] = True
                            self.used_candidates += 1
                    except Exception as exc:
                        safe_print_error(f"strong branching candidate {getattr(var, 'name', var)}", exc)
                        continue
            finally:
                self.model.endStrongbranch()

            if self.used_candidates > 0:
                self.graph = graph
                self.labels = labels
                self.label_mask = label_mask
                self.examples.append((graph, labels, label_mask))
                if len(self.examples) >= self.max_samples:
                    self.collected = True
                    self.model.interruptSolve()
            return {"result": pyscipopt.SCIP_RESULT.DIDNOTRUN}
        except Exception as exc:
            self.errors += 1
            safe_print_error("StrongBranchingCollector.branchexeclp", exc)
            try:
                import pyscipopt

                return {"result": pyscipopt.SCIP_RESULT.DIDNOTRUN}
            except Exception:
                return {"result": None}


def standardize(values: torch.Tensor) -> torch.Tensor:
    values = torch.nan_to_num(values.float(), nan=0.0, posinf=0.0, neginf=0.0)
    std = values.std()
    if not torch.isfinite(std) or float(std) < 1e-6:
        return torch.zeros_like(values)
    return (values - values.mean()) / std.clamp_min(1e-6)


def standardize_masked(values: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    try:
        values = torch.nan_to_num(values.float(), nan=0.0, posinf=0.0, neginf=0.0)
        if mask is None or int(mask.sum()) <= 1:
            return standardize(values)

        masked = values[mask]
        std = masked.std()
        result = torch.zeros_like(values)
        if not torch.isfinite(std) or float(std) < 1e-6:
            return result
        result[mask] = (masked - masked.mean()) / std.clamp_min(1e-6)
        return result
    except Exception as exc:
        safe_print_error("standardize_masked", exc)
        return standardize(values)


def heuristic_branching_labels(graph: Any) -> torch.Tensor:
    try:
        x_vars = graph.x_variables
        obj_score = torch.abs(x_vars[:, 0])
        width_score = torch.clamp(x_vars[:, 2] - x_vars[:, 1], min=0.0)

        if graph.edge_index.numel() > 0:
            var_idx = graph.edge_index[1].cpu()
            degree = torch.zeros(x_vars.shape[0], dtype=torch.float32)
            degree = degree.index_add(0, var_idx, torch.ones(var_idx.shape[0]))
        else:
            degree = torch.ones(x_vars.shape[0], dtype=torch.float32)

        labels = 0.45 * standardize(obj_score) + 0.45 * standardize(torch.log1p(degree)) + 0.10 * standardize(width_score)
        return torch.nan_to_num(labels.float(), nan=0.0, posinf=0.0, neginf=0.0)
    except Exception as exc:
        safe_print_error("heuristic_branching_labels", exc)
        return torch.zeros(graph.x_variables.shape[0], dtype=torch.float32)


def collect_strong_branching_example(
    mps_path: str | Path,
    time_limit: int = 10,
    itlim: int = 100,
    max_candidates: int = 64,
) -> tuple[Any | None, torch.Tensor | None, torch.Tensor | None, bool]:
    examples = collect_strong_branching_examples(
        mps_path=mps_path,
        time_limit=time_limit,
        itlim=itlim,
        max_candidates=max_candidates,
        max_samples=1,
    )
    if examples:
        graph, labels, label_mask = examples[0]
        return graph, labels, label_mask, True
    return None, None, None, False


def collect_strong_branching_examples(
    mps_path: str | Path,
    time_limit: int = 10,
    itlim: int = 100,
    max_candidates: int = 64,
    max_samples: int = 1,
) -> list[tuple[Any, torch.Tensor, torch.Tensor]]:
    try:
        import pyscipopt

        scip = pyscipopt.Model()
        scip.hideOutput(True)
        scip.setParam("limits/time", float(time_limit))
        scip.readProblem(str(mps_path))
        collector = StrongBranchingCollector(itlim=itlim, max_candidates=max_candidates, max_samples=max_samples)
        scip.includeBranchrule(
            collector,
            "collect_strong_branching",
            "collect strong branching labels",
            priority=1000000,
            maxdepth=-1,
            maxbounddist=1.0,
        )
        scip.optimize()
        return collector.examples
    except Exception as exc:
        safe_print_error(f"collect_strong_branching_examples({mps_path})", exc)
        return []


def collect_training_example(mps_path: str | Path) -> tuple[Any | None, torch.Tensor | None, torch.Tensor | None, bool]:
    try:
        graph, labels, label_mask, used_strong = collect_strong_branching_example(mps_path)
        if graph is not None and labels is not None:
            return graph, labels, label_mask, used_strong

        graph = milp_to_bipartite_graph(mps_path)
        if graph is None:
            return None, None, None, False
        print(f"NOTE: using deterministic heuristic branching pseudo-labels for {Path(mps_path).name}")
        return graph, heuristic_branching_labels(graph), torch.ones(graph.x_variables.shape[0], dtype=torch.bool), False
    except Exception as exc:
        safe_print_error(f"collect_training_example({mps_path})", exc)
        return None, None, None, False


def collect_training_examples(
    mps_path: str | Path,
    max_strong_samples: int = 1,
    strong_time_limit: int = 20,
) -> list[tuple[Any, torch.Tensor, torch.Tensor, bool]]:
    try:
        strong_examples = collect_strong_branching_examples(
            mps_path,
            time_limit=strong_time_limit,
            max_samples=max_strong_samples,
        )
        if strong_examples:
            return [(graph, labels, label_mask, True) for graph, labels, label_mask in strong_examples]

        graph = milp_to_bipartite_graph(mps_path)
        if graph is None:
            return []
        print(f"NOTE: using deterministic heuristic branching pseudo-labels for {Path(mps_path).name}")
        return [
            (
                graph,
                heuristic_branching_labels(graph),
                torch.ones(graph.x_variables.shape[0], dtype=torch.bool),
                False,
            )
        ]
    except Exception as exc:
        safe_print_error(f"collect_training_examples({mps_path})", exc)
        return []


def train_gnn(
    instance_paths: list[str | Path],
    n_epochs: int = 1,
    max_strong_samples_per_instance: int = 1,
    strong_time_limit: int = 20,
    model_family: str = "local",
) -> BipartiteGNN:
    try:
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)

        if model_family == "learn2branch":
            model = Learn2BranchPolicy()
        else:
            model = BipartiteGNN()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        dataset = []
        strong_examples = 0
        heuristic_examples = 0

        for path in instance_paths:
            for graph, labels, label_mask, used_strong in collect_training_examples(
                path,
                max_strong_samples=max_strong_samples_per_instance,
                strong_time_limit=strong_time_limit,
            ):
                if graph is not None and labels is not None and labels.numel() == graph.x_variables.shape[0]:
                    if label_mask is None:
                        label_mask = torch.ones(labels.shape[0], dtype=torch.bool)
                    dataset.append((graph, standardize_masked(labels, label_mask), label_mask))
                    if used_strong:
                        strong_examples += 1
                    else:
                        heuristic_examples += 1

        print(
            f"training examples: {len(dataset)}; "
            f"strong-branching labels: {strong_examples}; "
            f"heuristic pseudo-label examples: {heuristic_examples}"
        )
        if not dataset:
            return model

        for epoch in range(n_epochs):
            losses = []
            random.shuffle(dataset)
            for graph, labels, label_mask in dataset:
                try:
                    optimizer.zero_grad()
                    predictions = torch.clamp(model(graph).float(), min=-20.0, max=20.0)
                    if label_mask is not None and bool(label_mask.any()):
                        loss = F.smooth_l1_loss(predictions[label_mask], labels[label_mask])
                    else:
                        loss = F.smooth_l1_loss(predictions, labels)
                    if not torch.isfinite(loss):
                        continue
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    losses.append(float(loss.item()))
                except Exception as exc:
                    safe_print_error("training step", exc)
            print(f"epoch {epoch + 1}/{n_epochs}: loss={np.mean(losses) if losses else float('nan'):.4f}")

        return model
    except Exception as exc:
        safe_print_error("train_gnn", exc)
        return BipartiteGNN()
