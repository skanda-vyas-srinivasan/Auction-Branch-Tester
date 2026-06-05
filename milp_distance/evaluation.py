from __future__ import annotations

from pathlib import Path
from typing import Any

from .branching import MLBranchingRule
from .common import safe_print_error


def evaluate_branching(model: Any, mps_path: str | Path, time_limit: int = 60) -> dict[str, float | int] | None:
    try:
        import pyscipopt

        def run_default() -> int | None:
            try:
                scip = pyscipopt.Model()
                scip.hideOutput(True)
                scip.setParam("limits/time", float(time_limit))
                scip.readProblem(str(mps_path))
                scip.optimize()
                return int(scip.getNNodes())
            except Exception as exc:
                safe_print_error(f"default SCIP solve {mps_path}", exc)
                return None

        def run_ml() -> int | None:
            try:
                scip = pyscipopt.Model()
                scip.hideOutput(True)
                scip.setParam("limits/time", float(time_limit))
                scip.readProblem(str(mps_path))
                rule = MLBranchingRule(model)
                scip.includeBranchrule(rule, "ml_branching", "GNN branching rule", priority=1000000, maxdepth=-1, maxbounddist=1.0)
                scip.optimize()
                return int(scip.getNNodes())
            except Exception as exc:
                safe_print_error(f"ML SCIP solve {mps_path}", exc)
                return None

        scip_nodes = run_default()
        ml_nodes = run_ml()
        if scip_nodes is None or ml_nodes is None:
            return None

        return {
            "ml_nodes": int(ml_nodes),
            "scip_nodes": int(scip_nodes),
            "degradation": float(ml_nodes / max(1, scip_nodes)),
        }
    except Exception as exc:
        safe_print_error(f"evaluate_branching({mps_path})", exc)
        return None

