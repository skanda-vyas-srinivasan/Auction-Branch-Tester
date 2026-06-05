from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .common import safe_print_error, variable_name, variable_obj


def bucket_variable(var: Any) -> str:
    try:
        vtype = str(var.vtype()).upper()
        lb = float(var.getLbGlobal())
        ub = float(var.getUbGlobal())
        if "BINARY" in vtype or (("INTEGER" in vtype or "IMPLINT" in vtype) and lb == 0.0 and ub == 1.0):
            return "B"
        if "INTEGER" in vtype or "IMPLINT" in vtype:
            return "I"
        return "C"
    except Exception as exc:
        safe_print_error("bucketing variable", exc)
        return "C"


def bucket_weight(value: float) -> str:
    try:
        value = float(value)
        if math.isclose(value, -1.0, abs_tol=1e-9):
            return "-1"
        if math.isclose(value, 1.0, abs_tol=1e-9):
            return "1"
        return "other"
    except Exception as exc:
        safe_print_error("bucketing weight", exc)
        return "other"


def bucket_rhs(value: float) -> str:
    try:
        value = float(value)
        if math.isclose(value, 0.0, abs_tol=1e-9):
            return "0"
        if math.isclose(value, 1.0, abs_tol=1e-9):
            return "1"
        return "other"
    except Exception as exc:
        safe_print_error("bucketing RHS", exc)
        return "other"


def normalize_counter(counter: Counter) -> dict[Any, float]:
    try:
        total = float(sum(counter.values()))
        if total <= 0:
            return {}
        return {key: value / total for key, value in counter.items()}
    except Exception as exc:
        safe_print_error("normalize_counter", exc)
        return {}


def canonical_constraint_key(constraint_rep: dict[str, Any]) -> tuple[Any, ...]:
    try:
        pairs = tuple(sorted((pair[0], pair[1], round(prop, 8)) for pair, prop in constraint_rep["pairs"].items()))
        return pairs, constraint_rep["rhs_class"]
    except Exception as exc:
        safe_print_error("canonical_constraint_key", exc)
        return tuple()


def extract_normalized_representation(mps_path: str | Path) -> dict[str, Any] | None:
    try:
        import pyscipopt

        model = pyscipopt.Model()
        model.hideOutput(True)
        model.readProblem(str(mps_path))

        variables = model.getVars()
        variable_classes = {var.name: bucket_variable(var) for var in variables}

        objective_counts = Counter()
        for var in variables:
            coeff = variable_obj(var)
            if abs(coeff) > 1e-12:
                objective_counts[(bucket_weight(coeff), variable_classes.get(var.name, "C"))] += 1

        objective_rep = {
            "pairs": normalize_counter(objective_counts),
            "rhs_class": "0",
            "proportion": 1.0,
        }

        constraints: list[dict[str, Any]] = []
        for cons in model.getConss():
            try:
                vals = model.getValsLinear(cons)
                if not vals:
                    continue

                lhs = model.getLhs(cons)
                rhs = model.getRhs(cons)
                rows = []

                # Section 3.3: represent constraints as <= rows; equality/ranged rows become two inequalities.
                if rhs < model.infinity():
                    rows.append((vals, rhs))
                if lhs > -model.infinity():
                    rows.append(({name: -coef for name, coef in vals.items()}, -lhs))

                for row_vals, row_rhs in rows:
                    pair_counts = Counter()
                    for raw_name, coef in row_vals.items():
                        if abs(float(coef)) <= 1e-12:
                            continue
                        name = variable_name(raw_name)
                        pair_counts[(bucket_weight(coef), variable_classes.get(name, "C"))] += 1

                    if pair_counts:
                        # Section 3.3: pc is the proportion of each weight-variable pair in a constraint.
                        constraints.append(
                            {
                                "pairs": normalize_counter(pair_counts),
                                "rhs_class": bucket_rhs(row_rhs),
                            }
                        )
            except Exception as exc:
                safe_print_error(f"extracting constraint in {mps_path}", exc)
                continue

        unique_counts = Counter(canonical_constraint_key(cons) for cons in constraints)
        exemplars = {}
        for cons in constraints:
            exemplars.setdefault(canonical_constraint_key(cons), cons)

        normalized_constraints = []
        denom = max(1, len(constraints))
        for key, count in unique_counts.items():
            rep = dict(exemplars[key])
            # Section 3.3: pP is the proportion of each unique constraint in the instance.
            rep["proportion"] = count / denom
            normalized_constraints.append(rep)

        return {
            "path": str(mps_path),
            "objective": objective_rep,
            "constraints": normalized_constraints,
            "n_constraints": len(constraints),
            "n_unique_constraints": len(normalized_constraints),
            "n_variables": len(variables),
        }
    except Exception as exc:
        safe_print_error(f"extract_normalized_representation({mps_path})", exc)
        return None


def pair_distance(pair1: tuple[str, str], pair2: tuple[str, str], alpha: float = 1, beta: float = 1) -> float:
    try:
        # Equation 3: weight mismatch costs alpha; variable-class mismatch costs beta.
        return (alpha if pair1[0] != pair2[0] else 0) + (beta if pair1[1] != pair2[1] else 0)
    except Exception as exc:
        safe_print_error("pair_distance", exc)
        return float("inf")


def greedy_constraint_distance(
    c1: dict[str, Any] | None,
    c2: dict[str, Any] | None,
    alpha: float = 1,
    beta: float = 1,
    gamma: float = 1,
) -> float:
    try:
        if c1 is None or c2 is None:
            return float("inf")

        supply = {pair: float(prop) for pair, prop in c1.get("pairs", {}).items()}
        demand = {pair: float(prop) for pair, prop in c2.get("pairs", {}).items()}
        distance = gamma if c1.get("rhs_class") != c2.get("rhs_class") else 0.0

        # Equation 4 greedy EMD: repeatedly match remaining mass between closest weight-variable pairs.
        while supply and demand:
            best = None
            for pair1 in supply:
                for pair2 in demand:
                    cost = pair_distance(pair1, pair2, alpha=alpha, beta=beta)
                    if best is None or cost < best[0]:
                        best = (cost, pair1, pair2)

            if best is None:
                break

            cost, pair1, pair2 = best
            moved = min(supply[pair1], demand[pair2])
            distance += cost * moved
            supply[pair1] -= moved
            demand[pair2] -= moved
            if supply[pair1] <= 1e-12:
                del supply[pair1]
            if demand[pair2] <= 1e-12:
                del demand[pair2]

        return float(distance)
    except Exception as exc:
        safe_print_error("greedy_constraint_distance", exc)
        return float("inf")


def greedy_instance_distance(rep1: dict[str, Any] | None, rep2: dict[str, Any] | None, zeta: float = 1) -> float:
    try:
        if rep1 is None or rep2 is None:
            return float("inf")

        constraints1 = list(rep1.get("constraints", []))
        constraints2 = list(rep2.get("constraints", []))
        supply = {i: float(cons.get("proportion", 0.0)) for i, cons in enumerate(constraints1)}
        demand = {j: float(cons.get("proportion", 0.0)) for j, cons in enumerate(constraints2)}
        distance = 0.0
        cache = {}

        # Equation 5 greedy EMD: repeatedly match remaining mass between closest constraints.
        while supply and demand:
            best = None
            for i in supply:
                for j in demand:
                    key = (i, j)
                    if key not in cache:
                        cache[key] = greedy_constraint_distance(constraints1[i], constraints2[j])
                    cost = cache[key]
                    if best is None or cost < best[0]:
                        best = (cost, i, j)

            if best is None:
                break

            cost, i, j = best
            moved = min(supply[i], demand[j])
            distance += cost * moved
            supply[i] -= moved
            demand[j] -= moved
            if supply[i] <= 1e-12:
                del supply[i]
            if demand[j] <= 1e-12:
                del demand[j]

        # Equation 5: add zeta-weighted objective distance.
        distance += zeta * greedy_constraint_distance(rep1.get("objective"), rep2.get("objective"))
        return float(distance)
    except Exception as exc:
        safe_print_error("greedy_instance_distance", exc)
        return float("inf")


def compute_distribution_distance(test_rep: dict[str, Any], reference_reps: list[dict[str, Any] | None]) -> float:
    try:
        distances = [greedy_instance_distance(test_rep, ref) for ref in reference_reps if ref is not None]
        distances = [distance for distance in distances if np.isfinite(distance)]
        return float(np.mean(distances)) if distances else float("inf")
    except Exception as exc:
        safe_print_error("compute_distribution_distance", exc)
        return float("inf")

