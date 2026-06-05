from __future__ import annotations

import random
from pathlib import Path
from typing import Callable

from .common import MAX_VARIABLES, count_vars, safe_print_error


GENERATED_CLASSES = ["set_partitioning", "facility_location", "multidim_knapsack", "bin_packing"]


def solve_node_count(path: str | Path, time_limit: int = 10) -> tuple[int | None, str]:
    try:
        import pyscipopt

        model = pyscipopt.Model()
        model.hideOutput(True)
        model.setParam("limits/time", float(time_limit))
        model.readProblem(str(path))
        model.optimize()
        return int(model.getNNodes()), str(model.getStatus())
    except Exception as exc:
        safe_print_error(f"default SCIP solve for generated candidate {path}", exc)
        return None, "error"


def write_set_partitioning(path: str | Path, seed: int) -> Path | None:
    try:
        import pyscipopt

        rng = random.Random(seed)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        model = pyscipopt.Model()
        model.hideOutput(True)

        n_elements = rng.randint(35, 65)
        n_sets = rng.randint(180, 430)
        columns: list[list[int]] = []
        covering_sets = [[] for _ in range(n_elements)]
        variables = []

        for set_idx in range(n_sets):
            size = rng.randint(2, min(8, n_elements))
            elems = rng.sample(range(n_elements), size)
            columns.append(elems)
            cost = rng.randint(1, 80) + 0.1 * size
            var = model.addVar(vtype="B", name=f"s_{set_idx}", obj=cost)
            variables.append(var)
            for elem in elems:
                covering_sets[elem].append(var)

        for elem, vars_for_elem in enumerate(covering_sets):
            if not vars_for_elem:
                set_idx = rng.randrange(n_sets)
                columns[set_idx].append(elem)
                vars_for_elem.append(variables[set_idx])
            # Routing/column-generation style set partitioning formulation.
            model.addCons(sum(vars_for_elem) == 1, name=f"partition_{elem}")

        model.writeProblem(str(path))
        return path
    except Exception as exc:
        safe_print_error(f"write_set_partitioning({path})", exc)
        return None


def write_facility_location(path: str | Path, seed: int) -> Path | None:
    try:
        import pyscipopt

        rng = random.Random(seed)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        model = pyscipopt.Model()
        model.hideOutput(True)

        n_facilities = rng.randint(12, 18)
        n_customers = rng.randint(20, 26)
        demands = [rng.randint(3, 12) for _ in range(n_customers)]
        total_demand = sum(demands)
        capacities = [rng.randint(max(10, total_demand // n_facilities), max(20, total_demand // 3)) for _ in range(n_facilities)]
        fixed_costs = [rng.randint(20, 120) for _ in range(n_facilities)]

        open_vars = [model.addVar(vtype="B", name=f"y_{j}", obj=fixed_costs[j]) for j in range(n_facilities)]
        assign = []
        for i in range(n_customers):
            row = []
            for j in range(n_facilities):
                transport_cost = rng.randint(1, 40)
                row.append(model.addVar(vtype="B", name=f"x_{i}_{j}", obj=transport_cost))
            assign.append(row)

        for i in range(n_customers):
            model.addCons(sum(assign[i][j] for j in range(n_facilities)) == 1, name=f"assign_{i}")
        for j in range(n_facilities):
            model.addCons(
                sum(demands[i] * assign[i][j] for i in range(n_customers)) <= capacities[j] * open_vars[j],
                name=f"capacity_{j}",
            )
            for i in range(n_customers):
                model.addCons(assign[i][j] <= open_vars[j], name=f"link_{i}_{j}")

        model.writeProblem(str(path))
        return path
    except Exception as exc:
        safe_print_error(f"write_facility_location({path})", exc)
        return None


def write_multidim_knapsack(path: str | Path, seed: int) -> Path | None:
    try:
        import pyscipopt

        rng = random.Random(seed)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        model = pyscipopt.Model()
        model.hideOutput(True)

        n_items = rng.randint(160, 320)
        n_dims = rng.randint(5, 12)
        variables = []
        weights = [[rng.randint(1, 80) for _ in range(n_items)] for _ in range(n_dims)]
        profits = [rng.randint(10, 120) for _ in range(n_items)]

        for i in range(n_items):
            variables.append(model.addVar(vtype="B", name=f"x_{i}", obj=-profits[i]))
        for dim in range(n_dims):
            capacity = int(sum(weights[dim]) * rng.uniform(0.28, 0.48))
            model.addCons(sum(weights[dim][i] * variables[i] for i in range(n_items)) <= capacity, name=f"capacity_{dim}")

        model.writeProblem(str(path))
        return path
    except Exception as exc:
        safe_print_error(f"write_multidim_knapsack({path})", exc)
        return None


def write_bin_packing(path: str | Path, seed: int) -> Path | None:
    try:
        import pyscipopt

        rng = random.Random(seed)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        model = pyscipopt.Model()
        model.hideOutput(True)

        n_items = rng.randint(24, 34)
        n_bins = rng.randint(10, 14)
        capacity = rng.randint(45, 70)
        weights = [rng.randint(max(5, capacity // 5), max(6, capacity // 2)) for _ in range(n_items)]
        used = [model.addVar(vtype="B", name=f"y_{j}", obj=1.0) for j in range(n_bins)]
        assign = [
            [model.addVar(vtype="B", name=f"x_{i}_{j}") for j in range(n_bins)]
            for i in range(n_items)
        ]

        for i in range(n_items):
            model.addCons(sum(assign[i][j] for j in range(n_bins)) == 1, name=f"assign_{i}")
        for j in range(n_bins):
            model.addCons(
                sum(weights[i] * assign[i][j] for i in range(n_items)) <= capacity * used[j],
                name=f"capacity_{j}",
            )
        for j in range(n_bins - 1):
            model.addCons(used[j] >= used[j + 1], name=f"symmetry_{j}")

        model.writeProblem(str(path))
        return path
    except Exception as exc:
        safe_print_error(f"write_bin_packing({path})", exc)
        return None


WRITERS: dict[str, Callable[[str | Path, int], Path | None]] = {
    "set_partitioning": write_set_partitioning,
    "facility_location": write_facility_location,
    "multidim_knapsack": write_multidim_knapsack,
    "bin_packing": write_bin_packing,
}


def generate_filtered_class(
    cls: str,
    dest_root: str | Path,
    target_count: int = 50,
    max_attempts: int = 500,
    seed_offset: int = 0,
    max_variables: int = MAX_VARIABLES,
    min_nodes: int = 2,
    time_limit: int = 10,
) -> list[Path]:
    kept: list[Path] = []
    try:
        dest_dir = Path(dest_root) / cls
        dest_dir.mkdir(parents=True, exist_ok=True)
        kept = sorted(dest_dir.glob("*.mps"))[:target_count]
        writer = WRITERS[cls]
        attempt = 0

        while len(kept) < target_count and attempt < max_attempts:
            seed = seed_offset + 100000 * GENERATED_CLASSES.index(cls) + attempt
            candidate = dest_dir / f"{cls}_{len(kept):03d}_candidate_{attempt:04d}.mps"
            attempt += 1

            path = writer(candidate, seed)
            if path is None:
                continue

            n_vars = count_vars(path)
            if n_vars >= max_variables:
                path.unlink(missing_ok=True)
                print(f"{cls}: reject {candidate.name} vars={n_vars}")
                continue

            nodes, status = solve_node_count(path, time_limit=time_limit)
            if nodes is None or nodes < min_nodes:
                path.unlink(missing_ok=True)
                print(f"{cls}: reject {candidate.name} vars={n_vars} nodes={nodes} status={status}")
                continue

            final_path = dest_dir / f"{cls}_{len(kept):03d}.mps"
            path.replace(final_path)
            kept.append(final_path)
            print(f"{cls}: kept {final_path.name} vars={n_vars} nodes={nodes} status={status}")

        print(f"{cls}: generated {len(kept)}/{target_count} branchable instances")
        return kept
    except Exception as exc:
        safe_print_error(f"generate_filtered_class({cls})", exc)
        return kept


def generate_filtered_dataset(
    dest_root: str | Path = "generated_instances",
    classes: list[str] | None = None,
    target_count: int = 50,
    max_attempts_per_class: int = 500,
    seed_offset: int = 0,
    min_nodes: int = 2,
    time_limit: int = 10,
) -> dict[str, list[Path]]:
    results: dict[str, list[Path]] = {}
    for cls in classes or GENERATED_CLASSES:
        try:
            results[cls] = generate_filtered_class(
                cls=cls,
                dest_root=dest_root,
                target_count=target_count,
                max_attempts=max_attempts_per_class,
                seed_offset=seed_offset,
                min_nodes=min_nodes,
                time_limit=time_limit,
            )
        except Exception as exc:
            safe_print_error(f"generate_filtered_dataset class {cls}", exc)
            results[cls] = []
    return results
