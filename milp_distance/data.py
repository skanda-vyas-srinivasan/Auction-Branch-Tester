from __future__ import annotations

import gzip
import random
import shutil
import urllib.request
from io import BytesIO
from pathlib import Path

import pandas as pd

from .common import MAX_VARIABLES, is_small_mps, safe_print_error


CLASSES = ["vrp", "knapsack", "bin_packing"]
MIPLIB_INSTANCE_URL = "https://miplib.zib.de/WebData/instances/{name}.mps.gz"
MIPLIB_TAGS = {
    "vrp": "set_partitioning",
    "knapsack": "knapsack",
    "bin_packing": "binpacking",
}


def miplib_tag_candidates(tag: str, max_variables: int = MAX_VARIABLES) -> list[str]:
    try:
        tables = pd.read_html(f"https://miplib.zib.de/tag_{tag}.html")
        table = max(tables, key=len)
        table.columns = [str(col).strip() for col in table.columns]
        instance_col = next(col for col in table.columns if "Instance" in col)
        variable_col = next((col for col in table.columns if "Variables" in col), None)
        if variable_col is not None:
            table[variable_col] = pd.to_numeric(table[variable_col], errors="coerce")
            table = table[table[variable_col] < max_variables]
        return list(dict.fromkeys(str(value).strip() for value in table[instance_col].dropna().tolist()))
    except Exception as exc:
        safe_print_error(f"reading MIPLIB tag table {tag}", exc)
        return []


def download_miplib_instance(name: str, dest_dir: str | Path, max_variables: int = MAX_VARIABLES) -> Path | None:
    try:
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        target = dest_dir / f"{name}.mps"
        if target.exists() and target.stat().st_size > 0:
            return target if is_small_mps(target, max_variables=max_variables) else None

        req = urllib.request.Request(
            MIPLIB_INSTANCE_URL.format(name=name),
            headers={"User-Agent": "MILP distance branching experiment"},
        )
        with urllib.request.urlopen(req, timeout=45) as resp:
            compressed = resp.read()
        with gzip.open(BytesIO(compressed), "rb") as src, open(target, "wb") as dst:
            shutil.copyfileobj(src, dst)

        if is_small_mps(target, max_variables=max_variables):
            return target
        target.unlink(missing_ok=True)
        return None
    except Exception as exc:
        safe_print_error(f"downloading MIPLIB instance {name}", exc)
        return None


def write_vrp_like_mps(path: str | Path, seed: int) -> Path | None:
    try:
        import pyscipopt

        path = Path(path)
        rng = random.Random(seed)
        model = pyscipopt.Model()
        model.hideOutput(True)
        n_customers = rng.randint(14, 24)
        n_routes = rng.randint(45, 95)
        routes = [model.addVar(vtype="B", name=f"r_{idx}", obj=rng.randint(5, 40)) for idx in range(n_routes)]
        covers = [[] for _ in range(n_customers)]
        for route_idx in range(n_routes):
            route_size = rng.randint(2, min(5, n_customers))
            for customer in rng.sample(range(n_customers), route_size):
                covers[customer].append(routes[route_idx])
        for customer, route_vars in enumerate(covers):
            if not route_vars:
                route_vars.append(routes[rng.randrange(n_routes)])
            model.addCons(sum(route_vars) == 1, name=f"cover_{customer}")
        model.writeProblem(str(path))
        return path
    except Exception as exc:
        safe_print_error(f"writing synthetic VRP-like instance {path}", exc)
        return None


def write_knapsack_mps(path: str | Path, seed: int) -> Path | None:
    try:
        import pyscipopt

        path = Path(path)
        rng = random.Random(seed)
        model = pyscipopt.Model()
        model.hideOutput(True)
        n_items = rng.randint(80, 180)
        variables = []
        weights = []
        for idx in range(n_items):
            weight = rng.randint(1, 60)
            value = rng.randint(1, 100)
            weights.append(weight)
            variables.append(model.addVar(vtype="B", name=f"x_{idx}", obj=-value))
        capacity = int(sum(weights) * rng.uniform(0.25, 0.55))
        model.addCons(sum(weights[idx] * variables[idx] for idx in range(n_items)) <= capacity, name="capacity")
        model.writeProblem(str(path))
        return path
    except Exception as exc:
        safe_print_error(f"writing synthetic knapsack instance {path}", exc)
        return None


def write_binpacking_mps(path: str | Path, seed: int) -> Path | None:
    try:
        import pyscipopt

        path = Path(path)
        rng = random.Random(seed)
        model = pyscipopt.Model()
        model.hideOutput(True)
        n_items = rng.randint(18, 30)
        n_bins = rng.randint(8, 14)
        capacity = rng.randint(35, 60)
        weights = [rng.randint(4, max(5, capacity // 2)) for _ in range(n_items)]
        bin_used = [model.addVar(vtype="B", name=f"y_{bin_idx}", obj=1.0) for bin_idx in range(n_bins)]
        assign = [
            [model.addVar(vtype="B", name=f"x_{item_idx}_{bin_idx}") for bin_idx in range(n_bins)]
            for item_idx in range(n_items)
        ]
        for item_idx in range(n_items):
            model.addCons(sum(assign[item_idx][bin_idx] for bin_idx in range(n_bins)) == 1, name=f"assign_{item_idx}")
        for bin_idx in range(n_bins):
            model.addCons(
                sum(weights[item_idx] * assign[item_idx][bin_idx] for item_idx in range(n_items))
                <= capacity * bin_used[bin_idx],
                name=f"cap_{bin_idx}",
            )
        model.writeProblem(str(path))
        return path
    except Exception as exc:
        safe_print_error(f"writing synthetic bin-packing instance {path}", exc)
        return None


def supplement_synthetic(cls: str, dest_dir: str | Path, min_instances: int = 50) -> None:
    try:
        dest_dir = Path(dest_dir)
        writers = {
            "vrp": write_vrp_like_mps,
            "knapsack": write_knapsack_mps,
            "bin_packing": write_binpacking_mps,
        }
        idx = len(list(dest_dir.glob("*.mps")))
        while len(list(dest_dir.glob("*.mps"))) < min_instances:
            target = dest_dir / f"synthetic_{cls}_{idx:03d}.mps"
            if not target.exists():
                writers[cls](target, seed=10000 * (CLASSES.index(cls) + 1) + idx)
            idx += 1
    except Exception as exc:
        safe_print_error(f"supplementing {cls}", exc)


def load_instances(root: str | Path, min_instances: int = 50, max_variables: int = MAX_VARIABLES) -> dict[str, list[Path]]:
    loaded: dict[str, list[Path]] = {}
    root = Path(root)
    for cls in CLASSES:
        try:
            dest_dir = root / cls
            dest_dir.mkdir(parents=True, exist_ok=True)
            paths = [path for path in sorted(dest_dir.glob("*.mps")) if is_small_mps(path, max_variables=max_variables)]
            if len(paths) < min_instances:
                names = miplib_tag_candidates(MIPLIB_TAGS[cls], max_variables=max_variables)
                for name in names:
                    if len(paths) >= min_instances:
                        break
                    path = download_miplib_instance(name, dest_dir, max_variables=max_variables)
                    if path is not None and path not in paths:
                        paths.append(path)
            if len(paths) < min_instances:
                supplement_synthetic(cls, dest_dir, min_instances=min_instances)
                paths = [path for path in sorted(dest_dir.glob("*.mps")) if is_small_mps(path, max_variables=max_variables)]
            loaded[cls] = paths[:min_instances]
            print(f"{cls}: {len(loaded[cls])} instances loaded")
        except Exception as exc:
            safe_print_error(f"loading class {cls}", exc)
            loaded[cls] = []
    return loaded

