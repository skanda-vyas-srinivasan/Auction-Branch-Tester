from __future__ import annotations

import math
from pathlib import Path
from typing import Any


MAX_VARIABLES = 500


def safe_print_error(context: str, exc: BaseException) -> None:
    try:
        print(f"[ERROR] {context}: {type(exc).__name__}: {exc}")
    except Exception:
        print("[ERROR] could not format exception")


def variable_name(var_or_name: Any) -> str:
    try:
        return var_or_name if isinstance(var_or_name, str) else getattr(var_or_name, "name", str(var_or_name))
    except Exception as exc:
        safe_print_error("variable_name", exc)
        return str(var_or_name)


def variable_obj(var: Any) -> float:
    try:
        return float(var.getObj())
    except Exception as exc:
        safe_print_error(f"objective coefficient for {var}", exc)
        return 0.0


def finite_or_zero(value: Any) -> float:
    try:
        value = float(value)
        return value if math.isfinite(value) else 0.0
    except Exception:
        return 0.0


def count_vars(mps_path: str | Path) -> int:
    try:
        import pyscipopt

        model = pyscipopt.Model()
        model.hideOutput(True)
        model.readProblem(str(mps_path))
        return len(model.getVars())
    except Exception as exc:
        safe_print_error(f"counting variables in {mps_path}", exc)
        return 10**9


def is_small_mps(mps_path: str | Path, max_variables: int = MAX_VARIABLES) -> bool:
    try:
        return count_vars(mps_path) < max_variables
    except Exception as exc:
        safe_print_error(f"filtering {mps_path}", exc)
        return False

