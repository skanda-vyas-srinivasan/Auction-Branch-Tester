from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .branching import train_gnn
from .common import safe_print_error
from .data import CLASSES, load_instances
from .distance import compute_distribution_distance, extract_normalized_representation
from .evaluation import evaluate_branching


def run_experiment(
    root: str | Path = "local_instances",
    results_dir: str | Path = "local_results",
    training_class: str = "vrp",
    test_classes: Iterable[str] = CLASSES,
    min_instances: int = 50,
    n_reference: int = 40,
    n_train: int = 40,
    n_test_per_class: int | None = None,
    n_epochs: int = 20,
    time_limit: int = 60,
    max_strong_samples_per_instance: int = 1,
    strong_time_limit: int = 20,
    model_family: str = "local",
) -> pd.DataFrame:
    try:
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        root = Path(root)
        needed_classes = list(dict.fromkeys([training_class] + list(test_classes)))
        instances = {
            cls: sorted((root / cls).glob("*.mps"))[:min_instances]
            for cls in needed_classes
            if (root / cls).exists()
        }
        missing_classes = [cls for cls in needed_classes if len(instances.get(cls, [])) < min_instances]
        if missing_classes and set(needed_classes).issubset(set(CLASSES)):
            instances = load_instances(root, min_instances=min_instances)
        elif missing_classes:
            print(f"WARNING: these classes have fewer than {min_instances} local instances: {missing_classes}")

        reference_paths = instances.get(training_class, [])[:n_reference]
        print(f"reference instances: {len(reference_paths)} from {training_class}")

        reference_reps = []
        for idx, path in enumerate(reference_paths, start=1):
            reference_reps.append(extract_normalized_representation(path))
            if idx % 10 == 0:
                print(f"precomputed {idx}/{len(reference_paths)} reference representations")

        train_paths = instances.get(training_class, [])[:n_train]
        model = train_gnn(
            train_paths,
            n_epochs=n_epochs,
            max_strong_samples_per_instance=max_strong_samples_per_instance,
            strong_time_limit=strong_time_limit,
            model_family=model_family,
        )

        rows = []
        for cls in test_classes:
            paths = instances.get(cls, [])
            if n_test_per_class is not None:
                paths = paths[:n_test_per_class]
            for idx, path in enumerate(paths, start=1):
                try:
                    rep = extract_normalized_representation(path)
                    distance = compute_distribution_distance(rep, reference_reps) if rep is not None else float("inf")
                    evaluation = evaluate_branching(model, path, time_limit=time_limit)
                    if evaluation is None:
                        continue
                    rows.append(
                        {
                            "instance_name": Path(path).name,
                            "class": cls,
                            "distance": distance,
                            "ml_nodes": evaluation["ml_nodes"],
                            "scip_nodes": evaluation["scip_nodes"],
                            "degradation": evaluation["degradation"],
                        }
                    )
                    pd.DataFrame(
                        rows,
                        columns=["instance_name", "class", "distance", "ml_nodes", "scip_nodes", "degradation"],
                    ).to_csv(results_dir / "raw_results.csv", index=False)
                    if len(rows) % 10 == 0:
                        print(f"processed {len(rows)} test instances")
                except Exception as exc:
                    safe_print_error(f"experiment row {cls}/{path}", exc)
                    continue

        df = pd.DataFrame(rows, columns=["instance_name", "class", "distance", "ml_nodes", "scip_nodes", "degradation"])
        df.to_csv(results_dir / "raw_results.csv", index=False)
        print(f"saved {len(df)} rows to {results_dir / 'raw_results.csv'}")
        return df
    except Exception as exc:
        safe_print_error("run_experiment", exc)
        return pd.DataFrame()


def plot_results(df: pd.DataFrame, results_dir: str | Path = "local_results") -> None:
    try:
        from scipy.stats import pearsonr

        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        mpl_config_dir = results_dir / ".matplotlib"
        xdg_cache_dir = results_dir / ".cache"
        mpl_config_dir.mkdir(parents=True, exist_ok=True)
        xdg_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
        os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache_dir))
        import matplotlib.pyplot as plt

        if df.empty:
            print("no results to plot")
            return

        colors = {"vrp": "blue", "knapsack": "orange", "bin_packing": "green"}
        plt.figure(figsize=(8, 5))
        for cls, group in df.groupby("class"):
            plt.scatter(group["distance"], group["degradation"], label=cls, color=colors.get(cls, "gray"), alpha=0.8)
        plt.axhline(1.0, color="black", linestyle="--", linewidth=1)
        if len(df) >= 2 and df["distance"].nunique() > 1:
            coef = np.polyfit(df["distance"], df["degradation"], 1)
            x_values = np.linspace(float(df["distance"].min()), float(df["distance"].max()), 100)
            plt.plot(x_values, coef[0] * x_values + coef[1], color="red", linewidth=1.5)
        plt.xlabel("distribution distance to training set")
        plt.ylabel("degradation ratio (ML nodes / SCIP nodes)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(results_dir / "scatter.png", dpi=200)
        plt.close()

        if len(df) >= 2 and df["distance"].nunique() > 1 and df["degradation"].nunique() > 1:
            corr, p_value = pearsonr(df["distance"], df["degradation"])
            print(f"pearson correlation: r={corr:.4f}, p={p_value:.4g}")
        else:
            print("pearson correlation: not enough variation")

        thresholds = sorted(df["distance"].dropna().unique())
        best_threshold = None
        best_accuracy = -1.0
        labels = df["degradation"] > 1.0
        for threshold in thresholds:
            predictions = df["distance"] > threshold
            accuracy = float((predictions == labels).mean())
            if accuracy > best_accuracy:
                best_threshold = float(threshold)
                best_accuracy = accuracy
        print(f"best threshold: {best_threshold} accuracy={best_accuracy:.4f}")
        if best_threshold is not None:
            for cls, group in df.groupby("class"):
                above = int((group["distance"] > best_threshold).sum())
                below = int((group["distance"] <= best_threshold).sum())
                print(f"{cls}: above_threshold={above}, below_or_equal_threshold={below}")

        plt.figure(figsize=(7, 5))
        ordered = [cls for cls in list(CLASSES) + sorted(set(df["class"])) if cls in set(df["class"])]
        ordered = list(dict.fromkeys(ordered))
        plt.boxplot([df[df["class"] == cls]["degradation"] for cls in ordered], labels=ordered)
        plt.axhline(1.0, color="black", linestyle="--", linewidth=1)
        plt.ylabel("degradation ratio (ML nodes / SCIP nodes)")
        plt.tight_layout()
        plt.savefig(results_dir / "boxplot.png", dpi=200)
        plt.close()
        print(f"saved plots to {results_dir}")
    except Exception as exc:
        safe_print_error("plot_results", exc)
