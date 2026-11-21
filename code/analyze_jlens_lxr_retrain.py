"""Summarize validation and held-out paired LXR retraining results."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def validation_summary(results_dir, datasets, seeds):
    rows = []
    for data in datasets:
        for seed in seeds:
            path = results_dir / f"{data}_lxr_retrain_validation_s{seed}.csv"
            frame = pd.read_csv(path)
            for beta in (0.0, 1.0):
                arm = frame[frame["beta"] == beta]
                best = arm.loc[arm["val_lxr_loss"].idxmin()]
                rows.append(best.to_dict())
    result = pd.DataFrame(rows)
    columns = ["data", "seed", "beta", "epoch", "val_lxr_loss",
               "val_effect", "val_gap50", "val_gap90", "val_tail"]
    print("validation checkpoints selected by unregularized LXR loss")
    print(result[columns].round(4).to_string(index=False))

    metrics = ["val_lxr_loss", "val_effect", "val_gap50",
               "val_gap90", "val_tail"]
    grouped = result.groupby(["data", "beta"])[metrics].agg(["mean", "std"])
    print("\nvalidation mean +/- SD over independent initializations")
    print(grouped.round(4).to_string())

    wide = result.pivot(index=["data", "seed"], columns="beta")
    changes = pd.DataFrame({
        "effect_ratio": wide["val_effect"][1.0] / wide["val_effect"][0.0],
        "gap50_change": wide["val_gap50"][1.0] - wide["val_gap50"][0.0],
        "gap90_change": wide["val_gap90"][1.0] - wide["val_gap90"][0.0],
        "tail_change": wide["val_tail"][1.0] - wide["val_tail"][0.0],
    })
    print("\nbeta=1 minus paired beta=0")
    print(changes.round(4).to_string())
    print("\nmean paired change")
    print(changes.groupby("data").agg(["mean", "std"])
          .round(4).to_string())
    return result, changes


def load_test(path):
    frame = pd.read_csv(path)
    z = frame[frame["tap"] == "z"].copy()
    z["gap"] = np.clip(1.0 - z["mr"], 0.0, 1.0)
    positive = z[z["delta_beh"] > 0.01]
    return {
        "n": len(z),
        "effect": z["delta_beh"].mean(),
        "aligned": z["jcomp"].mean(),
        "remainder": z["clean_unrecovered"].mean(),
        "gap50": positive["gap"].median(),
        "gap90": positive["gap"].quantile(.9),
        "tail": ((z["delta_beh"] > .2) & (z["gap"] > .5)).mean(),
    }


def test_summary(results_dir, datasets, seeds):
    rows = []
    for data in datasets:
        for seed in seeds:
            for beta in (0.0, 1.0):
                tag = f"b{beta:g}".replace(".", "p")
                path = (results_dir
                        / f"{data}_vae_lxr_retrain_{tag}_k5_s{seed}.csv")
                rows.append({"data": data, "seed": seed, "beta": beta,
                             **load_test(path)})
    result = pd.DataFrame(rows)
    print("held-out test")
    print(result.round(4).to_string(index=False))
    metrics = ["effect", "aligned", "remainder", "gap50", "gap90", "tail"]
    print("\nheld-out test mean +/- SD")
    print(result.groupby(["data", "beta"])[metrics].agg(["mean", "std"])
          .round(4).to_string())
    return result


def objective_summary(results_dir, datasets, seeds):
    frames = []
    for data in datasets:
        for seed in seeds:
            path = results_dir / f"{data}_lxr_retrain_test_objective_s{seed}.csv"
            frames.append(pd.read_csv(path))
    result = pd.concat(frames, ignore_index=True)
    metrics = ["total", "positive", "negative", "sparsity"]
    print("unmodified LXR objective on all held-out test users")
    print(result.groupby(["data", "beta"])[metrics].agg(["mean", "std"])
          .round(4).to_string())
    wide = result.pivot(index=["data", "seed"], columns="beta")
    paired = pd.DataFrame({
        "objective_ratio": wide["total"][1.0] / wide["total"][0.0],
        "objective_change": wide["total"][1.0] - wide["total"][0.0],
    })
    print("\npaired beta=1 / beta=0 objective")
    print(paired.round(4).to_string())
    return result, paired


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="jlens_results")
    parser.add_argument("--stage", choices=["validation", "test", "objective"],
                        default="validation")
    parser.add_argument("--datasets", nargs="+", default=["yahoo", "pinterest"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 2027])
    args = parser.parse_args(argv)
    root = Path(args.results_dir)
    if args.stage == "validation":
        return validation_summary(root, args.datasets, args.seeds)
    if args.stage == "test":
        return test_summary(root, args.datasets, args.seeds)
    return objective_summary(root, args.datasets, args.seeds)


if __name__ == "__main__":
    main()
