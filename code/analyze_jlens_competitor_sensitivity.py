"""Aggregate competitor-count and competitor-composition sensitivity runs."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def summarize_file(path):
    frame = pd.read_csv(path)
    if "clean_gap" not in frame:
        frame["clean_gap"] = np.clip(1.0 - frame["mr"], 0.0, 1.0)
    positive = frame[frame["delta_beh"] > 0.01]
    keys = ["architecture", "data", "method", "strategy", "n_competitors"]
    return positive.groupby(keys, as_index=False).agg(
        n=("user", "nunique"),
        gap50=("clean_gap", "median"),
        gap90=("clean_gap", lambda values: values.quantile(0.9)),
        basis_rank50=("basis_rank", "median"),
        joint_tail=("clean_gap", lambda values: np.nan),
    ).drop(columns="joint_tail").merge(
        frame.assign(joint=(frame["delta_beh"] > 0.2)
                     & (frame["clean_gap"] > 0.5)).groupby(
            keys, as_index=False).agg(joint_tail=("joint", "mean")),
        on=keys, how="left")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="jlens_results")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    root = Path(args.results_dir)
    paths = sorted(root.glob("*_competitor_sensitivity_k*.csv"))
    if not paths:
        raise FileNotFoundError("no competitor-sensitivity CSVs found")
    summary = pd.concat([summarize_file(path) for path in paths],
                        ignore_index=True)
    print(summary.round(4).to_string(index=False))
    print("\nrange over all tested count/composition settings")
    ranges = summary.groupby(["architecture", "data", "method"]).agg(
        gap50_min=("gap50", "min"), gap50_max=("gap50", "max"),
        gap90_min=("gap90", "min"), gap90_max=("gap90", "max"),
        tail_min=("joint_tail", "min"), tail_max=("joint_tail", "max"),
    )
    print(ranges.round(4).to_string())
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.out, index=False)
        print(f"\nsaved {len(summary)} rows -> {args.out}")
    return summary


if __name__ == "__main__":
    main()
