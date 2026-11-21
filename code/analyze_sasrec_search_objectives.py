"""Reproduce the SASRec audit-guided selection table.

The three inputs must come from the same cohort and fixed-budget local search;
only the optimized scalar differs: Delta, U = Delta - J, or J.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULTS = {
    "max_delta": "ml1m_sasrec_search_max_delta_k5_s42.csv",
    "max_u": "ml1m_sasrec_search_max_u_k5_s42.csv",
    "max_j": "ml1m_sasrec_search_max_j_k5_s42.csv",
}


def load_one(path: Path, tap: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"user", "tap", "delta_beh", "clean_gap", "symmetric_gap"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    frame = frame.loc[frame["tap"] == tap].copy().set_index("user")
    if frame.empty:
        raise ValueError(f"{path}: tap {tap!r} is absent")
    if not frame.index.is_unique:
        raise ValueError(f"{path}: duplicate users at tap {tap!r}")
    return frame.sort_index()


def summarize(frame: pd.DataFrame) -> dict[str, float | int]:
    measurable = frame["delta_beh"] > 0.01
    tail = ((frame["delta_beh"] > 0.2) & (frame["clean_gap"] > 0.5))
    return {
        "n": len(frame),
        "n_gap": int(measurable.sum()),
        "mean_delta": float(frame["delta_beh"].mean()),
        "median_gap": float(frame.loc[measurable, "clean_gap"].median()),
        "median_symmetric_gap": float(
            frame.loc[measurable, "symmetric_gap"].median()),
        "tail_rate": float(tail.mean()),
    }


def paired_median_interval(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    samples: int,
    seed: int,
) -> tuple[int, float, float, float]:
    users = left.index.intersection(right.index)
    keep = ((left.loc[users, "delta_beh"] > 0.01)
            & (right.loc[users, "delta_beh"] > 0.01))
    users = users[keep]
    a = left.loc[users, "clean_gap"].to_numpy()
    b = right.loc[users, "clean_gap"].to_numpy()
    observed = float(np.median(a) - np.median(b))
    rng = np.random.default_rng(seed)
    boot = np.empty(samples)
    for i in range(samples):
        draw = rng.integers(0, len(users), len(users))
        boot[i] = np.median(a[draw]) - np.median(b[draw])
    low, high = np.quantile(boot, [0.025, 0.975])
    return len(users), observed, float(low), float(high)


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parent.parent / "jlens_results"
    parser = argparse.ArgumentParser(description=__doc__)
    for key, filename in DEFAULTS.items():
        parser.add_argument(f"--{key.replace('_', '-')}", type=Path,
                            default=root / filename)
    parser.add_argument("--tap", default="block1")
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args(argv)

    frames = {
        "max-Delta (behavior)": load_one(args.max_delta, args.tap),
        "max-U (stress)": load_one(args.max_u, args.tap),
        "max-J (guided)": load_one(args.max_j, args.tap),
    }
    cohorts = [set(frame.index) for frame in frames.values()]
    if any(cohort != cohorts[0] for cohort in cohorts[1:]):
        raise ValueError("search-objective files do not contain the same cohort")

    print("objective\tmean_delta\tmedian_gap\ttail_rate\tmedian_symmetric_gap")
    for label, frame in frames.items():
        row = summarize(frame)
        print(f"{label}\t{row['mean_delta']:.3f}\t{row['median_gap']:.3f}"
              f"\t{row['tail_rate']:.3f}\t{row['median_symmetric_gap']:.3f}")

    n, diff, low, high = paired_median_interval(
        frames["max-J (guided)"], frames["max-Delta (behavior)"],
        samples=args.bootstrap, seed=args.seed)
    print(f"paired max-J minus max-Delta median-gap difference (n={n}): "
          f"{diff:.3f}, 95% bootstrap CI [{low:.3f}, {high:.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
