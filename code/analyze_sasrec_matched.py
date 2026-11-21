"""Effect-matched comparison of mechanism gaps across SASRec explainers.

Raw gap summaries are not directly comparable across explainers because the
ratio is interpreted only for a measurable effect and methods produce
different effect-size distributions.

The manuscript therefore compares gaps *after matching deletion effect*. This
implements that matching explicitly: rows are stratified by decile of the
pooled effect distribution, the median gap is taken within each stratum, and
strata are averaged with equal weight. Methods are then compared at
like-for-like effect sizes.

Usage:
  python3 analyze_sasrec_matched.py --prefix ml1m --seeds 42 7 123
  python3 analyze_sasrec_matched.py --prefix ml1m --seeds 42 7 123 \
      --include-published --include-published-attack
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "jlens_results"
METHODS = [
    "random", "pop", "occlusion", "lime", "ig", "saliency",
    "attack_lam1",
]


def load(prefix, seed, k, tap):
    tag = f"_s{seed}" if seed is not None else ""
    path = RESULTS / f"{prefix}_sasrec_explainers_k{k}{tag}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[(df.tap == tap) & (df.delta_beh > 0.01)].copy()
    df["gap"] = np.clip((df.delta_beh - df.jfwd) / df.delta_beh, 0.0, 1.0)
    return df


def load_published_attack(prefix, k, tap):
    """Load the saved white-box sets for the published checkpoint."""
    path = RESULTS / f"{prefix}_sasrec_attack_k{k}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[(df.tap == tap) & (df.delta_beh > 0.01)].copy()
    df["gap"] = np.clip((df.delta_beh - df.jfwd) / df.delta_beh, 0.0, 1.0)
    return df


def matched(df, n_strata=10, comparison_methods=None):
    """Median gap per method over effect strata shared by every method."""
    if comparison_methods is not None:
        available = [m for m in comparison_methods
                     if m in set(df.explainer)]
        df = df[df.explainer.isin(available)].copy()
    else:
        available = list(df.explainer.unique())
    edges = np.quantile(df.delta_beh, np.linspace(0, 1, n_strata + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    df = df.assign(stratum=pd.cut(df.delta_beh, edges, labels=False,
                                  include_lowest=True))
    common_strata = set.intersection(*(
        set(df.loc[df.explainer == method, "stratum"])
        for method in available
    ))
    out = {}
    for method, g in df.groupby("explainer"):
        per = g.loc[g.stratum.isin(common_strata)].groupby(
            "stratum"
        )["gap"].median()
        # Equal weight over the same strata, so neither concentration in a
        # favorable effect range nor method-specific support changes the
        # comparison.
        out[method] = {"matched_gap": float(per.mean()),
                       "raw_gap": float(g["gap"].median()),
                       "n": int(len(g)), "strata": int(len(common_strata))}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", nargs="+", default=["ml1m", "sports"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--tap", default="block1")
    ap.add_argument("--strata", type=int, default=10)
    ap.add_argument(
        "--comparison-methods", nargs="+",
        default=["random", "occlusion", "lime", "attack_lam1"],
        help="methods whose common effect strata define the comparison",
    )
    ap.add_argument("--include-published", action="store_true",
                    help="prepend the untagged published checkpoint")
    ap.add_argument("--include-published-attack", action="store_true",
                    help="include its saved white-box sets in effect matching")
    args = ap.parse_args(argv)

    for prefix in args.prefix:
        rows = []
        checkpoints = [(f"s{seed}", seed) for seed in args.seeds]
        if args.include_published:
            checkpoints.insert(0, ("published", None))
        for checkpoint, seed in checkpoints:
            df = load(prefix, seed, args.k, args.tap)
            if df is None:
                continue
            if checkpoint == "published" and args.include_published_attack:
                attack = load_published_attack(prefix, args.k, args.tap)
                if attack is not None:
                    df = pd.concat([df, attack], ignore_index=True)
            res = matched(df, args.strata, args.comparison_methods)
            for method, v in res.items():
                rows.append({"checkpoint": checkpoint, "seed": seed,
                             "method": method, **v})
        if not rows:
            print(f"[{prefix}] no result files")
            continue
        table = pd.DataFrame(rows)
        agg = table.groupby("method").agg(
            matched_mean=("matched_gap", "mean"),
            matched_std=("matched_gap", "std"),
            raw_mean=("raw_gap", "mean"),
            n_mean=("n", "mean"))
        agg = agg.reindex([m for m in METHODS if m in agg.index])
        print(f"\n=== {prefix}, tap {args.tap}, K={args.k}, "
              f"{args.strata} effect strata, seeds {args.seeds} ===")
        print(f"{'method':12s} {'gap @matched':>14s} {'+-':>6s} "
              f"{'gap raw':>9s} {'rows':>7s}")
        for m, r in agg.iterrows():
            print(f"{m:12s} {r.matched_mean:14.3f} {r.matched_std:6.3f} "
                  f"{r.raw_mean:9.3f} {r.n_mean:7.0f}")
        out = RESULTS / f"{prefix}_sasrec_matched_k{args.k}.csv"
        table.to_csv(out, index=False)
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
