"""Analysis of Item-Jacobian lens benchmark results.

Consumes the per-row CSVs from run_jlens_ml1m.py and reports:
- behavioral vs mechanistic leaderboards (rank shift),
- per-explainer Spearman correlation between deletion effect and JComp,
- the four-quadrant taxonomy (outcome sensitivity x mechanistic mediation),
- disruption diagnostics (collateral effect, selectivity),
- random-subspace calibration,
- the MLP positive control (MR must be ~1).

Usage: python3 analyze_jlens.py [--vae CSV] [--mlp CSV] [--tap z]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VAE = ROOT / "jlens_results" / "ml1m_vae_k5.csv"
DEFAULT_MLP = ROOT / "jlens_results" / "ml1m_mlp_k5.csv"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", default=str(DEFAULT_VAE))
    ap.add_argument("--mlp", default=str(DEFAULT_MLP))
    ap.add_argument("--tap", default="z",
                    help="informative tap for mechanistic metrics")
    ap.add_argument("--beh-thresh", type=float, default=0.01)
    ap.add_argument("--mr-thresh", type=float, default=0.8)
    args = ap.parse_args(argv)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    df = pd.read_csv(args.vae)
    z = df[df["tap"] == args.tap].copy()
    n_users = z["user"].nunique()
    print(f"=== VAE @ tap '{args.tap}', {n_users} users, "
          f"K={int(z['k'].iloc[0])} ===")

    # -------- leaderboards
    beh = z.groupby("explainer")["delta_beh"].mean().sort_values(
        ascending=False)
    mech = z.groupby("explainer")["jcomp"].mean().sort_values(ascending=False)
    strong = z[z["delta_beh"].abs() > args.beh_thresh]
    mr_med = strong.groupby("explainer")["mr"].median()
    jr_med = strong.groupby("explainer")["jrestore"].median()

    board = pd.DataFrame({
        "DEL_mean(delta_beh)": beh,
        "beh_rank": beh.rank(ascending=False).astype(int),
        "JComp_mean": mech,
        "mech_rank": mech.rank(ascending=False).astype(int),
        "MR_median(filt)": mr_med,
        "JRestore_median(filt)": jr_med,
    }).sort_values("beh_rank")
    print("\n-- Leaderboard: behavioral vs mechanistic --")
    print(board.round(4))

    # -------- per-user agreement
    print("\n-- Spearman(delta_beh, jcomp) across users, per explainer --")
    for name, g in z.groupby("explainer"):
        rho = g["delta_beh"].corr(g["jcomp"], method="spearman")
        print(f"  {name:10s} rho={rho:.4f}")

    # -------- quadrant taxonomy
    beh_split = z["delta_beh"].median()
    z["quadrant"] = np.select(
        [(z["delta_beh"] > beh_split) & (z["mr"] >= args.mr_thresh),
         (z["delta_beh"] > beh_split) & (z["mr"] < args.mr_thresh),
         (z["delta_beh"] <= beh_split) & (z["mr"] >= args.mr_thresh)],
        ["aligned", "disruptive", "compensated"], default="weak")
    print(f"\n-- Quadrants (beh split at median={beh_split:.3f}, "
          f"MR thresh={args.mr_thresh}) --")
    print(z.groupby("explainer")["quadrant"].value_counts(normalize=True)
          .unstack(fill_value=0).round(3))

    # -------- disruption diagnostics
    print("\n-- Disruption: collateral & selectivity (means) --")
    print(z.groupby("explainer")[
        ["collateral_all", "selectivity_del", "selectivity_patch"]]
        .mean().round(4))

    # -------- random-subspace calibration
    print("\n-- Random-subspace control (same-dim): jcomp vs random --")
    cal = z.groupby("explainer")[["jcomp", "jcomp_rand_mean"]].mean()
    cal["ratio"] = cal["jcomp_rand_mean"] / cal["jcomp"]
    print(cal.round(4))

    # -------- MR identity check at dec_h1 (sanity on real model)
    dec = df[df["tap"] == "dec_h1"]
    dec_strong = dec[dec["delta_beh"].abs() > args.beh_thresh]
    dev = (dec_strong["mr"] - 1).abs()
    print(f"\n-- Sanity: dec_h1 MR==1 identity: median|MR-1|={dev.median():.2e},"
          f" max={dev.max():.2e} --")

    # -------- MLP positive control
    if Path(args.mlp).exists():
        m = pd.read_csv(args.mlp)
        ms = m[m["delta_beh"].abs() > args.beh_thresh]
        dev = (ms["mr"] - 1).abs()
        print(f"\n=== MLP positive control ({m['user'].nunique()} users): "
              f"median|MR-1|={dev.median():.2e}, max={dev.max():.2e} ===")
    return df


if __name__ == "__main__":
    main()
