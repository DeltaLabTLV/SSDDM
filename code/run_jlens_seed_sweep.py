"""Audit the MultVAE seed replicates and test capacity as a predictor.

The manuscript reports that a checkpoint's dormant carrier capacity coincides
with its white-box stress-test success rate, across one checkpoint per
dataset. That is three points and supports only a descriptive statement. This
script re-runs the identical stress-test and carrier protocol on the seed
replicates produced by ``train_vae_seeds.py`` and assembles a capacity /
success table over all available checkpoints, so the association can be
evaluated within and across datasets and tested out of sample.

Protocol is unchanged from the manuscript runs: K=10, the first 100 test users
as the eligible prefix cohort, tap ``z``, and the paired summary of
``paper/make_tables.py`` with success = (Delta > 0.2 and clean gap > 0.5).

Usage:
  python3 run_jlens_seed_sweep.py --data ML1M Yahoo Pinterest --seeds 1 2 3
  python3 run_jlens_seed_sweep.py --report-only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import run_jlens_adversarial
import run_jlens_switch_diagnostic
from run_jlens_ml1m import CONFIGS, ROOT

sys.path.insert(0, str(ROOT / "paper"))
from make_tables import paired_attack_summary  # noqa: E402

RESULTS = ROOT / "jlens_results"
SEED_CKPTS = ROOT / "check" / "seeds"
METHODS = ["saliency", "attack_lam1", "constraint"]
MAIN_TAP = {"VAE": "z", "NCF": "user_state"}
# The exact carrier share the switch diagnostic reports differs by architecture:
# VAE decomposes its single ReLU decoder layer, NeuMF splits the two branches.
CARRIER_COL = {"VAE": "switch_share", "NCF": "mlp_share"}


def checkpoints(datasets, seeds, model="VAE"):
    """Yield (dataset, label, tag, ckpt_override) for every audited model."""
    for data in datasets:
        yield data, "released", "", None
        for seed in seeds:
            path = SEED_CKPTS / f"{model}_{data}_s{seed}.pt"
            if path.exists():
                yield data, f"s{seed}", f"_s{seed}", f"seeds/{path.name}"


def adversarial_csv(data, tag, k, model="VAE"):
    return (RESULTS
            / f"{data.lower()}_{model.lower()}_adversarial_k{k}{tag}.csv")


def switchdiag_csv(data, tag, k, model="VAE"):
    return (RESULTS
            / f"{data.lower()}_{model.lower()}_switchdiag_k{k}{tag}.csv")


def run_one(data, tag, ckpt, args):
    adv = adversarial_csv(data, tag, args.k, args.model)
    if adv.exists() and not args.force:
        print(f"[skip] {adv.name} exists")
    else:
        argv = ["--data", data, "--model", args.model, "--k", str(args.k),
                "--n-users", str(args.n_users), "--tag", tag]
        if ckpt:
            argv += ["--ckpt", ckpt]
        t0 = time.time()
        run_jlens_adversarial.main(argv)
        print(f"[done] {adv.name} in {(time.time() - t0) / 60:.1f} min",
              flush=True)

    sw = switchdiag_csv(data, tag, args.k, args.model)
    if sw.exists() and not args.force:
        print(f"[skip] {sw.name} exists")
    else:
        argv = ["--data", data, "--model", args.model, "--k", str(args.k),
                "--tag", tag]
        if ckpt:
            argv += ["--ckpt", ckpt]
        run_jlens_switch_diagnostic.main(argv)
        print(f"[done] {sw.name}", flush=True)


def collect(datasets, seeds, k, model="VAE"):
    rows = []
    carrier = CARRIER_COL[model]
    for data, label, tag, _ in checkpoints(datasets, seeds, model):
        adv = adversarial_csv(data, tag, k, model)
        sw = switchdiag_csv(data, tag, k, model)
        if not (adv.exists() and sw.exists()):
            continue
        summary = paired_attack_summary(pd.read_csv(adv), METHODS,
                                        tap=MAIN_TAP[model])
        diag = pd.read_csv(sw)
        sal = diag[diag.method == "saliency"]
        attack = diag[diag.method == "attack_lam1"]
        rows.append({
            "dataset": data,
            "checkpoint": label,
            "n_users": summary["saliency"]["n_users"],
            "capacity_median": float(sal["capacity"].median()),
            "capacity_mean": float(sal["capacity"].mean()),
            "switch_share_median": float(attack[carrier].median()),
            "attack_success": summary["attack_lam1"]["success_rate"],
            "constraint_success": summary["constraint"]["success_rate"],
            "attack_gap_median": summary["attack_lam1"]["clean_gap_median"],
            "saliency_gap_median": summary["saliency"]["clean_gap_median"],
            "saliency_success": summary["saliency"]["success_rate"],
        })
    return pd.DataFrame(rows)


def spearman(a, b):
    ra, rb = pd.Series(a).rank(), pd.Series(b).rank()
    return float(np.corrcoef(ra, rb)[0, 1])


def leave_one_out(table, x="capacity_median", y="attack_success"):
    """Fit the capacity-success line without each checkpoint, predict it.

    Reports out-of-sample error against the mean-only baseline, which is the
    honest way to ask whether capacity carries predictive information rather
    than merely ordering three points.
    """
    xs, ys = table[x].to_numpy(), table[y].to_numpy()
    if len(xs) < 4:
        return None
    preds, base = [], []
    for i in range(len(xs)):
        mask = np.ones(len(xs), bool)
        mask[i] = False
        slope, intercept = np.polyfit(xs[mask], ys[mask], 1)
        preds.append(slope * xs[i] + intercept)
        base.append(ys[mask].mean())
    preds, base = np.array(preds), np.array(base)
    mae, mae_base = np.abs(preds - ys).mean(), np.abs(base - ys).mean()
    ss_res = float(((ys - preds) ** 2).sum())
    ss_tot = float(((ys - ys.mean()) ** 2).sum())
    return {"loo_mae": float(mae), "loo_mae_mean_baseline": float(mae_base),
            "loo_r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
            "n": int(len(xs))}


def report(table):
    if table.empty:
        print("no completed checkpoints yet")
        return {}
    pd.set_option("display.width", 200)
    cols = ["dataset", "checkpoint", "n_users", "capacity_median",
            "switch_share_median", "attack_success", "constraint_success",
            "saliency_gap_median"]
    print("\n=== per-checkpoint ===")
    print(table[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    stats = {"n_checkpoints": int(len(table))}
    # The released checkpoints come from the original authors' recipe; only the
    # sN replicates share one recipe with each other, so pure seed variance is
    # the spread among those.
    print("\n=== seed variance (replicates only, one shared recipe) ===")
    reps = table[table.checkpoint != "released"]
    for data, g in reps.groupby("dataset"):
        if len(g) < 2:
            continue
        released = table[(table.dataset == data)
                         & (table.checkpoint == "released")]
        rel = (f"released {released.attack_success.iloc[0]:.3f}"
               if len(released) else "released n/a")
        print(f"  {data:10s} n={len(g)}  attack_success "
              f"{g.attack_success.mean():.3f} +- {g.attack_success.std():.3f}"
              f"  (min {g.attack_success.min():.3f}, "
              f"max {g.attack_success.max():.3f})   [{rel}]")
        stats[f"{data}_replicate_success_mean"] = float(g.attack_success.mean())
        stats[f"{data}_replicate_success_std"] = float(g.attack_success.std())

    print("\n=== all checkpoints per dataset (replicates + released) ===")
    for data, g in table.groupby("dataset"):
        if len(g) < 2:
            continue
        print(f"  {data:10s} n={len(g)}  attack_success "
              f"{g.attack_success.mean():.3f} +- {g.attack_success.std():.3f}"
              f"  (min {g.attack_success.min():.3f}, "
              f"max {g.attack_success.max():.3f})  |  capacity "
              f"{g.capacity_median.mean():.4f} +- {g.capacity_median.std():.4f}")
        stats[f"{data}_success_mean"] = float(g.attack_success.mean())
        stats[f"{data}_success_std"] = float(g.attack_success.std())
        stats[f"{data}_capacity_mean"] = float(g.capacity_median.mean())
        stats[f"{data}_capacity_std"] = float(g.capacity_median.std())

    if len(table) >= 3:
        rho = spearman(table.capacity_median, table.attack_success)
        pear = float(np.corrcoef(table.capacity_median,
                                 table.attack_success)[0, 1])
        print(f"\n=== capacity vs attack success over {len(table)} checkpoints ===")
        print(f"  Spearman {rho:.3f}   Pearson {pear:.3f}")
        stats.update(spearman=rho, pearson=pear)
        loo = leave_one_out(table)
        if loo:
            print(f"  leave-one-out MAE {loo['loo_mae']:.3f} vs "
                  f"mean-baseline {loo['loo_mae_mean_baseline']:.3f}"
                  f"   (LOO R^2 {loo['loo_r2']:.3f})")
            stats["leave_one_out"] = loo
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=list(CONFIGS),
                    choices=list(CONFIGS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--model", default="VAE", choices=["VAE", "NCF"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n-users", type=int, default=100)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.out is None:
        args.out = str(RESULTS / f"{args.model.lower()}_seed_capacity.csv")

    if not args.report_only:
        for data, label, tag, ckpt in checkpoints(args.data, args.seeds,
                                                  args.model):
            print(f"\n########## {args.model} {data} / {label} ##########",
                  flush=True)
            run_one(data, tag, ckpt, args)

    table = collect(args.data, args.seeds, args.k, args.model)
    stats = report(table)
    if not table.empty:
        table.to_csv(args.out, index=False)
        Path(args.out).with_suffix(".json").write_text(
            json.dumps(stats, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
