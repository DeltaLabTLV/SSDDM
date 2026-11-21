"""Audit the SASRec seed replicates and report the strong-gap rate spread.

The manuscript's sequential finding -- that a sizeable share of ordinary
occlusion and LIME deletions on SASRec are already strong mechanism-gap events
-- comes from a single checkpoint. This runs the pre-registered validation gate
and then the identical explainer audit on every replicate produced by
``train_sasrec_seeds.py``, and reports the rate per checkpoint so the finding
can be stated with a spread instead of as a point estimate.

The gate is not optional: a checkpoint whose random-deletion recovery at the
inner tap falls below 0.5 does not support the audit, and is reported as failed
rather than silently averaged in.

Usage:
  python3 run_jlens_sasrec_seed_sweep.py --seeds 42 7 123
  python3 run_jlens_sasrec_seed_sweep.py --report-only
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LXSR = ROOT / "external" / "LXSR"
SPLIT_ROOT = LXSR / "data" / "splitted" / "global_timesplit" / "val_by_time"
CKPT_DIR = ROOT / "check" / "sasrec_seeds"
RESULTS = ROOT / "jlens_results"
PYTHON = str(ROOT / ".venv-lxsr" / "bin" / "python")

ORDINARY = ["occlusion", "lime"]
TAP = "block1"


def csv_path(tag, k, prefix="ml1m"):
    return RESULTS / f"{prefix}_sasrec_explainers_k{k}{tag}.csv"


def checkpoints(seeds, dataset="Movielens-1m", prefix="ml1m"):
    """(label, tag, ckpt) for the published checkpoint and each replicate."""
    published = csv_path("", 5, prefix)
    if published.exists():
        yield "published", "", None
    for seed in seeds:
        ckpt = CKPT_DIR / f"sasrec_{dataset.lower()}_s{seed}.pth"
        if ckpt.exists():
            yield f"s{seed}", f"_s{seed}", ckpt


def run(cmd):
    print("$ " + " ".join(str(c) for c in cmd), flush=True)
    proc = subprocess.run([str(c) for c in cmd], cwd=ROOT / "code",
                          capture_output=True, text=True)
    sys.stdout.write(proc.stdout[-4000:])
    if proc.returncode != 0:
        sys.stdout.write(proc.stderr[-4000:])
    return proc.returncode, proc.stdout


def gate(ckpt, args):
    code, out = run([PYTHON, "validate_jlens_sasrec.py",
                     "--lxsr-root", LXSR, "--ckpt", ckpt,
                     "--data-dir", SPLIT_ROOT / args.dataset / "q08",
                     "--n-users", args.gate_users, "--k", args.k,
                     "--min-hist", args.min_hist,
                     "--device", args.device])
    passed = code == 0 and "KILL" not in out.upper().replace("KILL CRITERION", "")
    return passed, out


def audit(ckpt, tag, args):
    out = csv_path(tag, args.k, args.prefix)
    if out.exists() and not args.force:
        print(f"[skip] {out.name} exists")
        return True
    code, _ = run([PYTHON, "run_jlens_sasrec.py",
                   "--lxsr-root", LXSR,
                   "--data-dir", SPLIT_ROOT / args.dataset / "q08",
                   "--ckpt", ckpt, "--mode", "explainers",
                   "--n-users", args.n_users, "--k", args.k,
                   "--device", args.device, "--out-dir", RESULTS,
                   "--tag", tag, "--prefix", args.prefix,
                   "--min-hist", args.min_hist])
    return code == 0


def strong_rate(df, explainer, delta_min=0.2, gap_min=0.5):
    """Return the manuscript's joint-tail rate on the full method cohort."""
    g = df[(df.tap == TAP) & (df.explainer == explainer)].copy()
    if g.empty:
        return None
    gap = np.clip((g.delta_beh - g.jfwd) / g.delta_beh, 0.0, 1.0)
    measurable = g.delta_beh > 0.01
    return {"n": int(len(g)),
            "mr_median": float(g.loc[measurable, "mr"].median()),
            "gap_median": float(gap.loc[measurable].median()),
            "strong_rate": float(
                ((g.delta_beh > delta_min) & (gap > gap_min)).mean()
            )}


def collect(seeds, k, dataset="Movielens-1m", prefix="ml1m"):
    rows = []
    # Reporting depends only on the released per-user CSVs.  Checkpoint files
    # are needed to rerun an audit, but should not be required to summarize
    # one that has already been run (as in the compact artifact).
    result_sets = [("published", "")]
    result_sets.extend((f"s{seed}", f"_s{seed}") for seed in seeds)
    for label, tag in result_sets:
        path = csv_path(tag, k, prefix)
        if not path.exists():
            continue
        df = pd.read_csv(path)
        row = {"checkpoint": label, "users": int(df.user.nunique())}
        for explainer in ORDINARY:
            stat = strong_rate(df, explainer)
            if stat:
                row[f"{explainer}_n"] = stat["n"]
                row[f"{explainer}_mr_med"] = stat["mr_median"]
                row[f"{explainer}_strong"] = stat["strong_rate"]
        rows.append(row)
    return pd.DataFrame(rows)


def report(table):
    if table.empty:
        print("no SASRec checkpoints audited yet")
        return {}
    pd.set_option("display.width", 200)
    print("\n=== SASRec strong-gap rate for ordinary explainers "
          f"(tap {TAP}) ===")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    stats = {"n_checkpoints": int(len(table))}
    reps = table[table.checkpoint != "published"]
    for explainer in ORDINARY:
        col = f"{explainer}_strong"
        if col not in table:
            continue
        published = table.loc[table.checkpoint == "published", col]
        line = (f"  {explainer:10s} published "
                f"{published.iloc[0]:.3f}" if len(published)
                else f"  {explainer:10s} published n/a")
        if len(reps) >= 2:
            line += (f" | replicates {reps[col].mean():.3f}"
                     f" +- {reps[col].std():.3f}"
                     f" (min {reps[col].min():.3f}, max {reps[col].max():.3f})")
            stats[f"{explainer}_replicate_mean"] = float(reps[col].mean())
            stats[f"{explainer}_replicate_std"] = float(reps[col].std())
        print(line)
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123])
    ap.add_argument("--dataset", default="Movielens-1m",
                    help="LXSR dataset name (the split directory)")
    ap.add_argument("--prefix", default="ml1m",
                    help="dataset prefix of the result CSV names")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n-users", type=int, default=300)
    ap.add_argument("--gate-users", type=int, default=25)
    ap.add_argument("--min-hist", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip-gate", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.out is None:
        args.out = str(RESULTS / f"{args.prefix}_sasrec_seed_strong.csv")

    if not args.report_only:
        for label, tag, ckpt in checkpoints(args.seeds, args.dataset,
                                            args.prefix):
            if ckpt is None:
                continue
            print(f"\n########## SASRec {label} ##########", flush=True)
            if not args.skip_gate:
                passed, _ = gate(ckpt, args)
                print(f"[gate] {label}: {'PASS' if passed else 'FAIL'}")
                if not passed:
                    print(f"[gate] skipping audit for {label}")
                    continue
            audit(ckpt, tag, args)

    table = collect(args.seeds, args.k, args.dataset, args.prefix)
    stats = report(table)
    if not table.empty:
        table.to_csv(args.out, index=False)
        Path(args.out).with_suffix(".json").write_text(
            json.dumps(stats, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
