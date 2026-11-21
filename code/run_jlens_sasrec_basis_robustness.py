"""Reference-state robustness for the SASRec four-state audit.

The attention mask is discrete.  Consequently, the activation path is
evaluated at both fixed endpoint contexts rather than by inventing a
continuous mask interpolation.  The reported bases are clean, corrupted,
clean-context path, deleted-context path, their itemwise average, and the
endpoint union.

Usage:
  python run_jlens_sasrec_basis_robustness.py --ckpt <checkpoint> \
      --methods occlusion lime attack --n-users 100 --k 5
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import jlens_sasrec as js
from run_jlens_sasrec import (build_cohort, greedy_attack,
                               target_and_competitors)

DEFAULT_LXSR = Path(__file__).resolve().parent.parent / "external" / "LXSR"


def deletion_sets(aud, tokens, y, k, uid, methods):
    sets = {}
    if "saliency" in methods:
        sets["saliency"] = js.topk_positions(
            js.grad_x_input_map(aud, tokens, y), k)
    if "occlusion" in methods:
        sets["occlusion"] = js.topk_positions(
            js.occlusion_map(aud, tokens, y), k)
    if "lime" in methods:
        sets["lime"] = js.topk_positions(
            js.lime_map(aud, tokens, y, seed=uid), k)
    if "random" in methods:
        sets["random"] = list(
            np.random.default_rng(uid).choice(len(tokens), size=k,
                                              replace=False))
    return sets


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lxsr-root", default=str(DEFAULT_LXSR))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--n-users", type=int, default=100)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--tap", default="block1")
    ap.add_argument("--path-steps", type=int, default=8)
    ap.add_argument("--methods", nargs="+",
                    default=["occlusion", "lime", "attack"],
                    choices=["saliency", "occlusion", "lime", "random",
                             "attack"])
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", default="jlens_results")
    ap.add_argument("--tag", default="")
    ap.add_argument("--prefix", default="ml1m")
    args = ap.parse_args(argv)

    lxsr_root = Path(args.lxsr_root)
    data_dir = Path(args.data_dir) if args.data_dir else \
        lxsr_root / "data/splitted/global_timesplit/val_by_time/Movielens-1m/q08"
    device = args.device if torch.cuda.is_available() else "cpu"
    model, tok, _ = js.load_sasrec(
        lxsr_root, args.ckpt, data_dir / "train.csv", device=device)
    aud = js.SasRecAuditor(model, device=device)
    if args.tap not in aud.taps:
        raise ValueError(f"unknown tap {args.tap!r}; choose from {aud.taps}")

    users, _ = build_cohort(lxsr_root, data_dir, args.k)
    users = users[:args.n_users]
    rows = []
    t0 = time.time()
    for pos, (uid, seq) in enumerate(users):
        tokens = [tok.item2id[i] for i in seq][-tok.max_seq_length:]
        y, comps = target_and_competitors(aud, tok, tokens)
        sets = deletion_sets(aud, tokens, y, args.k, uid, args.methods)
        if "attack" in args.methods:
            S, _ = greedy_attack(aud, tokens, y, comps, args.k,
                                 tap=args.tap, lam=args.lam, seed=uid)
            sets[f"attack_lam{args.lam:g}"] = S

        for method, S in sets.items():
            result = js.audit_set(
                aud, tokens, S, y, comps, taps=[args.tap],
                ref_states=("all",), path_steps=args.path_steps)[0]
            rows.append({"user": uid, "method": method,
                         "S": json.dumps([int(i) for i in sorted(S)]),
                         "path_steps": args.path_steps, **result})
        if (pos + 1) % 5 == 0 or pos + 1 == len(users):
            elapsed = time.time() - t0
            print(f"{pos + 1}/{len(users)} users "
                  f"({elapsed / (pos + 1):.1f}s/user)", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (
        f"{args.prefix}_sasrec_basis_robustness_k{args.k}{args.tag}.csv")
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"saved {len(df)} rows -> {out}")

    positive = df[df.delta_beh > 0.01].copy()
    gap_cols = [
        "clean_gap", "symmetric_gap",
        "clean_gap_corrupt", "symmetric_gap_corrupt",
        "clean_gap_path_clean", "symmetric_gap_path_clean",
        "clean_gap_path_deleted", "symmetric_gap_path_deleted",
        "clean_gap_path_avg", "symmetric_gap_path_avg",
        "clean_gap_union", "symmetric_gap_union",
    ]
    print(positive.groupby("method")[gap_cols].median().round(3).to_string())
    return df


if __name__ == "__main__":
    main()
