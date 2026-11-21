"""Append a pure-DEL white-box-search control to an existing attack CSV.

This avoids recomputing saliency, lambda=1, and constrained searches when an
older artifact lacks only the lambda=0 control.  Existing rows and user ids are
preserved exactly.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch

import jlens
import jlens_adversarial as ja
from run_jlens_ml1m import CONFIGS, ROOT, load_model, target_and_competitors


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="ML1M", choices=list(CONFIGS))
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--tap", default="z")
    args = ap.parse_args(argv)

    path = (ROOT / "jlens_results"
            / f"{args.data.lower()}_vae_adversarial_k{args.k}.csv")
    old = pd.read_csv(path)
    old = old[old["method"] != "attack_lam0"].copy()
    users = sorted(old["user"].unique())

    classes = jlens.load_notebook_classes()
    inspector = load_model("VAE", classes, args.data)
    test = pd.read_csv(
        ROOT / "processed_data" / args.data
        / f"test_data_{args.data}.csv", index_col=0)
    X = torch.tensor(test.to_numpy(), dtype=torch.float32)

    rows = []
    for pos, u in enumerate(users):
        x = X[int(u)]
        y, comps = target_and_competitors(inspector, x, 5)
        S = ja.greedy_attack(
            inspector, x, y, args.k, args.tap, comps, lam=0.0, seed=int(u))
        for r in jlens.mechanistic_eval(
                inspector, x, S, y, competitors=comps, n_random=3,
                seed=int(u)):
            rows.append({"user": int(u), "method": "attack_lam0",
                         "k": args.k, "S": json.dumps(sorted(S)), **r})
        if (pos + 1) % 10 == 0:
            print(f"{pos + 1}/{len(users)} users", flush=True)

    out = pd.concat([old, pd.DataFrame(rows)], ignore_index=True, sort=False)
    out.to_csv(path, index=False)
    print(f"appended {len(rows)} lambda=0 rows -> {path}")
    return out


if __name__ == "__main__":
    main()
