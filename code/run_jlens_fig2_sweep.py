"""Full-test-set K-sweep for the Figure 2 explainers (tail-of-MR trend).

For every eligible test user (hist > K) and K in {1,3,5,10,20}: top-K
explanations from the four methods plotted in Figure 2 (saliency, SPINRec,
LXR, popularity) and the mechanistic evaluation at tap z with one random-set
control.  One output CSV per K so runs are independently restartable.

Usage: python3 run_jlens_fig2_sweep.py --k 5 [--data ML1M]
Writes jlens_results/<data>_vae_fig2_k<K>.csv
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch

import jlens
import jlens_explainers as jx
from run_jlens_ml1m import ROOT, load_model, load_pop_array, \
    target_and_competitors
from run_jlens_lxr_shap import load_lxr


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="ML1M")
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--n-users", type=int, default=-1)
    ap.add_argument("--out-dir", default=str(ROOT / "jlens_results"))
    args = ap.parse_args(argv)

    classes = jlens.load_notebook_classes()
    insp = load_model("VAE", classes, args.data)
    lxr = load_lxr(args.data)
    pop = load_pop_array(args.data)
    ddir = ROOT / "processed_data" / args.data
    test = pd.read_csv(ddir / f"test_data_{args.data}.csv", index_col=0)
    train = pd.read_csv(ddir / f"train_data_{args.data}.csv", index_col=0)
    X = torch.tensor(test.to_numpy(), dtype=torch.float32)
    train_X = torch.tensor(train.to_numpy(), dtype=torch.float32)
    n_items = X.shape[1]

    n_users = len(X) if args.n_users < 0 else min(args.n_users, len(X))
    rows = []
    t0 = time.time()
    done = 0
    for u in range(n_users):
        x = X[u]
        if int(x.sum()) <= args.k:
            continue
        y, comps = target_and_competitors(insp, x, 5)
        onehot_y = torch.zeros(1, n_items)
        onehot_y[0, y] = 1.0
        with torch.no_grad():
            lxr_scores = lxr(x.unsqueeze(0), onehot_y)[0]
        try:
            expl = {
                "saliency": jx.topk_from_map(jx.saliency(insp, x, y),
                                             x, args.k),
                "spinrec": jx.topk_from_map(
                    jx.spinrec_real(insp, x, y, args.k, train_X,
                                    n_baselines=10, steps=10, seed=u),
                    x, args.k),
                "lxr": jx.topk_from_map(lxr_scores, x, args.k),
                "pop": jx.pop_topk(x, pop, args.k),
            }
        except RuntimeError as e:
            print(f"user {u}: explainer failed: {e}", file=sys.stderr)
            continue
        for name, S in expl.items():
            for r in jlens.mechanistic_eval(insp, x, S, y, competitors=comps,
                                            taps=["z"], n_random=1, seed=u):
                rows.append({"user": u, "hist_len": int(x.sum()), "y": y,
                             "explainer": name, "k": args.k, **r})
        done += 1
        if done % 50 == 0:
            el = time.time() - t0
            print(f"K={args.k}: {done} users, {el:.0f}s "
                  f"({el / done:.2f}s/user)", flush=True)

    df = pd.DataFrame(rows)
    out = Path(args.out_dir) / f"{args.data.lower()}_vae_fig2_k{args.k}.csv"
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"saved {len(df)} rows ({df.user.nunique()} users) -> {out}")
    strong = df[df.delta_beh > 0.01]
    for e in ["saliency", "spinrec", "lxr", "pop"]:
        z = strong[strong.explainer == e]
        print(f"K={args.k} {e}: n={z.user.nunique()} "
              f"median MR {z.mr.median():.3f} | q10 {z.mr.quantile(0.1):.3f}")


if __name__ == "__main__":
    main()
