"""Complete the Figure 2 K-sweep: LIME, occlusion, cluster-SHAP, random.

Same cohort and aggregation as run_jlens_fig2_sweep.py (full ML1M test set,
tap z, n_random=1), all K values in one pass: the attribution maps are
K-independent, so each user's LIME/occlusion/SHAP maps are computed once and
only the mechanistic evaluation runs per K.

Usage: python3 run_jlens_fig2_extras.py
Writes jlens_results/ml1m_vae_fig2x_k<K>.csv (same schema as fig2 files).
"""
import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import jlens
import jlens_explainers as jx
from run_jlens_ml1m import ROOT, load_model, target_and_competitors
from run_jlens_lime_occlusion import lime_map, occlusion_map


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10, 20])
    ap.add_argument("--n-users", type=int, default=-1)
    args = ap.parse_args(argv)

    classes = jlens.load_notebook_classes()
    insp = load_model("VAE", classes, "ML1M")
    ddir = ROOT / "processed_data" / "ML1M"
    test = pd.read_csv(ddir / "test_data_ML1M.csv", index_col=0)
    X = torch.tensor(test.to_numpy(), dtype=torch.float32)
    shap = pickle.load(open(ddir / "shap_values_VAE_ML1M.pkl", "rb"))
    i2c = pickle.load(open(ddir / "item_to_cluster_VAE_ML1M.pkl", "rb"))
    n_items = X.shape[1]
    cluster_of = torch.tensor([int(i2c.get(i, 0)) for i in range(n_items)])
    cluster_of = cluster_of.clamp(0, shap.shape[1] - 1)

    n_users = len(X) if args.n_users < 0 else min(args.n_users, len(X))
    rows = {k: [] for k in args.ks}
    t0 = time.time()
    for u in range(n_users):
        x = X[u]
        hist = int(x.sum())
        if hist <= min(args.ks):
            continue
        y, comps = target_and_competitors(insp, x, 5)
        maps = {
            "lime": lime_map(insp, x, y, n_samples=256, seed=u),
            "occlusion": occlusion_map(insp, x, y),
            "shap": torch.tensor(shap[u])[cluster_of].float(),
        }
        for k in args.ks:
            if hist <= k:
                continue
            expl = {name: jx.topk_from_map(m, x, k)
                    for name, m in maps.items()}
            expl["random"] = jx.random_topk(x, k, seed=u)
            for name, S in expl.items():
                for r in jlens.mechanistic_eval(insp, x, S, y,
                                                competitors=comps,
                                                taps=["z"], n_random=1,
                                                seed=u):
                    rows[k].append({"user": u, "hist_len": hist, "y": y,
                                    "explainer": name, "k": k, **r})
        if (u + 1) % 100 == 0:
            el = time.time() - t0
            print(f"{u + 1}/{n_users} users ({el / (u + 1):.2f}s/user)",
                  flush=True)

    for k in args.ks:
        df = pd.DataFrame(rows[k])
        out = ROOT / "jlens_results" / f"ml1m_vae_fig2x_k{k}.csv"
        df.to_csv(out, index=False)
        strong = df[df.delta_beh > 0.01]
        for e in ["lime", "occlusion", "shap", "random"]:
            z = strong[strong.explainer == e]
            print(f"K={k} {e:9s}: n={z.user.nunique():4d} "
                  f"q10 {z.mr.quantile(0.1):5.2f} med {z.mr.median():5.2f}")


if __name__ == "__main__":
    main()
