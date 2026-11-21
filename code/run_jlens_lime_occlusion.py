"""LIME-style local surrogate and single-item occlusion as explainers.

LIME: for each user, sample B binary keep/drop perturbations of the history,
score the target on each, and fit a weighted ridge surrogate (exponential
kernel on kept-fraction distance); the coefficients attribute the target
score to individual interactions.  Occlusion: the deletion effect of each
single interaction, batched.  Both are audited with the standard mechanistic
evaluation on the same cohorts as the published-explainer table (the user
ids of the LXR/SHAP runs), K=5, tap z.

Usage: python3 run_jlens_lime_occlusion.py [--data ML1M ...]
Writes jlens_results/<data>_vae_lime_occl_k5.csv and prints table-ready rows.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import jlens
import jlens_explainers as jx
from run_jlens_ml1m import ROOT, load_model, target_and_competitors


def lime_map(insp, x, y, n_samples=256, keep_prob=0.5, ridge=1.0,
             kernel_width=0.25, seed=0, batch=128):
    hist = x.nonzero().flatten()
    h = len(hist)
    gen = torch.Generator().manual_seed(seed)
    masks = (torch.rand(n_samples, h, generator=gen) < keep_prob).float()
    masks[0] = 1.0  # anchor at the unperturbed history
    xs = x.unsqueeze(0).repeat(n_samples, 1)
    xs[:, hist] = masks
    scores = []
    with torch.no_grad():
        for i in range(0, n_samples, batch):
            logits, _ = insp.run(xs[i:i + batch])
            scores.append(logits[:, y])
    s = torch.cat(scores)
    kept_frac = masks.mean(dim=1)
    w = torch.exp(-((1.0 - kept_frac) ** 2) / kernel_width ** 2)
    Xd = masks - masks.mean(dim=0, keepdim=True)
    sd = s - s.mean()
    A = Xd.T @ (w[:, None] * Xd) + ridge * torch.eye(h)
    b = Xd.T @ (w * sd)
    coef = torch.linalg.solve(A, b)
    m = torch.full_like(x, float("-inf"))
    m[hist] = coef
    return m


def occlusion_map(insp, x, y, batch=256):
    hist = x.nonzero().flatten()
    xs = x.unsqueeze(0).repeat(len(hist), 1)
    xs[torch.arange(len(hist)), hist] = 0.0
    with torch.no_grad():
        s0 = insp.run(x)[0][0, y]
        drops = []
        for i in range(0, len(hist), batch):
            logits, _ = insp.run(xs[i:i + batch])
            drops.append(s0 - logits[:, y])
    m = torch.full_like(x, float("-inf"))
    m[hist] = torch.cat(drops)
    return m


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["ML1M", "Yahoo", "Pinterest"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n-samples", type=int, default=256)
    args = ap.parse_args(argv)

    classes = jlens.load_notebook_classes()
    for data in args.data:
        insp = load_model("VAE", classes, data)
        test = pd.read_csv(ROOT / "processed_data" / data
                           / f"test_data_{data}.csv", index_col=0)
        X = torch.tensor(test.to_numpy(), dtype=torch.float32)
        cohort_src = pd.read_csv(ROOT / "jlens_results"
                                 / f"{data.lower()}_vae_lxr_shap_k5.csv")
        cohort = sorted(int(u) for u in cohort_src.user.unique())

        rows = []
        for pos, u in enumerate(cohort):
            x = X[u]
            if int(x.sum()) <= args.k:
                continue
            y, comps = target_and_competitors(insp, x, 5)
            expl = {
                "lime": jx.topk_from_map(
                    lime_map(insp, x, y, n_samples=args.n_samples, seed=u),
                    x, args.k),
                "occlusion": jx.topk_from_map(occlusion_map(insp, x, y),
                                              x, args.k),
            }
            for name, S in expl.items():
                for r in jlens.mechanistic_eval(insp, x, S, y,
                                                competitors=comps,
                                                taps=["z"], n_random=3,
                                                seed=u):
                    rows.append({"user": u, "explainer": name,
                                 "k": args.k, **r})
            if (pos + 1) % 50 == 0:
                print(f"  {data}: {pos + 1}/{len(cohort)}", flush=True)

        df = pd.DataFrame(rows)
        out = (ROOT / "jlens_results"
               / f"{data.lower()}_vae_lime_occl_k{args.k}.csv")
        df.to_csv(out, index=False)
        print(f"saved {len(df)} rows -> {out}")
        for e in ["lime", "occlusion"]:
            z = df[df.explainer == e]
            strong = z[z.delta_beh > 0.01]
            print(f"{data} {e}: Del {z.delta_beh.mean():.2f} | "
                  f"MRf {strong.mr.median():.2f} | "
                  f"MRs {strong.smr.median():.2f} | "
                  f"Sel {z.selectivity_del.median():.2f} | n={z.user.nunique()}")


if __name__ == "__main__":
    main()
