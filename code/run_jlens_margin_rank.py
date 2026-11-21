"""Outcome-function robustness for the saved K=10 white-box sets.

For every saved deletion set (saliency / attack_lam0 / attack_lam1 /
constraint) recompute the tap-z audit under two alternative outcomes:

- margin outcome  m = s_y - max_{c in C} s_c  (target vs. strongest of the
  five clean competitors), giving margin-Delta and margin-MR_fwd;
- rank outcome: rank of y among the user's original non-history items in the
  clean, corrupted, and comp-patched states -> does the deletion actually
  change the recommendation (flip@1 / leaves top-10), and does the clean
  subspace patch reproduce that behavioral flip?

Usage: python3 run_jlens_margin_rank.py [--data ML1M Yahoo Pinterest]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import jlens
from run_jlens_ml1m import CONFIGS, ROOT, load_model, target_and_competitors

TAP = "z"


def rank_of(logits_row, y, nonhist_mask):
    scores = logits_row[nonhist_mask]
    return int((scores > logits_row[y]).sum().item()) + 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["ML1M", "Yahoo", "Pinterest"],
                    choices=list(CONFIGS))
    ap.add_argument("--out-dir", default=str(ROOT / "jlens_results"))
    args = ap.parse_args(argv)

    classes = jlens.load_notebook_classes()
    out_dir = Path(args.out_dir)

    for data in args.data:
        insp = load_model("VAE", classes, data)
        test = pd.read_csv(ROOT / "processed_data" / data /
                           f"test_data_{data}.csv", index_col=0)
        X = torch.tensor(test.to_numpy(), dtype=torch.float32)
        adv = pd.read_csv(out_dir / f"{data.lower()}_vae_adversarial_k10.csv")
        adv = adv[adv.tap == TAP]

        rows = []
        for u, grp in adv.groupby("user"):
            x = X[int(u)]
            y, comps = target_and_competitors(insp, x, 5)
            nonhist = x == 0
            with torch.no_grad():
                l_clean, acts_clean = insp.run(x)
            m_clean = float(l_clean[0, y] - l_clean[0, comps].max())
            r_clean = rank_of(l_clean[0], y, nonhist)
            V = jlens.item_directions(insp, x, TAP, [y] + comps)
            Q = jlens.orthonormal_basis(V)
            for _, row in grp.iterrows():
                S = json.loads(row.S)
                with torch.no_grad():
                    l_corr, acts_corr = insp.run(jlens.delete_items(x, S))
                    l_patch = jlens.patch_logits(
                        insp, TAP, acts_clean[TAP], acts_corr[TAP], Q, "comp")
                m_corr = float(l_corr[0, y] - l_corr[0, comps].max())
                m_patch = float(l_patch[0, y] - l_patch[0, comps].max())
                dm = m_clean - m_corr
                jm = m_clean - m_patch
                mrm = jm / dm if abs(dm) > 1e-9 else np.nan
                rows.append({
                    "user": int(u), "method": row.method,
                    "delta_logit": row.delta_beh, "mr_logit": row.mr,
                    "margin_clean": m_clean, "margin_delta": dm,
                    "margin_mr_fwd": mrm,
                    "rank_clean": r_clean,
                    "rank_corr": rank_of(l_corr[0], y, nonhist),
                    "rank_patch": rank_of(l_patch[0], y, nonhist),
                })
        df = pd.DataFrame(rows)
        out = out_dir / f"{data.lower()}_vae_margin_rank_k10.csv"
        df.to_csv(out, index=False)

        print(f"\n=== {data} (n users {df.user.nunique()}) ===")
        for m in ["saliency", "attack_lam0", "attack_lam1", "constraint"]:
            d = df[df.method == m]
            ok = d[d.margin_delta > 0.01]
            strong = d[(d.margin_delta > 0.2) & (d.margin_mr_fwd < 0.5)]
            flip1 = (d.rank_corr > 1).mean()
            flip10 = (d.rank_corr > 10).mean()
            agree = ((d.rank_patch > 1) == (d.rank_corr > 1)).mean()
            sp = d[["margin_mr_fwd", "mr_logit"]].dropna()
            rho = (sp.margin_mr_fwd.corr(sp.mr_logit, method="spearman")
                   if len(sp) > 2 else np.nan)
            print(f"{m:>12}: dm med {d.margin_delta.median():.2f} | "
                  f"margin-MRf med {ok.margin_mr_fwd.median():.2f} "
                  f"(n={len(ok)}) | strong {len(strong)}/{len(d)} "
                  f"({100*len(strong)/max(len(d),1):.0f}%) | "
                  f"flip@1 {100*flip1:.0f}% flip@10 {100*flip10:.0f}% | "
                  f"patch-flip agree {100*agree:.0f}% | "
                  f"rho(margin,logit) {rho:.2f}")


if __name__ == "__main__":
    main()
