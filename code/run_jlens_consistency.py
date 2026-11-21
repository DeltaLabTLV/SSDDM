"""Consistency experiment following the TP protocol of Mohammadi et al.
(RecSys'24/'25): evaluate explainers against VAE checkpoints at 25/50/75/100%
of training and compare how stable each evaluation metric's explainer
ranking is across performance levels.

Per user x explainer x TP we record:
- flip1: does the top-1 recommendation change after deleting the top-K
  explanation (the volatile event their POS@1-style metrics count)
- disp3: rank-discounted share of the clean top-3 that leaves post-deletion
  top-5 (their beyond-top-1 aggregation, single-shot variant)
- del_logit, jcomp, mr (tap z), rdlm3 (rank-discounted logit mediation),
  selectivity, margin (clean logit gap between y and the best competitor)

Usage: python3 run_jlens_consistency.py [--n-users 300] [--k 5]
"""
import argparse
import math
import time
from pathlib import Path

import pandas as pd
import torch

import jlens
from run_jlens_ml1m import (CHECK, CONFIGS, DATA, ROOT, VAE_CONF,
                            explanations_for, load_pop_array,
                            target_and_competitors)

TPS = (25, 50, 75, 100)


def load_tp_model(tp, classes):
    kw = dict(device=torch.device("cpu"),
              num_features=CONFIGS["ML1M"]["n_items"],
              num_items=CONFIGS["ML1M"]["n_items"], demographic=False)
    model = classes["VAE"](VAE_CONF, **kw)
    model.load_state_dict(torch.load(CHECK / "tp" / f"VAE_ML1M_tp{tp}.pt",
                                     map_location="cpu"))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return jlens.VAEInspector(model)


def rank_after_deletion(inspector, x, S, items, top_t=5):
    with torch.no_grad():
        logits, _ = inspector.run(jlens.delete_items(x, S))
    masked = logits[0].clone()
    masked[x > 0] = float("-inf")
    top_after = masked.argsort(descending=True)[:top_t].tolist()
    flip1 = items[0] != top_after[0]
    w = [1.0 / math.log2(j + 2) for j in range(min(3, len(items)))]
    disp = sum(wj for wj, it in zip(w, items[:3]) if it not in top_after)
    return flip1, disp / sum(w)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-users", type=int, default=300)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out-dir", default=str(ROOT / "jlens_results"))
    args = ap.parse_args(argv)

    classes = jlens.load_notebook_classes()
    pop = load_pop_array()
    test = pd.read_csv(DATA / "test_data_ML1M.csv", index_col=0)
    X = torch.tensor(test.to_numpy(), dtype=torch.float32)
    train = pd.read_csv(DATA / "train_data_ML1M.csv", index_col=0)
    train_X = torch.tensor(train.to_numpy(), dtype=torch.float32)

    rows = []
    t0 = time.time()
    for tp in TPS:
        insp = load_tp_model(tp, classes)
        n_users = min(args.n_users, len(X))
        for u in range(n_users):
            x = X[u]
            if int(x.sum()) <= args.k:
                continue
            y, comps = target_and_competitors(insp, x, 5)
            with torch.no_grad():
                logits, _ = insp.run(x)
            masked = logits[0].clone()
            masked[x > 0] = float("-inf")
            top3 = masked.argsort(descending=True)[:3].tolist()
            margin = float(masked[y] - masked[comps[0]])

            hist_len = int(x.sum())
            spin_probs = torch.clamp(pop / pop.sum() * hist_len, 0.0, 1.0)
            expl = explanations_for(insp, x, y, args.k, pop, spin_probs,
                                    seed=u, train_X=train_X)
            for name, S in expl.items():
                flip1, disp3 = rank_after_deletion(insp, x, S, top3)
                r = jlens.mechanistic_eval(insp, x, S, y, competitors=comps,
                                           taps=["z"])[0]
                rdlm3 = jlens.rank_discounted_logit_mediation(
                    insp, x, S, top3, tap="z")
                rows.append({
                    "tp": tp, "user": u, "explainer": name, "k": args.k,
                    "margin": margin, "flip1": int(flip1), "disp3": disp3,
                    "rdlm3": rdlm3, **r})
        print(f"TP{tp} done ({time.time() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    out = Path(args.out_dir) / f"ml1m_vae_consistency_k{args.k}.csv"
    df.to_csv(out, index=False)
    print(f"saved {len(df)} rows -> {out}")

    # ---- E1: ranking stability across TP levels, per metric
    pd.set_option("display.width", 200)
    metrics = {"flip1": True, "disp3": True, "delta_beh": True,
               "jcomp": True, "mr": True, "rdlm3": True}
    print("\n=== Explainer ranking per TP (by metric) ===")
    taus = {}
    for m, _ in metrics.items():
        per_tp = df.groupby(["tp", "explainer"])[m].mean().unstack()
        ranks = per_tp.rank(axis=1, ascending=False)
        tau_vals = []
        tps = list(per_tp.index)
        for i in range(len(tps)):
            for j in range(i + 1, len(tps)):
                tau = ranks.loc[tps[i]].corr(ranks.loc[tps[j]],
                                             method="kendall")
                tau_vals.append(tau)
        taus[m] = sum(tau_vals) / len(tau_vals)
    print("\n-- mean pairwise Kendall tau of explainer rankings across TP --")
    for m, t in sorted(taus.items(), key=lambda kv: kv[1]):
        print(f"  {m:10s} tau = {t:.3f}")

    # ---- E2: flip volatility vs margin and selectivity
    print("\n=== E2: flip1 rate by clean-margin quartile ===")
    df["margin_q"] = pd.qcut(df["margin"], 4, labels=["q1", "q2", "q3", "q4"])
    print(df.groupby("margin_q", observed=True)["flip1"].mean().round(3))
    print("\nflip1 rate by selectivity half (within |delta_beh|>0.01):")
    s = df[df["delta_beh"].abs() > 0.01].copy()
    s["sel_half"] = s["selectivity_del"] > s["selectivity_del"].median()
    print(s.groupby("sel_half")["flip1"].mean().round(3))
    print("\nSpearman(margin, flip1) =",
          round(df["margin"].corr(df["flip1"], method="spearman"), 3))
    return df


if __name__ == "__main__":
    main()
