"""One concrete ML1M case: same score drop, different mechanism, with titles.

Picks a user from the saved K=10 white-box cohort where the attack set is
behaviorally strong and low-recovery while saliency is clean-recoverable,
maps item indices to MovieLens titles, and reports score/rank effects plus
the opened-decoder-unit share for both sets.

Internal item index -> MovieLens id: ratings with rating>=4, items with >=2
such ratings, ids sorted ascending (validated against pop_dict, corr 0.9998).

Usage: python3 run_jlens_case_study.py [--movies /path/to/movies.dat]
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch

import jlens
from run_jlens_ml1m import ROOT, load_model, target_and_competitors

TAP = "z"


def build_title_map(movies_path):
    r = pd.read_csv(ROOT / "processed_data/ML1M/ratings.dat", sep="::",
                    engine="python", names=["u", "i", "rt", "ts"])
    cnt = r[r.rt >= 4].groupby("i").size()
    kept = sorted(cnt[cnt >= 2].index)
    titles = {}
    with open(movies_path, encoding="latin-1") as f:
        for line in f:
            mid, title, genres = line.strip().split("::")
            titles[int(mid)] = (title, genres)
    return {idx: titles.get(mid, (f"movie {mid}", "?"))
            for idx, mid in enumerate(kept)}


def describe(name, S, insp, x, y, comps, Q, acts_clean, l_clean, tmap):
    nonhist = x == 0
    with torch.no_grad():
        l_corr, acts_corr = insp.run(jlens.delete_items(x, S))
        l_patch = jlens.patch_logits(insp, TAP, acts_clean[TAP],
                                     acts_corr[TAP], Q, "comp")
    delta = float(l_clean[0, y] - l_corr[0, y])
    mrf = float(l_clean[0, y] - l_patch[0, y]) / delta
    rank_corr = int((l_corr[0][nonhist] > l_corr[0, y]).sum()) + 1
    # exact opened-unit share of the decoder deletion effect (single ReLU)
    z0, z1 = acts_clean[TAP][0], acts_corr[TAP][0]
    dec_lin = insp.model.decoder[0]
    W, b = dec_lin.weight, dec_lin.bias
    pre0, pre1 = W @ z0 + b, W @ z1 + b
    opened = (pre0 <= 0) & (pre1 > 0)
    W_out = insp.model.decoder[-1].weight[y]
    eff = W_out * (torch.relu(pre1) - torch.relu(pre0))  # per-unit effect on s_y
    switch_share = float(-eff[opened].sum() / delta)
    print(f"\n--- {name}: Delta {delta:.2f}, clean MRf {mrf:.2f}, "
          f"rank of target after deletion {rank_corr}, "
          f"opened units {int(opened.sum())}, switch share {switch_share:.2f}")
    for i in sorted(S):
        t, g = tmap[i]
        print(f"    {t}  [{g.split('|')[0]}]")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--movies", required=True)
    ap.add_argument("--user", type=int, default=None)
    args = ap.parse_args(argv)

    tmap = build_title_map(args.movies)
    classes = jlens.load_notebook_classes()
    insp = load_model("VAE", classes, "ML1M")
    test = pd.read_csv(ROOT / "processed_data/ML1M/test_data_ML1M.csv",
                       index_col=0)
    X = torch.tensor(test.to_numpy(), dtype=torch.float32)

    adv = pd.read_csv(ROOT / "jlens_results/ml1m_vae_adversarial_k10.csv")
    adv = adv[adv.tap == TAP]
    att = adv[adv.method == "attack_lam1"].set_index("user")
    sal = adv[adv.method == "saliency"].set_index("user")
    if args.user is None:
        cand = att[(att.mr < 0.35) & (att.delta_beh > 0.5)
                   & (sal.mr > 0.95) & (sal.delta_beh > 0.5)]
        cand = cand.sort_values("delta_beh", ascending=False)
        print("candidates:", cand.index.tolist())
        user = int(cand.index[0])
    else:
        user = args.user

    x = X[user]
    y, comps = target_and_competitors(insp, x, 5)
    with torch.no_grad():
        l_clean, acts_clean = insp.run(x)
    V = jlens.item_directions(insp, x, TAP, [y] + comps)
    Q = jlens.orthonormal_basis(V)

    print(f"\nUSER {user}: history {int(x.sum())} items")
    print(f"TARGET: {tmap[y][0]}  [{tmap[y][1]}]")
    print("competitors:", "; ".join(tmap[c][0] for c in comps))
    hist_items = torch.where(x > 0)[0].tolist()
    genre_counts = pd.Series(
        [tmap[i][1].split("|")[0] for i in hist_items]).value_counts()
    print("history top genres:", dict(genre_counts.head(5)))

    for name, key in [("saliency", "saliency"), ("white-box l=1",
                                                 "attack_lam1")]:
        S = json.loads(adv[(adv.user == user)
                           & (adv.method == key)].iloc[0].S)
        describe(name, S, insp, x, y, comps, Q, acts_clean, l_clean, tmap)


if __name__ == "__main__":
    main()
