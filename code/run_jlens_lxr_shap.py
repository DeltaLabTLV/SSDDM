"""LXR and precomputed cluster-SHAP as real explainers on ML1M VAE.

LXR: architecture reconstructed from the checkpoint's state dict (matches the
ExplainRec/LXR Explainer: users_fc/items_fc -> concat -> Tanh -> Linear ->
Tanh -> Linear -> per-item scores). Behavioral sanity gate: LXR was trained
for counterfactual quality, so its DEL must clearly beat random — otherwise
the reconstruction is considered wrong and results are discarded.

SHAP: precomputed cluster-level values (1208 test users x clusters), mapped
to items via item_to_cluster; item attribution = its cluster's SHAP value.

Usage: python3 run_jlens_lxr_shap.py [--n-users 300] [--k 5]
"""
import argparse
import pickle
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

import jlens
import jlens_explainers as jx
from run_jlens_ml1m import CONFIGS, ROOT, CHECK, load_model, \
    target_and_competitors

LXR_CKPTS = {
    "ML1M": ("LXR_ML1M_VAE_26_38_128_3.185652725834087_1.420642300151426.pt",
             128),
    "Yahoo": ("LXR_Yahoo_VAE_neg-1.5pos_combined_19_26_128_"
              "18.958765029913238_4.92235962483309.pt", 128),
    "Pinterest": ("LXR_Pinterest_VAE_comb_4_27_32_"
                  "6.3443735346179855_1.472868807603448.pt", 32),
}


class LXRExplainer(nn.Module):
    def __init__(self, user_size, item_size, hidden_size):
        super().__init__()
        self.users_fc = nn.Linear(user_size, hidden_size)
        self.items_fc = nn.Linear(item_size, hidden_size)
        self.bottleneck = nn.Sequential(
            nn.Tanh(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, user_size))

    def forward(self, user_tensor, item_tensor):
        u = self.users_fc(user_tensor)
        i = self.items_fc(item_tensor)
        return self.bottleneck(torch.cat((u, i), dim=-1))


def load_lxr(data_name):
    ckpt, hidden = LXR_CKPTS[data_name]
    n_items = CONFIGS[data_name]["n_items"]
    model = LXRExplainer(n_items, n_items, hidden)
    model.load_state_dict(torch.load(CHECK / ckpt, map_location="cpu"))
    model.eval()
    return model


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="ML1M", choices=list(LXR_CKPTS))
    ap.add_argument("--n-users", type=int, default=300)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out-dir", default=str(ROOT / "jlens_results"))
    args = ap.parse_args(argv)

    classes = jlens.load_notebook_classes()
    insp = load_model("VAE", classes, args.data)
    lxr = load_lxr(args.data)
    ddir = ROOT / "processed_data" / args.data
    test = pd.read_csv(ddir / f"test_data_{args.data}.csv", index_col=0)
    X = torch.tensor(test.to_numpy(), dtype=torch.float32)
    shap = pickle.load(open(ddir / f"shap_values_VAE_{args.data}.pkl", "rb"))
    i2c = pickle.load(
        open(ddir / f"item_to_cluster_VAE_{args.data}.pkl", "rb"))
    n_items = X.shape[1]
    cluster_of = torch.tensor([int(i2c.get(i, 0)) for i in range(n_items)])
    n_sh = shap.shape[1]
    cluster_of = cluster_of.clamp(0, n_sh - 1)

    rows = []
    n_users = min(args.n_users, len(X))
    for u in range(n_users):
        x = X[u]
        if int(x.sum()) <= args.k:
            continue
        y, comps = target_and_competitors(insp, x, 5)

        onehot_y = torch.zeros(1, n_items)
        onehot_y[0, y] = 1.0
        with torch.no_grad():
            lxr_scores = lxr(x.unsqueeze(0), onehot_y)[0]
        shap_map = torch.tensor(shap[u])[cluster_of].float()

        expl = {
            "lxr": jx.topk_from_map(lxr_scores, x, args.k),
            "shap": jx.topk_from_map(shap_map, x, args.k),
            "saliency": jx.topk_from_map(jx.saliency(insp, x, y), x, args.k),
            "random": jx.random_topk(x, args.k, seed=u),
        }
        for name, S in expl.items():
            for r in jlens.mechanistic_eval(insp, x, S, y, competitors=comps,
                                            n_random=3, seed=u):
                rows.append({"user": u, "explainer": name, "k": args.k, **r})

    df = pd.DataFrame(rows)
    out = (Path(args.out_dir)
           / f"{args.data.lower()}_vae_lxr_shap_k{args.k}.csv")
    df.to_csv(out, index=False)
    print(f"saved {len(df)} rows -> {out}")

    pd.set_option("display.width", 200)
    z = df[df["tap"] == "z"]
    strong = z[z["delta_beh"].abs() > 0.01]
    agg = z.groupby("explainer").agg(
        DEL_mean=("delta_beh", "mean"),
        JComp_mean=("jcomp", "mean"),
        MR_median=("mr", "median"),
        selectivity=("selectivity_del", "median"),
    )
    mr_med = strong.groupby("explainer")["mr"].median().rename("MR_med_filt")
    agg = agg.join(mr_med)
    print(f"\n=== LXR/SHAP @ z, {z['user'].nunique()} users, K={args.k} ===")
    print(agg.round(4).sort_values("DEL_mean", ascending=False))

    del_lxr = agg.loc["lxr", "DEL_mean"]
    del_rnd = agg.loc["random", "DEL_mean"]
    gate = "PASS" if del_lxr > 2 * del_rnd else "FAIL"
    print(f"\nLXR sanity gate (DEL_lxr > 2*DEL_random): {gate} "
          f"({del_lxr:.3f} vs {del_rnd:.3f})")
    return df


if __name__ == "__main__":
    main()
