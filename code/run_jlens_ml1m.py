"""Item-Jacobian lens benchmark on trained ML1M recommenders.

For every test user: recommend y (top-1 non-history item), produce top-K
explanations with five explainers (saliency, IG-zero, SPINRec-lite,
popularity, random), and run the mechanistic evaluation (deletion effect,
causal patching in target-aligned coordinates, MR/JComp/JRestore,
selectivity, random-subspace control) at every tap.

Usage:
    python3 run_jlens_ml1m.py --model VAE --n-users -1 --k 5
Outputs a per-row CSV into <repo>/jlens_results/ and prints aggregates.
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import jlens
import jlens_explainers as jx

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "processed_data" / "ML1M"
CHECK = ROOT / "check"
N_ITEMS = 3381
VAE_CONF = {"enc_dims": [512, 128], "dropout": 0.5,
            "anneal_cap": 0.2, "total_anneal_steps": 200000}

CONFIGS = {
    "ML1M": {"n_items": 3381, "enc_dims": [512, 128],
             "vae_ckpt": "VAE_ML1M_0.0007_128_10.pt",
             "mlp_ckpt": "MLP1_ML1M_0.0076_256_7.pt", "mlp_hidden": 32,
             "ncf_ckpt": "NCF_ML1M_5e-05_64_16.pt", "ncf_factor": 8},
    "Yahoo": {"n_items": 4604, "enc_dims": [512, 128],
              "vae_ckpt": "VAE_Yahoo_0.0001_128_13.pt",
              "ncf_ckpt": "NCF_Yahoo_0.001_64_21_0.pt", "ncf_factor": 8},
    "Pinterest": {"n_items": 9362, "enc_dims": [256, 64],
                  "vae_ckpt": "VAE_Pinterest_12_18_0.0001_256.pt",
                  "ncf_ckpt": "NCF2_Pinterest_9e-05_32_9_10.pt",
                  "ncf_factor": 64},
}


def load_model(name, classes, data_name="ML1M", ckpt=None):
    """Build `name` for `data_name` and load its weights.

    `ckpt` overrides the released checkpoint with another file of the same
    architecture (used for the seed replicates in check/seeds/); it accepts a
    bare filename resolved under check/ or an explicit path.
    """
    cfg = CONFIGS[data_name]
    n_items = cfg["n_items"]
    kw = dict(device=torch.device("cpu"), num_features=n_items,
              num_items=n_items, demographic=False)
    if name == "VAE":
        conf = dict(VAE_CONF, enc_dims=cfg["enc_dims"])
        model = classes["VAE"](conf, **kw)
        path = CHECK / cfg["vae_ckpt"]
    elif name == "MLP":
        model = classes["MLP"](cfg["mlp_hidden"], **kw)
        path = CHECK / cfg["mlp_ckpt"]
    elif name == "NCF":
        factor, layers = cfg["ncf_factor"], 3
        gmf = classes["GMF_model"](hidden_size=factor, **kw)
        mlp = classes["MLP_model"](factor, layers, **kw)
        model = classes["NCF"](
            factor, layers, 0.5, "NeuMF-pre", GMF_model=gmf,
            MLP_model=mlp, **kw)
        path = CHECK / cfg["ncf_ckpt"]
    else:
        raise ValueError(name)
    if ckpt is not None:
        ckpt_path = Path(ckpt)
        path = ckpt_path if ckpt_path.is_absolute() else CHECK / ckpt
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    inspector_cls = {"VAE": jlens.VAEInspector,
                     "MLP": jlens.MLPInspector,
                     "NCF": jlens.NCFInspector}[name]
    insp = inspector_cls(model)
    return insp


def load_pop_array(data_name="ML1M"):
    data_dir = ROOT / "processed_data" / data_name
    with open(data_dir / f"pop_dict_{data_name}.pkl", "rb") as f:
        pop_dict = pickle.load(f)
    pop = np.zeros(CONFIGS[data_name]["n_items"])
    for key, value in pop_dict.items():
        pop[int(key)] = value
    return torch.tensor(pop, dtype=torch.float32)


def target_and_competitors(inspector, x, n_comp):
    with torch.no_grad():
        logits, _ = inspector.run(x)
    masked = logits[0].clone()
    masked[x > 0] = float("-inf")
    order = masked.argsort(descending=True)
    return int(order[0]), order[1:1 + n_comp].tolist()


def explanations_for(inspector, x, y, k, pop, spin_probs, seed,
                     train_X=None):
    expl = {
        "saliency": jx.topk_from_map(jx.saliency(inspector, x, y), x, k),
        "ig_zero": jx.topk_from_map(
            jx.integrated_gradients(inspector, x, y, steps=20), x, k),
        "spinrec_lite": jx.topk_from_map(
            jx.spinrec_lite(inspector, x, y, k, n_baselines=10, steps=10,
                            seed=seed, baseline_probs=spin_probs), x, k),
        "pop": jx.pop_topk(x, pop, k),
        "random": jx.random_topk(x, k, seed),
    }
    if train_X is not None:
        expl["spinrec"] = jx.topk_from_map(
            jx.spinrec_real(inspector, x, y, k, train_X, n_baselines=10,
                            steps=10, seed=seed), x, k)
    return expl


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="VAE", choices=["VAE", "MLP", "NCF"])
    ap.add_argument("--data", default="ML1M", choices=list(CONFIGS))
    ap.add_argument("--n-users", type=int, default=-1,
                    help="-1 = all test users")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n-competitors", type=int, default=5)
    ap.add_argument("--n-random", type=int, default=5)
    ap.add_argument("--user-seed", type=int, default=None,
                    help="sample eligible users reproducibly; default keeps prefix protocol")
    ap.add_argument("--out-dir", default=str(ROOT / "jlens_results"))
    args = ap.parse_args(argv)

    classes = jlens.load_notebook_classes()
    inspector = load_model(args.model, classes, args.data)
    pop = load_pop_array(args.data)

    data_dir = ROOT / "processed_data" / args.data
    test = pd.read_csv(data_dir / f"test_data_{args.data}.csv", index_col=0)
    train = pd.read_csv(data_dir / f"train_data_{args.data}.csv", index_col=0)
    train_X = torch.tensor(train.to_numpy(), dtype=torch.float32)
    X = torch.tensor(test.to_numpy(), dtype=torch.float32)
    if args.user_seed is None:
        n_users = len(X) if args.n_users < 0 else min(args.n_users, len(X))
        user_ids = list(range(n_users))
    else:
        eligible = torch.where(X.sum(1) > args.k)[0]
        gen = torch.Generator().manual_seed(args.user_seed)
        perm = eligible[torch.randperm(len(eligible), generator=gen)]
        take = len(perm) if args.n_users < 0 else min(args.n_users, len(perm))
        user_ids = perm[:take].tolist()
        n_users = len(user_ids)

    rows = []
    t0 = time.time()
    for pos, u in enumerate(user_ids):
        x = X[u]
        if int(x.sum()) <= args.k:
            continue
        y, comps = target_and_competitors(inspector, x, args.n_competitors)
        hist_len = int(x.sum())
        spin_probs = torch.clamp(pop / pop.sum() * hist_len, 0.0, 1.0)
        try:
            expl = explanations_for(inspector, x, y, args.k, pop,
                                    spin_probs, seed=u, train_X=train_X)
        except RuntimeError as e:
            print(f"user {u}: explainer failed: {e}", file=sys.stderr)
            continue
        for name, S in expl.items():
            for r in jlens.mechanistic_eval(inspector, x, S, y,
                                            competitors=comps,
                                            n_random=args.n_random, seed=u):
                rows.append({"user": u, "hist_len": hist_len, "y": y,
                             "explainer": name, "k": args.k, **r})
        if (pos + 1) % 25 == 0:
            el = time.time() - t0
            print(f"{pos + 1}/{n_users} users, {el:.0f}s "
                  f"({el / (pos + 1):.2f}s/user)", flush=True)

    df = pd.DataFrame(rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.data.lower()}_{args.model.lower()}_k{args.k}.csv"
    df.to_csv(out, index=False)
    print(f"\nsaved {len(df)} rows -> {out}")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    informative = df[df["tap"].isin(["enc_h1", "z"])] \
        if args.model == "VAE" else df
    strong = informative[informative["delta_beh"].abs() > 0.01]
    print(f"\n=== {args.model} aggregate over {df['user'].nunique()} users "
          f"(ratio metrics on |delta_beh|>0.01: "
          f"{strong['user'].nunique()} users) ===")
    means = df.groupby(["explainer", "tap"])[
        ["delta_beh", "jcomp", "collateral_all", "selectivity_del",
         "selectivity_patch", "jcomp_rand_mean"]].mean()
    medians = strong.groupby(["explainer", "tap"])[
        ["mr", "jrestore"]].median()
    print(means.round(4))
    print("\nmedians (filtered):")
    print(medians.round(4))
    return df


if __name__ == "__main__":
    main()
