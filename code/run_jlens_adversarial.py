"""White-box search for high-DEL sets with low clean-state forward mediation.

Methods per user:
- saliency (reference explainer)
- attack_lam0: pure DEL attack (upper bound of behavioral effect)
- attack_lam1: DEL - JComp clean-regime red-team

Usage: python3 run_jlens_adversarial.py --model NCF --tap user_state
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch

import jlens
import jlens_adversarial as ja
import jlens_explainers as jx
from run_jlens_ml1m import (CONFIGS, ROOT, load_model, load_pop_array,
                            target_and_competitors)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="ML1M", choices=list(CONFIGS))
    ap.add_argument("--model", default="VAE", choices=["VAE", "NCF"])
    ap.add_argument("--n-users", type=int, default=100)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--tap", default=None)
    ap.add_argument("--lams", type=float, nargs="+", default=[1.0])
    ap.add_argument("--mr-max", type=float, default=0.5)
    ap.add_argument("--user-seed", type=int, default=None,
                    help="sample eligible users reproducibly; default keeps prefix protocol")
    ap.add_argument("--out-dir", default=str(ROOT / "jlens_results"))
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint override (path, or filename under check/)")
    ap.add_argument("--tag", default="",
                    help="suffix for the output CSV name, e.g. _s1")
    args = ap.parse_args(argv)

    classes = jlens.load_notebook_classes()
    insp = load_model(args.model, classes, args.data, ckpt=args.ckpt)
    tap = args.tap or ("z" if args.model == "VAE" else "user_state")
    data_dir = ROOT / "processed_data" / args.data
    test = pd.read_csv(data_dir / f"test_data_{args.data}.csv", index_col=0)
    X = torch.tensor(test.to_numpy(), dtype=torch.float32)

    rows = []
    t0 = time.time()
    if args.user_seed is None:
        n_users = min(args.n_users, len(X))
        user_ids = list(range(n_users))
    else:
        eligible = torch.where(X.sum(1) > args.k)[0]
        gen = torch.Generator().manual_seed(args.user_seed)
        perm = eligible[torch.randperm(len(eligible), generator=gen)]
        user_ids = perm[:min(args.n_users, len(perm))].tolist()
        n_users = len(user_ids)
    for pos, u in enumerate(user_ids):
        x = X[u]
        if int(x.sum()) <= args.k:
            continue
        y, comps = target_and_competitors(insp, x, 5)
        methods = {"saliency": (jx.topk_from_map(
            jx.saliency(insp, x, y), x, args.k), {})}
        for lam in args.lams:
            methods[f"attack_lam{lam:g}"] = (ja.greedy_attack(
                insp, x, y, args.k, tap, comps, lam=lam, seed=u), {})
        constraint_S, constraint_info = ja.constraint_attack(
            insp, x, y, args.k, tap, comps, mr_max=args.mr_max, seed=u,
            return_info=True)
        methods["constraint"] = (constraint_S, {
            "constraint_feasible": bool(constraint_info["feasible"]),
            "constraint_search_mr": constraint_info["mr"],
            "constraint_search_delta": constraint_info["delta"],
        })
        for name, (S, metadata) in methods.items():
            for r in jlens.mechanistic_eval(insp, x, S, y, competitors=comps,
                                            n_random=3, seed=u):
                rows.append({"user": u, "method": name, "k": args.k,
                             "S": json.dumps(sorted(S)), **metadata, **r})
        if (pos + 1) % 10 == 0:
            el = time.time() - t0
            print(f"{pos + 1}/{n_users}, {el:.0f}s ({el / (pos + 1):.2f}s/user)",
                  flush=True)

    df = pd.DataFrame(rows)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    out = (Path(args.out_dir)
           / f"{args.data.lower()}_{args.model.lower()}_adversarial_k{args.k}"
             f"{args.tag}.csv")
    df.to_csv(out, index=False)
    print(f"saved {len(df)} rows -> {out}")

    pd.set_option("display.width", 200)
    z = df[df["tap"] == tap]
    strong = z[z["delta_beh"].abs() > 0.01]
    agg = z.groupby("method").agg(
        DEL_mean=("delta_beh", "mean"),
        JComp_mean=("jcomp", "mean"),
        MR_median=("mr", "median"),
        MR_q10=("mr", lambda s: s.quantile(0.10)),
        selectivity=("selectivity_del", "median"),
        collateral=("collateral_all", "mean"),
    )
    print(f"\n=== {args.model}: adversarial vs explainer @ {tap}, K={args.k} ===")
    print(agg.round(4))
    frac = (strong.groupby("method")
            .apply(lambda g: ((g["delta_beh"] > g["delta_beh"].median() * 0)
                              & (g["mr"] < 0.5)).mean(), include_groups=False))
    print("\nfraction of users with MR < 0.5:")
    print(frac.round(3))
    return df


if __name__ == "__main__":
    main()
