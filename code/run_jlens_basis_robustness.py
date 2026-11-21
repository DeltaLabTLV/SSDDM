"""Audit whether low mediation survives reasonable Jacobian-basis choices.

The adversarial search is performed once with the clean local basis.  This
script then evaluates the saved deletion sets without further optimization
under clean/corrupted/path/union endpoint directions and stronger nulls that
contain the target gradient.  It therefore separates a robust nonlinear
blind spot from an artifact of one convenient basis definition.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch

import jlens
from run_jlens_ml1m import CONFIGS, ROOT, load_model, target_and_competitors


def basis_matrices(inspector, x, S, y, competitors, tap, path_steps, seed):
    C = [y] + [c for c in competitors if c != y]
    with torch.no_grad():
        _, clean = inspector.run(x)
        _, corr = inspector.run(jlens.delete_items(x, S))
    hc, hr = clean[tap], corr[tap]
    v_clean = jlens.directions_at_activation(inspector, tap, hc, C)
    v_corr = jlens.directions_at_activation(inspector, tap, hr, C)
    v_path = jlens.path_item_directions(
        inspector, tap, hc, hr, C, steps=path_steps)

    gen = torch.Generator(device=hc.device).manual_seed(seed)
    random_extra = torch.randn(
        v_clean.shape[0], max(0, len(C) - 1), generator=gen,
        device=hc.device, dtype=hc.dtype)
    random_full = torch.randn(
        v_clean.shape[0], len(C), generator=gen,
        device=hc.device, dtype=hc.dtype)
    return {
        "clean_target": v_clean[:, :1],
        "clean_target_comp": v_clean,
        "clean_target_random": torch.cat(
            [v_clean[:, :1], random_extra], dim=1),
        "clean_comp_only": v_clean[:, 1:],
        "clean_random": random_full,
        "corrupt_target_comp": v_corr,
        "path_target_comp": v_path,
        "union_endpoints": torch.cat([v_clean, v_corr], dim=1),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="ML1M", choices=list(CONFIGS))
    ap.add_argument("--model", default="VAE", choices=["VAE", "NCF"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--tap", default=None)
    ap.add_argument("--n-users", type=int, default=60)
    ap.add_argument("--path-steps", type=int, default=8)
    ap.add_argument("--user-seed", type=int, default=None,
                    help="sample saved attack users reproducibly")
    ap.add_argument("--out-dir", default=str(ROOT / "jlens_results"))
    args = ap.parse_args(argv)

    classes = jlens.load_notebook_classes()
    inspector = load_model(args.model, classes, args.data)
    tap = args.tap or ("z" if args.model == "VAE" else "user_state")
    data_dir = ROOT / "processed_data" / args.data
    test = pd.read_csv(
        data_dir / f"test_data_{args.data}.csv", index_col=0)
    X = torch.tensor(test.to_numpy(), dtype=torch.float32)

    attack_file = (ROOT / "jlens_results"
                   / f"{args.data.lower()}_{args.model.lower()}_adversarial_k{args.k}.csv")
    attacks = pd.read_csv(attack_file)
    attacks = attacks[attacks["tap"] == tap].drop_duplicates(
        ["user", "method"])
    keep_methods = {"saliency", "attack_lam1", "constraint"}
    attacks = attacks[attacks["method"].isin(keep_methods)]
    available = sorted(attacks["user"].unique())
    if args.user_seed is None:
        users = available[:args.n_users]
    else:
        gen = torch.Generator().manual_seed(args.user_seed)
        perm = torch.randperm(len(available), generator=gen)[:args.n_users]
        users = [available[i] for i in perm.tolist()]

    rows = []
    for pos, u in enumerate(users):
        x = X[int(u)]
        y, comps = target_and_competitors(inspector, x, 5)
        for _, saved in attacks[attacks["user"] == u].iterrows():
            S = json.loads(saved["S"])
            mats = basis_matrices(
                inspector, x, S, y, comps, tap, args.path_steps,
                seed=1009 * int(u) + len(saved["method"]))
            for basis, V in mats.items():
                result = jlens.mechanistic_eval(
                    inspector, x, S, y, competitors=comps, taps=[tap],
                    directions=lambda _tap, _C, V=V: V)[0]
                rows.append({
                    "data": args.data, "user": int(u),
                    "method": saved["method"], "k": args.k,
                    "basis": basis,
                    "basis_rank": jlens.orthonormal_basis(V).shape[1],
                    **result,
                })
        if (pos + 1) % 10 == 0:
            print(f"{pos + 1}/{len(users)} users", flush=True)

    out = (Path(args.out_dir)
           / f"{args.data.lower()}_{args.model.lower()}_basis_robustness_k{args.k}.csv")
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"saved {len(df)} rows -> {out}")
    positive = df[df["delta_beh"] > 0.01]
    print(positive.groupby(["method", "basis"])[["mr", "smr"]]
          .median().round(3))
    return df


if __name__ == "__main__":
    main()
