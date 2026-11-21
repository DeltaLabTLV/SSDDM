"""Population run of the synthetic two-pathway benchmark.

Ground-truth explanations (genre / anchor_t / anchor_u / filler) plus real
explainers (saliency, IG-zero, SPINRec-lite) on 200 sampled users.

Usage: python3 run_jlens_synthetic.py [--n-users 200] [--k 5]
"""
import argparse
from pathlib import Path

import pandas as pd
import torch

import jlens
import jlens_explainers as jx
import jlens_synthetic as js

ROOT = Path(__file__).resolve().parent.parent


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-users", type=int, default=200)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n-random", type=int, default=5)
    ap.add_argument("--out-dir", default=str(ROOT / "jlens_results"))
    args = ap.parse_args(argv)

    insp = js.SyntheticInspector(js.SyntheticTwoPath())
    rows = []
    for u in range(args.n_users):
        x = js.make_user(seed=u)
        y, comps = js.target_and_competitors(insp, x)
        assert y == js.TARGET
        expl = js.ground_truth_explanations(x, args.k, seed=u)
        expl["saliency"] = jx.topk_from_map(jx.saliency(insp, x, y), x, args.k)
        expl["ig_zero"] = jx.topk_from_map(
            jx.integrated_gradients(insp, x, y, steps=50), x, args.k)
        expl["spinrec"] = jx.topk_from_map(
            jx.spinrec_lite(insp, x, y, args.k, n_baselines=10, steps=10,
                            seed=u), x, args.k)
        for name, S in expl.items():
            frac_anchor = sum(js.is_anchor(i) for i in S) / len(S)
            frac_target_anchor = sum(i in js.A_T for i in S) / len(S)
            frac_uniform_anchor = sum(i in js.A_U for i in S) / len(S)
            # Store the path-basis diagnosis in the generated artifact.  In
            # this construction it should recover the gate direction after
            # the clean--corrupted path crosses the ReLU boundary.
            with torch.no_grad():
                _, clean = insp.run(x)
                _, corr = insp.run(jlens.delete_items(x, S))
            items = [y] + [c for c in comps if c != y]
            v_path = jlens.path_item_directions(
                insp, "h1", clean["h1"], corr["h1"], items, steps=8)
            path_result = jlens.mechanistic_eval(
                insp, x, S, y, competitors=comps, taps=["h1"],
                directions=lambda _tap, _items, V=v_path: V)[0]
            standard = jlens.mechanistic_eval(
                insp, x, S, y, competitors=comps,
                n_random=args.n_random, seed=u)
            for r in standard:
                rows.append({"user": u, "explainer": name, "k": args.k,
                             "frac_anchor": frac_anchor,
                             "frac_target_anchor": frac_target_anchor,
                             "frac_uniform_anchor": frac_uniform_anchor,
                             "path_smr": (path_result["smr"]
                                          if r["tap"] == "h1"
                                          else float("nan")),
                             **r})

    df = pd.DataFrame(rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"synthetic_twopath_k{args.k}.csv"
    df.to_csv(out, index=False)
    print(f"saved {len(df)} rows -> {out}")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    h1 = df[df["tap"] == "h1"]
    agg = h1.groupby("explainer").agg(
        DEL=("delta_beh", "mean"),
        MR_h1=("mr", "median"),
        JComp=("jcomp", "mean"),
        selectivity=("selectivity_del", "median"),
        collateral=("collateral_all", "mean"),
        jcomp_rand=("jcomp_rand_mean", "mean"),
        frac_anchor=("frac_anchor", "mean"),
        frac_target_anchor=("frac_target_anchor", "mean"),
        frac_uniform_anchor=("frac_uniform_anchor", "mean"),
    ).sort_values("DEL", ascending=False)
    print(f"\n=== Synthetic two-pathway @ h1, {args.n_users} users, "
          f"K={args.k} ===")
    print(agg.round(4))

    h2 = df[df["tap"] == "h2"]
    strong = h2[h2["delta_beh"].abs() > 0.5]
    print("\nMR @ h2 (output-adjacent, must be 1):",
          f"median={strong['mr'].median():.4f}",
          f"max|MR-1|={(strong['mr'] - 1).abs().max():.2e}")
    return df


if __name__ == "__main__":
    main()
