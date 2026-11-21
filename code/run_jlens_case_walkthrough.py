"""Step-by-step walkthrough of the MR_fwd computation for one real user.

Prints every intermediate quantity for ML1M user 23 and the two saved K=10
deletion sets (saliency and the lambda=1 white-box search): the clean and
corrupted latents, the readout direction at the clean state, how much of the
latent displacement lies inside the audited subspace, the four-corner scores,
and a unit-level decomposition of the drop grouped by ReLU gate status.

Usage: python3 run_jlens_case_walkthrough.py --movies /path/to/movies.dat
"""
import argparse
import json

import pandas as pd
import torch

import jlens
from run_jlens_ml1m import ROOT, load_model, target_and_competitors
from run_jlens_case_study import build_title_map

TAP = "z"


def unit_table(insp, z0, z1, y):
    """Per-unit contribution to the drop, with gate status at both states."""
    lin = insp.model.decoder[0]
    W, b = lin.weight, lin.bias
    pre0, pre1 = W @ z0 + b, W @ z1 + b
    w_out = insp.model.decoder[-1].weight[y]
    # contribution to the DROP (positive = pushes the score down)
    contrib = -w_out * (torch.relu(pre1) - torch.relu(pre0))
    on0, on1 = pre0 > 0, pre1 > 0
    status = torch.where(on0 & on1, 0, torch.where(~on0 & on1, 1,
             torch.where(on0 & ~on1, 2, 3)))  # stay-on, opened, closed, off
    return pre0, pre1, w_out, contrib, status


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--movies", required=True)
    ap.add_argument("--user", type=int, default=23)
    args = ap.parse_args(argv)

    tmap = build_title_map(args.movies)
    classes = jlens.load_notebook_classes()
    insp = load_model("VAE", classes, "ML1M")
    test = pd.read_csv(ROOT / "processed_data/ML1M/test_data_ML1M.csv",
                       index_col=0)
    X = torch.tensor(test.to_numpy(), dtype=torch.float32)
    adv = pd.read_csv(ROOT / "jlens_results/ml1m_vae_adversarial_k10.csv")
    adv = adv[adv.tap == TAP]

    x = X[args.user]
    y, comps = target_and_competitors(insp, x, 5)
    with torch.no_grad():
        l0, a0 = insp.run(x)
    z0 = a0[TAP][0]
    V = jlens.item_directions(insp, x, TAP, [y] + comps)
    Q = jlens.orthonormal_basis(V)
    v_y = V[:, 0]

    n_lat = z0.numel()
    lin = insp.model.decoder[0]
    pre0_all = lin.weight @ z0 + lin.bias
    print(f"=== USER {args.user} ===")
    print(f"target y = {tmap[y][0]}  (index {y})")
    print(f"clean score s_y(z0) = {float(l0[0, y]):.4f}")
    print(f"latent dim = {n_lat}; decoder hidden units = {pre0_all.numel()}")
    print(f"units active at the clean state: "
          f"{int((pre0_all > 0).sum())} / {pre0_all.numel()}")
    print(f"audited subspace: target + {len(comps)} competitors "
          f"-> rank {Q.shape[1]} of {n_lat}")
    print(f"||grad s_y at z0|| = {float(v_y.norm()):.4f}")

    for name, key in [("SALIENCY", "saliency"), ("WHITE-BOX", "attack_lam1")]:
        row = adv[(adv.user == args.user) & (adv.method == key)].iloc[0]
        S = json.loads(row.S)
        with torch.no_grad():
            l1, a1 = insp.run(jlens.delete_items(x, S))
        z1 = a1[TAP][0]
        delta_z = z1 - z0
        proj = Q @ (Q.T @ delta_z)
        orth = delta_z - proj
        with torch.no_grad():
            F00 = float(insp.run_from(TAP, z0.unsqueeze(0))[0, y])
            F10 = float(insp.run_from(TAP, (z0 + proj).unsqueeze(0))[0, y])
        drop = float(l0[0, y] - l1[0, y])
        jfwd = F00 - F10

        print(f"\n----- {name} ({len(S)} films) -----")
        for i in sorted(S):
            print(f"    {tmap[i][0]}")
        print(f"  s_y clean      = {F00:.4f}")
        print(f"  s_y deleted    = {float(l1[0, y]):.4f}")
        print(f"  Delta          = {drop:.4f}")
        print(f"  ||delta_z||            = {float(delta_z.norm()):.4f}")
        print(f"  ||P delta_z||  (inside) = {float(proj.norm()):.4f}")
        print(f"  ||(I-P) delta_z|| (out) = {float(orth.norm()):.4f}")
        cos = float(torch.dot(delta_z, v_y)
                    / (delta_z.norm() * v_y.norm() + 1e-12))
        print(f"  cos(delta_z, grad s_y)  = {cos:+.4f}")
        print(f"  s_y(z0 + P delta_z)     = {F10:.4f}")
        print(f"  J_fwd = F00 - F10       = {jfwd:.4f}")
        print(f"  MR_fwd = J_fwd / Delta  = {jfwd / drop:.4f}")

        pre0, pre1, w_out, contrib, status = unit_table(insp, z0, z1, y)
        names = ["stayed on ", "OPENED    ", "closed    ", "stayed off"]
        print(f"  unit-level decomposition of Delta = {drop:.4f}:")
        for s in range(4):
            m = status == s
            print(f"    {names[s]}: {int(m.sum()):4d} units, "
                  f"carry {float(contrib[m].sum()):+.4f} "
                  f"({100 * float(contrib[m].sum()) / drop:+.1f}%)")
        print("  five largest single-unit contributions:")
        order = contrib.abs().argsort(descending=True)[:5]
        for j in order.tolist():
            print(f"    unit {j:4d} [{names[int(status[j])].strip():10s}] "
                  f"pre0={float(pre0[j]):+7.3f} pre1={float(pre1[j]):+7.3f} "
                  f"w_out={float(w_out[j]):+6.3f} "
                  f"contrib={float(contrib[j]):+7.4f}")


if __name__ == "__main__":
    main()
