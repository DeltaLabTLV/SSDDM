"""Is the clean-state mediation ratio an artifact of layer parametrization?

At the VAE latent tap z, any invertible diagonal rescaling D can be absorbed
into the decoder weights without changing a single model output, yet it
changes the Euclidean projector P = QQ^T, the hybrid activations, and hence
potentially MR.  This script measures how much, on the saved K=10 red-team
sets:

1. Output-preserving diagonal rescalings: for R draws of D = diag(exp(u)),
   u ~ U(-ln s, ln s), recompute MR_fwd / MR_Sh with directions and
   displacement expressed in the rescaled coordinates (closed form in the
   original coordinates: patch h0 + D^{-1} P' D delta with P' built from
   QR(D^{-1}V)).
2. Covariance whitening: the same audit in Sigma^{-1/2}-coordinates, with
   Sigma the z covariance over train users (the natural scale-free choice).
3. A ||delta||-matched control: fresh random K-sets on the same users, with
   ||delta|| and MR reported next to saliency and the attack, isolating the
   MultVAE renormalization confound.

Usage: python3 run_jlens_invariance.py [--scales 3 10] [--draws 8]
Writes jlens_results/<data>_vae_invariance_k10.csv and prints aggregates.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

import jlens
import jlens_explainers as jx
from run_jlens_ml1m import CONFIGS, ROOT, load_model, target_and_competitors

TAP = "z"


def resume_y(insp, h, y):
    with torch.no_grad():
        return float(insp.run_from(TAP, h)[0, y])


def audit_in_coords(insp, h0, h1, delta, V, y, delta_beh, fwd, inv):
    """MR_fwd/MR_Sh with directions/displacement mapped by x -> fwd(x).

    fwd maps original coordinates to the reparametrized ones for activations
    (gradients transform with inv, its inverse-transpose for symmetric maps).
    Both callables act on 1-D tensors of the tap dimension.
    """
    Vp = torch.stack([inv(V[:, j]) for j in range(V.shape[1])], dim=1)
    Q = jlens.orthonormal_basis(Vp)
    d_p = fwd(delta[0])
    proj = Q @ (Q.T @ d_p)
    patch = inv(proj)
    h10 = h0 + patch.unsqueeze(0)
    h01 = h0 + delta - patch.unsqueeze(0)
    f00 = resume_y(insp, h0, y)
    f11 = resume_y(insp, h1, y)
    jf = f00 - resume_y(insp, h10, y)
    jr = resume_y(insp, h01, y) - f11
    return jf / delta_beh, 0.5 * (jf + jr) / delta_beh


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["ML1M", "Yahoo", "Pinterest"])
    ap.add_argument("--scales", type=float, nargs="+", default=[3.0, 10.0])
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2027)
    args = ap.parse_args(argv)

    classes = jlens.load_notebook_classes()
    for data in args.data:
        insp = load_model("VAE", classes, data)
        ddir = ROOT / "processed_data" / data
        test = pd.read_csv(ddir / f"test_data_{data}.csv", index_col=0)
        train = pd.read_csv(ddir / f"train_data_{data}.csv", index_col=0)
        X = torch.tensor(test.to_numpy(), dtype=torch.float32)
        Xtr = torch.tensor(train.to_numpy(), dtype=torch.float32)

        # z covariance over train users -> symmetric whitening maps
        with torch.no_grad():
            zs = []
            for i in range(0, min(len(Xtr), 4096), 512):
                _, acts = insp.run(Xtr[i:i + 512])
                zs.append(acts[TAP])
            Z = torch.cat(zs)
        Zc = Z - Z.mean(0, keepdim=True)
        cov = Zc.T @ Zc / (len(Z) - 1)
        evals, evecs = torch.linalg.eigh(cov)
        evals = evals.clamp_min(1e-8)
        W_fwd_mat = evecs @ torch.diag(evals.rsqrt()) @ evecs.T   # Sigma^-1/2
        W_inv_mat = evecs @ torch.diag(evals.sqrt()) @ evecs.T    # Sigma^+1/2

        adv = pd.read_csv(ROOT / "jlens_results"
                          / f"{data.lower()}_vae_adversarial_k{args.k}.csv")
        adv = adv[adv.tap == TAP]
        gen = torch.Generator().manual_seed(args.seed)
        dim = insp.model.enc_dims[-1]

        rows = []
        for (u, method), g in adv.groupby(["user", "method"]):
            if method not in ("saliency", "attack_lam1"):
                continue
            u = int(u)
            x = X[u]
            S = json.loads(g["S"].iloc[0])
            y, comps = target_and_competitors(insp, x, 5)
            C = [y] + [c for c in comps if c != y]
            with torch.no_grad():
                _, a0 = insp.run(x)
                _, a1 = insp.run(jlens.delete_items(x, S))
            h0, h1 = a0[TAP].detach(), a1[TAP]
            delta = h1 - h0
            delta_beh = float(g["delta_beh"].iloc[0])
            if abs(delta_beh) < 1e-9:
                continue
            V = jlens.item_directions(insp, x, TAP, C)

            ident = lambda v: v
            mr0, smr0 = audit_in_coords(insp, h0, h1, delta, V, y,
                                        delta_beh, ident, ident)
            mrw, smrw = audit_in_coords(
                insp, h0, h1, delta, V, y, delta_beh,
                lambda v: W_fwd_mat @ v, lambda v: W_inv_mat @ v)
            row = {"data": data, "user": u, "method": method,
                   "delta_beh": delta_beh, "mr_orig": mr0, "smr_orig": smr0,
                   "mr_white": mrw, "smr_white": smrw,
                   "delta_norm": float(delta.norm())}
            for s in args.scales:
                mrs = []
                for _ in range(args.draws):
                    logd = (torch.rand(dim, generator=gen) * 2 - 1) \
                        * float(np.log(s))
                    d = torch.exp(logd)
                    mr_d, _ = audit_in_coords(
                        insp, h0, h1, delta, V, y, delta_beh,
                        lambda v, d=d: v * d, lambda v, d=d: v / d)
                    mrs.append(mr_d)
                mrs = np.array(mrs)
                row[f"mr_s{s:g}_mean"] = mrs.mean()
                row[f"mr_s{s:g}_absdev"] = np.abs(mrs - mr0).mean()
                row[f"mr_s{s:g}_min"] = mrs.min()
                row[f"mr_s{s:g}_max"] = mrs.max()
            # ||delta||-matched random control (fresh K-set, same user)
            if method == "saliency":
                S_r = jx.random_topk(x, args.k, seed=u)
                with torch.no_grad():
                    _, ar = insp.run(jlens.delete_items(x, S_r))
                d_r = ar[TAP] - h0
                db_r = float(a0[TAP].new_tensor(0))  # placeholder
                logits0, _ = insp.run_items(x, [y])
                db_r = float(logits0[0, 0]
                             - insp.run_from_items(TAP, ar[TAP], [y])[0, 0])
                if abs(db_r) > 0.01:
                    mr_r, _ = audit_in_coords(insp, h0, ar[TAP], d_r, V, y,
                                              db_r, ident, ident)
                else:
                    mr_r = np.nan
                row["rand_delta_norm"] = float(d_r.norm())
                row["rand_mr"] = mr_r
                row["rand_delta_beh"] = db_r
            rows.append(row)

        df = pd.DataFrame(rows)
        out = (ROOT / "jlens_results"
               / f"{data.lower()}_vae_invariance_k{args.k}.csv")
        df.to_csv(out, index=False)

        print(f"\n=== {data}: parametrization sensitivity @ z, K={args.k} ===")
        for method in ("saliency", "attack_lam1"):
            m = df[(df.method == method) & (df.delta_beh > 0.01)]
            rho_w = spearmanr(m.mr_orig, m.mr_white)
            print(f"{method}: n={len(m)}")
            print(f"  MR median orig {m.mr_orig.median():.3f} | "
                  f"whitened {m.mr_white.median():.3f} | "
                  f"Spearman(orig,white) rho={rho_w.statistic:.3f}")
            for s in args.scales:
                print(f"  rescale s={s:g}: mean|dMR|={m[f'mr_s{s:g}_absdev'].mean():.3f}, "
                      f"per-user range [{m[f'mr_s{s:g}_min'].mean():.3f},"
                      f"{m[f'mr_s{s:g}_max'].mean():.3f}]")
            succ0 = ((m.delta_beh > 0.2) & (m.mr_orig < 0.5)).mean()
            succw = ((m.delta_beh > 0.2) & (m.mr_white < 0.5)).mean()
            print(f"  success orig {succ0:.1%} | whitened {succw:.1%}")
        sal = df[df.method == "saliency"]
        att = df[df.method == "attack_lam1"].set_index("user")
        print("delta norms: saliency %.3f | attack %.3f | random %.3f" % (
            sal.delta_norm.median(),
            att.delta_norm.median(),
            sal.rand_delta_norm.median()))
        print("random-set MR median: %.3f (delta_beh median %.3f)" % (
            sal.rand_mr.median(), sal.rand_delta_beh.median()))


if __name__ == "__main__":
    main()
