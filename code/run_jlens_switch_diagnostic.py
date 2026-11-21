"""Why does the red-team transfer to some checkpoints and not others?

For every saved deletion set in the K=10 adversarial CSVs, decompose the
target's downstream computation into linear-region changes:

- flip_frac    fraction of downstream ReLU units (decoder layer for VAE at
               tap z; the target-conditional MLP stack for NeuMF at
               user_state) whose activation state differs between the clean
               and the corrupted reference activation;
- opened_frac  fraction that switch off->on (dormant units woken by the
               deletion);
- switch_share (VAE, exact) the part of the behavioral effect carried by
               opened units, which have exactly zero clean-state gradient:
               Delta = W2[y] . [ (p0-p1)*stay_on + p0*closed - p1*opened ],
               switch_share = -W2[y].(p1*opened) / Delta.
- gate_mass (NeuMF, exact) the part of the behavioral effect caused by
               replacing the clean MLP ReLU gates with the deleted-state
               gates at the deleted user state.  A two-player Shapley split
               assigns this mass to opening and closing gates.

Also computes a per-user *clean-state* vulnerability statistic that does not
look at any attack: the fraction of off units whose pre-activation distance
to the ReLU boundary is smaller than the pre-activation movement induced by
this user's saliency deletion (off-near-boundary capacity).

Usage: python3 run_jlens_switch_diagnostic.py [--data ML1M --model VAE]
Writes jlens_results/<data>_<model>_switchdiag_k10.csv and prints aggregates.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

import jlens
from run_jlens_ml1m import CONFIGS, ROOT, load_model

MAIN_TAP = {"VAE": "z", "NCF": "user_state"}


def vae_decomp(insp, z0, z1, y):
    """Exact stay-on/closed/opened decomposition of the decoder effect."""
    W1, b1 = insp.model.decoder[0].weight, insp.model.decoder[0].bias
    w2y = insp.model.decoder[2].weight[y]
    p0 = (z0 @ W1.T + b1)[0]
    p1 = (z1 @ W1.T + b1)[0]
    on0, on1 = p0 > 0, p1 > 0
    stay_on = on0 & on1
    closed = on0 & ~on1
    opened = ~on0 & on1
    delta = float(w2y @ (torch.relu(p0) - torch.relu(p1)))
    switch_mass = float(-(w2y[opened] @ p1[opened])) if opened.any() else 0.0
    n = len(p0)
    return {
        "delta_dec": delta,
        "flip_frac": float((on0 != on1).float().mean()),
        "opened_frac": float(opened.float().mean()),
        "closed_frac": float(closed.float().mean()),
        "switch_mass": switch_mass,
        "n_units": n,
    }


def ncf_target_preacts(insp, state, y):
    """Pre-ReLU activations of the target-conditional MLP stack."""
    m = insp.model
    item_mlp = m.embed_item_MLP.weight[:, y]
    user_mlp = state[:, insp.gmf_dim:]
    h = torch.cat([user_mlp[0], item_mlp])
    pres = []
    for layer in m.MLP_layers:
        if isinstance(layer, torch.nn.Linear):
            h = layer(h)
            pres.append(h.clone())
        elif isinstance(layer, torch.nn.ReLU):
            h = torch.relu(h)
        # Dropout is identity in eval mode
    return pres


def ncf_gate_masks(insp, state, y):
    """Endpoint ReLU masks for the target-conditional NeuMF MLP."""
    return [p > 0 for p in ncf_target_preacts(insp, state, y)]


def ncf_score_with_gate_masks(insp, state, y, masks):
    """Target logit with externally supplied masks at every MLP ReLU.

    Dropout is the identity in evaluation mode.  Holding all masks fixed
    makes the MLP affine in the user state; the GMF branch is already affine.
    """
    m = insp.model
    item_gmf = m.embed_item_GMF.weight[:, y]
    item_mlp = m.embed_item_MLP.weight[:, y]
    out_gmf = state[0, :insp.gmf_dim] * item_gmf
    h = torch.cat([state[0, insp.gmf_dim:], item_mlp])
    mask_idx = 0
    for layer in m.MLP_layers:
        if isinstance(layer, torch.nn.Linear):
            h = layer(h)
        elif isinstance(layer, torch.nn.ReLU):
            h = h * masks[mask_idx].to(dtype=h.dtype)
            mask_idx += 1
        else:
            h = layer(h)  # Dropout is identity because the model is in eval.
    if mask_idx != len(masks):
        raise ValueError("number of masks does not match the MLP ReLU layers")
    joint = torch.cat([out_gmf, h])
    return float(m.predict_layer(joint).detach())


def ncf_mixed_gate_masks(clean_masks, deleted_masks, open_gates, close_gates):
    """Move clean masks toward deleted masks by gate-transition type."""
    mixed = []
    for clean, deleted in zip(clean_masks, deleted_masks):
        mask = clean.clone()
        opened = ~clean & deleted
        closed = clean & ~deleted
        if open_gates:
            mask |= opened
        if close_gates:
            mask &= ~closed
        mixed.append(mask)
    return mixed


def ncf_gate_decomp(insp, s0, s1, y):
    """Exact clean-affine/gate decomposition of a NeuMF target-score drop.

    At the deleted user state, the four scores use clean gates, only newly
    opened gates, only newly closed gates, or all deleted-state gates.  The
    two transition groups receive their order-symmetrized Shapley effects.
    """
    clean_masks = ncf_gate_masks(insp, s0, y)
    deleted_masks = ncf_gate_masks(insp, s1, y)

    def score(open_gates, close_gates):
        masks = ncf_mixed_gate_masks(
            clean_masks, deleted_masks, open_gates, close_gates)
        return ncf_score_with_gate_masks(insp, s1, y, masks)

    f00 = score(False, False)
    f10 = score(True, False)
    f01 = score(False, True)
    f11 = score(True, True)
    f0 = ncf_score_with_gate_masks(insp, s0, y, clean_masks)
    clean_affine = f0 - f00
    gate_mass = f00 - f11
    opened_mass = 0.5 * ((f00 - f10) + (f01 - f11))
    closed_mass = 0.5 * ((f00 - f01) + (f10 - f11))
    return {
        "clean_affine_mass": clean_affine,
        "gate_mass": gate_mass,
        "opened_gate_mass": opened_mass,
        "closed_gate_mass": closed_mass,
        "gate_interaction": (f00 - f10) - (f01 - f11),
        "gate_shapley_error": gate_mass - opened_mass - closed_mass,
        "gate_endpoint_delta": f0 - f11,
    }


def ncf_decomp(insp, s0, s1, y):
    pres0 = ncf_target_preacts(insp, s0, y)
    pres1 = ncf_target_preacts(insp, s1, y)
    flips, opened, total = 0, 0, 0
    for p0, p1 in zip(pres0, pres1):
        on0, on1 = p0 > 0, p1 > 0
        flips += int((on0 != on1).sum())
        opened += int((~on0 & on1).sum())
        total += len(p0)
    g0, m0 = ncf_branch_scores(insp, s0, y)
    g1, m1 = ncf_branch_scores(insp, s1, y)
    delta = (g0 + m0) - (g1 + m1)
    out = {
        "flip_frac": flips / total,
        "opened_frac": opened / total,
        "n_units": total,
        "delta_gmf": g0 - g1,
        "delta_mlp": m0 - m1,
        "mlp_share": (m0 - m1) / delta if abs(delta) > 0.05 else np.nan,
    }
    out.update(ncf_gate_decomp(insp, s0, s1, y))
    out["gate_share"] = (out["gate_mass"] / delta
                         if abs(delta) > 0.01 else np.nan)
    out["opened_gate_share"] = (out["opened_gate_mass"] / delta
                                if abs(delta) > 0.01 else np.nan)
    out["closed_gate_share"] = (out["closed_gate_mass"] / delta
                                if abs(delta) > 0.01 else np.nan)
    out["gate_endpoint_error"] = out["gate_endpoint_delta"] - delta
    return out


def ncf_branch_scores(insp, state, y):
    """Split the pre-sigmoid score into its GMF and MLP branch terms.

    The GMF term is linear in the user state for a fixed item, so its
    contribution to any deletion effect is always clean-mediated; the MLP
    term is the only pathway that can host a dormant mechanism.
    """
    m = insp.model
    item_gmf = m.embed_item_GMF.weight[:, y]
    item_mlp = m.embed_item_MLP.weight[:, y]
    out_gmf = state[0, :insp.gmf_dim] * item_gmf
    h = torch.cat([state[0, insp.gmf_dim:], item_mlp])
    for layer in m.MLP_layers:
        h = layer(h)
    W = m.predict_layer.weight[0]
    g = insp.gmf_dim
    return float(W[:g] @ out_gmf), float(W[g:] @ h)


def vae_capacity(insp, z0, z_sal, y):
    """Fraction of off decoder units within reach of a typical deletion."""
    W1, b1 = insp.model.decoder[0].weight, insp.model.decoder[0].bias
    p0 = (z0 @ W1.T + b1)[0]
    move = ((z_sal - z0) @ W1.T)[0].abs()
    off = p0 <= 0
    if not off.any():
        return 0.0
    return float(((-p0[off]) < move[off]).float().mean())


def ncf_capacity(insp, s0, s_sal, y):
    m = insp.model
    item_mlp = m.embed_item_MLP.weight[:, y]
    user_mlp0 = torch.cat([s0[0, insp.gmf_dim:], item_mlp])
    user_mlps = torch.cat([s_sal[0, insp.gmf_dim:], item_mlp])
    lin = [l for l in m.MLP_layers if isinstance(l, torch.nn.Linear)][0]
    p0 = lin(user_mlp0)
    move = (lin(user_mlps) - p0).abs()
    off = p0 <= 0
    if not off.any():
        return 0.0
    return float(((-p0[off]) < move[off]).float().mean())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["ML1M", "Yahoo", "Pinterest"])
    ap.add_argument("--model", nargs="+", default=["VAE", "NCF"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint override (path, or filename under check/)")
    ap.add_argument("--tag", default="",
                    help="suffix matching the adversarial CSV, e.g. _s1")
    ap.add_argument("--output-dir", default=str(ROOT / "jlens_results"),
                    help="directory for the per-set carrier CSVs")
    args = ap.parse_args(argv)

    classes = jlens.load_notebook_classes()
    for model_name in args.model:
        tap = MAIN_TAP[model_name]
        for data in args.data:
            src = (ROOT / "jlens_results"
                   / f"{data.lower()}_{model_name.lower()}_adversarial_k{args.k}"
                     f"{args.tag}.csv")
            if not src.exists():
                print(f"skip {src.name}: not found")
                continue
            df = pd.read_csv(src)
            df = df[df.tap == tap].copy()
            insp = load_model(model_name, classes, data, ckpt=args.ckpt)
            test = pd.read_csv(ROOT / "processed_data" / data
                               / f"test_data_{data}.csv", index_col=0)
            X = torch.tensor(test.to_numpy(), dtype=torch.float32)

            # per-user clean state and saliency corrupted state for capacity
            sal = df[df.method == "saliency"].set_index("user")
            rows = []
            for (u, method), g in df.groupby(["user", "method"]):
                x = X[u]
                S = json.loads(g["S"].iloc[0])
                y_logits, acts0 = insp.run(x)
                y = int(torch.where(x > 0, float("-inf"),
                                    y_logits[0]).argmax())
                with torch.no_grad():
                    _, acts1 = insp.run(jlens.delete_items(x, S))
                h0, h1 = acts0[tap].detach(), acts1[tap]
                if model_name == "VAE":
                    d = vae_decomp(insp, h0, h1, y)
                    d["switch_share"] = (d["switch_mass"] / d["delta_dec"]
                                         if abs(d["delta_dec"]) > 0.01
                                         else np.nan)
                else:
                    d = ncf_decomp(insp, h0, h1, y)
                cap = np.nan
                if u in sal.index:
                    S_sal = json.loads(sal.loc[u, "S"])
                    with torch.no_grad():
                        _, acts_s = insp.run(jlens.delete_items(x, S_sal))
                    cap_fn = vae_capacity if model_name == "VAE" else ncf_capacity
                    cap = cap_fn(insp, h0, acts_s[tap], y)
                rows.append({"user": u, "method": method,
                             "mr": float(g["mr"].iloc[0]),
                             "delta_beh": float(g["delta_beh"].iloc[0]),
                             "capacity": cap, **d})

            out = pd.DataFrame(rows)
            dst = (Path(args.output_dir)
                   / f"{data.lower()}_{model_name.lower()}_switchdiag_k{args.k}"
                     f"{args.tag}.csv")
            dst.parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(dst, index=False)

            print(f"\n=== {model_name} / {data} (tap {tap}) ===")
            agg = out.groupby("method").agg(
                flip=("flip_frac", "mean"),
                opened=("opened_frac", "mean"),
                mr_med=("mr", "median"),
            )
            if model_name == "VAE":
                agg["switch_share_med"] = out.groupby("method")[
                    "switch_share"].median()
            else:
                agg["mlp_share_med"] = out.groupby("method")[
                    "mlp_share"].median()
                agg["gate_share_med"] = out.groupby("method")[
                    "gate_share"].median()
                agg["opened_gate_share_med"] = out.groupby("method")[
                    "opened_gate_share"].median()
            print(agg.round(3))
            att = out[(out.method == "attack_lam1")
                      & (out.delta_beh > 0.01)].dropna(subset=["mr"])
            if model_name == "VAE":
                att2 = att.dropna(subset=["switch_share"])
                rho = spearmanr(att2.mr, att2.switch_share)
                print(f"Spearman(mr, switch_share) attack: "
                      f"rho={rho.statistic:.3f} p={rho.pvalue:.1e} n={len(att2)}")
            else:
                att2 = att.dropna(subset=["mlp_share"])
                rho = spearmanr(att2.mr, att2.mlp_share)
                print(f"Spearman(mr, mlp_share) attack:   "
                      f"rho={rho.statistic:.3f} p={rho.pvalue:.1e} n={len(att2)}")
                gap_mass = att.delta_beh * (1.0 - att.mr)
                rho_gate = spearmanr(gap_mass, att.gate_mass)
                median_error = np.median(
                    np.abs(att.gate_mass - gap_mass)
                    / np.abs(att.delta_beh))
                print(f"Exact frozen-gate carrier vs raw gap: "
                      f"rho={rho_gate.statistic:.3f} "
                      f"median normalized error={median_error:.3f} "
                      f"n={len(att)}")
                max_identity = max(
                    out.gate_shapley_error.abs().max(),
                    out.gate_endpoint_error.abs().max())
                print(f"Frozen-gate identity max error: {max_identity:.2e}")
            rho2 = spearmanr(att.mr, att.flip_frac)
            print(f"Spearman(mr, flip_frac) attack:    "
                  f"rho={rho2.statistic:.3f} p={rho2.pvalue:.1e} n={len(att)}")
            cap_u = out[out.method == "saliency"].capacity
            print(f"clean-state capacity (median over users): "
                  f"{np.nanmedian(cap_u):.3f}")


if __name__ == "__main__":
    main()
