"""Exact component-carrier diagnostic for the final SASRec block.

At the penultimate tap, the downstream computation is one pre-norm
Transformer block followed by an affine item readout.  We replace each of
the block's two residual sublayers by its clean first-order tangent:

  A: LayerNorm + self-attention + the attention-mask context
  B: LayerNorm + ReLU feed-forward network

The all-tangent state is the clean local computation evaluated at the full
deleted activation.  Replacing A and B by their actual deleted-state maps
gives four states whose two-player Shapley effects sum exactly to the
nonlinear/context residual.  This is independent of the item-Jacobian
projection used by the mechanism-gap audit.

Usage:
  python run_jlens_sasrec_carrier.py --ckpt <checkpoint> \
      --input ../jlens_results/ml1m_sasrec_basis_robustness_k5_s42.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

import jlens_sasrec as js
from run_jlens_sasrec import build_cohort, target_and_competitors

DEFAULT_LXSR = Path(__file__).resolve().parent.parent / "external" / "LXSR"


def attention_context_last(layer, H, attention_mask):
    """Last-position output of one pre-norm attention residual sublayer."""
    x = H.unsqueeze(1)  # [T, 1, D]
    z = layer.norm1(x)
    length = H.shape[0]
    causal = torch.triu(
        torch.ones(length, length, device=H.device), diagonal=1).bool()
    # need_weights=True avoids a CPU flash-attention path without JVP support.
    attended, _ = layer.self_attn(
        z, z, z, attn_mask=causal,
        key_padding_mask=(attention_mask == 0), need_weights=True,
        average_attn_weights=False)
    return (x + attended)[-1, 0]


def feed_forward_last(layer, state):
    """One last-position pre-norm FFN residual sublayer."""
    hidden = layer.linear1(layer.norm2(state))
    return state + layer.linear2(layer.activation(hidden))


def tangent_value(fn, clean_input, evaluation_input):
    """Value of the first-order Taylor map of ``fn`` at ``clean_input``."""
    value, directional = torch.autograd.functional.jvp(
        fn, clean_input, evaluation_input - clean_input,
        create_graph=False, strict=False)
    return value + directional


def target_readout(model, state, target):
    if model.adapter_W is not None:
        state = model.adapter_W(state)
    return model.fc_out(state)[target]


def decompose_last_block(aud, tap, H0, H1, m0, m1, target):
    """Exact clean-tangent and A/B Shapley decomposition for one deletion."""
    if tap == "emb":
        start = 0
    elif tap.startswith("block"):
        start = int(tap.replace("block", ""))
    else:
        raise ValueError(f"unknown tap {tap!r}")
    if len(aud.model.transformer_layers) - start != 1:
        raise ValueError(
            "carrier decomposition requires exactly one block after the tap")
    layer = aud.model.transformer_layers[start]
    if layer.training:
        raise ValueError("carrier decomposition requires eval mode")

    clean_attention = lambda h: attention_context_last(layer, h, m0)
    a0 = clean_attention(H0).detach()
    a_tangent = tangent_value(clean_attention, H0, H1).detach()
    a_actual = attention_context_last(layer, H1, m1).detach()

    b_tt = tangent_value(
        lambda a: feed_forward_last(layer, a), a0, a_tangent)
    b_at = tangent_value(
        lambda a: feed_forward_last(layer, a), a0, a_actual)
    b_ta = feed_forward_last(layer, a_tangent)
    b_aa = feed_forward_last(layer, a_actual)

    f0 = target_readout(aud.model, feed_forward_last(layer, a0), target)
    f_tt = target_readout(aud.model, b_tt, target)
    f_at = target_readout(aud.model, b_at, target)
    f_ta = target_readout(aud.model, b_ta, target)
    f_aa = target_readout(aud.model, b_aa, target)

    tangent_recovered = f0 - f_tt
    residual = f_tt - f_aa
    attention_mass = 0.5 * ((f_tt - f_at) + (f_ta - f_aa))
    ffn_mass = 0.5 * ((f_tt - f_ta) + (f_at - f_aa))
    scalar = lambda value: float(value.detach())
    return {
        "carrier_delta": scalar(f0 - f_aa),
        "tangent_recovered": scalar(tangent_recovered),
        "tangent_residual": scalar(residual),
        "attention_context_mass": scalar(attention_mass),
        "ffn_mass": scalar(ffn_mass),
        "component_interaction": scalar((f_tt - f_at) - (f_ta - f_aa)),
        "carrier_shapley_error": scalar(residual - attention_mass - ffn_mass),
    }


def add_ratios(row):
    denom = row["delta_beh"]
    for key in ("tangent_recovered", "tangent_residual",
                "attention_context_mass", "ffn_mass"):
        row[f"{key}_ratio"] = (row[key] / denom
                               if abs(denom) > js.EPS else np.nan)
    row["carrier_endpoint_error"] = row["carrier_delta"] - denom
    row["carrier_error_to_clean_gap"] = (
        row["tangent_residual"] - row["clean_unrecovered"])
    return row


def audit_file(aud, tokenizer, users, source, output_dir, tap):
    frame = pd.read_csv(source)
    if "tap" in frame:
        frame = frame[frame.tap == tap].copy()
    rows = []
    for pos, (_, old) in enumerate(frame.iterrows()):
        uid = int(old.user)
        if uid not in users:
            raise KeyError(f"user {uid} is absent from the reconstructed cohort")
        sequence = users[uid]
        tokens = [tokenizer.item2id[item] for item in sequence]
        tokens = tokens[-tokenizer.max_seq_length:]
        target, _ = target_and_competitors(aud, tokenizer, tokens)
        deletion = json.loads(old.S)

        ids0 = torch.tensor(
            tokens, dtype=torch.long, device=aud.device).unsqueeze(0)
        m0 = torch.ones_like(ids0)
        ids1 = js.delete_and_shift(tokens, deletion).to(aud.device).unsqueeze(0)
        m1 = (ids1 != 0).long()
        with torch.no_grad():
            stages0 = aud.stages(ids0, m0)
            stages1 = aud.stages(ids1, m1)
        carrier = decompose_last_block(
            aud, tap, stages0[tap].detach(), stages1[tap].detach(),
            m0, m1, target)
        row = {
            "user": uid,
            "method": old.method,
            "k": int(old.k),
            "S": old.S,
            "tap": tap,
            "delta_beh": float(old.delta_beh),
            "clean_unrecovered": float(old.clean_unrecovered),
            "symmetric_unrecovered": float(old.symmetric_unrecovered),
            **carrier,
        }
        rows.append(add_ratios(row))
        if (pos + 1) % 50 == 0 or pos + 1 == len(frame):
            print(f"{Path(source).name}: {pos + 1}/{len(frame)}", flush=True)

    out = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{Path(source).stem}_carrier.csv"
    out.to_csv(destination, index=False)
    print(f"saved {len(out)} rows -> {destination}")

    for method, group in out.groupby("method"):
        strong = group[(group.delta_beh > 0.2)
                       & ((group.clean_unrecovered / group.delta_beh)
                          .clip(0.0, 1.0) > 0.5)]
        if strong.empty:
            continue
        rho = spearmanr(strong.tangent_residual,
                        strong.clean_unrecovered).statistic
        error = ((strong.tangent_residual - strong.clean_unrecovered).abs()
                 / strong.delta_beh.abs()).median()
        print(
            f"{method}: strong n={len(strong)}, "
            f"U0/D={np.median(strong.clean_unrecovered / strong.delta_beh):.3f}, "
            f"tangent residual/D={strong.tangent_residual_ratio.median():.3f}, "
            f"attention/context={strong.attention_context_mass_ratio.median():.3f}, "
            f"FFN={strong.ffn_mass_ratio.median():.3f}, "
            f"rho={rho:.3f}, median normalized error={error:.3f}")
    return out


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--lxsr-root", default=str(DEFAULT_LXSR))
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output-dir", default="jlens_results")
    parser.add_argument("--tap", default="block1")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)

    lxsr_root = Path(args.lxsr_root)
    data_dir = Path(args.data_dir) if args.data_dir else \
        lxsr_root / "data/splitted/global_timesplit/val_by_time/Movielens-1m/q08"
    device = args.device if torch.cuda.is_available() else "cpu"
    model, tokenizer, _ = js.load_sasrec(
        lxsr_root, args.ckpt, data_dir / "train.csv", device=device)
    auditor = js.SasRecAuditor(model, device=device)

    sources = [Path(path) for path in args.input]
    max_k = max(int(pd.read_csv(path, usecols=["k"]).k.max())
                for path in sources)
    cohort, _ = build_cohort(lxsr_root, data_dir, max_k)
    users = dict(cohort)
    for source in sources:
        audit_file(auditor, tokenizer, users, source,
                   Path(args.output_dir), args.tap)


if __name__ == "__main__":
    main()
