"""Shared MultVAE utilities for the paired mechanism-aware LXR experiment."""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import jlens
import jlens_explainers as jx
from run_jlens_ml1m import target_and_competitors
from run_jlens_switch_diagnostic import vae_decomp


def encode_z(model, x):
    """Return the deterministic MultVAE mean used by the audit."""
    h = F.normalize(x, dim=-1)
    for layer in model.encoder:
        h = layer(h)
    latent = model.decoder[0].in_features
    return h[:, :latent]


def target_scores(model, z, items):
    """Evaluate one target logit per row without materializing all logits."""
    if not (len(model.decoder) == 3
            and isinstance(model.decoder[0], torch.nn.Linear)
            and isinstance(model.decoder[1], torch.nn.ReLU)
            and isinstance(model.decoder[2], torch.nn.Linear)):
        raise ValueError("LXR retraining expects a Linear-ReLU-Linear decoder")
    h = model.decoder[1](model.decoder[0](z))
    output = model.decoder[2]
    score = (h * output.weight[items]).sum(dim=1)
    if output.bias is not None:
        score = score + output.bias[items]
    return score


def all_scores(model, z):
    h = model.decoder[1](model.decoder[0](z))
    return model.decoder[2](h)


def clean_item_bases(model, z, items, tol=1e-6):
    """Build batched clean item-score bases for the MultVAE decoder."""
    first, last = model.decoder[0], model.decoder[2]
    active = (first(z) > 0).to(z.dtype)
    selected = last.weight[items] * active[:, None, :]
    directions = torch.matmul(selected, first.weight).transpose(1, 2)
    q, singular, _ = torch.linalg.svd(directions, full_matrices=False)
    keep = singular > tol * singular[:, :1].clamp_min(torch.finfo(z.dtype).eps)
    return q * keep[:, None, :], keep.sum(dim=1)


def project_batched(q, delta):
    coeff = torch.bmm(q.transpose(1, 2), delta.unsqueeze(-1))
    return torch.bmm(q, coeff).squeeze(-1)


def precompute_clean_state(inspector, X, device, n_competitors=5,
                           batch=256):
    """Cache clean targets, latent states, logits, and Jacobian bases."""
    model = inspector.model.to(device)
    targets, clean_zs, clean_scores, bases, ranks = [], [], [], [], []
    started = time.time()
    with torch.no_grad():
        for start in range(0, len(X), batch):
            xb = X[start:start + batch].to(device)
            z0 = encode_z(model, xb)
            logits = all_scores(model, z0)
            logits = logits.masked_fill(xb > 0, float("-inf"))
            items = logits.topk(n_competitors + 1, dim=1).indices
            q, rank = clean_item_bases(model, z0, items)
            y = items[:, 0]
            targets.append(y.cpu())
            clean_zs.append(z0.cpu())
            clean_scores.append(logits.gather(1, y[:, None]).squeeze(1).cpu())
            bases.append(q.cpu())
            ranks.append(rank.cpu())
            if start == 0 or start + batch >= len(X):
                print(f"  cached {min(start + batch, len(X))}/{len(X)} users",
                      flush=True)
    cache = {
        "target": torch.cat(targets),
        "z0": torch.cat(clean_zs),
        "s0": torch.cat(clean_scores),
        "q": torch.cat(bases),
        "rank": torch.cat(ranks),
    }
    print(f"  cache complete in {time.time() - started:.1f}s; "
          f"rank range {int(cache['rank'].min())}--{int(cache['rank'].max())}",
          flush=True)
    return cache


def evaluate(masker, inspector, X, cohort, data, k, beta, training_seed):
    """Audit hard top-k explanations from one trained LXR model."""
    rows = []
    n_items = X.shape[1]
    for pos, user in enumerate(cohort):
        x = X[user]
        y, competitors = target_and_competitors(inspector, x, 5)
        onehot = torch.zeros(1, n_items)
        onehot[0, y] = 1.0
        with torch.no_grad():
            scores = masker(x.unsqueeze(0), onehot)[0]
        deletion = jx.topk_from_map(scores, x, k)
        with torch.no_grad():
            _, clean = inspector.run(x)
            _, corrupted = inspector.run(jlens.delete_items(x, deletion))
        decomposition = vae_decomp(
            inspector, clean["z"], corrupted["z"], y)
        for result in jlens.mechanistic_eval(
                inspector, x, deletion, y, competitors=competitors):
            rows.append({
                "data": data,
                "user": int(user),
                "method": f"lxr_retrain_b{beta:g}",
                "beta": beta,
                "training_seed": training_seed,
                "k": k,
                "S": json.dumps(sorted(int(value) for value in deletion)),
                "switch_share": (
                    decomposition["switch_mass"] / decomposition["delta_dec"]
                    if abs(decomposition["delta_dec"]) > 0.01 else np.nan),
                **result,
            })
        if (pos + 1) % 20 == 0 or pos + 1 == len(cohort):
            print(f"  audited {pos + 1}/{len(cohort)} users", flush=True)
    return pd.DataFrame(rows)
