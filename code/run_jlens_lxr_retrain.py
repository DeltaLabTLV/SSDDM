"""Paired from-scratch LXR training with a mechanism-aware deletion branch.

This script deliberately separates model selection from test evaluation.

``train``
    Splits the original training users into train/validation users, creates one
    random LXR initialization per seed, and trains every beta arm from that
    identical state with identical epoch permutations.  Checkpoints are chosen
    using the *unregularized* LXR validation loss.  No test users are loaded.

``evaluate``
    Loads one already-selected beta checkpoint and audits hard top-K masks on
    the held-out test users.

``objective``
    Evaluates the unmodified public LXR loss on all held-out test users for
    one paired seed.  This diagnostic checks whether a lower hard-mask gap was
    purchased by degrading the explainer's original soft-mask objective.

The baseline reproduces the public LXR implementation's combined objective:
sufficiency, inverse-mask comprehensiveness, and sparsity.  This is a paired
retraining experiment, not a fine-tune or a reconstruction of any released
checkpoint: the original code selected epochs on test users, whereas this
script uses a held-out split of the training users and opens the test set only
in ``evaluate``.  For beta > 0, only the inverse-mask branch changes.  Its
target logit is

    s_deleted + beta * relu(s_clean-aligned - s_deleted).

At beta=1 the inverse branch receives no benefit from a positive deletion-only
remainder, while all logits and all other LXR loss terms remain unchanged.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import jlens
from run_jlens_lxr_shap import LXRExplainer, load_lxr
from jlens_lxr_training import (
    all_scores,
    encode_z,
    evaluate,
    precompute_clean_state,
    project_batched,
    target_scores,
)
from run_jlens_ml1m import ROOT, load_model


LXR_CONFIG = {
    # Architecture and loss coefficients match the combined VAE explainers
    # evaluated in this repository.  ML1M and Pinterest use the public LXR
    # learning-rate/batch recipes.  The later Yahoo checkpoint is not listed
    # in the public configuration sheet; it uses that release's coefficients
    # with the published Yahoo-VAE learning rate and batch size.
    "ML1M": {
        "hidden": 128, "lambda_pos": 3.185652725834087,
        "lambda_neg": 1.420642300151426, "lr": 0.005019,
        "batch": 128,
    },
    "Yahoo": {
        "hidden": 128, "lambda_pos": 18.958765029913238,
        "lambda_neg": 4.92235962483309, "lr": 0.0022989149911625,
        "batch": 32,
    },
    "Pinterest": {
        "hidden": 32, "lambda_pos": 6.3443735346179855,
        "lambda_neg": 1.472868807603448, "lr": 0.0420483968289197,
        "batch": 256,
    },
}


def beta_tag(beta):
    return f"b{beta:g}".replace(".", "p")


def split_indices(n, val_fraction=0.1, split_seed=314159):
    generator = torch.Generator().manual_seed(split_seed)
    order = torch.randperm(n, generator=generator)
    n_val = max(1, int(round(n * val_fraction)))
    return order[n_val:], order[:n_val]


def masked_inputs(explainer, xb, y):
    onehot = torch.zeros(len(xb), xb.shape[1], device=xb.device)
    onehot[torch.arange(len(xb), device=xb.device), y] = 1.0
    mask = torch.sigmoid(explainer(xb, onehot)) * xb
    return mask, xb * mask, xb * (1.0 - mask)


def lxr_loss(model, explainer, xb, y, z0, q, config, beta=0.0,
             alpha=1.0, inverse_loss="released_code"):
    """LXR loss with an audit-corrected inverse-mask target score."""
    mask, kept, deleted = masked_inputs(explainer, xb, y)
    kept_logits = all_scores(model, encode_z(model, kept))
    z1 = encode_z(model, deleted)
    deleted_logits = all_scores(model, z1)
    rows = torch.arange(len(xb), device=xb.device)

    positive = -F.log_softmax(kept_logits, dim=1)[rows, y].mean()
    deleted_target = deleted_logits[rows, y]
    if beta > 0:
        aligned = project_batched(q, z1 - z0)
        patched_target = target_scores(model, z0 + aligned, y)
        corrected_target = deleted_target + beta * F.relu(
            patched_target - deleted_target)
        corrected_logits = deleted_logits + F.one_hot(
            y, num_classes=deleted_logits.shape[1]).to(deleted_logits.dtype) \
            * (corrected_target - deleted_target).unsqueeze(1)
    else:
        corrected_logits = deleted_logits

    target_log_prob = F.log_softmax(corrected_logits, dim=1)[rows, y]
    if inverse_loss == "released_code":
        # The public LXR training notebook minimizes log p_y(x_{1-m}).
        negative = target_log_prob.mean()
    elif inverse_loss == "paper":
        target_prob = target_log_prob.exp().clamp(max=1.0 - 1e-7)
        negative = -torch.log1p(-target_prob).mean()
    else:
        raise ValueError(f"unknown inverse loss: {inverse_loss}")

    active = xb > 0
    sparsity = mask[active].mean() if active.any() else mask.mean()
    total = (config["lambda_pos"] * positive
             + config["lambda_neg"] * negative + alpha * sparsity)
    return total, {
        "positive": positive.detach(),
        "negative": negative.detach(),
        "sparsity": sparsity.detach(),
    }


@torch.no_grad()
def lxr_objective_metrics(model, explainer, X, cache, indices, config, device,
                          batch=256, inverse_loss="released_code"):
    totals = {key: 0.0 for key in
              ("total", "positive", "negative", "sparsity")}
    count = 0
    explainer.eval()
    for start in range(0, len(indices), batch):
        idx = indices[start:start + batch]
        xb = X[idx].to(device)
        y = cache["target"][idx].to(device)
        z0 = cache["z0"][idx].to(device)
        q = cache["q"][idx].to(device)
        loss, components = lxr_loss(
            model, explainer, xb, y, z0, q, config, beta=0.0,
            inverse_loss=inverse_loss)
        totals["total"] += float(loss) * len(idx)
        for key, value in components.items():
            totals[key] += float(value) * len(idx)
        count += len(idx)
    return {key: value / count for key, value in totals.items()}


@torch.no_grad()
def validation_lxr_loss(model, explainer, X, cache, indices, config, device,
                        batch=256, inverse_loss="released_code"):
    return lxr_objective_metrics(
        model, explainer, X, cache, indices, config, device, batch,
        inverse_loss)["total"]


@torch.no_grad()
def hard_topk_audit(model, explainer, X, cache, indices, k, device,
                    batch=256):
    """Fast vectorized clean-gap audit used only on validation users."""
    effects, gaps = [], []
    explainer.eval()
    for start in range(0, len(indices), batch):
        idx = indices[start:start + batch]
        xb = X[idx].to(device)
        y = cache["target"][idx].to(device)
        z0 = cache["z0"][idx].to(device)
        s0 = cache["s0"][idx].to(device)
        q = cache["q"][idx].to(device)

        onehot = torch.zeros(len(idx), X.shape[1], device=device)
        onehot[torch.arange(len(idx), device=device), y] = 1.0
        importance = explainer(xb, onehot).masked_fill(xb <= 0, float("-inf"))
        chosen = importance.topk(min(k, X.shape[1]), dim=1).indices
        hard = torch.zeros_like(xb)
        hard.scatter_(1, chosen, 1.0)
        hard *= xb

        z1 = encode_z(model, xb * (1.0 - hard))
        s1 = target_scores(model, z1, y)
        aligned = project_batched(q, z1 - z0)
        patched = target_scores(model, z0 + aligned, y)
        delta = s0 - s1
        remainder = patched - s1
        gap = torch.where(
            delta > 1e-8,
            (remainder / delta.clamp_min(1e-8)).clamp(0.0, 1.0),
            torch.zeros_like(delta),
        )
        effects.append(delta.cpu())
        gaps.append(gap.cpu())

    effect = torch.cat(effects).numpy()
    gap = torch.cat(gaps).numpy()
    positive = gap[effect > 0.01]
    return {
        "n": int(len(effect)),
        "effect": float(effect.mean()),
        "gap50": float(np.median(positive)) if len(positive) else float("nan"),
        "gap90": float(np.quantile(positive, .9)) if len(positive) else float("nan"),
        "tail": float(np.mean((effect > .2) & (gap > .5))),
    }


def train_arm(model, X, cache, train_idx, val_idx, data, seed, beta, args,
              initial_state):
    config = LXR_CONFIG[data]
    device = args.device
    explainer = LXRExplainer(
        X.shape[1], X.shape[1], config["hidden"]).to(device)
    explainer.load_state_dict(initial_state)
    optimizer = torch.optim.Adam(explainer.parameters(), lr=config["lr"])
    checkpoint_dir = Path(args.checkpoint_dir) / data.lower() / f"s{seed}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / f"lxr_{beta_tag(beta)}.pth"

    best_loss = float("inf")
    best_epoch = -1
    history = []
    for epoch in range(args.epochs):
        if epoch in (15, 30):
            for group in optimizer.param_groups:
                group["lr"] *= 0.1
        generator = torch.Generator().manual_seed(seed * 1000 + epoch)
        order = train_idx[torch.randperm(len(train_idx), generator=generator)]
        explainer.train()
        total = 0.0
        seen = 0
        for start in range(0, len(order), config["batch"]):
            idx = order[start:start + config["batch"]]
            xb = X[idx].to(device)
            y = cache["target"][idx].to(device)
            z0 = cache["z0"][idx].to(device)
            q = cache["q"][idx].to(device)
            loss, _ = lxr_loss(
                model, explainer, xb, y, z0, q, config, beta=beta,
                inverse_loss=args.inverse_loss)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(explainer.parameters(), 10.0)
            optimizer.step()
            total += float(loss.detach()) * len(idx)
            seen += len(idx)

        val_loss = validation_lxr_loss(
            model, explainer, X, cache, val_idx, config, device,
            batch=args.eval_batch, inverse_loss=args.inverse_loss)
        audit = hard_topk_audit(
            model, explainer, X, cache, val_idx, args.k, device,
            batch=args.eval_batch)
        row = {
            "data": data, "seed": seed, "beta": beta,
            "epoch": epoch + 1, "train_loss": total / seen,
            "val_lxr_loss": val_loss, "lr": optimizer.param_groups[0]["lr"],
            **{f"val_{key}": value for key, value in audit.items()},
        }
        history.append(row)
        print(
            f"  {data} s{seed} beta={beta:g} epoch {epoch + 1}/{args.epochs}: "
            f"Lval={val_loss:.3f}, Delta={audit['effect']:.3f}, "
            f"G90={audit['gap90']:.3f}, tail={audit['tail']:.1%}",
            flush=True,
        )
        if np.isfinite(val_loss) and val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch + 1
            torch.save({
                "state_dict": copy.deepcopy(explainer.state_dict()),
                "data": data, "seed": seed, "beta": beta,
                "best_epoch": best_epoch, "val_lxr_loss": best_loss,
                "inverse_loss": args.inverse_loss, "config": config,
            }, checkpoint)

    return history, checkpoint, best_epoch


def train_stage(args):
    classes = jlens.load_notebook_classes()
    inspector = load_model("VAE", classes, args.data)
    model = inspector.model.to(args.device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    data_dir = ROOT / "processed_data" / args.data
    train = pd.read_csv(
        data_dir / f"train_data_{args.data}.csv", index_col=0)
    X = torch.tensor(train.to_numpy(), dtype=torch.float32)
    if args.train_limit > 0:
        X = X[:args.train_limit]
    train_idx, val_idx = split_indices(
        len(X), val_fraction=args.val_fraction, split_seed=args.split_seed)
    cache = precompute_clean_state(
        inspector, X, args.device, n_competitors=5, batch=args.eval_batch)

    config = LXR_CONFIG[args.data]
    torch.manual_seed(args.seed)
    initial = LXRExplainer(X.shape[1], X.shape[1], config["hidden"])
    initial_state = copy.deepcopy(initial.state_dict())

    # Scale reference only: the released checkpoint was trained on the full
    # original train pool, including these users, so this is *not* an honest
    # held-out comparison and is never used for model selection.
    released = load_lxr(args.data).to(args.device)
    reference = hard_topk_audit(
        model, released, X, cache, val_idx, args.k, args.device,
        batch=args.eval_batch)
    print(f"released train-pool scale reference: {reference}", flush=True)

    rows = []
    for beta in args.betas:
        arm_rows, _, _ = train_arm(
            model, X, cache, train_idx, val_idx, args.data, args.seed, beta,
            args, initial_state)
        rows.extend(arm_rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.data.lower()}_lxr_retrain_validation_s{args.seed}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    meta = {
        "data": args.data,
        "seed": args.seed,
        "betas": args.betas,
        "split_seed": args.split_seed,
        "val_fraction": args.val_fraction,
        "inverse_loss": args.inverse_loss,
        "released_train_pool_scale_reference": reference,
    }
    (out.with_suffix(".json")).write_text(json.dumps(meta, indent=2) + "\n")
    print(f"saved validation-only results -> {out}", flush=True)
    return pd.DataFrame(rows)


def evaluate_stage(args):
    if len(args.betas) != 1:
        raise ValueError("evaluate stage requires exactly one selected --betas value")
    beta = args.betas[0]
    checkpoint = (Path(args.checkpoint_dir) / args.data.lower()
                  / f"s{args.seed}" / f"lxr_{beta_tag(beta)}.pth")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = LXR_CONFIG[args.data]

    classes = jlens.load_notebook_classes()
    inspector = load_model("VAE", classes, args.data)
    data_dir = ROOT / "processed_data" / args.data
    test = pd.read_csv(
        data_dir / f"test_data_{args.data}.csv", index_col=0)
    X_test = torch.tensor(test.to_numpy(), dtype=torch.float32)
    explainer = LXRExplainer(
        X_test.shape[1], X_test.shape[1], config["hidden"])
    explainer.load_state_dict(payload["state_dict"])
    explainer.eval()

    reference_file = (Path(args.out_dir)
                      / f"{args.data.lower()}_vae_lxr_shap_k{args.k}.csv")
    reference = pd.read_csv(reference_file)
    cohort = sorted(int(user) for user in reference["user"].unique())
    frame = evaluate(
        explainer, inspector, X_test, cohort, args.data, args.k, beta,
        args.seed)
    frame["method"] = f"lxr_retrain_{beta_tag(beta)}"
    frame["objective"] = "released_lxr_mechanism_aware"
    frame["best_epoch"] = payload["best_epoch"]
    out = (Path(args.out_dir)
           / f"{args.data.lower()}_vae_lxr_retrain_{beta_tag(beta)}_k{args.k}_s{args.seed}.csv")
    frame.to_csv(out, index=False)
    print(f"saved held-out test audit -> {out}", flush=True)
    return frame


def objective_stage(args):
    classes = jlens.load_notebook_classes()
    inspector = load_model("VAE", classes, args.data)
    model = inspector.model.to(args.device).eval()
    data_dir = ROOT / "processed_data" / args.data
    test = pd.read_csv(
        data_dir / f"test_data_{args.data}.csv", index_col=0)
    X = torch.tensor(test.to_numpy(), dtype=torch.float32)
    cache = precompute_clean_state(
        inspector, X, args.device, n_competitors=5, batch=args.eval_batch)
    indices = torch.arange(len(X))
    config = LXR_CONFIG[args.data]

    rows = []
    for beta in args.betas:
        checkpoint = (Path(args.checkpoint_dir) / args.data.lower()
                      / f"s{args.seed}" / f"lxr_{beta_tag(beta)}.pth")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        explainer = LXRExplainer(
            X.shape[1], X.shape[1], config["hidden"]).to(args.device)
        explainer.load_state_dict(payload["state_dict"])
        metrics = lxr_objective_metrics(
            model, explainer, X, cache, indices, config, args.device,
            batch=args.eval_batch, inverse_loss=args.inverse_loss)
        row = {"data": args.data, "seed": args.seed, "beta": beta,
               "n": len(X), **metrics}
        rows.append(row)
        print(row, flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.data.lower()}_lxr_retrain_test_objective_s{args.seed}.csv"
    frame = pd.DataFrame(rows)
    frame.to_csv(out, index=False)
    print(f"saved held-out LXR-objective diagnostic -> {out}", flush=True)
    return frame


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["train", "evaluate", "objective"],
                        default="train")
    parser.add_argument("--data", choices=list(LXR_CONFIG), default="Pinterest")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--betas", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=314159)
    parser.add_argument("--eval-batch", type=int, default=256)
    parser.add_argument("--train-limit", type=int, default=-1,
                        help="smoke-test only; final runs use all training users")
    parser.add_argument("--inverse-loss", choices=["released_code", "paper"],
                        default="released_code")
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--checkpoint-dir", default=str(ROOT / "check" / "lxr_retrain"))
    parser.add_argument("--out-dir", default=str(ROOT / "jlens_results"))
    args = parser.parse_args(argv)
    if any(beta < 0 or beta > 1 for beta in args.betas):
        raise ValueError("mechanism interpolation beta must lie in [0, 1]")
    if args.stage == "train":
        return train_stage(args)
    if args.stage == "evaluate":
        return evaluate_stage(args)
    return objective_stage(args)


if __name__ == "__main__":
    main()
