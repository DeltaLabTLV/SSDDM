"""Sensitivity of the mechanism gap to competitor count and selection.

The default audit spans the target and the five highest-scored clean
non-history items.  This runner keeps the model, user, target, deletion set,
and tap fixed while changing only that competitor set.  It supports the
static MultVAE/NeuMF checkpoints and the SASRec seed checkpoints used by the
paper.

Static example:
  python run_jlens_competitor_sensitivity.py static --data ML1M --model VAE

SASRec example:
  python run_jlens_competitor_sensitivity.py sasrec \
      --ckpt ../check/sasrec_seeds/sasrec_movielens-1m_s42.pth \
      --input ../jlens_results/ml1m_sasrec_basis_robustness_k5_s42.csv
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import jlens
import jlens_sasrec as js
from run_jlens_ml1m import CONFIGS, ROOT, load_model
from run_jlens_sasrec import DEFAULT_LXSR, build_cohort


DEFAULT_COUNTS = (0, 1, 3, 5, 10, 20)
DEFAULT_STRATEGIES = ("top", "sampled_top100", "uniform")


def competitor_sets(ranked_nonhistory, counts=DEFAULT_COUNTS, seed=0,
                    strategies=DEFAULT_STRATEGIES, local_pool=100):
    """Return nested competitor sets for each requested strategy and count.

    ``ranked_nonhistory`` starts with the target.  ``sampled_top100`` samples
    without replacement from the next locally competitive items; ``uniform``
    samples from the remaining non-history catalog.  One permutation per
    strategy makes the sets nested as M grows, so changes are attributable to
    adding directions rather than redrawing the smaller set.
    """
    ranked = [int(i) for i in ranked_nonhistory]
    if not ranked:
        raise ValueError("ranked_nonhistory must contain the target")
    pool = ranked[1:]
    max_count = max(counts)
    if len(pool) < max_count:
        raise ValueError(f"need {max_count} competitors, found {len(pool)}")

    rng = np.random.default_rng(seed)
    local_n = min(max(local_pool, max_count), len(pool))
    orders = {
        "top": pool,
        "sampled_top100": rng.permutation(pool[:local_n]).tolist(),
        "uniform": rng.permutation(pool).tolist(),
    }
    result = {}
    for strategy in strategies:
        if strategy not in orders:
            raise ValueError(f"unknown strategy: {strategy}")
        for m in counts:
            # M=0 is identical under every strategy; retain one copy.
            if m == 0 and strategy != "top":
                continue
            result[(strategy, int(m))] = orders[strategy][:int(m)]
    return result


def ranked_static(inspector, x):
    with torch.no_grad():
        logits, _ = inspector.run(x)
    scores = logits[0].clone()
    scores[x > 0] = float("-inf")
    return scores.argsort(descending=True).tolist()


def run_static(args):
    classes = jlens.load_notebook_classes()
    inspector = load_model(args.model, classes, args.data, ckpt=args.ckpt)
    tap = args.tap or ("z" if args.model == "VAE" else "user_state")
    data_file = (ROOT / "processed_data" / args.data
                 / f"test_data_{args.data}.csv")
    X = torch.tensor(pd.read_csv(data_file, index_col=0).to_numpy(),
                     dtype=torch.float32)

    source = Path(args.input) if args.input else (
        ROOT / "jlens_results"
        / f"{args.data.lower()}_{args.model.lower()}_adversarial_k{args.k}.csv")
    saved = pd.read_csv(source)
    saved = saved[saved["tap"] == tap]
    saved = saved[saved["method"].isin(args.methods)]
    saved = saved.drop_duplicates(["user", "method"])
    users = sorted(saved["user"].unique())[:args.n_users]

    rows = []
    t0 = time.time()
    for pos, user in enumerate(users):
        x = X[int(user)]
        ranked = ranked_static(inspector, x)
        y = int(ranked[0])
        sets = competitor_sets(ranked, args.counts, int(user) + args.seed,
                               args.strategies, args.local_pool)
        user_rows = saved[saved["user"] == user]
        for _, saved_row in user_rows.iterrows():
            deletion = json.loads(saved_row["S"])
            for (strategy, m), competitors in sets.items():
                result = jlens.mechanistic_eval(
                    inspector, x, deletion, y, competitors=competitors,
                    taps=[tap])[0]
                rows.append({
                    "architecture": args.model,
                    "data": args.data,
                    "user": int(user),
                    "method": saved_row["method"],
                    "k": int(saved_row["k"]),
                    "strategy": strategy,
                    "n_competitors": m,
                    "basis_rank": result["n_directions"],
                    **result,
                })
        if (pos + 1) % 10 == 0 or pos + 1 == len(users):
            elapsed = time.time() - t0
            print(f"{pos + 1}/{len(users)} users "
                  f"({elapsed / (pos + 1):.2f}s/user)", flush=True)

    out = Path(args.out) if args.out else (
        ROOT / "jlens_results"
        / f"{args.data.lower()}_{args.model.lower()}_competitor_sensitivity_k{args.k}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out, index=False)
    print(f"saved {len(frame)} rows -> {out}")
    report(frame)
    return frame


def ranked_sasrec(auditor, tokens):
    ids = torch.tensor(tokens, dtype=torch.long,
                       device=auditor.device).unsqueeze(0)
    logits = auditor.logits(ids, torch.ones_like(ids)).clone()
    logits[0] = float("-inf")
    logits[list(set(tokens))] = float("-inf")
    return logits.argsort(descending=True).tolist()


def run_sasrec(args):
    lxsr_root = Path(args.lxsr_root)
    data_dir = Path(args.data_dir) if args.data_dir else (
        lxsr_root
        / "data/splitted/global_timesplit/val_by_time/Movielens-1m/q08")
    device = args.device if torch.cuda.is_available() else "cpu"
    model, tokenizer, _ = js.load_sasrec(
        lxsr_root, args.ckpt, data_dir / "train.csv", device=device)
    auditor = js.SasRecAuditor(model, device=device)

    saved = pd.read_csv(args.input)
    method_col = "method" if "method" in saved else "explainer"
    saved = saved[saved[method_col].isin(args.methods)]
    if "tap" in saved:
        saved = saved[saved["tap"] == args.tap]
    saved = saved.drop_duplicates(["user", method_col])
    users, _ = build_cohort(lxsr_root, data_dir, args.k)
    sequence_by_user = {int(uid): seq for uid, seq in users}
    selected_users = [int(u) for u in sorted(saved["user"].unique())
                      if int(u) in sequence_by_user][:args.n_users]

    rows = []
    t0 = time.time()
    for pos, user in enumerate(selected_users):
        seq = sequence_by_user[user]
        tokens = [tokenizer.item2id[item] for item in seq]
        tokens = tokens[-tokenizer.max_seq_length:]
        ranked = ranked_sasrec(auditor, tokens)
        y = int(ranked[0])
        sets = competitor_sets(ranked, args.counts, user + args.seed,
                               args.strategies, args.local_pool)
        user_rows = saved[saved["user"] == user]
        for _, saved_row in user_rows.iterrows():
            deletion = json.loads(saved_row["S"])
            for (strategy, m), competitors in sets.items():
                result = js.audit_set(
                    auditor, tokens, deletion, y, competitors,
                    taps=[args.tap], ref_states=("clean",))[0]
                rows.append({
                    "architecture": "SASRec",
                    "data": args.dataset_label,
                    "user": user,
                    "method": saved_row[method_col],
                    "k": len(deletion),
                    "strategy": strategy,
                    "n_competitors": m,
                    "basis_rank": result["basis_rank"],
                    **result,
                })
        if (pos + 1) % 5 == 0 or pos + 1 == len(selected_users):
            elapsed = time.time() - t0
            print(f"{pos + 1}/{len(selected_users)} users "
                  f"({elapsed / (pos + 1):.2f}s/user)", flush=True)

    out = Path(args.out) if args.out else (
        ROOT / "jlens_results"
        / f"{args.dataset_label.lower()}_sasrec_competitor_sensitivity_k{args.k}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out, index=False)
    print(f"saved {len(frame)} rows -> {out}")
    report(frame)
    return frame


def report(frame):
    positive = frame[frame["delta_beh"] > 0.01].copy()
    if "clean_gap" not in positive:
        positive["clean_gap"] = (1.0 - positive["mr"]).clip(0.0, 1.0)
    summary = positive.groupby(
        ["method", "strategy", "n_competitors"], dropna=False).agg(
        n=("user", "nunique"),
        gap50=("clean_gap", "median"),
        gap90=("clean_gap", lambda s: s.quantile(0.9)),
        rank50=("basis_rank", "median"),
    )
    print(summary.round(3).to_string())


def parser():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="kind", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--counts", type=int, nargs="+",
                        default=list(DEFAULT_COUNTS))
    common.add_argument("--strategies", nargs="+",
                        choices=list(DEFAULT_STRATEGIES),
                        default=list(DEFAULT_STRATEGIES))
    common.add_argument("--local-pool", type=int, default=100)
    common.add_argument("--n-users", type=int, default=60)
    common.add_argument("--seed", type=int, default=2027)
    common.add_argument("--out", default=None)

    static = sub.add_parser("static", parents=[common])
    static.add_argument("--data", default="ML1M", choices=list(CONFIGS))
    static.add_argument("--model", default="VAE", choices=["VAE", "NCF"])
    static.add_argument("--k", type=int, default=10)
    static.add_argument("--tap", default=None)
    static.add_argument("--methods", nargs="+",
                        default=["saliency", "attack_lam1"])
    static.add_argument("--input", default=None)
    static.add_argument("--ckpt", default=None)
    static.set_defaults(func=run_static)

    sasrec = sub.add_parser("sasrec", parents=[common])
    sasrec.add_argument("--lxsr-root", default=str(DEFAULT_LXSR))
    sasrec.add_argument("--ckpt", required=True)
    sasrec.add_argument("--input", required=True)
    sasrec.add_argument("--data-dir", default=None)
    sasrec.add_argument("--dataset-label", default="ML1M")
    sasrec.add_argument("--k", type=int, default=5)
    sasrec.add_argument("--tap", default="block1")
    sasrec.add_argument("--methods", nargs="+",
                        default=["occlusion", "lime", "attack_lam1"])
    sasrec.add_argument("--device", default="cuda:0")
    sasrec.set_defaults(func=run_sasrec)
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    main()
