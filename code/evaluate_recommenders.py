"""Evaluate released recommenders on the authors' static leave-one-out split.

The positive item stored in ``pos`` is removed from the input history. All
remaining observed items are masked from the ranked catalogue. The script
reports full-catalogue HR@10/50/100, MRR, and mean percentile rank.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch

import jlens
from run_jlens_ml1m import CONFIGS, ROOT, load_model


def evaluate(inspector, frame, n_items, batch_size=32, n_users=-1):
    if n_users >= 0:
        frame = frame.iloc[:n_users]
    X = torch.tensor(frame.iloc[:, :n_items].to_numpy(), dtype=torch.float32)
    pos = torch.tensor(frame["pos"].to_numpy(), dtype=torch.long)
    X[torch.arange(len(X)), pos] = 0.0
    ranks = []
    for start in range(0, len(X), batch_size):
        xb = X[start:start + batch_size]
        pb = pos[start:start + batch_size]
        with torch.no_grad():
            scores, _ = inspector.run(xb)
        scores[xb > 0] = float("-inf")
        target = scores[torch.arange(len(xb)), pb]
        # Rank 1 is best. Ties receive pessimistic rank, deterministic for an
        # audit and immaterial for the continuously valued neural scores.
        ranks.append((scores >= target[:, None]).sum(1))
    rank = torch.cat(ranks).float()
    return {
        "n_users": len(rank),
        "hr10": float((rank <= 10).float().mean()),
        "hr50": float((rank <= 50).float().mean()),
        "hr100": float((rank <= 100).float().mean()),
        "mrr": float((1.0 / rank).mean()),
        "mean_percentile_rank": float(((rank - 1) / (n_items - 1)).mean()),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=list(CONFIGS),
                    choices=list(CONFIGS))
    ap.add_argument("--models", nargs="+", default=["VAE", "NCF"],
                    choices=["VAE", "NCF"])
    ap.add_argument("--n-users", type=int, default=-1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default=str(
        ROOT / "jlens_results" / "recommender_quality.json"))
    args = ap.parse_args(argv)
    classes = jlens.load_notebook_classes()
    results = []
    for data in args.data:
        cfg = CONFIGS[data]
        path = (ROOT / "processed_data" / data
                / f"static_test_data_{data}.csv")
        frame = pd.read_csv(path, index_col=0)
        for model_name in args.models:
            inspector = load_model(model_name, classes, data)
            batch = args.batch_size
            if model_name == "NCF" and data == "Pinterest":
                batch = min(batch, 4)
            metrics = evaluate(inspector, frame, cfg["n_items"], batch,
                               args.n_users)
            row = {"data": data, "model": model_name, **metrics}
            results.append(row)
            print(row, flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"saved -> {out}")
    return results


if __name__ == "__main__":
    main()
