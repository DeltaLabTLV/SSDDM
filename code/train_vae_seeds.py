"""Train MultVAE seed replicates on ML1M / Yahoo / Pinterest.

The paper's carrier claim ("dormant capacity tracks white-box vulnerability")
rests on one checkpoint per architecture-dataset pair, which supports only a
descriptive association. This script produces independently trained replicates
of the released MultVAE checkpoints so the association can be tested across
checkpoints that differ only in training seed.

Recipe per dataset: the released hyperparameters, a fixed 90/10 user split for
early stopping (the split is seeded independently of the model seed, so
replicates differ only in initialization and batch order), and a quality gate
reporting test cross-entropy plus top-10 overlap with the released checkpoint.

Usage:
  python3 train_vae_seeds.py --data ML1M Yahoo Pinterest --seeds 1 2 3
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import jlens
from run_jlens_ml1m import CONFIGS, ROOT, VAE_CONF, load_model

# Released-checkpoint hyperparameters, read off the checkpoint filenames in
# check/ (VAE_<DATA>_<lr>_<batch>_<epoch>.pt).
RECIPES = {
    "ML1M": {"lr": 7e-4, "batch_size": 128},
    "Yahoo": {"lr": 1e-4, "batch_size": 128},
    "Pinterest": {"lr": 1e-4, "batch_size": 256},
}

VAL_SPLIT_SEED = 20260809  # fixed across replicates on purpose


def data_dir(name):
    return ROOT / "processed_data" / name


def load_matrices(name):
    train = pd.read_csv(data_dir(name) / f"train_data_{name}.csv", index_col=0)
    test = pd.read_csv(data_dir(name) / f"test_data_{name}.csv", index_col=0)
    return (train.to_numpy().astype(np.float32),
            test.to_numpy().astype(np.float32))


def mean_ce(model, X, device, batch_size=256):
    """Reconstruction cross-entropy in eval mode (deterministic, z = mu)."""
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i + batch_size]).to(device)
            probs = model(xb)
            total += float(-(torch.log(probs + 1e-10) * xb).sum(1).sum())
            n += len(xb)
    return total / max(n, 1)


def top10(model, X, device, batch_size=256):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i + batch_size]).to(device)
            probs = model(xb).clone()
            probs[xb > 0] = -1.0
            out.append(probs.argsort(dim=1, descending=True)[:, :10].cpu())
    return torch.cat(out)


def overlap(a, b):
    return float(np.mean([len(set(a[i].tolist()) & set(b[i].tolist())) / 10
                          for i in range(len(a))]))


def build_model(name, classes, device):
    cfg = CONFIGS[name]
    conf = dict(VAE_CONF, enc_dims=cfg["enc_dims"])
    return classes["VAE"](conf, device=device, num_features=cfg["n_items"],
                          num_items=cfg["n_items"], demographic=False)


def train_one(name, seed, args, classes, ref_top10, X_tr, X_val, X_test):
    device = torch.device(args.device)
    torch.manual_seed(seed)
    np.random.seed(seed)

    recipe = RECIPES[name]
    lr = args.lr if args.lr is not None else recipe["lr"]
    bs = args.batch_size if args.batch_size is not None else recipe["batch_size"]

    model = build_model(name, classes, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_ce, best_state, best_epoch, stale = float("inf"), None, 0, 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        model.train_one_epoch(X_tr, optimizer, bs)
        ce = mean_ce(model, X_val, device)
        if ce < best_ce - args.min_delta:
            best_ce, best_epoch, stale = ce, epoch, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            stale += 1
        print(f"  [{name} s{seed}] epoch {epoch:3d}  val CE {ce:9.3f}"
              f"  best {best_ce:9.3f} @{best_epoch}", flush=True)
        if stale >= args.patience:
            break

    model.load_state_dict(best_state)
    model.to(device)
    model.device = device

    out_dir = ROOT / "check" / "seeds"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"VAE_{name}_s{seed}.pt"
    torch.save(best_state, path)

    rec = {
        "dataset": name, "seed": seed, "lr": lr, "batch_size": bs,
        "epochs_run": epoch, "best_epoch": best_epoch,
        "val_ce": best_ce,
        "test_ce": mean_ce(model, X_test, device),
        "top10_overlap_released": overlap(top10(model, X_test, device),
                                          ref_top10),
        "minutes": (time.time() - t0) / 60.0,
        "checkpoint": path.name,
    }
    print(f"  [{name} s{seed}] DONE test CE {rec['test_ce']:.3f} "
          f"overlap {rec['top10_overlap_released']:.3f} "
          f"({rec['minutes']:.1f} min)", flush=True)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["ML1M", "Yahoo", "Pinterest"],
                    choices=list(CONFIGS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--epochs", type=int, default=60, help="max epochs")
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--min-delta", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=None, help="override recipe")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--out", default=str(ROOT / "jlens_results" /
                                         "vae_seed_replicates.json"))
    args = ap.parse_args(argv)

    classes = jlens.load_notebook_classes()
    device = torch.device(args.device)
    records = []
    out_path = Path(args.out)
    if out_path.exists():
        records = json.loads(out_path.read_text())

    for name in args.data:
        X_train_all, X_test = load_matrices(name)
        rng = np.random.default_rng(VAL_SPLIT_SEED)
        perm = rng.permutation(len(X_train_all))
        n_val = max(1, int(round(args.val_frac * len(X_train_all))))
        X_val = X_train_all[perm[:n_val]]
        X_tr = X_train_all[perm[n_val:]]

        ref = load_model("VAE", classes, name)
        ref.model.to(device)
        ref.model.device = device
        ref_top10 = top10(ref.model, X_test, device)
        del ref
        print(f"[{name}] train {X_tr.shape} val {X_val.shape} "
              f"test {X_test.shape}", flush=True)

        done = {(r["dataset"], r["seed"]) for r in records}
        for seed in args.seeds:
            if (name, seed) in done:
                print(f"  [skip] {name} s{seed} already recorded", flush=True)
                continue
            records.append(train_one(name, seed, args, classes, ref_top10,
                                     X_tr, X_val, X_test))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(records, indent=2))

    print(f"wrote {out_path} ({len(records)} records)")


if __name__ == "__main__":
    main()
