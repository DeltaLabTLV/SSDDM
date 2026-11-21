"""Train a MultVAE on ML1M with checkpoints at 25/50/75/100% of training,
replicating the training-percentage (TP) protocol of Mohammadi et al.
(RecSys'24/'25) as a performance-level ladder.

Quality per checkpoint: mean test cross-entropy and top-10 overlap with the
released VAE_ML1M checkpoint (sanity that TP100 approaches it).

Usage: python3 train_vae_tp.py [--epochs 40] [--device cuda]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import jlens
from run_jlens_ml1m import CONFIGS, DATA, CHECK, VAE_CONF, load_model

TP_STEPS = (25, 50, 75, 100)


def quality(model, X_test, ref_top10, device):
    model.eval()
    with torch.no_grad():
        probs = model(X_test.to(device)).cpu()
    ce = -(torch.log(probs + 1e-10) * X_test).sum(1).mean()
    masked = probs.clone()
    masked[X_test > 0] = -1.0
    top10 = masked.argsort(dim=1, descending=True)[:, :10]
    overlap = np.mean([
        len(set(top10[u].tolist()) & set(ref_top10[u].tolist())) / 10
        for u in range(len(top10))])
    return float(ce), float(overlap)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=7e-4)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    train = pd.read_csv(DATA / "train_data_ML1M.csv", index_col=0)
    train_array = train.to_numpy().astype(np.float32)
    test = pd.read_csv(DATA / "test_data_ML1M.csv", index_col=0)
    X_test = torch.tensor(test.to_numpy()[:300], dtype=torch.float32)

    classes = jlens.load_notebook_classes()
    ref = load_model("VAE", classes)  # released checkpoint (inspector), CPU
    with torch.no_grad():
        rp = ref.model(X_test)
        rp_masked = rp.clone()
        rp_masked[X_test > 0] = -1.0
        ref_top10 = rp_masked.argsort(dim=1, descending=True)[:, :10]

    kw = dict(device=device, num_features=CONFIGS["ML1M"]["n_items"],
              num_items=CONFIGS["ML1M"]["n_items"], demographic=False)
    model = classes["VAE"](VAE_CONF, **kw)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    out_dir = CHECK / "tp"
    out_dir.mkdir(exist_ok=True)
    per_tp = {tp: max(1, round(args.epochs * tp / 100)) for tp in TP_STEPS}
    for epoch in range(1, args.epochs + 1):
        loss = model.train_one_epoch(train_array, optimizer,
                                     args.batch_size)
        if epoch in per_tp.values():
            tp = [t for t, e in per_tp.items() if e == epoch][0]
            model_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
            path = out_dir / f"VAE_ML1M_tp{tp}.pt"
            torch.save(model_cpu, path)
            model.to(device)
            model.device = device
            was = model.training
            model.eval()
            model.device = device
            ce, ov = quality(model, X_test, ref_top10, device)
            if was:
                model.train()
            print(f"TP{tp}: epoch {epoch}, loss {float(loss):.1f}, "
                  f"test CE {ce:.2f}, top10-overlap with released {ov:.3f}",
                  flush=True)
    print("done")


if __name__ == "__main__":
    main()
