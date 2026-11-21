"""Train NeuMF (NCF) seed replicates on ML1M / Yahoo / Pinterest.

Companion to ``train_vae_seeds.py``: the manuscript evaluates one NeuMF
checkpoint per dataset, so its carrier/vulnerability statements for that
architecture rest on three models. This reproduces the original training
procedure -- NeuMF-pre initialised from the released GMF and MLP components,
popularity-sampled negatives, the pos/neg objective of the project notebook --
with a configurable seed.

The recipe is recovered from the released checkpoint filenames
(``NCF_<data>_<lr>_<batch>_<trial>_<epoch>.pt``) and then corrected empirically;
see ``RECIPES``.

Scope caveat: NeuMF-pre replicates are initialised from the *same* released GMF
and MLP components, so they differ only in negative sampling, batch order and
the fine-tuning trajectory. Their spread is therefore a lower bound on true
training variance for this architecture -- a full replicate would retrain those
components too.

Usage:
  python3 train_ncf_seeds.py --data ML1M --seeds 1 2 3
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
from run_jlens_ml1m import CONFIGS, ROOT, load_model

CHECK = ROOT / "check"

# Pretrained NeuMF-pre components, downloaded from the project's release drive.
SUBMODELS = {
    "ML1M": ("GMF_best_ML1M_0.0001_32_17.pt",
             "MLP_model_ML1M_0.0001_64_27.pt"),
    "Yahoo": ("GMF_model2_Yahoo_0.0_128_0_49.pt",
              "MLP_model_Yahoo_5e-05_32_29.pt"),
    "Pinterest": ("GMF_model_Pinterest_0.001_512_3_19.pt",
                  "MLP_model_Pinterest_4e-05_1024_7_18.pt"),
}

# Starting learning rate and batch size per dataset. The filename-implied
# 5e-3 for ML1M destroys the pretrained initialisation (HR@10 falls from 0.113
# at epoch 1 to 0.089 by epoch 4); 5e-4, the bottom of the notebook's search
# range, reaches HR@10 0.125 against the released checkpoint's 0.118. Because
# NeuMF-pre starts from converged components, the optimum is an early epoch.
RECIPES = {
    "ML1M": {"lr": 5e-4, "batch_size": 64, "beta": 1.0},
    "Yahoo": {"lr": 5e-4, "batch_size": 64, "beta": 1.0},
    "Pinterest": {"lr": 5e-4, "batch_size": 32, "beta": 1.0},
}


def load_notebook_functions(nb_path):
    """Exec the function-defining cells of a project notebook."""
    import torch.nn.functional as F
    nb = json.loads(Path(nb_path).read_text())
    ns = {"torch": torch, "nn": torch.nn, "F": F, "np": np, "pd": pd}
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if any(line.startswith("def ") for line in src.splitlines()):
            exec(compile(src, str(nb_path), "exec"), ns)
    return ns


def load_context(name, device):
    """Assemble the kw_dict the project's helper functions expect."""
    import pickle

    data_dir = ROOT / "processed_data" / name
    n_items = CONFIGS[name]["n_items"]

    train = pd.read_csv(data_dir / f"train_data_{name}.csv", index_col=0)
    static_test = pd.read_csv(data_dir / f"static_test_data_{name}.csv",
                              index_col=0)
    # The notebook masks each held-out positive out of its own test row.
    for row in range(static_test.shape[0]):
        static_test.iloc[row, static_test.iloc[row, -2]] = 0

    with open(data_dir / f"pop_dict_{name}.pkl", "rb") as fh:
        pop_dict = pickle.load(fh)
    pop = np.zeros(n_items)
    for key, value in pop_dict.items():
        pop[int(key)] = value

    items_array = np.eye(n_items, dtype=np.float32)
    kw = {"device": device, "num_items": n_items, "num_features": n_items,
          "demographic": False, "pop_array": pop,
          "all_items_tensor": torch.tensor(items_array).to(device),
          "static_test_data": static_test, "items_array": items_array,
          "output_type": "single", "recommender_name": "NCF"}
    return train, kw


def build_ncf(name, classes, device):
    cfg = CONFIGS[name]
    n_items, factor, layers = cfg["n_items"], cfg["ncf_factor"], 3
    kw = dict(device=device, num_features=n_items, num_items=n_items,
              demographic=False)
    gmf_file, mlp_file = SUBMODELS[name]
    gmf = classes["GMF_model"](hidden_size=factor, **kw)
    gmf.load_state_dict(torch.load(CHECK / gmf_file, map_location="cpu"))
    mlp = classes["MLP_model"](factor, layers, **kw)
    mlp.load_state_dict(torch.load(CHECK / mlp_file, map_location="cpu"))
    return classes["NCF"](factor, layers, 0.5, "NeuMF-pre", GMF_model=gmf,
                          MLP_model=mlp, **kw)


def train_one(name, seed, args, classes, funcs, train_df, kw, ref_hr10):
    device = kw["device"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    recipe = RECIPES[name]
    lr0 = args.lr if args.lr is not None else recipe["lr"]
    batch_size = (args.batch_size if args.batch_size is not None
                  else recipe["batch_size"])
    beta = args.beta if args.beta is not None else recipe["beta"]

    model = build_ncf(name, classes, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr0)
    items_array = kw["items_array"]
    n_train = train_df.shape[0]
    n_batches = int(np.ceil(n_train / batch_size))

    best_hr10, best_state, best_epoch, stale = -1.0, None, 0, 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        if epoch > 1 and (epoch - 1) % 10 == 0:
            for group in optimizer.param_groups:
                group["lr"] *= 0.1
        model.train()
        matrix = funcs["sample_indices"](train_df.copy(), **kw)
        perm = np.random.permutation(n_train)
        for b in range(n_batches):
            optimizer.zero_grad()
            idx = perm[b * batch_size:(b + 1) * batch_size]
            if len(idx) == 0:
                continue
            users = torch.tensor(matrix[idx, :-2].astype(np.float32)).to(device)
            pos = torch.tensor(items_array[matrix[idx, -2].astype(int)]).to(device)
            neg = torch.tensor(items_array[matrix[idx, -1].astype(int)]).to(device)
            pos_out = model(users, pos).clamp(1e-7, 1 - 1e-7)
            neg_out = model(users, neg).clamp(1e-7, 1 - 1e-7)
            loss = -torch.log(pos_out).mean() - beta * torch.log(1 - neg_out).mean()
            if torch.isfinite(loss):
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            hr10, hr50, hr100, mrr, mpr = funcs["recommender_evaluations"](
                model, **kw)
        if hr10 > best_hr10:
            best_hr10, best_epoch, stale = hr10, epoch, 0
            if not args.save_final:
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
        else:
            stale += 1
        if args.save_final:
            # Keep the last epoch instead of the HR@10 optimum: used to test
            # whether vulnerability grows with training beyond the point where
            # ranking quality stops improving.
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            best_epoch = epoch
        print(f"  [{name} s{seed}] epoch {epoch:3d}  HR@10 {hr10:.4f}  "
              f"best {best_hr10:.4f} @{best_epoch}", flush=True)
        if stale >= args.patience:
            break

    out_dir = CHECK / "seeds"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"NCF_{name}_s{seed}.pt"
    torch.save(best_state, path)
    rec = {"dataset": name, "seed": seed, "lr": lr0, "batch_size": batch_size,
           "beta": beta, "epochs_run": epoch, "best_epoch": best_epoch,
           "hr10": float(best_hr10), "hr10_released": float(ref_hr10),
           "minutes": (time.time() - t0) / 60.0, "checkpoint": path.name}
    print(f"  [{name} s{seed}] DONE HR@10 {best_hr10:.4f} "
          f"(released {ref_hr10:.4f}) in {rec['minutes']:.1f} min", flush=True)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=list(CONFIGS),
                    choices=list(CONFIGS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--save-final", action="store_true",
                    help="save the last epoch rather than the HR@10 optimum")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--out", default=str(ROOT / "jlens_results" /
                                         "ncf_seed_replicates.json"))
    args = ap.parse_args(argv)

    classes = jlens.load_notebook_classes()
    funcs = load_notebook_functions(Path(__file__).resolve().parent
                                    / "help_functions.ipynb")
    device = torch.device(args.device)
    out_path = Path(args.out)
    records = json.loads(out_path.read_text()) if out_path.exists() else []

    for name in args.data:
        train_df, kw = load_context(name, device)
        released = load_model("NCF", classes, name)
        released.model.to(device)
        released.model.device = device
        with torch.no_grad():
            ref_hr10 = funcs["recommender_evaluations"](released.model, **kw)[0]
        del released
        print(f"[{name}] released HR@10 {ref_hr10:.4f}, train {train_df.shape}",
              flush=True)

        done = {(r["dataset"], r["seed"]) for r in records}
        for seed in args.seeds:
            if (name, seed) in done:
                print(f"  [skip] {name} s{seed} already recorded", flush=True)
                continue
            records.append(train_one(name, seed, args, classes, funcs,
                                     train_df, kw, ref_hr10))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(records, indent=2))

    print(f"wrote {out_path} ({len(records)} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
