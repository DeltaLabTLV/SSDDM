"""Global (averaged-Jacobian) vs local lens directions on the trained VAE.

Three direction providers at tap z:
- local:  v_c = grad of logit_c at the explained user's clean run (default)
- global: v_c = mean over a training-user batch of local directions —
          the reusable, J-lens-style "item workspace" directions
- linear: v_c = (W_out @ W_dec1)[c] — the decoder linearized through ReLU
          (logit-lens analog; ignores the activation pattern entirely)

If global/linear mediate as well as local, the recsys "item workspace" is
input-independent and the J-lens framing transfers; if only local works,
the honest claim is attribution patching, not a global lens.

Usage: python3 run_jlens_global.py [--n-users 200] [--batch 256]
"""
import argparse
from pathlib import Path

import pandas as pd
import torch

import jlens
import jlens_explainers as jx
from run_jlens_ml1m import (DATA, ROOT, load_model, target_and_competitors)


class GlobalDirections:
    """Averaged-Jacobian directions at tap z over a training-user batch,
    cached per item."""

    def __init__(self, inspector, X_train, tap="z"):
        self.inspector, self.tap = inspector, tap
        with torch.no_grad():
            _, acts = inspector.run(X_train)
        self.h = acts[tap].detach().requires_grad_(True)
        self.logits = inspector.run_from(tap, self.h)
        self.cache = {}

    def __call__(self, tap, C):
        assert tap == self.tap
        cols = []
        for c in C:
            if c not in self.cache:
                (g,) = torch.autograd.grad(self.logits[:, c].sum(), self.h,
                                           retain_graph=True)
                self.cache[c] = g.mean(0)
            cols.append(self.cache[c])
        return torch.stack(cols, dim=1)


def linear_lens_directions(model):
    W1 = model.decoder[0].weight          # (512, 128)
    W_out = model.decoder[-1].weight      # (n_items, 512)
    M = W_out @ W1                        # (n_items, 128)

    def provider(tap, C):
        assert tap == "z"
        return M[C, :].T
    return provider


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-users", type=int, default=200)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out-dir", default=str(ROOT / "jlens_results"))
    args = ap.parse_args(argv)

    classes = jlens.load_notebook_classes()
    insp = load_model("VAE", classes)
    train = pd.read_csv(DATA / "train_data_ML1M.csv", index_col=0)
    X_train = torch.tensor(train.to_numpy()[:args.batch], dtype=torch.float32)
    test = pd.read_csv(DATA / "test_data_ML1M.csv", index_col=0)
    X = torch.tensor(test.to_numpy(), dtype=torch.float32)

    providers = {
        "local": None,
        "global": GlobalDirections(insp, X_train),
        "linear": linear_lens_directions(insp.model),
    }

    rows = []
    n_users = min(args.n_users, len(X))
    for u in range(n_users):
        x = X[u]
        if int(x.sum()) <= args.k:
            continue
        y, comps = target_and_competitors(insp, x, 5)
        expl = {
            "saliency": jx.topk_from_map(jx.saliency(insp, x, y), x, args.k),
            "random": jx.random_topk(x, args.k, seed=u),
        }
        # direction similarity local vs global/linear for the target item
        v_loc = jlens.item_directions(insp, x, "z", [y])[:, 0]
        sims = {}
        for pname in ["global", "linear"]:
            v_alt = providers[pname]("z", [y])[:, 0]
            sims[pname] = float(torch.cosine_similarity(
                v_loc, v_alt, dim=0))
        for name, S in expl.items():
            for pname, prov in providers.items():
                r = jlens.mechanistic_eval(insp, x, S, y, competitors=comps,
                                           taps=["z"], directions=prov)[0]
                rows.append({"user": u, "explainer": name,
                             "directions": pname,
                             "cos_local": sims.get(pname, 1.0), **r})

    df = pd.DataFrame(rows)
    out = Path(args.out_dir) / f"ml1m_vae_global_vs_local_k{args.k}.csv"
    df.to_csv(out, index=False)
    print(f"saved {len(df)} rows -> {out}")

    pd.set_option("display.width", 200)
    strong = df[df["delta_beh"].abs() > 0.01]
    agg = strong.groupby(["explainer", "directions"]).agg(
        MR_median=("mr", "median"),
        MR_q10=("mr", lambda s: s.quantile(0.10)),
        JComp_mean=("jcomp", "mean"),
        cos_to_local=("cos_local", "mean"),
        erank=("erank", "mean"),
    )
    print(f"\n=== Direction providers @ z, {n_users} users, K={args.k} ===")
    print(agg.round(4))
    return df


if __name__ == "__main__":
    main()
