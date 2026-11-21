"""End-to-end demo of the Item-Jacobian lens pipeline on random-init models.

Runs the full metric pipeline (deletion, lens directions, causal patching,
MR/JComp/JRestore/selectivity, random-subspace control) for two toy
explainers — input-gradient saliency and random — on a small VAE and the
MLP positive control. No checkpoints or data needed; plugging a trained
model goes through `load_recommender(...)` from help_functions instead of
the random init below.

Usage: python3 jlens_demo.py
"""
import pandas as pd
import torch

import jlens

N_ITEMS = 300
HISTORY = 25
TOP_K = 5


def _make_models(classes):
    torch.manual_seed(0)
    vae_conf = {"enc_dims": [64, 32], "dropout": 0.5,
                "anneal_cap": 0.2, "total_anneal_steps": 200000}
    kw = dict(device=torch.device("cpu"), num_features=N_ITEMS,
              num_items=N_ITEMS, demographic=False)
    vae = classes["VAE"](vae_conf, **kw).eval()
    mlp = classes["MLP"](32, **kw).eval()
    return vae, mlp


def _make_users(n_users):
    torch.manual_seed(1)
    users = []
    for _ in range(n_users):
        x = torch.zeros(N_ITEMS)
        x[torch.randperm(N_ITEMS)[:HISTORY]] = 1.0
        users.append(x)
    return users


def saliency_topk(inspector, x, y, k):
    """Input-gradient explainer: top-k history items by |grad * input|."""
    x = x.clone().requires_grad_(True)
    logits, _ = inspector.run(x)
    (g,) = torch.autograd.grad(logits[0, y], x)
    scores = (g * x).abs()
    scores[x == 0] = -1.0
    return scores.topk(k).indices.tolist()


def random_topk(x, k, seed):
    gen = torch.Generator().manual_seed(seed)
    hist = x.nonzero().flatten()
    return hist[torch.randperm(len(hist), generator=gen)[:k]].tolist()


def main(n_users=5, verbose=True):
    classes = jlens.load_notebook_classes()
    vae, mlp = _make_models(classes)
    users = _make_users(n_users)

    tables = {}
    for name, model, inspector_cls in [("VAE", vae, jlens.VAEInspector),
                                       ("MLP", mlp, jlens.MLPInspector)]:
        insp = inspector_cls(model)
        rows = []
        for u, x in enumerate(users):
            with torch.no_grad():
                logits, _ = insp.run(x)
            y = int(logits[0].argmax())
            explanations = {
                "grad_x_input": saliency_topk(insp, x, y, TOP_K),
                "random": random_topk(x, TOP_K, seed=100 + u),
            }
            for explainer, S in explanations.items():
                for r in jlens.mechanistic_eval(insp, x, S, y,
                                                n_random=3, seed=u):
                    rows.append({"user": u, "explainer": explainer, **r})
        tables[name] = pd.DataFrame(rows)

    if verbose:
        pd.set_option("display.width", 160)
        pd.set_option("display.max_columns", None)
        cols = ["delta_beh", "jcomp", "jrestore", "mr", "mr_full",
                "selectivity_del", "selectivity_patch", "jcomp_rand_mean"]
        tap_order = {"VAE": ["enc_h1", "z", "dec_h1"], "MLP": ["user_vec"]}
        for name, df in tables.items():
            print(f"\n=== {name} (mean over {n_users} users) ===")
            agg = df.groupby(["explainer", "tap"])[cols].mean()
            agg = agg.reindex(tap_order[name], level="tap")
            print(agg.round(4))
        print("\nReminder: dec_h1 (and MLP user_vec) MR == 1 by construction;"
              "\nthe informative taps are enc_h1 and z.")
    return tables


if __name__ == "__main__":
    main()
