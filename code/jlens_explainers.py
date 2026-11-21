"""Explainers producing attribution maps over a user's history.

All gradient-based explainers work on the pre-softmax logit of the target
item via a jlens Inspector, so they are consistent with the mechanistic
metrics. `spinrec_lite` is a light reimplementation of the SPINRec recipe:
integrated gradients from stochastically sampled baselines, selecting the
attribution map that maximizes a deletion fidelity proxy — exactly the
metric-adapted selection whose mechanistic faithfulness the benchmark
questions.
"""
import torch

import jlens


def topk_from_map(attr, x, k):
    """Top-k history items (x > 0) by attribution value, best first."""
    masked = attr.clone()
    masked[x == 0] = float("-inf")
    k = min(k, int((x > 0).sum()))
    return masked.topk(k).indices.tolist()


def saliency(inspector, x, y):
    """Gradient-times-input attribution map (signed)."""
    x = x.clone().requires_grad_(True)
    logits, _ = inspector.run_items(x, [y])
    (g,) = torch.autograd.grad(logits[0, 0], x)
    return (g * x).detach()


def integrated_gradients(inspector, x, y, baseline=None, steps=20):
    """Midpoint-rule IG on the target logit; zero baseline by default."""
    if baseline is None:
        baseline = torch.zeros_like(x)
    alphas = ((torch.arange(steps, device=x.device, dtype=x.dtype) + 0.5)
              / steps)
    points = (baseline.unsqueeze(0)
              + alphas.unsqueeze(1) * (x - baseline).unsqueeze(0))
    points = points.detach().requires_grad_(True)
    logits, _ = inspector.run_items(points, [y])
    (grads,) = torch.autograd.grad(logits[:, 0].sum(), points)
    return ((x - baseline) * grads.mean(0)).detach()


def spinrec_lite(inspector, x, y, k, n_baselines=10, steps=10, seed=0,
                 baseline_probs=None, return_info=False):
    """SPINRec-style stochastic path integration with metric-adapted selection.

    Candidates: IG from the zero baseline plus IG from `n_baselines` sampled
    binary baselines (item j drawn with probability `baseline_probs[j]`, or
    uniformly with the user's history density when not given). The returned
    map is the candidate whose top-k deletion maximally drops the target
    logit (the DEL proxy).
    """
    gen = torch.Generator().manual_seed(seed)
    density = float(x.sum() / x.numel())
    candidates = [integrated_gradients(inspector, x, y, None, steps)]
    for _ in range(n_baselines):
        r = torch.rand(x.shape, generator=gen)
        p = baseline_probs if baseline_probs is not None else density
        b = (r < p).float()
        candidates.append(integrated_gradients(inspector, x, y, b, steps))

    with torch.no_grad():
        f_x, _ = inspector.run_items(x, [y])
        fy = float(f_x[0, 0])
    proxies = []
    for m in candidates:
        S = topk_from_map(m, x, k)
        with torch.no_grad():
            f_d, _ = inspector.run_items(jlens.delete_items(x, S), [y])
        proxies.append(fy - float(f_d[0, 0]))
    chosen = max(range(len(proxies)), key=lambda i: proxies[i])
    if return_info:
        return candidates[chosen], {"proxies": proxies, "chosen": chosen}
    return candidates[chosen]


def _path_avg_grads(inspector, x, y, baseline, steps, prob):
    """Right-endpoint Riemann average of input gradients along the path
    baseline -> x, of the target probability (as in published SPINRec)."""
    alphas = torch.arange(1, steps + 1, dtype=torch.float32) / steps
    pts = (baseline.unsqueeze(0)
           + alphas.unsqueeze(1) * (x - baseline).unsqueeze(0))
    pts = pts.detach().requires_grad_(True)
    logits, _ = inspector.run_items(pts, [y])
    if prob == "softmax":
        # Softmax needs the full catalogue.  This branch is used by VAE;
        # recover the full logits rather than silently softmaxing one item.
        logits, _ = inspector.run(pts)
        score = torch.softmax(logits, dim=1)[:, y].sum()
    elif prob == "sigmoid":
        score = torch.sigmoid(logits[:, 0]).sum()
    else:
        score = logits[:, 0].sum()
    (g,) = torch.autograd.grad(score, pts)
    return g.mean(0)


def spinrec_real(inspector, x, y, k, train_X, n_baselines=10, steps=10,
                 seed=0, prob="auto", include_zero_baseline=False,
                 return_info=False):
    """The published SPINRec (find_ip_mask, method='sample_random_user'):
    baselines are sampled TRAIN USERS, the attribution map is
    x * avg_grad * (x - baseline) with gradients of the model's probability
    output, and the best map is selected by the deletion fidelity proxy.
    prob='auto' matches the published per-model behavior (softmax for VAE,
    sigmoid for MLP/NCF); the selection proxy uses the target logit for
    consistency with the rest of this framework.
    """
    if prob == "auto":
        prob = ("sigmoid" if isinstance(
            inspector, (jlens.MLPInspector, jlens.NCFInspector))
                else "softmax")
    gen = torch.Generator().manual_seed(seed)
    baselines, ids = [], []
    if include_zero_baseline:
        baselines.append(torch.zeros_like(x))
        ids.append(-1)
    for _ in range(n_baselines):
        b = int(torch.randint(len(train_X), (1,), generator=gen))
        baselines.append(train_X[b])
        ids.append(b)

    candidates = [(x * _path_avg_grads(inspector, x, y, b, steps, prob)
                   * (x - b)).detach() for b in baselines]

    with torch.no_grad():
        f_x, _ = inspector.run_items(x, [y])
        fy = float(f_x[0, 0])
    proxies = []
    for m in candidates:
        S = topk_from_map(m, x, k)
        with torch.no_grad():
            f_d, _ = inspector.run_items(jlens.delete_items(x, S), [y])
        proxies.append(fy - float(f_d[0, 0]))
    chosen = max(range(len(proxies)), key=lambda i: proxies[i])
    if return_info:
        return candidates[chosen], {"proxies": proxies, "chosen": chosen,
                                    "baseline_ids": ids}
    return candidates[chosen]


def random_topk(x, k, seed):
    gen = torch.Generator().manual_seed(seed)
    hist = x.nonzero().flatten()
    perm = torch.randperm(len(hist), generator=gen)
    return hist[perm[:k]].tolist()


def pop_topk(x, pop, k):
    hist = x.nonzero().flatten().tolist()
    return sorted(hist, key=lambda i: -float(pop[i]))[:k]
