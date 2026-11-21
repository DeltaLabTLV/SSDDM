"""White-box search for deletion sets that switch clean-state mechanisms.

greedy_attack(...) searches for a K-subset S of the user's history maximizing

    delta_beh(S) - lam * JComp(S, tap)

i.e. deletions that change the target while remaining poorly accessible from
the clean target-aligned subspace at `tap`. Endpoint/path audits determine
whether the resulting effect activates a dormant direction. lam=0 reduces to
a pure deletion (DEL) attack.

Myopic greedy alone cannot discover saturated pathways (early deletions have
exactly zero effect), so the search is seeded with candidate sets from
attribution maps that integrate over the input path (IG from zero and from
sampled baselines), plus a myopic greedy set and random sets, and the best
candidate is refined by single-swap local search under the full objective.
"""
import torch

import jlens
import jlens_explainers as jx


class _Objective:
    """Batched evaluator of (delta_beh, jcomp) for candidate deletion sets."""

    def __init__(self, inspector, x, y, tap, competitors):
        self.inspector, self.x, self.y, self.tap = inspector, x, y, tap
        with torch.no_grad():
            logits_clean, acts_clean = inspector.run_items(x, [y])
        C = [y] + [c for c in competitors if c != y]
        V = jlens.item_directions(inspector, x, tap, C)
        self.Q = jlens.orthonormal_basis(V)
        self.h_clean = acts_clean[tap]
        self.fy = float(logits_clean[0, 0])

    def __call__(self, sets):
        with torch.no_grad():
            if (self.tap == "user_state"
                    and hasattr(self.inspector, "deletion_states")):
                h_corr = self.inspector.deletion_states(self.x, sets)
                logits_corr = self.inspector.run_from_items(
                    self.tap, h_corr, [self.y])
            else:
                X = self.x.unsqueeze(0).repeat(len(sets), 1)
                for r, S in enumerate(sets):
                    X[r, list(S)] = 0.0
                logits_corr, acts_corr = self.inspector.run_items(
                    X, [self.y])
                h_corr = acts_corr[self.tap]
            h_patch = self.h_clean + jlens.project(self.Q,
                                                   h_corr - self.h_clean)
            logits_patch = self.inspector.run_from_items(
                self.tap, h_patch, [self.y])
        delta = self.fy - logits_corr[:, 0]
        jcomp = self.fy - logits_patch[:, 0]
        return delta, jcomp


def _myopic_greedy(obj, hist, k, lam):
    S = []
    for _ in range(k):
        cands = [i for i in hist if i not in S]
        delta, jcomp = obj([S + [i] for i in cands])
        scores = delta - lam * jcomp
        S.append(cands[int(scores.argmax())])
    return S


def _local_search(obj, hist, S, lam, max_iters):
    S = list(S)
    delta, jcomp = obj([S])
    best = float(delta[0] - lam * jcomp[0])
    for _ in range(max_iters):
        swaps, sets = [], []
        outside = [i for i in hist if i not in S]
        for pos in range(len(S)):
            for i in outside:
                cand = S.copy()
                cand[pos] = i
                swaps.append((pos, i))
                sets.append(cand)
        delta, jcomp = obj(sets)
        scores = delta - lam * jcomp
        top = int(scores.argmax())
        if float(scores[top]) <= best + 1e-6:
            break
        best = float(scores[top])
        pos, i = swaps[top]
        S[pos] = i
    return S


def _candidate_sets(inspector, obj, x, y, k, hist, lam, n_seeds, gen):
    density = float(x.sum() / x.numel())
    candidates = [
        jx.topk_from_map(jx.saliency(inspector, x, y), x, k),
        jx.topk_from_map(
            jx.integrated_gradients(inspector, x, y, steps=20), x, k),
        _myopic_greedy(obj, hist, k, lam),
    ]
    for _ in range(n_seeds):
        b = (torch.rand(x.shape, generator=gen) < density).float()
        m = jx.integrated_gradients(inspector, x, y, baseline=b, steps=10)
        candidates.append(jx.topk_from_map(m, x, k))
    for _ in range(n_seeds):
        perm = torch.randperm(len(hist), generator=gen)[:k]
        candidates.append([hist[j] for j in perm.tolist()])
    return candidates


def constraint_attack(inspector, x, y, k, tap, competitors, mr_max=0.5,
                      n_seeds=8, seed=0, refine_iters=25, eps=1e-6,
                      return_info=False):
    """max DEL subject to MR <= mr_max (constraint form of the attack).

    Phase 1 seeds with candidate sets scored by the penalized objective and
    swap-searches toward feasibility (minimizing MR); phase 2 maximizes DEL
    with swaps that keep the constraint satisfied.
    """
    obj = _Objective(inspector, x, y, tap, competitors)
    hist = x.nonzero().flatten().tolist()
    k = min(k, len(hist))
    gen = torch.Generator().manual_seed(seed)

    def mr_of(delta, jcomp):
        guard = torch.where(delta >= 0, torch.full_like(delta, eps),
                            torch.full_like(delta, -eps))
        return jcomp / (delta + guard)

    candidates = _candidate_sets(inspector, obj, x, y, k, hist, 1.0,
                                 n_seeds, gen)
    delta, jcomp = obj(candidates)
    mr = mr_of(delta, jcomp)
    feas = (mr <= mr_max) & (delta > 0)
    if feas.any():
        masked = torch.where(feas, delta, torch.tensor(float("-inf")))
        S = list(candidates[int(masked.argmax())])
    else:  # start from the lowest-forward-mediation positive-effect candidate
        masked = torch.where(delta > 0, delta - jcomp,
                             torch.tensor(float("-inf")))
        best = int(masked.argmax()) if torch.isfinite(masked).any() \
            else int((delta - jcomp).argmax())
        S = list(candidates[best])

    for _ in range(refine_iters):
        outside = [i for i in hist if i not in S]
        swaps, sets = [], []
        for pos in range(len(S)):
            for i in outside:
                cand = S.copy()
                cand[pos] = i
                swaps.append((pos, i))
                sets.append(cand)
        sets.append(S)  # current, as the comparison point
        delta, jcomp = obj(sets)
        mr = mr_of(delta, jcomp)
        feas = (mr <= mr_max) & (delta > 0)
        cur_delta, cur_mr = delta[-1], mr[-1]
        cur_feasible = bool((cur_mr <= mr_max) & (cur_delta > 0))
        if cur_feasible:
            # improve DEL among feasible swaps only
            masked = torch.where(feas, delta, torch.tensor(float("-inf")))
            top = int(masked.argmax())
            if top == len(sets) - 1 or float(delta[top]) <= float(cur_delta) + 1e-6:
                break
        else:
            # move toward feasibility: minimize MR among positive-DEL swaps
            masked = torch.where(delta > 0, mr, torch.tensor(float("inf")))
            top = int(masked.argmin())
            if top == len(sets) - 1 or float(mr[top]) >= float(cur_mr) - 1e-6:
                break
        S[swaps[top][0]] = swaps[top][1]

    delta, jcomp = obj([S])
    final_mr = float(mr_of(delta, jcomp)[0])
    info = {"feasible": final_mr <= mr_max and float(delta[0]) > 0,
            "delta": float(delta[0]), "mr": final_mr}
    return (S, info) if return_info else S


def greedy_attack(inspector, x, y, k, tap, competitors, lam=1.0,
                  n_seeds=8, seed=0, refine_iters=20):
    """Attribution-seeded search + myopic greedy + swap refinement."""
    obj = _Objective(inspector, x, y, tap, competitors)
    hist = x.nonzero().flatten().tolist()
    k = min(k, len(hist))
    gen = torch.Generator().manual_seed(seed)

    candidates = _candidate_sets(inspector, obj, x, y, k, hist, lam,
                                 n_seeds, gen)
    delta, jcomp = obj(candidates)
    scores = delta - lam * jcomp
    S = candidates[int(scores.argmax())]
    return _local_search(obj, hist, S, lam, refine_iters)
