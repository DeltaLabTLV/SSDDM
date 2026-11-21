"""Item-Jacobian lens: mechanistic evaluation of recommender explanations.

Given a trained PI4Rec recommender, a user history x, a target item y and an
explanation set S (interactions claimed to drive the recommendation of y),
this module audits how much of the behavioral deletion effect on the
pre-softmax logit of y is accessible through a target-item-aligned subspace at
a stated reference activation. Forward and reverse activation patches expose
order dependence; their two-player Shapley average is exactly additive.

Design notes:
- All metrics operate on pre-softmax logits: softmax mechanically
  anticorrelates target and competitor deltas and would contaminate
  selectivity.
- Item directions are local exact gradients v_c = d logit_c / d h_tap.
  Averaged-Jacobian ("global") directions are the batch mean of local ones,
  which is the same thing by linearity of the Jacobian-vector product.
- Lens directions are computed on the clean run.

Spec: docs/superpowers/specs/2026-07-10-jlens-mechanistic-eval-design.md
"""
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Model classes without the `ipynb` package
# --------------------------------------------------------------------------

def load_notebook_classes(nb_path=None):
    """Exec the class-defining cells of recommenders_architecture.ipynb.

    Returns the resulting namespace dict (contains MLP, VAE, GMF_model,
    MLP_model, NCF). Single source of truth: the notebook itself.
    """
    if nb_path is None:
        nb_path = Path(__file__).resolve().parent / "recommenders_architecture.ipynb"
    nb = json.loads(Path(nb_path).read_text())
    ns = {"torch": torch, "nn": torch.nn, "F": F, "np": np}
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if any(line.startswith("class ") for line in src.splitlines()):
            exec(compile(src, str(nb_path), "exec"), ns)
    return ns


# --------------------------------------------------------------------------
# Inspectors: tapped forward passes on pre-softmax logits
# --------------------------------------------------------------------------

class Inspector:
    """A model forward pass unrolled into stages with named tap points.

    run(x)            -> (logits, {tap: activation})
    run_from(tap, h)  -> logits, resuming the forward pass after `tap`
                         with the supplied activation.
    """

    _stages = ()  # list of (tap_name_or_None, fn)

    @property
    def taps(self):
        return [name for name, _ in self._stages if name is not None]

    def _prep(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return x

    def run(self, x):
        h = self._prep(x)
        acts = {}
        for name, fn in self._stages:
            h = fn(h)
            if name is not None:
                acts[name] = h
        return h, acts

    def run_from(self, tap, h):
        names = [name for name, _ in self._stages]
        idx = names.index(tap)
        if h.dim() == 1:
            h = h.unsqueeze(0)
        for _, fn in self._stages[idx + 1:]:
            h = fn(h)
        return h

    def run_items(self, x, items):
        """Return only selected item logits while retaining all tap values.

        The default implementation slices a full forward pass.  Inspectors
        with item-conditional decoders (notably NeuMF) override this method so
        attribution and red-team search do not materialize every item score.
        """
        logits, acts = self.run(x)
        return logits[:, list(items)], acts

    def run_from_items(self, tap, h, items):
        """Resume from ``tap`` and return only selected item logits."""
        return self.run_from(tap, h)[:, list(items)]


class VAEInspector(Inspector):
    """Taps: every encoder ReLU output (enc_h1, ...), the latent mean (z),
    every decoder ReLU output (dec_h1, ...). Logits are pre-softmax, with
    the demographic slice applied when the model uses it."""

    def __init__(self, model):
        if model.training:
            raise ValueError("model must be in eval mode")
        self.model = model
        latent = model.enc_dims[-1]
        stages = [(None, lambda h: F.normalize(h, dim=-1))]
        k = 0
        for m in model.encoder:
            if isinstance(m, torch.nn.ReLU):
                k += 1
                stages.append((f"enc_h{k}", m))
            else:
                stages.append((None, m))
        stages.append(("z", lambda h: h[:, :latent]))
        k = 0
        for m in model.decoder:
            if isinstance(m, torch.nn.ReLU):
                k += 1
                stages.append((f"dec_h{k}", m))
            else:
                stages.append((None, m))
        if model.demographic:
            stages.append((None, lambda h: h[:, :model.items_only]))
        self._stages = stages


class MLPInspector(Inspector):
    """Single tap `user_vec`; logits are user_vec @ items_fc(I).T
    (pre-sigmoid). The positive control: the mechanism gap is identically 0."""

    def __init__(self, model, all_items_tensor=None):
        if model.training:
            raise ValueError("model must be in eval mode")
        self.model = model
        device = model.users_fc.weight.device
        if all_items_tensor is None:
            all_items_tensor = torch.eye(model.items_fc.in_features,
                                         device=device)
        with torch.no_grad():
            item_vecs = model.items_fc(all_items_tensor)
        self._stages = [("user_vec", model.users_fc),
                        (None, lambda h: h @ item_vecs.T)]


class NCFInspector(Inspector):
    """Inspector for NeuMF's GMF + nonlinear item-conditional MLP decoder.

    The tapped ``user_state`` concatenates the two history encodings.  The
    downstream function combines it with each item's learned embeddings,
    applies the released NeuMF MLP, and returns *pre-sigmoid* logits.  This
    vectorized implementation is algebraically identical to the notebook
    model but avoids constructing dense one-hot item matrices.
    """

    def __init__(self, model):
        if model.training:
            raise ValueError("model must be in eval mode")
        if model.model in {"MLP", "GMF"}:
            raise ValueError("NCFInspector requires a two-branch NeuMF model")
        self.model = model
        self.gmf_dim = model.embed_user_GMF.out_features
        self.registered_item_ids = torch.arange(
            model.embed_item_GMF.in_features,
            device=model.embed_item_GMF.weight.device)
        self._stages = [
            ("user_state", self._encode_user),
            (None, lambda h: self._score(h, self.registered_item_ids)),
        ]

    def _encode_user(self, x):
        return torch.cat([
            self.model.embed_user_GMF(x),
            self.model.embed_user_MLP(x),
        ], dim=-1)

    def _score(self, state, items):
        items = torch.as_tensor(items, dtype=torch.long, device=state.device)
        # nn.Linear(one_hot(i)) is exactly column i of its weight matrix.
        item_gmf = self.model.embed_item_GMF.weight[:, items].T
        item_mlp = self.model.embed_item_MLP.weight[:, items].T
        user_gmf = state[:, :self.gmf_dim]
        user_mlp = state[:, self.gmf_dim:]
        b, n = state.shape[0], len(items)
        out_gmf = user_gmf[:, None, :] * item_gmf[None, :, :]
        interaction = torch.cat([
            user_mlp[:, None, :].expand(b, n, -1),
            item_mlp[None, :, :].expand(b, n, -1),
        ], dim=-1)
        out_mlp = self.model.MLP_layers(interaction)
        joint = torch.cat([out_gmf, out_mlp], dim=-1)
        return self.model.predict_layer(joint).squeeze(-1)

    def run_items(self, x, items):
        x = self._prep(x)
        state = self._encode_user(x)
        return self._score(state, items), {"user_state": state}

    def run_from_items(self, tap, h, items):
        if tap != "user_state":
            raise ValueError(f"unknown tap: {tap}")
        if h.dim() == 1:
            h = h.unsqueeze(0)
        return self._score(h, items)

    def deletion_states(self, x, sets):
        """Exact ``user_state`` values for many deletion sets.

        NeuMF's history encoders are bias-free linear maps.  Subtracting the
        corresponding weight columns is therefore identical to rebuilding a
        large dense corrupted-history batch, but is substantially faster.
        """
        x = self._prep(x)
        if x.shape[0] != 1:
            raise ValueError("deletion_states expects one clean user")
        clean = self._encode_user(x)
        weights = torch.cat([
            self.model.embed_user_GMF.weight,
            self.model.embed_user_MLP.weight,
        ], dim=0)
        states = clean.repeat(len(sets), 1)
        for row, S in enumerate(sets):
            if S:
                idx = torch.as_tensor(S, dtype=torch.long, device=x.device)
                states[row] -= (weights[:, idx] * x[0, idx]).sum(dim=1)
        return states


# --------------------------------------------------------------------------
# Item directions
# --------------------------------------------------------------------------

def directions_at_activation(inspector, tap, h, items):
    """Exact downstream item gradients at an explicitly supplied activation.

    Separating this primitive from :func:`item_directions` makes the reference
    state auditable: callers can construct clean-, corrupted-, endpoint-union,
    or path-averaged Jacobian bases without changing the patching machinery.
    """
    h = h.detach().requires_grad_(True)
    logits = inspector.run_from_items(tap, h, items)
    cols = []
    for j, _c in enumerate(items):
        (g,) = torch.autograd.grad(logits[0, j], h, retain_graph=True)
        cols.append(g[0])
    return torch.stack(cols, dim=1)


def item_directions(inspector, x, tap, items):
    """Local exact lens directions v_c = d logit_c / d h_tap at input x.

    Returns a (d, len(items)) matrix, one column per item.
    """
    with torch.no_grad():
        _, acts = inspector.run_items(x, items)
    return directions_at_activation(inspector, tap, acts[tap], items)


def path_item_directions(inspector, tap, h_clean, h_corr, items, steps=8):
    """Average downstream Jacobian along the clean--corrupted activation path.

    This is a robustness provider rather than the default lens.  It exposes
    directions that are dormant at the clean endpoint but become locally
    visible after the deletion crosses a downstream non-linearity.
    """
    if steps < 1:
        raise ValueError("steps must be positive")
    mats = []
    for j in range(steps):
        alpha = (j + 0.5) / steps
        h = h_clean + alpha * (h_corr - h_clean)
        mats.append(directions_at_activation(inspector, tap, h, items))
    return torch.stack(mats).mean(0)


def global_item_directions(inspector, X, tap, items):
    """Averaged-Jacobian directions: batch mean of local directions over the
    rows of X (equal to E[J]^T w by linearity)."""
    vs = [item_directions(inspector, X[b], tap, items)
          for b in range(X.shape[0])]
    return torch.stack(vs).mean(0)


# --------------------------------------------------------------------------
# Subspace patching
# --------------------------------------------------------------------------

def orthonormal_basis(V, tol=1e-6):
    """QR-orthonormalize the columns of V, dropping (near-)dependent ones."""
    if V.numel() == 0 or V.shape[1] == 0:
        return V.new_zeros((V.shape[0], 0))
    Q, R = torch.linalg.qr(V, mode="reduced")
    diag = R.diagonal().abs()
    if float(diag.max()) == 0.0:
        return V.new_zeros((V.shape[0], 0))
    keep = diag > tol * diag.max()
    return Q[:, keep]

def project(Q, delta):
    """Orthogonal projection of row vector(s) delta onto span(Q)."""
    return (delta @ Q) @ Q.T


def patch_logits(inspector, tap, h_clean, h_corr, Q, mode):
    """Resume the forward pass from `tap` with a subspace-patched activation.

    comp:    clean activation, Item-J coordinates taken from the corrupt run
    restore: corrupt activation, Item-J coordinates taken from the clean run
    full:    the whole corrupt activation (subspace-free upper bound)
    """
    if mode == "comp":
        h = h_clean + project(Q, h_corr - h_clean)
    elif mode == "restore":
        h = h_corr + project(Q, h_clean - h_corr)
    elif mode == "full":
        h = h_corr
    else:
        raise ValueError(f"unknown mode: {mode}")
    return inspector.run_from(tap, h)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def delete_items(x, S):
    x_del = x.clone()
    x_del[..., list(S)] = 0.0
    return x_del


def effective_rank(V):
    """Participation ratio of the singular values of V."""
    s = torch.linalg.svdvals(V)
    return float(s.sum() ** 2 / (s ** 2).sum())


def _safe_div(a, b, eps):
    return a / (b + (eps if b >= 0 else -eps))


def bounded_mechanism_gap(delta, recovered, eps=1e-6):
    """Return the one-sided unrecovered share of a positive score drop.

    ``delta - recovered`` is the raw mechanism gap in logit units.  This
    reporting transform maps full recovery (or overshoot) to 0, no recovery
    or wrong-direction movement to 1, and partial recovery to the fraction of
    the natural deletion effect left unexplained.  Non-positive or tiny
    effects return NaN because the paper filters them before normalization.
    """
    if delta <= eps:
        return float("nan")
    return float(np.clip((delta - recovered) / delta, 0.0, 1.0))


def bounded_target_share(target_effect, mean_collateral, eps=1e-6):
    """Return target movement as a bounded share of target plus collateral.

    A value of 0.5 means target movement equals mean competitor movement; 1
    means only the target moves.  The legacy selectivity ratio remains in the
    artifacts for backward compatibility.
    """
    target = abs(float(target_effect))
    collateral = abs(float(mean_collateral))
    denom = target + collateral
    if denom <= eps:
        return float("nan")
    return target / denom


def rank_discounted_logit_mediation(inspector, x, S, items, tap,
                                    competitors=None, eps=1e-6):
    """Mediation of a rank-discounted *logit* functional.

    The weights use the familiar logarithmic rank discount, but this is not
    DCG: logits are not relevance labels and the clean ordering is held fixed.
    The explicit name prevents the statistic from being mistaken for a
    list-ranking metric.
    """
    C = list(items) + [c for c in (competitors or []) if c not in items]
    with torch.no_grad():
        logits_clean, acts_clean = inspector.run(x)
        logits_corr, acts_corr = inspector.run(delete_items(x, S))
    V = item_directions(inspector, x, tap, C)
    Q = orthonormal_basis(V)
    with torch.no_grad():
        l_patch = patch_logits(inspector, tap, acts_clean[tap],
                               acts_corr[tap], Q, "comp")
    w = torch.tensor([1.0 / math.log2(j + 2) for j in range(len(items))])
    idx = torch.tensor(items)
    beh = float((w * (logits_clean - logits_corr)[0, idx]).sum())
    med = float((w * (logits_clean - l_patch)[0, idx]).sum())
    return med / (beh + (eps if beh >= 0 else -eps))


# Backward-compatible API for existing artifacts.  New code and the paper use
# the precise name above.
listwise_mediation = rank_discounted_logit_mediation


def mechanistic_eval(inspector, x, S, y, competitors=None, taps=None,
                     n_competitors=5, eps=1e-6, n_random=0, seed=0,
                     directions=None):
    """Per-tap mechanistic metrics for one user / one explanation set.

    x            user history (multi-hot 1D tensor)
    S            explanation set: indices of interactions to delete
    y            target item
    competitors  competitor item ids for selectivity and the lens subspace;
                 default: top-`n_competitors` clean-run items excluding y
    n_random     draws of a same-dimension random-subspace control
    directions   optional callable (tap, C) -> direction matrix (d, |C|);
                 default: local exact gradients at x (item_directions)
    Returns a list of dicts, one per tap (DataFrame-friendly). Callers should
    filter users with |delta_beh| below a threshold before aggregating the
    ratio metrics (eps only guards the division).
    """
    with torch.no_grad():
        logits_clean, acts_clean = inspector.run(x)
        x_del = delete_items(x, S)
        logits_corr, acts_corr = inspector.run(x_del)

    if competitors is None:
        ranked = logits_clean[0].argsort(descending=True).tolist()
        competitors = [c for c in ranked if c != y][:n_competitors]
    C = [y] + [c for c in competitors if c != y]

    delta_beh_all = (logits_clean - logits_corr)[0]
    delta_beh = float(delta_beh_all[y])
    collateral_all = float(delta_beh_all.abs().mean())
    mean_competitor_effect = (
        float(delta_beh_all[C[1:]].abs().mean()) if len(C) > 1 else 0.0
    )
    sel_del = _safe_div(abs(delta_beh), mean_competitor_effect, eps)
    target_share_del = bounded_target_share(
        delta_beh, mean_competitor_effect, eps=eps)

    gen = torch.Generator().manual_seed(seed)
    rows = []
    for tap in taps or inspector.taps:
        h_clean, h_corr = acts_clean[tap], acts_corr[tap]
        if directions is None:
            V = item_directions(inspector, x, tap, C)
        else:
            V = directions(tap, C)
        Q = orthonormal_basis(V)

        with torch.no_grad():
            l_comp = patch_logits(inspector, tap, h_clean, h_corr, Q, "comp")
            l_rest = patch_logits(inspector, tap, h_clean, h_corr, Q, "restore")
            l_full = patch_logits(inspector, tap, h_clean, h_corr, Q, "full")

        jcomp = float(logits_clean[0, y] - l_comp[0, y])
        jrestore_raw = float(l_rest[0, y] - logits_corr[0, y])
        jrestore = _safe_div(jrestore_raw, delta_beh, eps)
        mr = _safe_div(jcomp, delta_beh, eps)
        # Exact two-player Shapley decomposition between the target-aligned
        # subspace P and its orthogonal complement.  JComp is P's marginal
        # contribution when the complement is clean; JRestoreRaw is P's
        # marginal contribution when the complement is corrupted.  Averaging
        # the two orders yields an additive allocation, unlike the one-sided
        # ratio `mr` in a nonlinear downstream network.
        jshap = 0.5 * (jcomp + jrestore_raw)
        smr = _safe_div(jshap, delta_beh, eps)
        jshap_complement = delta_beh - jshap
        clean_unrecovered = delta_beh - jcomp
        clean_gap = bounded_mechanism_gap(delta_beh, jcomp, eps=eps)
        symmetric_gap = bounded_mechanism_gap(delta_beh, jshap, eps=eps)
        interaction_gap = jcomp - jrestore_raw
        mr_full = _safe_div(float(logits_clean[0, y] - l_full[0, y]),
                            delta_beh, eps)
        patch_delta = (logits_clean - l_comp)[0]
        sel_patch = _safe_div(
            abs(jcomp),
            float(patch_delta[C[1:]].abs().mean()) if len(C) > 1 else 0.0,
            eps)

        rand_jcomps = []
        for _ in range(n_random):
            # dimension matched to the KEPT basis, not to len(C): near-parallel
            # item directions collapse under QR and the control must match that
            G = torch.randn(V.shape[0], Q.shape[1], generator=gen)
            Qr = orthonormal_basis(G)
            with torch.no_grad():
                l_r = patch_logits(inspector, tap, h_clean, h_corr, Qr, "comp")
            rand_jcomps.append(float(logits_clean[0, y] - l_r[0, y]))
        rand_jcomps = np.array(rand_jcomps)

        rows.append({
            "tap": tap,
            "delta_beh": delta_beh,
            "jcomp": jcomp,
            "jrestore_raw": jrestore_raw,
            "jrestore": jrestore,
            # mr is the one-sided clean-reference ratio kept for artifact
            # compatibility; smr is the additive order-average.
            "mr": mr,
            "jshap": jshap,
            "smr": smr,
            "jshap_complement": jshap_complement,
            "clean_unrecovered": clean_unrecovered,
            "clean_gap": clean_gap,
            "symmetric_gap": symmetric_gap,
            "interaction_gap": interaction_gap,
            "mr_full": mr_full,
            "selectivity_del": sel_del,
            "target_share_del": target_share_del,
            "selectivity_patch": sel_patch,
            "collateral_all": collateral_all,
            "erank": effective_rank(V),
            "n_directions": len(C),
            "jcomp_rand_mean": float(rand_jcomps.mean()) if n_random else float("nan"),
            "jcomp_rand_std": float(rand_jcomps.std()) if n_random else float("nan"),
        })
    return rows
