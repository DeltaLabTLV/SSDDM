"""Item-Jacobian activation-patching audit for the LXSR SasRec recommender.

Sequential generalization of the multi-hot audit (jlens.py).  The state at a
tap is the FULL right-aligned activation matrix H [seq_len, d]; the
item-readout directions are the gradients of the target/competitor logits
wrt the ENTIRE tap state (flattened to R^{T*d}); the multi-hot audit is the
special case T=1.  The two Shapley players are:

  A: P @ delta               (item-readout component of the full state
                              displacement, directions at the clean state)
  B: everything else         ((I-P) @ delta plus the attention-mask change
                              from delete-and-shift)

The four corners are exact by construction: F00 = clean score and
F11 = corrupted score (both verified at runtime).

Deletion follows the LXSR fidelity protocol exactly (pos_perturbation):
remove the chosen positions, shift right, left-pad; positional indices are
reverse (distance from the end), so right-aligned tokens keep their
positional embeddings.

Taps: 'emb' (post input-LayerNorm), 'block1', ..., 'blockL'. The last block
tap is output-adjacent: logits are affine in h_T there, so the bounded
mechanism gap is identically zero (runtime sanity check), mirroring the VAE
dec_h1 tap.
"""
from pathlib import Path

import numpy as np
import torch

EPS = 1e-6


def bounded_mechanism_gap(delta, recovered, eps=EPS):
    """One-sided unrecovered share for a positive deletion effect."""
    if delta <= eps:
        return float("nan")
    return float(np.clip((delta - recovered) / delta, 0.0, 1.0))


def bounded_target_share(target_effect, mean_collateral, eps=EPS):
    """Bounded target share; 0.5 means target equals mean collateral."""
    target = abs(float(target_effect))
    collateral = abs(float(mean_collateral))
    denom = target + collateral
    if denom <= eps:
        return float("nan")
    return target / denom


# --------------------------------------------------------------------------
# model / data loading
# --------------------------------------------------------------------------

def load_sasrec(lxsr_root, ckpt_path, train_csv, model_cfg=None, device="cpu"):
    """Rebuild the tokenizer exactly as Trainer does in --phase final
    (vocab from train+validation merged) and load the SasRec checkpoint."""
    import sys
    sys.path.insert(0, str(lxsr_root))
    import pandas as pd
    from src.recommenders.tokenizer import SequenceTokenizer
    from src.recommenders.models import ModelFactory

    cfg = {
        "model_type": "sasrec", "d_model": 64, "n_head": 2, "num_layers": 2,
        "dropout": 0.2, "attention_dropout": 0.2, "layer_norm_eps": 1e-6,
        "max_seq_length": 200, "embedding_type": "tied",
        "positional_direction": "reverse", "depth_agg_mode": "baseline",
    }
    if model_cfg:
        cfg.update(model_cfg)

    train_csv = Path(train_csv)
    train_df = pd.concat([pd.read_csv(train_csv),
                          pd.read_csv(train_csv.parent / "validation.csv")]) \
        .drop_duplicates()
    item_vocab = sorted(train_df["item_id"].unique().tolist())
    tokenizer = SequenceTokenizer(item_vocab=item_vocab,
                                  max_seq_length=cfg["max_seq_length"],
                                  model_type="sasrec")
    model = ModelFactory.create(model_type="sasrec",
                                vocab_size=tokenizer.get_vocab_size(),
                                config=cfg)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not missing, f"missing keys: {missing[:5]}"
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device), tokenizer, train_df


def delete_and_shift(tokens, positions):
    """LXSR pos_perturbation for one sequence: drop `positions`, keep order,
    left-pad with 0 to the original length."""
    n = len(tokens)
    keep = [t for i, t in enumerate(tokens) if i not in set(positions)]
    out = torch.zeros(n, dtype=torch.long)
    if keep:
        out[n - len(keep):] = torch.tensor(keep, dtype=torch.long)
    return out


# --------------------------------------------------------------------------
# staged forward
# --------------------------------------------------------------------------

class SasRecAuditor:
    """Tapped forward passes on a single sequence (batch of 1)."""

    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = device
        self.n_layers = len(model.transformer_layers)
        self.taps = ["emb"] + [f"block{i+1}" for i in range(self.n_layers)]

    def _prep(self, input_ids, attn_mask):
        m = self.model
        x = m.embedding(input_ids) * np.sqrt(m.d_model)
        seq_len = input_ids.size(1)
        if m.positional_direction == "reverse":
            pos = torch.arange(1, seq_len + 1, device=x.device).flip(0) \
                .unsqueeze(0).expand(input_ids.size(0), -1)
        else:
            pos = torch.cumsum(attn_mask, dim=1)
        x = x + m.pos_embedding(pos)
        x = m.layernorm(x)
        return x  # [B, T, D]; dropout is identity in eval

    def _masks(self, attn_mask):
        seq_len = attn_mask.size(1)
        causal = torch.triu(torch.ones(seq_len, seq_len, device=self.device),
                            diagonal=1).bool()
        key_pad = attn_mask == 0
        return causal, key_pad

    def stages(self, input_ids, attn_mask):
        """Return {tap: H [T, D]} for a batch of one."""
        x = self._prep(input_ids, attn_mask)
        causal, key_pad = self._masks(attn_mask)
        acts = {"emb": x[0]}
        hidden = x.transpose(0, 1)
        for i, layer in enumerate(self.model.transformer_layers):
            hidden = layer(hidden, src_mask=causal,
                           src_key_padding_mask=key_pad)
            acts[f"block{i+1}"] = hidden.transpose(0, 1)[0]
        return acts

    def advance_from_emb(self, tap, H, attn_mask):
        """Run the prefix ending at ``tap`` from an embedding-tap state.

        This is used only for path controls: interpolating at the embedding
        tap and passing the midpoint through the lower blocks gives a
        model-induced, generally curved anchor at a deeper audited tap.
        """
        if tap == "emb":
            return H
        end = int(tap.replace("block", ""))
        if end < 1 or end > self.n_layers:
            raise ValueError(f"unknown tap {tap!r}")
        causal, key_pad = self._masks(attn_mask)
        hidden = H.unsqueeze(0).transpose(0, 1)
        for layer in self.model.transformer_layers[:end]:
            hidden = layer(hidden, src_mask=causal,
                           src_key_padding_mask=key_pad)
        return hidden.transpose(0, 1)[0]

    def resume(self, tap, H, attn_mask):
        """Run the model downstream of `tap` from state H [T, D]; return
        last-position logits [V]."""
        start = 0 if tap == "emb" else int(tap.replace("block", ""))
        causal, key_pad = self._masks(attn_mask)
        h = H.unsqueeze(0).transpose(0, 1)
        for layer in self.model.transformer_layers[start:]:
            h = layer(h, src_mask=causal, src_key_padding_mask=key_pad)
        out = h.transpose(0, 1)
        if self.model.embedding_type == "adapter":
            out = self.model.adapter_W(out)
        return self.model.fc_out(out)[0, -1]

    def logits(self, input_ids, attn_mask):
        with torch.no_grad():
            out = self.model(input_ids, attn_mask)
        return out[0, -1]

    def readout_directions(self, tap, H, attn_mask, items):
        """v_c = grad of s_c wrt the FULL tap state H [T, d] (the item-Jacobian
        direction in the sequence state space; the multi-hot audit is the
        special case T=1).  Returns [T*d, |items|]."""
        Hv = H.detach().clone().requires_grad_(True)
        logit = self.resume(tap, Hv, attn_mask)
        cols = []
        for c in items:
            g = torch.autograd.grad(logit[c], Hv, retain_graph=True)[0]
            cols.append(g.detach().flatten().clone())
        return torch.stack(cols, dim=1)


def orthonormal_basis(V):
    Q, R = torch.linalg.qr(V)
    keep = R.diagonal().abs() > 1e-8
    return Q[:, keep]


# --------------------------------------------------------------------------
# the audit
# --------------------------------------------------------------------------

def path_readout_directions(aud, tap, H0, H1, attn_mask, items, steps=8):
    """Average item-score Jacobians along the activation path at fixed context.

    The attention mask is discrete, so a delete-and-shift intervention has no
    canonical continuous context path.  Callers therefore evaluate the same
    activation path once with the clean mask and once with the deleted mask.
    """
    if steps < 1:
        raise ValueError("steps must be positive")
    mats = []
    for j in range(steps):
        alpha = (j + 0.5) / steps
        H = H0 + alpha * (H1 - H0)
        mats.append(aud.readout_directions(tap, H, attn_mask, items))
    return torch.stack(mats).mean(0)


def allocation_from_directions(aud, tap, H0, H1, m0, m1, V, y,
                               delta_beh, F00=None, F11=None):
    """Evaluate both patch orders for one supplied direction matrix.

    Player A is the projection of H1-H0 onto span(V).  Player B is the
    complementary activation displacement together with m0 -> m1.
    """
    dlt = (H1 - H0).flatten()
    Q = orthonormal_basis(V)
    pdlt = (Q @ (Q.T @ dlt)).view_as(H0)
    with torch.no_grad():
        if F00 is None:
            F00 = float(aud.resume(tap, H0, m0)[y])
        if F11 is None:
            F11 = float(aud.resume(tap, H1, m1)[y])
        F10 = float(aud.resume(tap, H0 + pdlt, m0)[y])
        F01 = float(aud.resume(tap, H1 - pdlt, m1)[y])
    jfwd, jrev = F00 - F10, F01 - F11
    jshap = 0.5 * (jfwd + jrev)
    denom = delta_beh if abs(delta_beh) > EPS else EPS
    return {
        "jfwd": jfwd,
        "jrev": jrev,
        "jshap": jshap,
        "mr": jfwd / denom,
        "smr": jshap / denom,
        "clean_unrecovered": delta_beh - jfwd,
        "clean_gap": bounded_mechanism_gap(delta_beh, jfwd),
        "symmetric_unrecovered": delta_beh - jshap,
        "symmetric_gap": bounded_mechanism_gap(delta_beh, jshap),
        "interaction_gap": jfwd - jrev,
        "basis_rank": int(Q.shape[1]),
    }


def _add_suffixed(row, allocation, suffix):
    for key, value in allocation.items():
        row[f"{key}_{suffix}"] = value

def audit_set(aud, tokens, S, y, competitors, taps=None, n_random=0,
              seed=0, ref_states=("clean",), path_steps=8):
    """Mechanistic evaluation of one deletion set S (positions into `tokens`).

    Returns per-tap dictionaries mirroring ``jlens.mechanistic_eval``. New
    reports should use ``clean_unrecovered``, ``clean_gap``,
    ``symmetric_gap``, and ``target_share_del``. Legacy ratio fields remain
    available for old artifacts.  Optional reference states add full
    two-order allocations for corrupted, clean-context path, deleted-context
    path, context-averaged path, and endpoint-union direction sets.
    """
    device = aud.device
    n = len(tokens)
    ids0 = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    m0 = torch.ones_like(ids0)
    ids1 = delete_and_shift(tokens, S).to(device).unsqueeze(0)
    m1 = (ids1 != 0).long()

    l0, l1 = aud.logits(ids0, m0), aud.logits(ids1, m1)
    delta_beh = float(l0[y] - l1[y])
    competitor_moves = [abs(float(l0[c] - l1[c])) for c in competitors]
    coll = float(np.mean(competitor_moves)) if competitor_moves else 0.0
    sel = abs(delta_beh) / max(coll, EPS)
    target_share = bounded_target_share(delta_beh, coll)

    A0, A1 = aud.stages(ids0, m0), aud.stages(ids1, m1)
    refs = set(ref_states)
    if "all" in refs:
        refs.update({"corrupted", "path", "union"})
    rows = []
    for tap in (taps or aud.taps):
        H0, H1 = A0[tap], A1[tap]
        items = [y] + list(competitors)
        with torch.no_grad():
            F00 = float(aud.resume(tap, H0, m0)[y])
            F11 = float(aud.resume(tap, H1, m1)[y])
        V0 = aud.readout_directions(tap, H0, m0, items)
        clean = allocation_from_directions(
            aud, tap, H0, H1, m0, m1, V0, y, delta_beh, F00, F11)
        row = {"tap": tap, "delta_beh": delta_beh,
               "delta_patch": F00 - F11,
               "selectivity_del": sel, "target_share_del": target_share,
               "k": len(S), "hist_len": n,
               **clean}

        V1 = None
        if refs.intersection({"corrupted", "union"}):
            V1 = aud.readout_directions(tap, H1, m1, items)
        if "corrupted" in refs:
            corrupt = allocation_from_directions(
                aud, tap, H0, H1, m0, m1, V1, y, delta_beh, F00, F11)
            _add_suffixed(row, corrupt, "corrupt")
            row["mr_corr"] = corrupt["mr"]  # legacy name

        if "path" in refs:
            Vpath_clean = path_readout_directions(
                aud, tap, H0, H1, m0, items, steps=path_steps)
            Vpath_deleted = path_readout_directions(
                aud, tap, H0, H1, m1, items, steps=path_steps)
            path_sets = {
                "path_clean": Vpath_clean,
                "path_deleted": Vpath_deleted,
                "path_avg": 0.5 * (Vpath_clean + Vpath_deleted),
            }
            for suffix, Vpath in path_sets.items():
                allocation = allocation_from_directions(
                    aud, tap, H0, H1, m0, m1, Vpath, y, delta_beh,
                    F00, F11)
                _add_suffixed(row, allocation, suffix)

        if "union" in refs:
            if V1 is None:
                V1 = aud.readout_directions(tap, H1, m1, items)
            union = allocation_from_directions(
                aud, tap, H0, H1, m0, m1, torch.cat([V0, V1], dim=1),
                y, delta_beh, F00, F11)
            _add_suffixed(row, union, "union")

        rows.append(row)

    if n_random:
        gen = np.random.default_rng(seed)
        for j in range(n_random):
            Sr = list(gen.choice(n, size=len(S), replace=False))
            for r in audit_set(aud, tokens, Sr, y, competitors, taps=taps,
                               n_random=0, ref_states=ref_states,
                               path_steps=path_steps):
                r["explainer_suffix"] = f"rand{j}"
                rows.append(r)
    return rows


# --------------------------------------------------------------------------
# explainers over positions
# --------------------------------------------------------------------------

def grad_x_input_map(aud, tokens, y, steps=0):
    """Saliency (grad x input) per position; steps>0 = integrated gradients
    along the 0 -> emb path."""
    device = aud.device
    m = aud.model
    ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    mask = torch.ones_like(ids)
    emb = m.embedding(ids) * np.sqrt(m.d_model)

    def path_grad(e):
        e = e.detach().clone().requires_grad_(True)
        seq_len = ids.size(1)
        if m.positional_direction == "reverse":
            pos = torch.arange(1, seq_len + 1, device=device).flip(0) \
                .unsqueeze(0)
        else:
            pos = torch.cumsum(mask, dim=1)
        x = m.layernorm(e + m.pos_embedding(pos))
        causal = torch.triu(torch.ones(seq_len, seq_len, device=device),
                            diagonal=1).bool()
        h = x.transpose(0, 1)
        for layer in m.transformer_layers:
            h = layer(h, src_mask=causal,
                      src_key_padding_mask=(mask == 0))
        s = m.fc_out(h.transpose(0, 1))[0, -1, y]
        return torch.autograd.grad(s, e)[0][0]

    if steps == 0:
        g = path_grad(emb)
    else:
        g = torch.zeros_like(emb[0])
        for t in range(1, steps + 1):
            g = g + path_grad(emb * (t / steps))
        g = g / steps
    return (g * emb[0]).sum(-1).detach()  # [T]


def occlusion_map(aud, tokens, y):
    n = len(tokens)
    drops = torch.zeros(n)
    ids0 = torch.tensor(tokens, dtype=torch.long,
                        device=aud.device).unsqueeze(0)
    s0 = float(aud.logits(ids0, torch.ones_like(ids0))[y])
    batch, masks = [], []
    for i in range(n):
        ids = delete_and_shift(tokens, [i])
        batch.append(ids)
    ids = torch.stack(batch).to(aud.device)
    mask = (ids != 0).long()
    with torch.no_grad():
        out = aud.model(ids, mask)[:, -1, y]
    return s0 - out.cpu()  # [T]


def lime_map(aud, tokens, y, n_samples=256, keep_prob=0.5, ridge=1.0,
             kernel_width=0.25, seed=0, batch=64):
    n = len(tokens)
    gen = torch.Generator().manual_seed(seed)
    masks = (torch.rand(n_samples, n, generator=gen) < keep_prob).float()
    masks[0] = 1.0
    scores = []
    for start in range(0, n_samples, batch):
        chunk = masks[start:start + batch]
        ids = torch.stack([
            delete_and_shift(tokens, torch.where(row == 0)[0].tolist())
            for row in chunk]).to(aud.device)
        m = (ids != 0).long()
        with torch.no_grad():
            scores.append(aud.model(ids, m)[:, -1, y].cpu())
    s = torch.cat(scores)
    kept = masks.mean(dim=1)
    w = torch.exp(-((1.0 - kept) ** 2) / kernel_width ** 2)
    Xd = masks - masks.mean(dim=0, keepdim=True)
    sd = s - s.mean()
    A = Xd.T @ (w[:, None] * Xd) + ridge * torch.eye(n)
    b = Xd.T @ (w * sd)
    return torch.linalg.solve(A, b)  # [T]


def topk_positions(score_map, k):
    return score_map.argsort(descending=True)[:k].tolist()
