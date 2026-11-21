"""Regression tests for paper aggregation, where selection bias can hide."""
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "paper"))
import make_tables as mt  # noqa: E402
from run_jlens_competitor_sensitivity import competitor_sets  # noqa: E402
from jlens_lxr_training import (  # noqa: E402
    all_scores, clean_item_bases, encode_z, project_batched, target_scores)
from run_jlens_lxr_retrain import (  # noqa: E402
    lxr_loss as retrain_lxr_loss, split_indices)


@pytest.mark.parametrize("data,k", [
    ("ml1m", 5), ("ml1m", 10), ("ml1m", 20),
    ("yahoo", 10), ("pinterest", 10),
])
def test_attack_summary_uses_identical_users_and_unconditional_means(data, k):
    df = pd.read_csv(ROOT / "jlens_results"
                     / f"{data}_vae_adversarial_k{k}.csv")
    methods = ["saliency", "attack_lam1", "constraint"]
    out = mt.paired_attack_summary(df, methods)
    user_sets = [out[m]["users"] for m in methods]
    assert all(s == user_sets[0] for s in user_sets[1:])
    z = df[(df["tap"] == "z") & (df["user"].isin(user_sets[0]))]
    for m in methods:
        raw = z[z["method"] == m]["delta_beh"].mean()
        assert out[m]["delta_mean"] == pytest.approx(raw)


def test_additive_mediation_backfill_matches_order_average():
    df = pd.read_csv(ROOT / "jlens_results/ml1m_vae_k5.csv", nrows=20)
    out = mt._augment_mediation(df)
    assert ((out["smr"] - 0.5 * (out["mr"] + out["jrestore"])).abs()
            < 1e-12).all()
    positive = out["delta_beh"] > 1e-6
    assert out.loc[positive, "clean_gap"].between(0, 1).all()
    assert out.loc[positive, "symmetric_gap"].between(0, 1).all()
    expected = ((out.loc[positive, "delta_beh"]
                 - out.loc[positive, "jcomp"])
                / out.loc[positive, "delta_beh"]).clip(0, 1)
    assert (out.loc[positive, "clean_gap"] - expected).abs().max() < 1e-12


def test_attack_gap_threshold_matches_legacy_recovery_threshold():
    df = pd.read_csv(ROOT / "jlens_results/ml1m_vae_adversarial_k10.csv")
    out = mt._augment_mediation(df)
    strong = out["delta_beh"] > 0.2
    legacy = strong & (out["mr"] < 0.5)
    bounded = strong & (out["clean_gap"] > 0.5)
    assert legacy.equals(bounded)


@pytest.mark.parametrize("data", ["ml1m", "yahoo", "pinterest"])
def test_mechanism_switch_signature_survives_stronger_null(data):
    df = pd.read_csv(ROOT / "jlens_results"
                     / f"{data}_vae_basis_robustness_k10.csv")
    g = df[(df["method"] == "attack_lam1") & (df["delta_beh"] > 0.01)]
    med = g.groupby("basis")[["mr", "smr"]].median()
    # Adding random directions while retaining the target is a fairer null
    # than a random-only space and must not erase the clean-reference gap.
    assert abs(med.loc["clean_target", "mr"]
               - med.loc["clean_target_random", "mr"]) < 0.15
    # The same saved deletion sets become mediated once dormant directions
    # along the path are admitted: the empirical mechanism-switch signature.
    assert med.loc["path_target_comp", "smr"] > 0.95


def test_competitor_sweep_is_nested_deterministic_and_excludes_target():
    ranked = list(range(50))
    first = competitor_sets(ranked, counts=[0, 2, 5], seed=17,
                            local_pool=20)
    second = competitor_sets(ranked, counts=[0, 2, 5], seed=17,
                             local_pool=20)
    assert first == second
    assert first[("top", 0)] == []
    assert first[("top", 5)] == [1, 2, 3, 4, 5]
    for strategy in ("top", "sampled_top100", "uniform"):
        assert first[(strategy, 5)][:2] == first[(strategy, 2)]
        assert len(set(first[(strategy, 5)])) == 5
        assert 0 not in first[(strategy, 5)]


def test_batched_decoder_basis_matches_autograd_projection():
    torch.manual_seed(3)

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = torch.nn.ModuleList([
                torch.nn.Linear(3, 7), torch.nn.ReLU(),
                torch.nn.Linear(7, 11)])

    model = Tiny().eval()
    z = torch.randn(4, 3)
    items = torch.tensor([[0, 1, 2], [3, 4, 5],
                          [6, 7, 8], [8, 9, 10]])
    q, rank = clean_item_bases(model, z, items)
    assert (rank > 0).all()
    for row in range(len(z)):
        zv = z[row:row + 1].detach().requires_grad_(True)
        logits = all_scores(model, zv)
        columns = []
        for item in items[row]:
            (grad,) = torch.autograd.grad(
                logits[0, item], zv, retain_graph=True)
            columns.append(grad[0])
        directions = torch.stack(columns, dim=1)
        expected = directions @ torch.linalg.pinv(directions)
        actual = q[row] @ q[row].T
        assert torch.allclose(actual, expected, atol=2e-5, rtol=2e-5)

    delta = torch.randn(4, 3)
    projected = project_batched(q, delta)
    explicit = torch.stack([q[i] @ (q[i].T @ delta[i])
                            for i in range(len(delta))])
    assert torch.allclose(projected, explicit)


def test_lxr_retrain_split_is_deterministic_and_disjoint():
    train_a, val_a = split_indices(101, val_fraction=.2, split_seed=17)
    train_b, val_b = split_indices(101, val_fraction=.2, split_seed=17)
    assert torch.equal(train_a, train_b)
    assert torch.equal(val_a, val_b)
    assert len(val_a) == 20
    assert set(train_a.tolist()).isdisjoint(val_a.tolist())
    assert sorted(torch.cat([train_a, val_a]).tolist()) == list(range(101))


def test_beta_zero_retrain_loss_is_released_lxr_code_objective():
    torch.manual_seed(18)

    class TinyVAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.ModuleList([torch.nn.Linear(5, 6)])
            self.decoder = torch.nn.ModuleList([
                torch.nn.Linear(3, 7), torch.nn.ReLU(),
                torch.nn.Linear(7, 5)])

    class TinyExplainer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(10, 5)

        def forward(self, x, item):
            return self.layer(torch.cat([x, item], dim=1))

    from run_jlens_lxr_retrain import masked_inputs

    model = TinyVAE().eval()
    explainer = TinyExplainer()
    xb = torch.tensor([[1., 0., 1., 1., 0.],
                       [0., 1., 1., 0., 1.]])
    y = torch.tensor([4, 0])
    z0 = encode_z(model, xb)
    q = torch.zeros(2, 3, 2)
    config = {"lambda_pos": 3.0, "lambda_neg": 2.0}

    actual, _ = retrain_lxr_loss(
        model, explainer, xb, y, z0, q, config, beta=0.0,
        inverse_loss="released_code")
    mask, kept, deleted = masked_inputs(explainer, xb, y)
    rows = torch.arange(len(xb))
    kept_logits = all_scores(model, encode_z(model, kept))
    deleted_logits = all_scores(model, encode_z(model, deleted))
    positive = -torch.log_softmax(kept_logits, dim=1)[rows, y].mean()
    negative = torch.log_softmax(deleted_logits, dim=1)[rows, y].mean()
    expected = 3.0 * positive + 2.0 * negative + mask[xb > 0].mean()
    assert float(actual.detach()) == pytest.approx(
        float(expected.detach()), abs=1e-6)


def test_beta_one_replaces_positive_remainder_in_inverse_branch():
    torch.manual_seed(19)

    class TinyVAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.ModuleList([torch.nn.Linear(5, 6)])
            self.decoder = torch.nn.ModuleList([
                torch.nn.Linear(3, 7), torch.nn.ReLU(),
                torch.nn.Linear(7, 5)])

    class TinyExplainer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(10, 5)

        def forward(self, x, item):
            return self.layer(torch.cat([x, item], dim=1))

    from run_jlens_lxr_retrain import masked_inputs

    model = TinyVAE().eval()
    explainer = TinyExplainer()
    xb = torch.tensor([[1., 0., 1., 1., 0.],
                       [0., 1., 1., 0., 1.]])
    y = torch.tensor([4, 0])
    z0 = encode_z(model, xb)
    q = torch.zeros(2, 3, 2)
    config = {"lambda_pos": 3.0, "lambda_neg": 2.0}

    actual, _ = retrain_lxr_loss(
        model, explainer, xb, y, z0, q, config, beta=1.0,
        inverse_loss="released_code")
    mask, kept, deleted = masked_inputs(explainer, xb, y)
    rows = torch.arange(len(xb))
    kept_logits = all_scores(model, encode_z(model, kept))
    z1 = encode_z(model, deleted)
    deleted_logits = all_scores(model, z1)
    deleted_target = deleted_logits[rows, y]
    patched_target = target_scores(model, z0, y)
    corrected_target = torch.maximum(deleted_target, patched_target)
    corrected = deleted_logits.clone()
    corrected[rows, y] = corrected_target
    positive = -torch.log_softmax(kept_logits, dim=1)[rows, y].mean()
    negative = torch.log_softmax(corrected, dim=1)[rows, y].mean()
    expected = 3.0 * positive + 2.0 * negative + mask[xb > 0].mean()
    assert float(actual.detach()) == pytest.approx(
        float(expected.detach()), abs=1e-6)

