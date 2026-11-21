"""Identity tests for the Item-Jacobian lens module (jlens.py).

All tests run on small random-init models — no checkpoints or data needed.
The assertions are mathematical identities of the patching construction,
not smoke tests: if any fails, the implementation is wrong, not the model.
"""
import pytest
import torch
import torch.nn.functional as F

import jlens
import run_jlens_sasrec as sas_run
import run_jlens_sasrec_carrier as sas_carrier
import run_jlens_switch_diagnostic as switch_diag

N_ITEMS = 120
ENC_DIMS = [32, 16]
VAE_CONF = {"enc_dims": ENC_DIMS, "dropout": 0.5,
            "anneal_cap": 0.2, "total_anneal_steps": 200000}


@pytest.fixture(scope="module")
def classes():
    return jlens.load_notebook_classes()


@pytest.fixture(scope="module")
def vae(classes):
    torch.manual_seed(0)
    kw = dict(device=torch.device("cpu"), num_features=N_ITEMS,
              num_items=N_ITEMS, demographic=False)
    model = classes["VAE"](VAE_CONF, **kw)
    model.eval()
    return model


@pytest.fixture(scope="module")
def mlp(classes):
    torch.manual_seed(1)
    kw = dict(device=torch.device("cpu"), num_items=N_ITEMS)
    model = classes["MLP"](24, **kw)
    model.eval()
    return model


@pytest.fixture()
def x():
    torch.manual_seed(2)
    v = torch.zeros(N_ITEMS)
    v[torch.randperm(N_ITEMS)[:20]] = 1.0
    return v


@pytest.fixture()
def S(x):
    return x.nonzero().flatten()[:8].tolist()


# ---------------------------------------------------------------- inspectors

def test_vae_logits_match_model_forward(vae, x):
    insp = jlens.VAEInspector(vae)
    logits, _ = insp.run(x)
    with torch.no_grad():
        expected = vae(x)
    assert torch.allclose(torch.softmax(logits, dim=1), expected, atol=1e-6)


def test_vae_taps_present(vae):
    insp = jlens.VAEInspector(vae)
    assert insp.taps == ["enc_h1", "z", "dec_h1"]


def test_vae_run_from_resumes_each_tap(vae, x):
    insp = jlens.VAEInspector(vae)
    logits, acts = insp.run(x)
    for tap in insp.taps:
        resumed = insp.run_from(tap, acts[tap])
        assert torch.allclose(resumed, logits, atol=1e-6), tap


def test_vae_demographic_slice(classes):
    torch.manual_seed(3)
    kw = dict(device=torch.device("cpu"), num_features=140,
              num_items=N_ITEMS, demographic=True)
    model = classes["VAE"](VAE_CONF, **kw)
    model.eval()
    insp = jlens.VAEInspector(model)
    xd = torch.zeros(140)
    xd[:30] = 1.0
    logits, _ = insp.run(xd)
    assert logits.shape == (1, N_ITEMS)
    with torch.no_grad():
        expected = model(xd)
    assert torch.allclose(torch.softmax(logits, dim=1), expected, atol=1e-6)


def test_mlp_logits_match_model_forward(mlp, x):
    insp = jlens.MLPInspector(mlp)
    logits, acts = insp.run(x)
    assert "user_vec" in acts
    with torch.no_grad():
        expected = mlp(x.unsqueeze(0), torch.eye(N_ITEMS))
    assert torch.allclose(torch.sigmoid(logits), expected, atol=1e-6)
    resumed = insp.run_from("user_vec", acts["user_vec"])
    assert torch.allclose(resumed, logits, atol=1e-6)


def test_ncf_vectorized_forward_matches_notebook_model(classes):
    kw = {"device": torch.device("cpu"), "num_features": 11,
          "num_items": 11, "demographic": False}
    gmf = classes["GMF_model"](hidden_size=2, **kw)
    mlp = classes["MLP_model"](2, 2, **kw)
    model = classes["NCF"](2, 2, 0.0, "NeuMF-pre",
                           GMF_model=gmf, MLP_model=mlp, **kw).eval()
    inspector = jlens.NCFInspector(model)
    user = torch.tensor([1., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0.])
    items = [1, 4, 10]
    logits, _ = inspector.run_items(user, items)
    one_hot = torch.eye(11)[items]
    with torch.no_grad():
        notebook_prob = model(user.unsqueeze(0), one_hot)
    assert torch.allclose(logits[0], torch.logit(notebook_prob), atol=1e-6)
    full, acts = inspector.run(user)
    assert torch.allclose(logits, full[:, items], atol=1e-6)
    assert set(acts) == {"user_state"}
    sets = [[0], [2, 5], []]
    fast = inspector.deletion_states(user, sets)
    for row, S in enumerate(sets):
        deleted = jlens.delete_items(user, S)
        _, expected_acts = inspector.run_items(deleted, items)
        assert torch.allclose(fast[row], expected_acts["user_state"][0],
                              atol=1e-6)


def test_ncf_frozen_gate_decomposition_is_exact(classes):
    torch.manual_seed(31)
    kw = {"device": torch.device("cpu"), "num_features": 11,
          "num_items": 11, "demographic": False}
    gmf = classes["GMF_model"](hidden_size=2, **kw)
    mlp = classes["MLP_model"](2, 2, **kw)
    model = classes["NCF"](2, 2, 0.0, "NeuMF-pre",
                           GMF_model=gmf, MLP_model=mlp, **kw).eval()
    inspector = jlens.NCFInspector(model)
    x0 = torch.tensor([1., 0., 1., 0., 0., 1., 0., 1., 0., 0., 0.])
    x1 = x0.clone()
    x1[[0, 5]] = 0.0
    _, acts0 = inspector.run_items(x0, [4])
    _, acts1 = inspector.run_items(x1, [4])
    result = switch_diag.ncf_gate_decomp(
        inspector, acts0["user_state"], acts1["user_state"], 4)
    expected = float((
        inspector.run_from_items("user_state", acts0["user_state"], [4])[0, 0]
        - inspector.run_from_items("user_state", acts1["user_state"], [4])[0, 0]
    ).detach())
    assert result["gate_endpoint_delta"] == pytest.approx(expected, abs=1e-6)
    assert result["clean_affine_mass"] + result["gate_mass"] \
        == pytest.approx(expected, abs=1e-6)
    assert result["gate_shapley_error"] == pytest.approx(0.0, abs=1e-7)


def test_sasrec_component_carrier_decomposition_is_exact():
    torch.manual_seed(32)
    layer = torch.nn.TransformerEncoderLayer(
        d_model=8, nhead=2, dim_feedforward=16, dropout=0.0,
        batch_first=False, norm_first=True).eval()

    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer_layers = torch.nn.ModuleList([layer])
            self.fc_out = torch.nn.Linear(8, 5)
            self.adapter_W = None

    class ToyAuditor:
        def __init__(self):
            self.model = ToyModel().eval()

    auditor = ToyAuditor()
    H0 = torch.randn(6, 8)
    H1 = H0 + 0.3 * torch.randn(6, 8)
    m0 = torch.ones(1, 6, dtype=torch.long)
    m1 = torch.tensor([[0, 1, 1, 1, 1, 1]])
    result = sas_carrier.decompose_last_block(
        auditor, "emb", H0, H1, m0, m1, target=2)
    assert result["tangent_recovered"] + result["tangent_residual"] \
        == pytest.approx(result["carrier_delta"], abs=1e-6)
    assert result["carrier_shapley_error"] == pytest.approx(0.0, abs=1e-7)


def test_sasrec_search_objectives_are_distinct():
    remainder = {"delta_beh": 2.0, "jfwd": 0.4}
    aligned = {"delta_beh": 1.0, "jfwd": 1.0}
    assert sas_run.search_score(remainder, score="delta_minus_j", lam=0.0) \
        == pytest.approx(2.0)
    assert sas_run.search_score(remainder, score="delta_minus_j", lam=1.0) \
        == pytest.approx(1.6)
    assert sas_run.search_score(remainder, score="jfwd") == pytest.approx(0.4)
    assert sas_run.search_score(aligned, score="jfwd") == pytest.approx(1.0)


def test_sasrec_attack_label_names_the_score():
    assert sas_run.attack_label("jfwd", lam=1.0) == "attack_jfwd"
    assert sas_run.attack_label("delta_minus_j", lam=1.0) == "attack_lam1"
    assert sas_run.attack_label("delta_minus_j", lam=0.0) == "attack_lam0"


# ----------------------------------------------------------------- patching

def test_full_patch_equals_corrupt_run(vae, x, S):
    insp = jlens.VAEInspector(vae)
    x_del = jlens.delete_items(x, S)
    logits_corr, acts_corr = insp.run(x_del)
    for tap in insp.taps:
        resumed = insp.run_from(tap, acts_corr[tap])
        assert torch.allclose(resumed, logits_corr, atol=1e-6), tap


def test_zero_patch_equals_clean_run(vae, x):
    insp = jlens.VAEInspector(vae)
    logits, acts = insp.run(x)
    y = int(logits[0].argmax())
    for tap in insp.taps:
        V = jlens.item_directions(insp, x, tap, [y])
        Q = jlens.orthonormal_basis(V)
        patched = jlens.patch_logits(insp, tap, acts[tap], acts[tap], Q, "comp")
        assert torch.allclose(patched, logits, atol=1e-6), tap


def test_full_space_restore_equals_clean(vae, x, S):
    insp = jlens.VAEInspector(vae)
    logits_clean, acts_clean = insp.run(x)
    x_del = jlens.delete_items(x, S)
    _, acts_corr = insp.run(x_del)
    for tap in insp.taps:
        d = acts_clean[tap].shape[1]
        Q = torch.eye(d)
        restored = jlens.patch_logits(insp, tap, acts_clean[tap],
                                      acts_corr[tap], Q, "restore")
        assert torch.allclose(restored, logits_clean, atol=1e-5), tap


# --------------------------------------------------------------- directions

def test_directions_last_dec_tap_analytic(vae, x):
    insp = jlens.VAEInspector(vae)
    items = [3, 17, 42]
    V = jlens.item_directions(insp, x, "dec_h1", items)
    W_out = vae.decoder[-1].weight
    expected = W_out[items, :].T
    assert torch.allclose(V, expected, atol=1e-6)


def test_directions_mlp_analytic(mlp, x):
    insp = jlens.MLPInspector(mlp)
    items = [5, 60]
    V = jlens.item_directions(insp, x, "user_vec", items)
    with torch.no_grad():
        item_vecs = mlp.items_fc(torch.eye(N_ITEMS))
    assert torch.allclose(V, item_vecs[items, :].T, atol=1e-6)


def test_global_directions_are_mean_of_local(vae):
    torch.manual_seed(4)
    X = (torch.rand(3, N_ITEMS) < 0.15).float()
    insp = jlens.VAEInspector(vae)
    V_g = jlens.global_item_directions(insp, X, "z", [7])
    locals_ = torch.stack(
        [jlens.item_directions(insp, X[b], "z", [7]) for b in range(3)])
    assert torch.allclose(V_g, locals_.mean(0), atol=1e-6)


# ------------------------------------------------------- linear-alg helpers

def test_orthonormal_basis_props():
    torch.manual_seed(5)
    V = torch.randn(30, 4)
    Q = jlens.orthonormal_basis(V)
    assert Q.shape == (30, 4)
    assert torch.allclose(Q.T @ Q, torch.eye(4), atol=1e-5)
    delta = torch.randn(1, 30)
    p1 = jlens.project(Q, delta)
    assert torch.allclose(jlens.project(Q, p1), p1, atol=1e-5)


def test_orthonormal_basis_drops_dependent_columns():
    torch.manual_seed(6)
    v = torch.randn(30, 1)
    V = torch.cat([v, 2 * v, torch.randn(30, 1)], dim=1)
    Q = jlens.orthonormal_basis(V)
    assert Q.shape[1] == 2


def test_effective_rank():
    V = torch.eye(10)[:, :4]
    assert jlens.effective_rank(V) == pytest.approx(4.0, abs=1e-4)
    v = torch.randn(10, 1)
    assert jlens.effective_rank(v.repeat(1, 3)) == pytest.approx(1.0, abs=1e-4)


# ------------------------------------------------- the core identities (MR)

def test_mr_identity_last_hidden_tap_vae(vae, x, S):
    """At the last decoder hidden tap the lens is exact (J = W_out), so the
    comp-patch effect on the target logit equals the full behavioral delta."""
    insp = jlens.VAEInspector(vae)
    logits_clean, acts_clean = insp.run(x)
    y = int(logits_clean[0].argmax())
    x_del = jlens.delete_items(x, S)
    logits_corr, acts_corr = insp.run(x_del)
    delta_beh = (logits_clean - logits_corr)[0, y]

    V = jlens.item_directions(insp, x, "dec_h1", [y])
    Q = jlens.orthonormal_basis(V)
    patched = jlens.patch_logits(insp, "dec_h1", acts_clean["dec_h1"],
                                 acts_corr["dec_h1"], Q, "comp")
    jcomp = (logits_clean - patched)[0, y]
    assert torch.allclose(jcomp, delta_beh, atol=1e-5)


def test_mr_identity_mlp(mlp, x, S):
    """MLP has one linear layer before the item dot-product: MR must be 1."""
    insp = jlens.MLPInspector(mlp)
    logits_clean, acts_clean = insp.run(x)
    y = int(logits_clean[0].argmax())
    x_del = jlens.delete_items(x, S)
    logits_corr, acts_corr = insp.run(x_del)
    delta_beh = (logits_clean - logits_corr)[0, y]

    V = jlens.item_directions(insp, x, "user_vec", [y])
    Q = jlens.orthonormal_basis(V)
    patched = jlens.patch_logits(insp, "user_vec", acts_clean["user_vec"],
                                 acts_corr["user_vec"], Q, "comp")
    jcomp = (logits_clean - patched)[0, y]
    assert torch.allclose(jcomp, delta_beh, atol=1e-5)


# ------------------------------------------------------------------- runner

def test_mechanistic_eval_rows(vae, x, S):
    insp = jlens.VAEInspector(vae)
    logits, _ = insp.run(x)
    y = int(logits[0].argmax())
    rows = jlens.mechanistic_eval(insp, x, S, y, n_competitors=4,
                                  n_random=3, seed=0)
    assert [r["tap"] for r in rows] == insp.taps
    for r in rows:
        for key in ["tap", "delta_beh", "jcomp", "jrestore", "mr", "mr_full",
                    "jrestore_raw", "jshap", "smr", "jshap_complement",
                    "clean_unrecovered", "clean_gap", "symmetric_gap",
                    "interaction_gap", "selectivity_del", "target_share_del",
                    "selectivity_patch", "erank",
                    "jcomp_rand_mean", "jcomp_rand_std", "n_directions",
                    "collateral_all"]:
            assert key in r, key
        assert r["collateral_all"] >= 0
        assert r["n_directions"] == 5
        assert r["erank"] <= 5.0 + 1e-6
        assert r["jshap"] + r["jshap_complement"] == pytest.approx(
            r["delta_beh"], abs=1e-6)
        if r["delta_beh"] > 1e-6:
            assert 0.0 <= r["clean_gap"] <= 1.0
            assert 0.0 <= r["symmetric_gap"] <= 1.0
            assert 0.0 <= r["target_share_del"] <= 1.0
    # the by-construction identity, now via the runner:
    last = rows[-1]
    assert last["tap"] == "dec_h1"
    assert last["mr"] == pytest.approx(1.0, abs=1e-3)
    assert last["smr"] == pytest.approx(1.0, abs=1e-3)
    assert last["mr_full"] == pytest.approx(1.0, abs=1e-3)


def test_mechanistic_eval_mlp_control(mlp, x, S):
    insp = jlens.MLPInspector(mlp)
    logits, _ = insp.run(x)
    y = int(logits[0].argmax())
    rows = jlens.mechanistic_eval(insp, x, S, y, n_competitors=4)
    assert len(rows) == 1
    assert rows[0]["mr"] == pytest.approx(1.0, abs=1e-3)


def test_mechanistic_eval_random_control_reproducible(vae, x, S):
    insp = jlens.VAEInspector(vae)
    logits, _ = insp.run(x)
    y = int(logits[0].argmax())
    r1 = jlens.mechanistic_eval(insp, x, S, y, n_random=2, seed=7)
    r2 = jlens.mechanistic_eval(insp, x, S, y, n_random=2, seed=7)
    assert r1[0]["jcomp_rand_mean"] == pytest.approx(r2[0]["jcomp_rand_mean"])


# --------------------------------------------------------------------- demo

def test_demo_runs_end_to_end():
    import jlens_demo
    tables = jlens_demo.main(n_users=2, verbose=False)
    assert set(tables) == {"VAE", "MLP"}
    for df in tables.values():
        assert len(df) > 0
        assert {"explainer", "tap", "mr"} <= set(df.columns)
    # MLP positive control: MR == 1 for every user/explainer
    assert (tables["MLP"]["mr"] - 1.0).abs().max() < 1e-3


def test_mechanistic_eval_directions_override(vae, x, S):
    """Passing a directions callable that returns the local directions must
    reproduce the default rows exactly."""
    insp = jlens.VAEInspector(vae)
    logits, _ = insp.run(x)
    y = int(logits[0].argmax())
    base = jlens.mechanistic_eval(insp, x, S, y)
    override = jlens.mechanistic_eval(
        insp, x, S, y,
        directions=lambda tap, C: jlens.item_directions(insp, x, tap, C))
    for a, b in zip(base, override):
        assert a["jcomp"] == pytest.approx(b["jcomp"], abs=1e-6)
        assert a["mr"] == pytest.approx(b["mr"], abs=1e-6)


def test_listwise_mediation_identity_last_hidden(vae, x, S):
    """List-wise MR over the top-k list obeys the same output-adjacent
    identity as the scalar MR: at dec_h1 the patch transmits every item
    direction in the basis exactly, so LMR == 1."""
    insp = jlens.VAEInspector(vae)
    logits, _ = insp.run(x)
    top = logits[0].argsort(descending=True)[:3].tolist()
    lmr = jlens.listwise_mediation(insp, x, S, top, tap="dec_h1")
    # med == beh exactly; the eps division guard bounds |LMR-1| by
    # eps/|beh|, ~2e-4 for this random-init model's tiny weighted deltas
    assert lmr == pytest.approx(1.0, abs=2e-3)


def test_listwise_mediation_weights_and_range(vae, x, S):
    insp = jlens.VAEInspector(vae)
    logits, _ = insp.run(x)
    top = logits[0].argsort(descending=True)[:3].tolist()
    lmr_z = jlens.listwise_mediation(insp, x, S, top, tap="z")
    assert isinstance(lmr_z, float)
    # single-item list must reduce to the scalar MR
    y = top[0]
    rows = jlens.mechanistic_eval(insp, x, S, y, competitors=top[1:],
                                  taps=["z"])
    lmr_single = jlens.listwise_mediation(insp, x, S, [y], tap="z",
                                          competitors=top[1:])
    assert lmr_single == pytest.approx(rows[0]["mr"], abs=1e-3)


def test_activation_and_path_directions_reduce_to_local(vae, x):
    insp = jlens.VAEInspector(vae)
    _, acts = insp.run(x)
    items = [3, 17]
    local = jlens.item_directions(insp, x, "z", items)
    explicit = jlens.directions_at_activation(insp, "z", acts["z"], items)
    path = jlens.path_item_directions(
        insp, "z", acts["z"], acts["z"], items, steps=3)
    assert torch.allclose(local, explicit, atol=1e-6)
    assert torch.allclose(local, path, atol=1e-6)


def test_shapley_order_average_is_exact(vae, x, S):
    insp = jlens.VAEInspector(vae)
    logits, _ = insp.run(x)
    y = int(logits[0].argmax())
    rows = jlens.mechanistic_eval(insp, x, S, y)
    for r in rows:
        expected = 0.5 * (r["jcomp"] + r["jrestore_raw"])
        assert r["jshap"] == pytest.approx(expected, abs=1e-7)
        assert r["jshap"] + r["jshap_complement"] == pytest.approx(
            r["delta_beh"], abs=1e-6)
