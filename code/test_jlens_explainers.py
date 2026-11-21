"""Tests for jlens_explainers.py — random-init models, no checkpoints.

The MLP recommender's logit is affine in the input, so gradient-based
explainers have closed forms there: saliency == grad * x with a constant
gradient, and integrated gradients is EXACT for any number of steps
(completeness holds to float precision). Those are identities, not
tolerances.
"""
import pytest
import torch

import jlens
import jlens_explainers as jx

N_ITEMS = 120


@pytest.fixture(scope="module")
def classes():
    return jlens.load_notebook_classes()


@pytest.fixture(scope="module")
def mlp_insp(classes):
    torch.manual_seed(1)
    model = classes["MLP"](24, device=torch.device("cpu"),
                           num_items=N_ITEMS).eval()
    return jlens.MLPInspector(model)


@pytest.fixture(scope="module")
def vae_insp(classes):
    torch.manual_seed(0)
    conf = {"enc_dims": [32, 16], "dropout": 0.5,
            "anneal_cap": 0.2, "total_anneal_steps": 200000}
    model = classes["VAE"](conf, device=torch.device("cpu"),
                           num_features=N_ITEMS, num_items=N_ITEMS,
                           demographic=False).eval()
    return jlens.VAEInspector(model)


@pytest.fixture(scope="module")
def ncf_insp(classes):
    torch.manual_seed(4)
    kw = {"device": torch.device("cpu"), "num_features": N_ITEMS,
          "num_items": N_ITEMS, "demographic": False}
    gmf = classes["GMF_model"](hidden_size=4, **kw)
    mlp = classes["MLP_model"](4, 2, **kw)
    model = classes["NCF"](4, 2, 0.0, "NeuMF-pre",
                           GMF_model=gmf, MLP_model=mlp, **kw).eval()
    return jlens.NCFInspector(model)


@pytest.fixture()
def x():
    torch.manual_seed(2)
    v = torch.zeros(N_ITEMS)
    v[torch.randperm(N_ITEMS)[:20]] = 1.0
    return v


def test_topk_from_map_restricted_to_history(x):
    attr = torch.arange(N_ITEMS, dtype=torch.float)  # favors largest indices
    top = jx.topk_from_map(attr, x, k=5)
    assert len(top) == 5
    hist = set(x.nonzero().flatten().tolist())
    assert set(top) <= hist
    hist_sorted = sorted(hist, reverse=True)[:5]
    assert top == hist_sorted


def test_saliency_analytic_on_mlp(mlp_insp, x):
    y = 7
    attr = jx.saliency(mlp_insp, x, y)
    model = mlp_insp.model
    with torch.no_grad():
        item_vecs = model.items_fc(torch.eye(N_ITEMS))
        grad = model.users_fc.weight.T @ item_vecs[y]
    assert torch.allclose(attr, grad * x, atol=1e-6)


def test_ig_exact_on_affine_mlp(mlp_insp, x):
    """Affine logit => IG == grad*(x - baseline) and completeness is exact."""
    y = 7
    for steps in (1, 5):
        attr = jx.integrated_gradients(mlp_insp, x, y, steps=steps)
        with torch.no_grad():
            f_x, _ = mlp_insp.run(x)
            f_0, _ = mlp_insp.run(torch.zeros_like(x))
        assert attr.sum() == pytest.approx(float(f_x[0, y] - f_0[0, y]),
                                           abs=1e-4)
    sal = jx.saliency(mlp_insp, x, y)
    assert torch.allclose(attr, sal, atol=1e-5)  # zero baseline, affine


def test_ig_vae_shape_and_offhistory_zero(vae_insp, x):
    attr = jx.integrated_gradients(vae_insp, x, y=3, steps=8)
    assert attr.shape == (N_ITEMS,)
    assert torch.isfinite(attr).all()
    # zero baseline: attribution is grad-average * (x - 0), so 0 off-history
    assert (attr[x == 0] == 0).all()


def test_spinrec_lite_selects_argmax_candidate(vae_insp, x):
    y = 3
    k = 5
    attr, info = jx.spinrec_lite(vae_insp, x, y, k=k, n_baselines=4,
                                 steps=4, seed=11, return_info=True)
    # returned map must be the candidate with the largest deletion proxy
    assert info["chosen"] == int(max(range(len(info["proxies"])),
                                     key=lambda i: info["proxies"][i]))
    # and the proxy must be reproducible from the returned map
    S = jx.topk_from_map(attr, x, k)
    with torch.no_grad():
        f_x, _ = vae_insp.run(x)
        f_d, _ = vae_insp.run(jlens.delete_items(x, S))
    assert info["proxies"][info["chosen"]] == pytest.approx(
        float(f_x[0, y] - f_d[0, y]), abs=1e-5)


def test_spinrec_lite_deterministic(vae_insp, x):
    a1, _ = jx.spinrec_lite(vae_insp, x, 3, k=5, n_baselines=3, steps=4,
                            seed=5, return_info=True)
    a2, _ = jx.spinrec_lite(vae_insp, x, 3, k=5, n_baselines=3, steps=4,
                            seed=5, return_info=True)
    assert torch.equal(a1, a2)


def test_random_and_pop_explainers(x):
    hist = set(x.nonzero().flatten().tolist())
    S = jx.random_topk(x, k=5, seed=3)
    assert len(S) == 5 and set(S) <= hist
    assert S == jx.random_topk(x, k=5, seed=3)

    pop = torch.rand(N_ITEMS)
    S = jx.pop_topk(x, pop, k=5)
    assert set(S) <= hist
    assert S == sorted(hist, key=lambda i: -pop[i])[:5]


# ---------------------------------------------------------------- real SPINRec

@pytest.fixture(scope="module")
def train_X():
    torch.manual_seed(9)
    return (torch.rand(30, N_ITEMS) < 0.15).float()


def test_spinrec_real_map_zero_off_history(vae_insp, x, train_X):
    attr, info = jx.spinrec_real(vae_insp, x, 3, k=5, train_X=train_X,
                                 n_baselines=4, steps=5, seed=1,
                                 return_info=True)
    assert attr.shape == x.shape
    assert (attr[x == 0] == 0).all()
    assert all(0 <= b < len(train_X) for b in info["baseline_ids"])
    assert info["chosen"] == int(max(range(len(info["proxies"])),
                                     key=lambda i: info["proxies"][i]))


def test_spinrec_real_deterministic(vae_insp, x, train_X):
    a1 = jx.spinrec_real(vae_insp, x, 3, k=5, train_X=train_X,
                         n_baselines=3, steps=5, seed=7)
    a2 = jx.spinrec_real(vae_insp, x, 3, k=5, train_X=train_X,
                         n_baselines=3, steps=5, seed=7)
    assert torch.equal(a1, a2)


def test_spinrec_real_auto_uses_sigmoid_for_ncf(ncf_insp, x, train_X):
    auto = jx.spinrec_real(ncf_insp, x, 3, k=5, train_X=train_X,
                           n_baselines=3, steps=4, seed=13, prob="auto")
    explicit = jx.spinrec_real(ncf_insp, x, 3, k=5, train_X=train_X,
                               n_baselines=3, steps=4, seed=13,
                               prob="sigmoid")
    assert torch.equal(auto, explicit)


def test_spinrec_real_zero_baseline_matches_ig_on_affine_mlp(mlp_insp, x):
    """With the zero baseline forced, the published SPINRec map is
    x * avg_grad * (x - 0) = IG on a binary history (x^2 = x); on the affine
    MLP the input gradient is constant along the path, so the probability
    rescaling cannot change the ranking."""
    y = 7
    attr = jx.spinrec_real(mlp_insp, x, y, k=5, train_X=None,
                           n_baselines=0, steps=5, seed=0,
                           include_zero_baseline=True)
    ig = jx.integrated_gradients(mlp_insp, x, y, steps=5)
    hist = x.nonzero().flatten().tolist()
    rank_a = sorted(hist, key=lambda i: -float(attr[i]))
    rank_b = sorted(hist, key=lambda i: -float(ig[i]))
    assert rank_a == rank_b
