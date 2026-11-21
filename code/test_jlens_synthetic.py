"""Tests for the synthetic two-pathway recommender.

Ground truth is built into the weights, so the dissociation assertions are
part of the test suite: same behavioral deletion effect, opposite
mechanistic mediation at the pre-gate tap.
"""
import pytest
import torch

import jlens
import jlens_explainers as jx
import jlens_synthetic as js


@pytest.fixture(scope="module")
def model():
    return js.SyntheticTwoPath()


@pytest.fixture(scope="module")
def insp(model):
    return js.SyntheticInspector(model)


@pytest.fixture()
def user():
    return js.make_user(seed=0)


def test_taps_and_run_from(insp, user):
    assert insp.taps == ["h1", "h2"]
    logits, acts = insp.run(user)
    for tap in insp.taps:
        assert torch.allclose(insp.run_from(tap, acts[tap]), logits,
                              atol=1e-6), tap


def test_clean_user_analytic(insp, user):
    """15 genre-0 items, closed gates, and a constant top-1 tie-break."""
    logits, acts = insp.run(user)
    assert int(acts["h1"][0, js.G0]) == 15
    assert float(acts["h1"][0, js.AT_COORD]) == 8.0
    assert float(logits[0, js.TARGET]) == pytest.approx(
        15 + js.TARGET_BIAS, abs=1e-5)
    masked = logits[0].clone()
    masked[user > 0] = float("-inf")
    assert int(masked.argmax()) == js.TARGET


def test_lens_blind_to_closed_gate(insp, user):
    """Clean-run gradient of the target logit has exactly zero weight on both
    anchor coordinates of h1 while the gates are saturated."""
    V = jlens.item_directions(insp, user, "h1", [js.TARGET])
    assert float(V[js.AT_COORD, 0]) == 0.0
    assert float(V[js.AU_COORD, 0]) == 0.0
    assert float(V[js.G0, 0]) == pytest.approx(1.0, abs=1e-6)


def test_dissociation(insp, user):
    """The core claim: equal DEL, opposite MR at the pre-gate tap."""
    expl = js.ground_truth_explanations(user, k=5, seed=0)
    y, comps = js.target_and_competitors(insp, user)
    res = {}
    for name, S in expl.items():
        rows = jlens.mechanistic_eval(insp, user, S, y, competitors=comps)
        res[name] = {r["tap"]: r for r in rows}

    for name in ["genre", "anchor_t", "anchor_u"]:
        assert res[name]["h1"]["delta_beh"] == pytest.approx(5.0, abs=1e-5), name
        # h2 is the last hidden layer -> MR == 1 by construction everywhere
        assert res[name]["h2"]["mr"] == pytest.approx(1.0, abs=1e-3), name

    assert res["genre"]["h1"]["mr"] > 0.9
    assert res["genre"]["h1"]["smr"] > 0.9
    assert res["anchor_t"]["h1"]["mr"] < 0.1
    assert res["anchor_u"]["h1"]["mr"] < 0.1
    assert res["anchor_t"]["h1"]["smr"] < 0.1
    assert res["anchor_u"]["h1"]["smr"] < 0.1

    # anchor_t hits ONLY the target: high selectivity, DEL+selectivity fooled
    assert res["anchor_t"]["h1"]["selectivity_del"] > 10
    # The uniform-gate and genre routes match exactly on both reported
    # behavioral quantities: target-score effect and bounded target share.
    assert res["genre"]["h1"]["delta_beh"] == pytest.approx(
        res["anchor_u"]["h1"]["delta_beh"], abs=1e-6)
    assert res["genre"]["h1"]["target_share_del"] == pytest.approx(
        res["anchor_u"]["h1"]["target_share_del"], abs=1e-6)
    assert res["genre"]["h1"]["target_share_del"] == pytest.approx(
        0.5, abs=1e-6)

    assert abs(res["filler"]["h1"]["delta_beh"]) < 0.5  # weak quadrant


def test_saliency_cannot_see_the_gate(insp, user):
    """Local gradients are zero on saturated anchors -> saliency picks only
    genre items; IG integrates through the gate-active region -> anchors."""
    S_sal = jx.topk_from_map(jx.saliency(insp, user, js.TARGET), user, 5)
    assert all(i < js.N_GENRE_ITEMS for i in S_sal)
    S_ig = jx.topk_from_map(
        jx.integrated_gradients(insp, user, js.TARGET, steps=50), user, 5)
    assert all(js.is_anchor(i) for i in S_ig)


# ------------------------------------------------------- adversarial search

def test_greedy_attack_finds_the_silent_channel(insp, user):
    """Ground-truth validation of the adversarial search: maximizing
    (DEL - JComp@h1) on the synthetic model must select anchor items
    (the locally-silent gate pathway), yielding high DEL and MR ~ 0."""
    import jlens_adversarial as ja
    y, comps = js.target_and_competitors(insp, user)

    S = ja.greedy_attack(insp, user, y, k=5, tap="h1",
                         competitors=comps, lam=1.0)
    assert len(S) == 5
    assert all(js.is_anchor(i) for i in S), S

    rows = jlens.mechanistic_eval(insp, user, S, y, competitors=comps)
    h1 = next(r for r in rows if r["tap"] == "h1")
    assert h1["delta_beh"] > 4.0
    assert h1["mr"] < 0.1


def test_greedy_attack_lam_zero_is_pure_del(insp, user):
    """lam=0 optimizes DEL only; on the synthetic model both pathways give
    ~5, so it must reach at least the genre explanation's effect."""
    import jlens_adversarial as ja
    y, comps = js.target_and_competitors(insp, user)
    S = ja.greedy_attack(insp, user, y, k=5, tap="h1",
                         competitors=comps, lam=0.0)
    rows = jlens.mechanistic_eval(insp, user, S, y, competitors=comps)
    h1 = next(r for r in rows if r["tap"] == "h1")
    assert h1["delta_beh"] >= 4.9


def test_constraint_attack_max_del_under_mr_cap(insp, user):
    """max DEL s.t. MR <= tau on the synthetic model must find the anchors:
    the only high-DEL sets satisfying the mediation cap are gate sets."""
    import jlens_adversarial as ja
    y, comps = js.target_and_competitors(insp, user)
    S, info = ja.constraint_attack(insp, user, y, k=5, tap="h1",
                                   competitors=comps, mr_max=0.3, seed=0,
                                   return_info=True)
    assert info["feasible"]
    assert all(js.is_anchor(i) for i in S), S
    rows = jlens.mechanistic_eval(insp, user, S, y, competitors=comps)
    h1 = next(r for r in rows if r["tap"] == "h1")
    assert h1["delta_beh"] > 4.0
    assert h1["mr"] < 0.3
