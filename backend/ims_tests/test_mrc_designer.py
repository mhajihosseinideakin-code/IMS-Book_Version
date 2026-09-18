"""
Tests for the General MRC Designer (ims_platform.mrc_designer). The Four-State
Stabilising MRC is the exact reference/regression case for the generic path.

Run: cd /app/backend && PYTHONPATH=/app/backend \
     python -m pytest ims_tests/test_mrc_designer.py -o addopts="" -q
"""
import numpy as np
import pytest

from ims_platform.mrc_designer import (
    feasibility_report, synthesize_mrc, verify_closed_loop, before_after, MRCNotEstablished,
    control_affine_split, manifold_gradient,
)
from ims_platform.models.stabilizing_mrc import StabilizingMRCModel

P = StabilizingMRCModel().default_params()


def _sys():
    return StabilizingMRCModel(params=P)


def _sim():
    return dict(t_end=0.06, n_eval=1000, initial_perturbation={"v_o": 20.0},
                disturbance={"P_nom": P["P"], "P_disturbed": 1.5 * P["P"], "t_start": 0.01, "t_clear": 0.03})


def test_control_affine_split_is_analytic_for_reference():
    s = _sys(); x = s.equilibrium_closed_form(P)
    f, G, method = control_affine_split(s, x, P)
    assert method == "analytic"
    assert G.shape == (4, 1)
    # only dv_o/dt carries the input u -> G = [0,0,0,1]
    assert np.allclose(G[:, 0], [0, 0, 0, 1], atol=1e-9)


def test_feasibility_reports_A_rank_and_establishes_mrc():
    s = _sys(); x = s.equilibrium_closed_form(P)
    fr = feasibility_report(s, x, P)
    assert fr["dim_phi"] == 1 and fr["dim_u"] == 1
    assert fr["A"] == [[1.0]]
    assert fr["rank_A"] == 1 and fr["relative_degree_one"] is True
    assert fr["condition_number"] == pytest.approx(1.0, abs=1e-9)
    assert fr["G_method"] == "analytic" and fr["manifold_method"] == "analytic"
    assert fr["mrc_established"] is True
    assert fr["establishment_basis"] == "analytic"


def test_finite_difference_split_matches_analytic():
    s = _sys(); x = s.equilibrium_closed_form(P)
    fa, Ga, _ = control_affine_split(s, x, P, prefer_symbolic=True)
    fn, Gn, method = control_affine_split(s, x, P, prefer_symbolic=False)
    assert method == "finite_difference"
    assert np.allclose(Ga, Gn, atol=1e-5)
    assert np.allclose(fa, fn, atol=1e-4)


def test_generic_synthesis_reproduces_reference_law():
    # regression anchor: the model-independent engine reproduces the exact
    # four-state stabilising-MRC control law
    d = synthesize_mrc("stabilizing_mrc", P)
    law = d["_numeric_law"]
    s = _sys()
    for x in [np.array([55., 395., 0.72, 415.]), np.array([40., 405., 0.6, 405.])]:
        assert abs(law(x) - s.closed_loop_control(x, P)) < 1e-9
    assert d["manifold_roles"]["validated_controlled_invariant"] is True


def test_verify_reproduces_reference_benchmark():
    v = verify_closed_loop("stabilizing_mrc", P, _sim())
    assert v["equilibrium"]["residual_norm"] < 1e-9
    assert v["local_stability"]["transverse_check_ok"] is True
    # nominal pole benchmark unchanged
    reals = sorted(e["re"] for e in v["local_stability"]["full_eigenvalues"])
    assert abs(min(reals) - (-500.0)) < 1e-3
    assert v["contraction_evidence"]["max_rel_error"] < 1e-4
    # claim levels attached and distinct
    assert v["contraction_evidence"]["claim_level"] == "ideal_residual_contraction"
    assert v["recovery_status"]["claim_level"] == "observed_finite_horizon_recovery"


def test_before_after_uses_identical_conditions_and_reports_evidence():
    ba = before_after("stabilizing_mrc", P, _sim())
    assert ba["claim_level"] == "observed_finite_horizon_recovery"
    assert "not a certified regional ims guarantee" in ba["disclaimer"].lower()
    # identical initial condition for both runs
    assert ba["open_loop"]["t"][0] == ba["mrc"]["t"][0]
    assert len(ba["open_loop"]["t"]) == len(ba["mrc"]["t"])
    # MRC returns closer to equilibrium than the uncontrolled plant
    m = ba["metrics"]["final_state_deviation"]
    assert m["mrc"] < m["open_loop"]


def test_unsupported_model_is_not_established_not_invented():
    with pytest.raises(MRCNotEstablished):
        synthesize_mrc("arbitrary_network", P)


def test_manifold_gradient_analytic_for_reference():
    s = _sys(); x = s.equilibrium_closed_form(P)
    Dphi, phi, method, _ = manifold_gradient(s, x, P)
    assert method == "analytic"
    assert Dphi.shape == (1, 4)
    # d phi / d(i_l, v_b, sigma, v_o) = (R_v, 0, -K_i, 1)
    assert np.allclose(Dphi[0], [P["R_v"], 0.0, -P["K_i"], 1.0], atol=1e-9)
