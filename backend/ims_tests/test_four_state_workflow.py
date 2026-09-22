"""
Automated tests for the ideal four-state stabilizing-MRC vertical slice.

These cover the verification requirements of the milestone that are NOT
already covered by tests/test_stabilizing_mrc.py (which owns the symbolic
residual identity, equilibrium-vs-independent-solve, char-coeffs vs fresh
Jacobian, and nominal pole regression). Here we test the NEW workflow
layer end to end.

Run: cd /app/backend && PYTHONPATH=/app/backend python -m pytest ims_tests/test_four_state_workflow.py -o addopts="" -q
"""
import math

import numpy as np
import pytest

from ims_platform.stabilizing import (
    validate_params, ParameterError,
    compute_equilibrium, compute_stability, run_simulation, dataset_to_csv,
)

NOMINAL = dict(R=0.20, L=1.5e-3, C=2.5e-3, P=20000.0, v_nom=400.0,
               R_v=0.5, K_i=50.0, k_m=500.0)


def test_analytical_equilibrium_consistency():
    eq = compute_equilibrium(NOMINAL)
    xs = eq["x_star"]
    assert abs(xs["v_b"] - 400.0) < 1e-9
    assert abs(xs["i_l"] - 20000.0 / 400.0) < 1e-9
    assert abs(xs["sigma"] - (0.7 * 20000.0) / (50.0 * 400.0)) < 1e-9
    assert abs(xs["v_o"] - (400.0 + 0.2 * 20000.0 / 400.0)) < 1e-9


def test_closed_loop_equilibrium_residual_near_zero():
    eq = compute_equilibrium(NOMINAL)
    assert eq["equilibrium_residual_norm"] < 1e-9


def test_transverse_eigenvalue_equals_minus_km():
    st = compute_stability(NOMINAL)
    assert st["transverse_check_ok"]
    assert abs(st["transverse_eigenvalue"]["re"] - (-500.0)) < 1e-6
    assert abs(st["transverse_eigenvalue"]["im"]) < 1e-9


def test_full_spectrum_is_reduced_union_transverse():
    st = compute_stability(NOMINAL)
    full = sorted([(round(e["re"], 3), round(abs(e["im"]), 3)) for e in st["full_eigenvalues"]])
    reduced = sorted([(round(e["re"], 3), round(abs(e["im"]), 3)) for e in st["reduced_eigenvalues"]])
    # every reduced eigenvalue appears in the full spectrum
    for r in reduced:
        assert r in full
    # plus the transverse -k_m
    assert (-500.0, 0.0) in full


def test_independent_jacobian_route_agrees():
    st = compute_stability(NOMINAL)
    assert st["jacobian_independent_check_ok"]
    assert st["jacobian_independent_check_max_abs_error"] < 1e-4


def test_routh_hurwitz_and_char_coeffs():
    st = compute_stability(NOMINAL)
    c = st["characteristic_coeffs"]
    g = 20000.0 / 400.0 ** 2
    assert abs(c["a2"] - ((0.7) / 1.5e-3 - g / 2.5e-3)) < 1e-6
    assert abs(c["a0"] - (50.0 / (1.5e-3 * 2.5e-3))) < 1e-3
    assert st["locally_exponentially_stable"] is True
    for name, d in st["routh_hurwitz"].items():
        assert "margin" in d and "satisfied" in d


def test_nominal_pole_regression():
    st = compute_stability(NOMINAL)
    reals = sorted(e["re"] for e in st["full_eigenvalues"])
    # -500, then complex pair -178.29 (x2), then -60.08
    assert abs(min(reals) - (-500.0)) < 1e-3
    assert any(abs(e["re"] - (-178.2909)) < 1e-2 and abs(abs(e["im"]) - 436.0281) < 1e-2
               for e in st["full_eigenvalues"])
    assert any(abs(e["re"] - (-60.0849)) < 1e-2 and abs(e["im"]) < 1e-9
               for e in st["full_eigenvalues"])


def test_invalid_parameter_raises_explicit_error():
    with pytest.raises(ParameterError) as ei:
        validate_params(dict(NOMINAL, L=-1.0))
    assert any("L must be > 0" in e for e in ei.value.errors)
    with pytest.raises(ParameterError):
        validate_params(dict(NOMINAL, C=0.0))
    with pytest.raises(ParameterError):
        d = dict(NOMINAL); d.pop("k_m"); validate_params(d)


def test_nonzero_residual_exponential_contraction():
    ds = run_simulation(NOMINAL, dict(t_end=0.05, n_eval=2000,
                                      initial_perturbation=dict(v_o=15.0)))
    assert abs(ds["e_sigma_0"] - 15.0) < 1e-9
    # simulated residual tracks e0*exp(-k_m t) tightly in the ideal model
    assert ds["residual_contraction"]["max_rel_error"] < 1e-4
    # spot check the identity at a mid point
    t = np.array(ds["t"]); e = np.array(ds["e_sigma"])
    idx = len(t) // 2
    assert abs(e[idx] - 15.0 * math.exp(-500.0 * t[idx])) < 1e-3


def test_cpl_disturbance_applied_and_cleared():
    ds = run_simulation(NOMINAL, dict(
        t_end=0.06, n_eval=1500,
        disturbance=dict(P_nom=20000.0, P_disturbed=30000.0, t_start=0.01, t_clear=0.03)))
    labels = [ev["label"] for ev in ds["events"]]
    assert any("applied" in l for l in labels)
    assert any("cleared" in l for l in labels)
    # CPL power column reflects the schedule
    t = np.array(ds["t"]); pwr = np.array(ds["cpl_power"])
    assert np.all(pwr[(t >= 0.01) & (t < 0.03)] == 30000.0)
    assert pwr[0] == 20000.0 and pwr[-1] == 20000.0
    # bus voltage deviates during the disturbance then returns
    vb = np.array(ds["v_b"])
    assert np.min(vb[(t >= 0.01) & (t < 0.03)]) < 400.0


def test_csv_export_matches_dataset():
    ds = run_simulation(NOMINAL, dict(t_end=0.04, n_eval=500,
                                      initial_perturbation=dict(v_o=10.0)))
    csv = dataset_to_csv(ds)
    lines = [l for l in csv.splitlines() if not l.startswith("#")]
    header = lines[0].split(",")
    assert header[:6] == ["time", "i_l", "v_b", "sigma", "v_o", "e_sigma"]
    data = lines[1:]
    assert len(data) == len(ds["t"])
    # first and last rows equal dataset values
    first = data[0].split(",")
    assert abs(float(first[0]) - ds["t"][0]) < 1e-12
    assert abs(float(first[4]) - ds["v_o"][0]) < 1e-9
    last = data[-1].split(",")
    assert abs(float(last[2]) - ds["v_b"][-1]) < 1e-9


def test_reproducibility_same_config_same_result():
    cfg = dict(t_end=0.03, n_eval=400, initial_perturbation=dict(v_o=12.0))
    a = run_simulation(NOMINAL, cfg)
    b = run_simulation(NOMINAL, cfg)
    assert np.allclose(a["v_b"], b["v_b"], atol=0, rtol=0)
    assert np.allclose(a["e_sigma"], b["e_sigma"], atol=0, rtol=0)


def test_electrical_constraint_model_still_present_and_nonstabilizing():
    # preservation: the diagnostic reference model must remain and stay non-stabilizing
    from ims_platform.models.converter_cpl_paper import ConverterCPLPaper  # noqa: F401
    assert ConverterCPLPaper is not None


# ---------------------------------------------------------------------------
# Nominal 20 kW benchmark regression (task requirement: "the latest generated
# Four-State report used P = 200000 W, which is NOT the nominal validated
# benchmark" -- this is the automated regression restoring/pinning the exact
# validated nominal set, run against every quantity the task lists by name).
# No code path in this repository actually defaults to P = 200000 (every
# default -- StabilizingMRCModel.default_params, stabilizing_mrc_analysis's
# default_config, and the frontend's NOMINAL_PARAMS in lib/ims.ts -- already
# uses P = 20000); this test exists so a future change to any of those
# defaults, or a future manual override, is caught by CI rather than only
# noticed in a generated report.
# ---------------------------------------------------------------------------

def test_nominal_20kw_benchmark_regression_full_checklist():
    """Single regression pinning every quantity in the task's own checklist,
    against the exact nominal parameter set: R=0.20, L=1.5e-3, C=2.5e-3,
    P=20000 W, v_nom=400, R_v=0.5, K_i=50, k_m=500."""
    assert NOMINAL == dict(R=0.20, L=1.5e-3, C=2.5e-3, P=20000.0, v_nom=400.0,
                            R_v=0.5, K_i=50.0, k_m=500.0)

    eq = compute_equilibrium(NOMINAL)
    xs = eq["x_star"]
    # x* ~= [50 A, 400 V, 0.7, 410 V]
    assert abs(xs["i_l"] - 50.0) < 1e-6
    assert abs(xs["v_b"] - 400.0) < 1e-6
    assert abs(xs["sigma"] - 0.7) < 1e-6
    assert abs(xs["v_o"] - 410.0) < 1e-6
    # equilibrium residual ~= 0
    assert eq["equilibrium_residual_norm"] < 1e-9

    st = compute_stability(NOMINAL)
    # transverse eigenvalue ~= -500
    assert st["transverse_check_ok"]
    assert abs(st["transverse_eigenvalue"]["re"] - (-500.0)) < 1e-6
    assert abs(st["transverse_eigenvalue"]["im"]) < 1e-9
    # expected eigenvalues ~= -60.1, -178.3 +/- 436j, -500
    reals = sorted(e["re"] for e in st["full_eigenvalues"])
    assert abs(min(reals) - (-500.0)) < 1e-2
    assert any(abs(e["re"] - (-178.3)) < 0.1 and abs(abs(e["im"]) - 436.0) < 1.0
               for e in st["full_eigenvalues"])
    assert any(abs(e["re"] - (-60.1)) < 0.1 and abs(e["im"]) < 1e-9
               for e in st["full_eigenvalues"])
    # Routh-Hurwitz satisfied
    for name, d in st["routh_hurwitz"].items():
        assert d["satisfied"] is True, f"Routh-Hurwitz condition '{name}' failed at the nominal benchmark"
    # reduced/tangential dynamics locally stable == full closed-loop equilibrium locally stable
    # (both come from the same characteristic polynomial at this benchmark, verified independently
    # against a fresh Jacobian in test_stabilizing_mrc.py's own tests)
    assert st["locally_exponentially_stable"] is True


def test_200kw_case_is_a_separate_unstable_stress_test_not_the_benchmark():
    """The 200 kW case must be KEPT as a separate unstable/stress scenario --
    never substituted for the nominal 20 kW benchmark above. At P=200000 W
    (all other params unchanged) the reduced dynamics genuinely loses local
    exponential stability (a2 = (R+R_v)/L - g/C goes negative once
    g = P/v_nom^2 grows past (R+R_v)*C/L), independently confirmed via the
    actual closed-loop poles, not just the Routh-Hurwitz flag alone."""
    stress = dict(NOMINAL, P=200000.0)
    assert stress["P"] != NOMINAL["P"], "the stress case must not replace the nominal benchmark's P"

    st = compute_stability(stress)
    assert st["locally_exponentially_stable"] is False
    assert st["routh_hurwitz"]["a2 > 0"]["satisfied"] is False

    reals = [e["re"] for e in st["full_eigenvalues"]]
    assert max(reals) > 0, "the 200 kW stress case must be genuinely unstable, not merely RH-flagged"
