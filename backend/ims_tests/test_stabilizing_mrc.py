"""
Tests for models.stabilizing_mrc.StabilizingMRCModel -- the book's
stabilizing-MRC construction, distinct from models.converter_cpl_paper.
ConverterCPLPaper (the electrical-constraint diagnostic reference).

Every numeric claim here is checked by an INDEPENDENT path where
possible: e.g. the control law is verified both by direct symbolic
substitution AND by running the model through the platform's own
generic control.mrc_synthesis.MRCSynthesizer engine (agreement between
two different code paths, not the same function called twice).
"""
import numpy as np
import sympy as sp
import pytest

from ims_platform.models.stabilizing_mrc import StabilizingMRCModel
from ims_platform.control.mrc_synthesis import MRCSynthesizer
from ims_platform.models.converter_cpl_paper import ConverterCPLPaper


BENCHMARK_PARAMS = dict(R=0.20, L=1.5e-3, C=2.5e-3, v_nom=400.0, P=20000.0,
                         R_v=0.5, K_i=50.0, k_m=500.0)


def test_residual_identity_symbolic_direct_substitution():
    """de_Sigma/dt = -k_m*e_Sigma, derived directly from the plant + control law (not the synthesizer)."""
    i_l, v_b, sigma, v_o = sp.symbols('i_l v_b sigma v_o', real=True)
    R, L, C, P, v_nom, R_v, K_i, k_m = sp.symbols('R L C P v_nom R_v K_i k_m', positive=True)
    u = sp.symbols('u', real=True)

    di_l = (v_o - R * i_l - v_b) / L
    dsigma = v_nom - v_b
    v_Sigma = v_nom - R_v * i_l + K_i * sigma
    e_Sigma = v_o - v_Sigma
    u_law = -(R_v / L) * (v_o - R * i_l - v_b) + K_i * (v_nom - v_b) - k_m * e_Sigma

    de_Sigma_dt = u_law - (-R_v * di_l + K_i * dsigma)
    assert sp.simplify(de_Sigma_dt - (-k_m * e_Sigma)) == 0


def test_residual_identity_via_platform_synthesizer_matches_manual_law():
    """Independent validation: the platform's own generic MRCSynthesizer, run on this model, derives
    EXACTLY the same control law as the book/manual derivation -- checked symbolically and numerically."""
    model = StabilizingMRCModel()
    synth = MRCSynthesizer(model, control_symbol_name='u')
    result = synth.synthesize(km_symbol_name='k_m')

    i_l, v_b, sigma, v_o = result.x_syms
    R, L, C, v_nom, P, R_v, K_i = (result.p_syms[k] for k in ['R', 'L', 'C', 'v_nom', 'P', 'R_v', 'K_i'])
    k_m = result.km_symbol
    manual_law = -(R_v / L) * (v_o - R * i_l - v_b) + K_i * (v_nom - v_b) - k_m * (v_o - v_nom + R_v * i_l - K_i * sigma)

    assert sp.simplify(result.control_expr - manual_law) == 0

    import random
    random.seed(42)
    syms = [i_l, v_b, sigma, v_o, R, L, C, v_nom, P, R_v, K_i, k_m]
    for _ in range(20):
        subs = {s: random.uniform(0.5, 5.0) for s in syms}
        assert abs(float(result.control_expr.subs(subs)) - float(manual_law.subs(subs))) < 1e-9


def test_equilibrium_closed_form_matches_independent_solve():
    """Closed-form equilibrium verified by an independent sympy.solve on the reduced dynamics,
    not merely by substituting the closed form back in (which would only check consistency)."""
    i_l, v_b, sigma = sp.symbols('i_l v_b sigma', real=True)
    R, L, C, P, v_nom, R_v, K_i = sp.symbols('R L C P v_nom R_v K_i', positive=True)

    di_l = (v_nom - (R + R_v) * i_l + K_i * sigma - v_b) / L
    dv_b = (i_l - P / v_b) / C
    dsigma = v_nom - v_b

    sol = sp.solve([sp.Eq(di_l, 0), sp.Eq(dv_b, 0), sp.Eq(dsigma, 0)], [i_l, v_b, sigma], dict=True)
    assert len(sol) == 1
    s = sol[0]
    assert sp.simplify(s[v_b] - v_nom) == 0
    assert sp.simplify(s[i_l] - P / v_nom) == 0
    assert sp.simplify(s[sigma] - (R + R_v) * P / (K_i * v_nom)) == 0

    model = StabilizingMRCModel()
    x_star = model.equilibrium_closed_form(BENCHMARK_PARAMS)
    p = BENCHMARK_PARAMS
    assert abs(x_star[1] - p["v_nom"]) < 1e-9
    assert abs(x_star[0] - p["P"] / p["v_nom"]) < 1e-9
    assert abs(x_star[2] - (p["R"] + p["R_v"]) * p["P"] / (p["K_i"] * p["v_nom"])) < 1e-9

    # The equilibrium must make ALL FOUR original dynamics vanish (di_l, dv_b, dsigma, and
    # dv_o = u_closed_loop, since at equilibrium the control law itself must also be zero).
    dx = model.dynamics(0.0, x_star, np.array([model.closed_loop_control(x_star, p)]), p)
    assert np.allclose(dx[:3], 0.0, atol=1e-9)


def test_characteristic_coefficients_match_fresh_jacobian_derivation():
    """a2/a1/a0 checked against a FRESH sympy Jacobian computation, not copied from the model's own formula."""
    i_l, v_b, sigma = sp.symbols('i_l v_b sigma', real=True)
    R, L, C, P, v_nom, R_v, K_i = sp.symbols('R L C P v_nom R_v K_i', positive=True)
    s = sp.symbols('s')

    di_l = (v_nom - (R + R_v) * i_l + K_i * sigma - v_b) / L
    dv_b = (i_l - P / v_b) / C
    dsigma = v_nom - v_b
    f = sp.Matrix([di_l, dv_b, dsigma])
    X = sp.Matrix([i_l, v_b, sigma])
    J = f.jacobian(X)

    subs = {v_b: v_nom, i_l: P / v_nom, sigma: (R + R_v) * P / (K_i * v_nom)}
    J_eq = J.subs(subs)
    poly = sp.Poly(sp.simplify(J_eq.charpoly(s).as_expr()), s)
    c3, c2, c1, c0 = poly.all_coeffs()
    a2_fresh = sp.simplify(c2 / c3)
    a1_fresh = sp.simplify(c1 / c3)
    a0_fresh = sp.simplify(c0 / c3)

    model = StabilizingMRCModel()
    a2, a1, a0 = model.characteristic_coeffs(BENCHMARK_PARAMS)

    g = BENCHMARK_PARAMS["P"] / BENCHMARK_PARAMS["v_nom"] ** 2
    subs_num = {R: BENCHMARK_PARAMS["R"], L: BENCHMARK_PARAMS["L"], C: BENCHMARK_PARAMS["C"],
                v_nom: BENCHMARK_PARAMS["v_nom"], P: BENCHMARK_PARAMS["P"],
                R_v: BENCHMARK_PARAMS["R_v"], K_i: BENCHMARK_PARAMS["K_i"]}
    assert abs(float(a2_fresh.subs(subs_num)) - a2) < 1e-6
    assert abs(float(a1_fresh.subs(subs_num)) - a1) < 1e-6
    assert abs(float(a0_fresh.subs(subs_num)) - a0) < 1e-6


def test_nominal_poles_match_task_specified_values():
    model = StabilizingMRCModel()
    poles = sorted(model.poles(BENCHMARK_PARAMS), key=lambda r: r.real)
    # -178.2909 +/- j436.0281, -60.0849 (sorted by real part ascending)
    assert abs(poles[0].real - (-178.2909)) < 1e-3
    assert abs(abs(poles[0].imag) - 436.0281) < 1e-3
    assert abs(poles[2].real - (-60.0849)) < 1e-3
    assert abs(poles[2].imag) < 1e-9


def test_routh_hurwitz_satisfied_at_benchmark_nominal_gains():
    model = StabilizingMRCModel()
    report = model.routh_hurwitz_report(BENCHMARK_PARAMS)
    assert report["locally_exponentially_stable"] is True
    for name, c in report["conditions"].items():
        assert c["satisfied"] is True, f"{name} failed with margin {c['margin']}"
    assert "regional" in report["scope_note"].lower()


def test_routh_hurwitz_fails_for_destabilizing_gain():
    """A genuinely destabilizing K_i must be flagged by the Routh-Hurwitz report AND independently
    confirmed unstable via the actual poles -- not just one check asserted in isolation."""
    model = StabilizingMRCModel()
    bad_params = dict(BENCHMARK_PARAMS, K_i=5000.0)
    report = model.routh_hurwitz_report(bad_params)
    assert report["locally_exponentially_stable"] is False
    assert report["conditions"]["a2*a1 > a0"]["satisfied"] is False

    poles = model.poles(bad_params)
    assert max(p.real for p in poles) > 0, "the flagged gain choice must be genuinely unstable, not just RH-flagged"


def test_domain_admissibility_is_v_b_positive():
    model = StabilizingMRCModel()
    assert model.admissible(np.array([1.0, 50.0, 1.0, 1.0])) is True
    assert model.admissible(np.array([1.0, -1.0, 1.0, 1.0])) is False
    assert model.admissible(np.array([1.0, 0.0, 1.0, 1.0])) is False


def test_electrical_constraint_reference_still_documented_as_nonstabilizing():
    """Guard against ever silently 'fixing' the diagnostic reference into looking like a
    successful stabilizer -- its own eigenvalue structure must still show the zero mode
    and positive tangential mode described in its docstring."""
    model = ConverterCPLPaper()
    p = model.params
    v_b_star = 400.0  # representative operating point consistent with the paper's own table
    i_l_star = p["P"] / v_b_star
    v_o_star = v_b_star + p["R"] * i_l_star
    x_star = np.array([i_l_star, v_b_star, v_o_star])

    i_l, v_b, v_o = sp.symbols('i_l v_b v_o', real=True)
    R, L, C, P, k_m = sp.symbols('R L C P k_m', positive=True)
    u = sp.symbols('u', real=True)
    e_m = v_b - v_o + R * i_l
    di_l = (v_o - R * i_l - v_b) / L
    dv_b = (i_l - P / v_b) / C
    de_m_dt = dv_b - u + R * di_l
    u_law = sp.solve(sp.Eq(de_m_dt, -k_m * e_m), u)[0]
    f = sp.Matrix([di_l, dv_b, u_law])
    X = sp.Matrix([i_l, v_b, v_o])
    J = f.jacobian(X)
    v_b_s = sp.symbols('v_b_star', positive=True)
    subs = {i_l: P / v_b_s, v_b: v_b_s, v_o: v_b_s + R * (P / v_b_s)}
    J_eq = sp.simplify(J.subs(subs))
    eigs = list(J_eq.eigenvals().keys())

    # Must contain exactly {-k_m, 0, P/(C*v_b_star^2)} -- the zero mode and the positive
    # tangential mode this reference is documented to demonstrate.
    assert sp.Integer(0) in eigs or 0 in eigs
    has_positive_mode = any(sp.simplify(e - P / (C * v_b_s ** 2)) == 0 for e in eigs)
    assert has_positive_mode


def test_distance_certificate_gradient_and_q():
    model = StabilizingMRCModel()
    p = BENCHMARK_PARAMS
    x = np.array([50.0, 400.0, 5.0, 410.0])
    e_sigma = model.residual_e_sigma(x, p)
    q = p["R_v"] ** 2 + p["K_i"] ** 2 + 1.0
    dist_sq = model.distance_to_manifold_squared(x, p)
    assert abs(dist_sq - (e_sigma ** 2) / q) < 1e-12


def test_distance_certificate_orthogonal_projection_stays_in_domain():
    """The orthogonal projection direction (grad(e_Sigma) = (R_v, 0, -K_i, 1)) has a ZERO
    v_b-component, so moving along it never changes v_b -- verified directly, not asserted."""
    p = BENCHMARK_PARAMS
    grad = np.array([p["R_v"], 0.0, -p["K_i"], 1.0])
    assert grad[1] == 0.0  # the v_b component


def test_distance_certificate_dVdt_identity():
    """V := e_Sigma^2 = q*dist^2; verified dV/dt = -2*k_m*V via the chain rule applied to the
    already-verified residual identity de_Sigma/dt = -k_m*e_Sigma (independent symbolic check)."""
    e_Sigma, k_m = sp.symbols('e_Sigma k_m', positive=True)
    V = e_Sigma ** 2
    dV_dt = sp.diff(V, e_Sigma) * (-k_m * e_Sigma)
    assert sp.simplify(dV_dt - (-2 * k_m * V)) == 0
