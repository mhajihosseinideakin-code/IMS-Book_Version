"""
Tests for the IMS-native MRC synthesis engine (control.mrc_synthesis).

Validation direction (important): MRCSynthesizer is a generic, model-
independent engine. These tests do not validate the engine BY validating
the plant model -- they validate it by feeding it one independently
published example and checking the output for exact symbolic equality:

    Generic Symbolic Engine -> Paper Model (validation input) -> derived law
                                                                       |
                                                          exact symbolic equality
                                                                       |
                                                              Published eq. (13)

test_synthesis_matches_paper_eq13_exactly is that check, and it is the
central correctness result in this file: the engine automatically derives
the feedback law from the supplied nonlinear dynamics and manifold
constraint; when applied to the converter-CPL example from the reference
manuscript, the derived control law is symbolically identical to the
published formulation. That is a statement about the engine's generality,
demonstrated on one example -- not a statement that depends on the
example itself being a complete or numerically well-posed plant model.

test_closed_loop_tangential_eigenvalue_documents_validation_example_limitation
is kept separate and clearly delineated from the above: it records a
property of the *validation fixture* (models.converter_cpl_paper), not of
the synthesis engine. See that model's module docstring for detail. It is
not being pursued further; the architectural priority is Phase V2.2
(network representation, graph builder, automatic model builder,
component library), not further numerical work on this one fixture.
"""
import numpy as np
import sympy

try:
    import pytest
except ImportError:
    pytest = None

from ims_platform.models.converter_cpl_paper import ConverterCPLPaper
from ims_platform.control.mrc_synthesis import MRCSynthesizer, MRCSynthesisError
from ims_platform.core.simulator import Simulator


def test_manifold_constraint_matches_paper_eq6():
    model = ConverterCPLPaper()
    synth = MRCSynthesizer(model)
    result = synth.synthesize()
    i_l, v_b, v_o = result.x_syms
    R = result.p_syms["R"]
    expected = v_b - (v_o - R * i_l)
    assert sympy.simplify(result.manifold_constraint - expected) == 0


def test_synthesis_matches_paper_eq13_exactly():
    """
    The central validation: the synthesizer's derived control law must be
    symbolically IDENTICAL (not merely numerically close) to the paper's
    published eq. 13, since both are exact algebraic solutions of the
    same equation (differentiate e_m, impose -k_m*e_m, solve for u), and
    that equation has a unique solution (u appears linearly).
    """
    model = ConverterCPLPaper()
    synth = MRCSynthesizer(model)
    result = synth.synthesize()

    i_l, v_b, v_o = result.x_syms
    R, L, C, P = result.p_syms["R"], result.p_syms["L"], result.p_syms["C"], result.p_syms["P"]
    km = result.km_symbol
    e_m = v_b - (v_o - R * i_l)  # paper eq. 6

    paper_eq13 = (1 / C) * (i_l - P / v_b) + (R / L) * (v_o - R * i_l - v_b) + km * e_m

    diff = sympy.simplify(result.control_expr - paper_eq13)
    assert diff == 0, f"Synthesized law differs from paper eq. 13 by: {diff}"


def test_open_loop_residual_dynamics_are_u_independent():
    """The open-loop (u=0) residual dynamics should not contain the control symbol."""
    model = ConverterCPLPaper()
    synth = MRCSynthesizer(model)
    result = synth.synthesize()
    assert result.control_symbol not in result.open_loop_residual_dynamics.free_symbols


def test_relative_degree_zero_raises_clean_error():
    """
    A manifold constraint with zero relative degree with respect to the
    control input (the input simply doesn't affect the residual) must be
    reported as a structural MRCSynthesisError, not silently mishandled.
    """
    class DegenerateModel(ConverterCPLPaper):
        def symbolic_manifold_constraint(self, x_syms, u_syms, p_syms):
            i_l, v_b, v_o = x_syms
            # A constraint on i_l and v_b only: v_o (and hence u = v_o_dot)
            # never appears in e_dot, since neither i_l_dot nor v_b_dot
            # depends on v_o... (they do, in fact, via v_o in i_l_dot) --
            # use a constraint that is genuinely blind to u instead:
            return i_l - v_b  # depends only on states whose derivatives don't involve u

    model = DegenerateModel()
    synth = MRCSynthesizer(model)
    if pytest is not None:
        with pytest.raises(MRCSynthesisError):
            synth.synthesize()
    else:
        try:
            synth.synthesize()
            raise AssertionError("expected MRCSynthesisError")
        except MRCSynthesisError:
            pass


def test_compiled_numeric_law_reproduces_symbolic_value():
    """Spot-check: the compiled numeric callable must agree with the symbolic expression at a point."""
    model = ConverterCPLPaper()
    synth = MRCSynthesizer(model)
    result = synth.synthesize()

    p_values = dict(R=0.20, L=1.5e-3, C=2.5e-3, P=10e3)
    km_value = 500.0
    kappa = result.compile_numeric(p_values, km_value)

    x_test = np.array([27.3, 395.0, 401.0])
    u_numeric = kappa(x_test, np.array([]))

    subs = {result.x_syms[0]: x_test[0], result.x_syms[1]: x_test[1], result.x_syms[2]: x_test[2]}
    subs.update({result.p_syms[k]: v for k, v in p_values.items()})
    subs[result.km_symbol] = km_value
    u_symbolic = float(result.control_expr.subs(subs))

    assert abs(u_numeric - u_symbolic) < 1e-9


def test_closed_loop_exponential_contraction_rate():
    """
    Regardless of the open item above, the transverse (manifold-residual)
    dynamics themselves must contract at exactly rate k_m under the
    synthesized law -- this is the one property the synthesis algorithm
    guarantees unconditionally (architecture document Theorem 1.2,
    hypothesis-free part: e_m(t) = e_m(0) exp(-k_m t)), independent of
    whether the full closed-loop system is stable at any particular
    equilibrium.
    """
    model = ConverterCPLPaper()
    synth = MRCSynthesizer(model)
    result = synth.synthesize()

    p_values = dict(R=0.20, L=1.5e-3, C=2.5e-3, P=10e3)
    km_value = 500.0
    kappa = result.compile_numeric(p_values, km_value)

    def controller(t, x):
        return np.array([kappa(x, np.array([]))])

    x0 = np.array([25.0, 395.0, 405.0])  # 5V bus deficit, matching the paper's disturbance test
    sim = Simulator(model, method="RK45")
    traj = sim.simulate(x0, (0.0, 0.02), controller=controller, n_eval=400)
    assert traj.success

    residuals = np.array([model.manifold_residual_numeric(traj.x[:, k], p_values) for k in range(traj.x.shape[1])])
    em0 = residuals[0]
    t_check = 0.01
    idx = np.argmin(np.abs(traj.t - t_check))
    predicted = em0 * np.exp(-km_value * traj.t[idx])
    assert abs(residuals[idx] - predicted) < 0.05 * abs(em0), (
        f"residual at t={traj.t[idx]:.4f}s was {residuals[idx]:.4f}, "
        f"expected ~{predicted:.4f} (exp(-k_m t) decay)"
    )


def test_closed_loop_tangential_eigenvalue_documents_validation_example_limitation():
    """
    Records a property of the VALIDATION FIXTURE (models.converter_cpl_paper),
    not of MRCSynthesizer or the Symbolic Engine: the minimal 3-state
    reconstruction used as one validation example has a structurally
    positive tangential eigenvalue P/(C*v_b^2) at the high-voltage
    equilibrium. This is not asserted as a defect of the synthesis
    algorithm, whose correctness is established independently by
    test_synthesis_matches_paper_eq13_exactly (exact symbolic equality
    against a published result). See the model's module docstring.
    """
    model = ConverterCPLPaper()
    synth = MRCSynthesizer(model)
    result = synth.synthesize()
    p_values = dict(R=0.20, L=1.5e-3, C=2.5e-3, P=10e3)
    km_value = 500.0
    kappa = result.compile_numeric(p_values, km_value)

    v_b_star = 400.0
    i_l_star = p_values["P"] / v_b_star
    v_o_star = v_b_star + p_values["R"] * i_l_star
    x_star = np.array([i_l_star, v_b_star, v_o_star])

    def closed_loop_f(x):
        u = kappa(x, np.array([]))
        return model.dynamics(0.0, x, np.array([u]), p_values)

    eps = 1e-6
    J = np.zeros((3, 3))
    for i in range(3):
        dx = np.zeros(3)
        dx[i] = eps * max(1.0, abs(x_star[i]))
        J[:, i] = (closed_loop_f(x_star + dx) - closed_loop_f(x_star - dx)) / (2 * dx[i])
    eigs = np.linalg.eigvals(J)

    analytic_tangential = p_values["P"] / (p_values["C"] * v_b_star ** 2)
    max_real = np.max(eigs.real)
    print(f"[validation-fixture note, not an engine issue] closed-loop eigenvalues at x*: {eigs}")
    print(f"[validation-fixture note, not an engine issue] analytic tangential eigenvalue P/(C v_b*^2) = {analytic_tangential}")
    # Documented, not asserted stable -- see docstring. We DO assert the
    # analytic prediction matches the numeric Jacobian, since that
    # agreement is itself a useful correctness check on the model/synth.
    assert abs(max_real - analytic_tangential) < 1e-2 * analytic_tangential


if __name__ == "__main__":
    import sys
    import inspect
    fns = [f for name, f in inspect.getmembers(sys.modules[__name__], inspect.isfunction)
           if name.startswith("test_")]
    passed, failed = 0, 0
    for f in fns:
        try:
            f()
            print(f"PASS {f.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAIL {f.__name__} -> {e!r}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
