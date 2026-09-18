"""
engine.symbolic
----------------

The Symbolic Engine subsystem (architecture document, Part V, Table 5.1):
derives vector fields, gradients, and total (chain-rule) derivatives
symbolically via SymPy, and compiles the results into fast NumPy-backed
callables via `sympy.lambdify`.

This module has exactly one scientific responsibility relevant to Part II
§2.6 of the architecture document: computing the total time derivative of
a scalar expression phi(x) along a vector field dx/dt = f(x, u; p),

    d(phi)/dt = sum_i  d(phi)/d(x_i) * f_i(x, u; p)

which is the "differentiate the residual" step (step 2) of the MRC
synthesis algorithm. Step 4 (solving the resulting equation for a control
input) is provided as a thin, explicit wrapper around `sympy.solve` rather
than folded in here, so that failure to solve (a genuine relative-degree
or invertibility problem, not a bug) is visible at the call site in
`control.mrc_synthesis`.
"""

from __future__ import annotations

import sympy
import numpy as np
from typing import Callable, Dict, List, Sequence, Tuple


def gradient(expr: sympy.Expr, x_syms: Sequence[sympy.Symbol]) -> List[sympy.Expr]:
    """d(expr)/d(x_i) for each x_i in x_syms."""
    return [sympy.diff(expr, xi) for xi in x_syms]


def total_derivative(
    expr: sympy.Expr,
    x_syms: Sequence[sympy.Symbol],
    xdot_exprs: Sequence[sympy.Expr],
) -> sympy.Expr:
    """
    Total (chain-rule) time derivative of a scalar expression phi(x) along
    trajectories of dx/dt = xdot_exprs(x, u; p):

        d(phi)/dt = grad(phi) . xdot_exprs

    This is exactly MRC synthesis step 2 (architecture document §1.7,
    §2.6): if phi is a function of x only (the usual case for a manifold
    constraint) but xdot_exprs itself depends on a control input u, the
    result correctly depends on u through the chain rule -- no separate
    "does phi depend on u directly" case is needed, since manifold
    constraints are defined purely on the state in this framework
    (architecture document §1.5).
    """
    if len(x_syms) != len(xdot_exprs):
        raise ValueError(
            f"x_syms has {len(x_syms)} entries but xdot_exprs has {len(xdot_exprs)}; "
            "they must be in the same state order."
        )
    grad = gradient(expr, x_syms)
    return sympy.expand(sum(g * xd for g, xd in zip(grad, xdot_exprs)))


def jacobian_matrix(
    exprs: Sequence[sympy.Expr], x_syms: Sequence[sympy.Symbol]
) -> sympy.Matrix:
    """Symbolic Jacobian d(exprs_i)/d(x_syms_j), as a SymPy Matrix."""
    return sympy.Matrix(exprs).jacobian(sympy.Matrix(list(x_syms)))


def substitute_params(expr, p_syms: Dict[str, sympy.Symbol], p_values: Dict[str, float]):
    """Substitute numeric parameter values into a symbolic expression (or list/Matrix of them)."""
    subs = {p_syms[k]: v for k, v in p_values.items() if k in p_syms}
    if isinstance(expr, (list, tuple)):
        return [e.subs(subs) if hasattr(e, "subs") else e for e in expr]
    return expr.subs(subs)


def lambdify_scalar(
    x_syms: Sequence[sympy.Symbol],
    u_syms: Sequence[sympy.Symbol],
    expr: sympy.Expr,
) -> Callable[[np.ndarray, np.ndarray], float]:
    """
    Compile a scalar SymPy expression (with parameters already substituted
    numerically, see `substitute_params`) into f(x_array, u_array) -> float.
    """
    all_syms = list(x_syms) + list(u_syms)
    fn = sympy.lambdify(all_syms, expr, modules="numpy")

    def wrapped(x: np.ndarray, u: np.ndarray) -> float:
        args = list(np.asarray(x, dtype=float)) + list(np.asarray(u, dtype=float))
        return float(fn(*args))

    return wrapped


def lambdify_vector(
    x_syms: Sequence[sympy.Symbol],
    u_syms: Sequence[sympy.Symbol],
    exprs: Sequence[sympy.Expr],
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """
    Compile a list of SymPy expressions (parameters already substituted
    numerically) into f(x_array, u_array) -> np.ndarray, in the given order.
    """
    all_syms = list(x_syms) + list(u_syms)
    fns = [sympy.lambdify(all_syms, e, modules="numpy") for e in exprs]

    def wrapped(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        args = list(np.asarray(x, dtype=float)) + list(np.asarray(u, dtype=float))
        return np.array([float(fn(*args)) for fn in fns])

    return wrapped
