"""
control.mrc_synthesis
----------------------

Genuine IMS-native Manifold-Reshaping Control synthesis (architecture
document Part I §1.7, Part II §2.6). This module implements the four-step
algorithm directly -- it is *not* a scheduled conventional controller
(compare `control.mrc.ScheduledLQRControl`, a manifold-scheduled LQR
design that belongs to the Conventional Control Library, not here):

    1. Identify the manifold constraint  phi(x), M = {x : phi(x) = 0}
    2. Differentiate the residual        e_dot = d(phi)/dt along f(x,u;p)
    3. Impose the desired transverse dynamics   e_dot = -k_m * phi
    4. Solve algebraically for the control input u that satisfies (3)

`MRCSynthesizer` is entirely model-independent: it operates on whatever
symbolic dynamics and manifold constraint a `DynamicalSystem` subclass
supplies, and has no knowledge of, or dependency on, any specific
converter topology. models.converter_cpl_paper.ConverterCPLPaper is one
validation example used to check this generic engine's output against an
independently published result (tests/test_mrc_synthesis.py); it is a
test fixture, not a component the engine is built around, and any
questions about that particular example's own closed-loop numerics
(tracked in that model's docstring) are questions about the validation
example, not about this module.

The output is a single closed-form control law u = kappa(x), derived once
per model (symbolically), not a numerical scheme evaluated at run time --
this is what makes it "IMS-native" rather than a numerical approximation:
no linearisation, no optimisation, no Riccati equation appears anywhere
in this module.
"""

from __future__ import annotations

import sympy
import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from ..core.system import DynamicalSystem
from ..engine.symbolic import total_derivative, substitute_params, lambdify_scalar


class MRCSynthesisError(Exception):
    """
    Raised when the four-step algorithm cannot produce a closed-form
    control law for a given model -- most commonly because the chosen
    control input has relative degree greater than one with respect to
    the manifold residual (d(e_dot)/du = 0), which step 4 requires to be
    nonzero. This is a genuine structural property of the model, not a
    numerical failure, and is reported as such rather than silently
    falling back to a numerical approximation.
    """


@dataclass
class MRCSynthesisResult:
    """The output of one MRC synthesis run: a derived, closed-form control law."""
    manifold_constraint: sympy.Expr          # phi(x)
    open_loop_residual_dynamics: sympy.Expr  # e_dot with u=0, i.e. the uncontrolled transverse dynamics
    control_expr: sympy.Expr                 # symbolic solution u = kappa(x; k_m)
    control_symbol: sympy.Symbol             # which input symbol was solved for
    km_symbol: sympy.Symbol                  # the symbolic contraction gain
    x_syms: tuple
    u_syms: tuple
    p_syms: dict
    _numeric_law: Optional[Callable] = None  # populated by `compile_numeric`

    def as_latex(self) -> str:
        """Render the derived control law as a LaTeX string, for direct inclusion in documentation."""
        return sympy.latex(sympy.Eq(self.control_symbol, self.control_expr))

    def compile_numeric(self, p_values: Dict[str, float], km_value: float) -> Callable[[np.ndarray], float]:
        """
        Substitute numeric parameter values and a numeric contraction gain
        into the symbolic control law, and compile it into a fast
        callable u = kappa(x) suitable for use as a Simulator controller.
        """
        expr = substitute_params(self.control_expr, self.p_syms, p_values)
        expr = expr.subs({self.km_symbol: km_value})
        other_u = [u for u in self.u_syms if u != self.control_symbol]
        fn = lambdify_scalar(self.x_syms, other_u, expr)
        self._numeric_law = fn
        return fn


class MRCSynthesizer:
    """
    Synthesizes a Manifold-Reshaping Control law for a `DynamicalSystem`
    that provides `symbolic_dynamics` and `symbolic_manifold_constraint`
    (core.system.DynamicalSystem's optional symbolic contract).
    """

    def __init__(self, component: DynamicalSystem, control_symbol_name: Optional[str] = None):
        """
        Parameters
        ----------
        component : DynamicalSystem
            Must implement `symbolic_symbols`, `symbolic_dynamics`, and
            `symbolic_manifold_constraint`.
        control_symbol_name : str, optional
            Name of the input to solve for (must match one of
            `component.input_names`). Defaults to the first input.
        """
        self.component = component
        self.control_symbol_name = control_symbol_name or component.input_names[0]

    def synthesize(self, km_symbol_name: str = "k_m") -> MRCSynthesisResult:
        """Run the four-step algorithm and return the derived control law (still symbolic in k_m)."""
        x_syms, u_syms, p_syms = self.component.symbolic_symbols()

        # Step 1: identify the manifold constraint.
        phi = self.component.symbolic_manifold_constraint(x_syms, u_syms, p_syms)

        # Step 2: differentiate the residual along the full nonlinear dynamics.
        f = self.component.symbolic_dynamics(x_syms, u_syms, p_syms)
        e_dot = total_derivative(phi, x_syms, f)

        name_to_sym = {str(u): u for u in u_syms}
        if self.control_symbol_name not in name_to_sym:
            raise MRCSynthesisError(
                f"control_symbol_name='{self.control_symbol_name}' is not one of this "
                f"model's inputs {list(name_to_sym)}."
            )
        u_target = name_to_sym[self.control_symbol_name]

        # Relative-degree check: the control input must appear in e_dot for
        # step 4 to be solvable in closed form (architecture document §1.7,
        # "whenever the control input enters e_dot with a nonzero,
        # invertible coefficient").
        open_loop_e_dot = e_dot.subs({u_target: 0})
        coefficient = sympy.diff(e_dot, u_target)
        if sympy.simplify(coefficient) == 0:
            raise MRCSynthesisError(
                f"'{self.control_symbol_name}' has relative degree greater than one with "
                f"respect to the manifold residual (d(e_dot)/d{self.control_symbol_name} = 0); "
                "MRC synthesis in closed form is not possible for this (constraint, input) pair. "
                "Choose a different control input, or a different manifold constraint."
            )

        # Step 3: impose the desired exponential transverse contraction.
        km = sympy.symbols(km_symbol_name, positive=True)
        target = -km * phi

        # Step 4: solve algebraically for the control law.
        solutions = sympy.solve(sympy.Eq(e_dot, target), u_target)
        if not solutions:
            raise MRCSynthesisError(
                f"sympy.solve could not isolate '{self.control_symbol_name}' from "
                f"e_dot = {e_dot} = {target}; the equation may be nonlinear in the "
                "control input in a way that has no closed-form solution."
            )
        control_expr = sympy.simplify(solutions[0])

        return MRCSynthesisResult(
            manifold_constraint=phi,
            open_loop_residual_dynamics=open_loop_e_dot,
            control_expr=control_expr,
            control_symbol=u_target,
            km_symbol=km,
            x_syms=x_syms,
            u_syms=u_syms,
            p_syms=p_syms,
        )
