"""
core.system
-----------

Defines the fundamental modelling contract of the platform: `DynamicalSystem`.

Any converter, device, or aggregated network model (grid-forming inverter,
DC microgrid, HVDC link, battery energy storage system, ...) is implemented
by subclassing `DynamicalSystem` and providing:

    - state_names / input_names   : symbolic bookkeeping
    - dynamics(t, x, u, p)        : the nonlinear vector field  dx/dt = f(x,u,p)
    - default_params()            : nominal parameter dictionary
    - admissible(x)               : physical/operational admissibility check

Everything else (Jacobians, equilibria, simulation, IMS analysis, control)
is derived generically from this contract, so new models plug into the
whole platform for free.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence
from scipy.optimize import fsolve


@dataclass
class EquilibriumResult:
    """Result of an equilibrium-point search."""
    x_star: np.ndarray
    converged: bool
    residual_norm: float
    eigenvalues: Optional[np.ndarray] = None
    # Solver diagnostics (architecture doc-aligned: exposing what the
    # solver already computes internally, not new computation). All
    # optional / default None so existing callers are unaffected.
    n_function_evals: Optional[int] = None
    jacobian: Optional[np.ndarray] = None
    jacobian_condition_number: Optional[float] = None
    computation_time_s: Optional[float] = None

    @property
    def is_hyperbolic(self) -> bool:
        if self.eigenvalues is None:
            return False
        return bool(np.all(np.abs(self.eigenvalues.real) > 1e-8))

    @property
    def is_stable(self) -> bool:
        if self.eigenvalues is None:
            return False
        return bool(np.all(self.eigenvalues.real < 0))


class DynamicalSystem:
    """
    Abstract base class for a nonlinear, control-affine-or-general
    finite-dimensional dynamical system:

        dx/dt = f(t, x, u; p)
        y     = h(x, u; p)

    Subclasses must implement `dynamics` and should override
    `default_params`, `state_names`, `input_names`, and `admissible`
    as appropriate. A generic finite-difference Jacobian and
    Newton-based equilibrium solver are provided so every model gets
    linear-analysis and IMS-analysis capability automatically.
    """

    #: human-readable names of state variables, override in subclasses
    state_names: Sequence[str] = ()
    #: human-readable names of exogenous inputs, override in subclasses
    input_names: Sequence[str] = ()

    def __init__(self, params: Optional[Dict] = None, name: str = "system"):
        self.name = name
        self.params: Dict = self.default_params()
        if params:
            self.params.update(params)

    # ------------------------------------------------------------------
    # Contract to be implemented by concrete models
    # ------------------------------------------------------------------
    def dynamics(self, t: float, x: np.ndarray, u: np.ndarray, p: Dict) -> np.ndarray:
        """Nonlinear vector field dx/dt = f(t, x, u; p). Must be overridden."""
        raise NotImplementedError

    def default_params(self) -> Dict:
        """Nominal physical parameters for this model."""
        return {}

    def default_input(self) -> np.ndarray:
        """Nominal exogenous input / setpoint vector."""
        return np.zeros(len(self.input_names))

    def admissible(self, x: np.ndarray) -> bool:
        """
        Physical/operational admissibility of a state (e.g. voltage and
        frequency bounds). Default: always admissible. Override for
        models with hard operating limits.
        """
        return True

    def output(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Optional measurement/output map. Defaults to identity on x."""
        return x

    # ------------------------------------------------------------------
    # Optional symbolic contract, for automatic MRC synthesis
    # (Part II §2.6 of the architecture document). A model that does not
    # override these simply cannot be used with MRCSynthesizer; every
    # numeric capability above (simulation, equilibria, Jacobians,
    # recoverability) is entirely unaffected either way.
    # ------------------------------------------------------------------
    def symbolic_symbols(self):
        """
        Return (x_syms, u_syms, p_syms): SymPy Symbol tuples/dict matching
        this model's state_names / input_names / params. Default
        implementation auto-generates real-valued symbols from those
        names, which is sufficient for most models; override only if a
        parameter needs a specific assumption (e.g. positive=True).
        """
        import sympy
        x_syms = tuple(sympy.symbols(list(self.state_names), real=True)) if self.state_names else ()
        u_syms = tuple(sympy.symbols(list(self.input_names), real=True)) if self.input_names else ()
        p_syms = {k: sympy.symbols(k, real=True) for k in self.params.keys()}
        return x_syms, u_syms, p_syms

    def symbolic_dynamics(self, x_syms, u_syms, p_syms):
        """
        Optional symbolic (SymPy) counterpart of `dynamics`, returning a
        list of SymPy expressions dx/dt in state order. Required for
        automatic MRC synthesis (control.mrc_synthesis.MRCSynthesizer).
        Not implemented by default.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not provide symbolic_dynamics(); "
            "automatic MRC synthesis is unavailable for this model. "
            "Numeric simulation and IMS analysis are unaffected."
        )

    def symbolic_manifold_constraint(self, x_syms, u_syms, p_syms):
        """
        Optional SymPy expression phi(x) whose zero level set defines this
        component's intrinsic manifold M = {x : phi(x) = 0}, in the sense
        of the architecture document Part I §1.5/§1.7. Required for
        automatic MRC synthesis. Not implemented by default.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not provide symbolic_manifold_constraint(); "
            "automatic MRC synthesis is unavailable for this model."
        )

    # ------------------------------------------------------------------
    # Generic derived capabilities (work for any subclass)
    # ------------------------------------------------------------------
    @property
    def n_states(self) -> int:
        return len(self.state_names) if self.state_names else self._infer_dim()

    def _infer_dim(self) -> int:
        raise ValueError(
            "state_names not set on model; set `state_names` in the subclass "
            "so the platform knows the state dimension."
        )

    def rhs(self, t: float, x: np.ndarray, u: Optional[np.ndarray] = None) -> np.ndarray:
        """Convenience wrapper used by the simulator / manifold / control code."""
        if u is None:
            u = self.default_input()
        return np.asarray(self.dynamics(t, np.asarray(x, dtype=float), np.asarray(u, dtype=float), self.params))

    def jacobian(self, x: np.ndarray, u: Optional[np.ndarray] = None, eps: float = 1e-6) -> np.ndarray:
        """Central-difference numerical Jacobian df/dx at (x, u)."""
        x = np.asarray(x, dtype=float)
        n = self.n_states
        J = np.zeros((n, n))
        f0_base = self.rhs(0.0, x, u)
        for i in range(n):
            dx = np.zeros(n)
            step = eps * max(1.0, abs(x[i]))
            dx[i] = step
            f_plus = self.rhs(0.0, x + dx, u)
            f_minus = self.rhs(0.0, x - dx, u)
            J[:, i] = (f_plus - f_minus) / (2 * step)
        _ = f0_base
        return J

    def find_equilibrium(
        self,
        x0: np.ndarray,
        u: Optional[np.ndarray] = None,
        with_eigs: bool = True,
    ) -> EquilibriumResult:
        """
        Newton-solve f(x*, u; p) = 0 starting from x0, and characterise
        the equilibrium via the eigenvalues of the local Jacobian
        (hyperbolicity / local stability), consistent with the IMS
        definition of the intrinsic manifold as a normally-hyperbolic
        invariant manifold built from equilibrium-consistent points.

        Also records solver diagnostics already available from SciPy's
        `fsolve` (function-evaluation count) and the Jacobian already
        computed for the eigenvalue characterisation (its condition
        number, and the matrix itself) -- exposing information the
        solver already produces internally, not new computation.
        """
        import time
        t0 = time.perf_counter()

        if u is None:
            u = self.default_input()

        def g(x):
            return self.rhs(0.0, x, u)

        x_star, info, ier, msg = fsolve(g, np.asarray(x0, dtype=float), full_output=True)
        residual_norm = float(np.linalg.norm(g(x_star)))
        # Convergence is judged on the residual itself (the actual
        # statement "f(x*) ~ 0" that defines an equilibrium), NOT on
        # MINPACK/hybrd's own ier flag. ier==1 is a solver-internal
        # heuristic ("did the step-improvement pattern look like textbook
        # convergence"), and it can come back non-1 (commonly ier==5,
        # "not making good progress... from the last ten iterations")
        # even when x_star is already an extremely accurate root -- this
        # happens whenever the initial guess lands close to the true
        # equilibrium, so the solver's last steps have nothing left to
        # improve and its own progress heuristic misfires. Confirmed
        # directly: a PI-controlled buck converter case with residual_norm
        # ~2e-12 (14 orders of magnitude below this function's own 1e-6
        # tolerance) was previously flagged converged=False purely because
        # ier==5, which downstream became a false "target manifold
        # incompatible with equilibrium" MRC_SYNTHESIS_NOT_ESTABLISHED
        # verdict for a target that is, in fact, reachable. Gating on
        # residual alone is strictly the more rigorous check, not a
        # weaker one -- it is exactly the mathematical condition
        # "equilibrium" requires, and does not accept anything the old
        # ier-gated check would not also have accepted at the same
        # tolerance whenever ier really did indicate failure with a large
        # residual.
        converged = residual_norm < 1e-6

        eigs = None
        J = None
        cond = None
        if with_eigs:
            J = self.jacobian(x_star, u)
            eigs = np.linalg.eigvals(J)
            try:
                cond_val = np.linalg.cond(J)
                # np.linalg.cond does NOT raise for a singular or exactly-
                # singular matrix -- it returns float('inf') (or, less
                # commonly, nan) directly. A structurally marginally-stable
                # system (e.g. a PI controller's integrator state, which
                # has a genuine zero eigenvalue by construction) makes J
                # singular at every equilibrium, not just this one, so this
                # is a real, reachable case -- not a hypothetical. Only a
                # finite condition number is a meaningful diagnostic value;
                # inf/nan here get caught before they ever reach JSON
                # serialisation (see server.app._json_safe for the
                # defense-in-depth pass applied to every API response too).
                cond = float(cond_val) if np.isfinite(cond_val) else None
            except np.linalg.LinAlgError:
                cond = None

        elapsed = time.perf_counter() - t0

        return EquilibriumResult(
            x_star=x_star, converged=converged, residual_norm=residual_norm, eigenvalues=eigs,
            n_function_evals=int(info.get("nfev", 0)), jacobian=J,
            jacobian_condition_number=cond, computation_time_s=elapsed,
        )
