"""
network.renewable_components
------------------------------

Component-library additions for the "Iberian 2025-Inspired Overvoltage
Cascade" case (see cases/iberian_2025_overvoltage_cascade.yaml and
server.iberian_scenario). These extend the existing bus-attached
component library (network.components) with the two device types the
28 April 2025 Iberian Peninsula event's official post-mortem singles
out as central to the overvoltage/reactive-power/protection chain:

    - GFLRenewableSource: an aggregated grid-following (GFL) PV/wind
      cluster whose reactive-current behaviour can be swept continuously
      from "fixed power factor" (no dynamic voltage support, KQ = 0 --
      the ENTSO-E report's description of how a portion of Spanish
      renewable generation was operating that day) to "dynamic Q/V
      droop support" (KQ > 0), subject to a converter current limit and
      a delayed overvoltage-trip protection element.

    - ShuntReactor: a switched reactive-absorption device (representing
      the report's shunt reactors, some of which were unavailable /
      required manual operation) whose *available* absorption capacity
      is itself a tunable parameter -- exactly the "shunt reactor
      availability" slider the platform's demo scenario exposes.

Positioning note (do not remove): this is a reduced-order, illustrative
model of the *mechanism class* the official ENTSO-E report describes
(insufficient dynamic voltage/reactive-power support, current limiting,
delayed overvoltage protection, cascading disconnection) -- it is
explicitly NOT a reconstruction of the real Spanish network and does
not reproduce or validate any specific numeric finding of that report.

Modelling conventions
----------------------
This platform's Bus/Line/BusComponent layer is a DC (or DC-analog,
per-unit "voltage magnitude") abstraction (network.bus, network.branch);
there is no AC phase angle or a genuine P/Q decomposition. Consistent
with how the rest of the codebase already treats "voltage" as a general
admissibility/large-signal state (see network.components.ConstantPowerLoad
and the IMS papers' own v_b large-signal notion), reactive-power support
here is represented as an *additional current contribution* that pushes
bus voltage down when it is high and up when it is low -- a standard
droop analogy (Q/V droop -> current/voltage droop in this abstraction),
not a literal dq-frame PLL/current-loop model. This keeps the new
components composable with the existing Automatic Model Builder
without requiring an AC/dq extension of the whole platform (noted as
future work in network.electrical_model's own module docstring).
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional, Tuple

from .components import BusComponent


def _smooth_relu(z: float, eps: float = 1e-3) -> float:
    """C1-continuous approximation of max(z, 0), used throughout for ODE-friendly (smooth) switching."""
    return 0.5 * (z + np.sqrt(z * z + eps * eps))


def _sigmoid(z: float) -> float:
    # Clipped to avoid overflow in exp() for very negative/positive z.
    z_c = min(max(z, -60.0), 60.0)
    return 1.0 / (1.0 + np.exp(-z_c))


class GFLRenewableSource(BusComponent):
    """
    Aggregated renewable cluster (PV + wind) whose capacity is split
    between two structurally DIFFERENT control architectures according
    to `gfm_fraction` (0-1, the fraction of the cluster's OWN capacity,
    not an additive gain):

      - a `1 - gfm_fraction` share operated grid-following (GFL): a
        controlled CURRENT source with a tunable reactive/voltage-
        support droop gain `KQ` (0 = fixed power factor, matching the
        ENTSO-E report's description of how a portion of Spanish
        renewable generation was operating that day). `KQ` is a free
        control-tuning parameter -- there is no physical ceiling on it
        inherent to the device.

      - a `gfm_fraction` share operated grid-forming (GFM): a Thevenin-
        equivalent VOLTAGE source behind a fixed internal reactance
        `X_gfm`,

            i_gfm = (E_gfm - v_bus) / X_gfm ,

        which is a structurally different mechanism from the GFL
        droop above, not a relabelled/rescaled version of it: `X_gfm`
        is a fixed converter-filter characteristic (not a tunable
        control gain), the restoring current is INHERENT to the
        voltage-source behaviour rather than a measured-and-corrected
        control loop, and it responds identically regardless of `KQ`.
        This is deliberately NOT an inertial/swing-equation model (no
        angle or frequency state is introduced -- this platform's
        Bus/Line layer has no such state, and adding one here would be
        unsupported invention rather than a reduction of validated
        physics); it captures only the voltage-forming/stiffness
        behaviour the funding-demo brief's item 4 (`GFL -> GFM`) and
        item 10 (GFM-penetration comparison) ask for.

    Each share is subject to its OWN smooth current limit, scaled by
    its OWN capacity share of the cluster's total rating `Imax`
    (`Imax*(1-gfm_fraction)` for the GFL share, `Imax*gfm_fraction` for
    the GFM share), using the same frozen clamp shape as before
    (`Imax_share * tanh(i_raw_share / Imax_share)`) -- so a GFM share
    with a small Imax allocation can still be current-limited during a
    severe disturbance, matching real grid-forming converters' own
    current-limited fault-ride-through mode; see the class-level
    current-limiter note in `_clamp` for why this exact shape.

    At gfm_fraction=0, Imax_gfm=0 and the GFM share's clamp identically
    returns 0 (guarded explicitly, not just via tanh(0)), so the whole
    component's behaviour reduces EXACTLY to the pre-GFM formula -- this
    is verified directly in tests/test_iberian_scenario.py and
    tests/test_gfm_mechanism.py, not merely asserted here.

    Also carries:
      - a smooth, delayed overvoltage-trip protection element shared by
        both shares (once bus voltage exceeds `v_trip` for longer than
        `t_delay`, BOTH shares' output ramps toward zero together) --
        kept as a single shared mechanism rather than inventing two
        separate, unvalidated protection-response characteristics for
        GFL vs. GFM;
      - a passive voltage-damping current proportional to each share's
        OWN currently-online active power, `k_damp * P_share` (see the
        original docstring note on this term, preserved unchanged in
        mechanism, now applied per-share).

    Local states (both dimensionless, 0 by default):
        trip_timer : accumulates while v_bus > v_trip, decays otherwise

    Sign convention: current_injection > 0 means power flows FROM the
    source INTO the bus (generation).
    """

    n_local_states = 1
    _local_state_names = ("trip_timer",)

    def __init__(
        self,
        id: str,
        bus: str,
        P_set: float,
        KQ: float = 0.0,
        v_ref: float = 1.0,
        Imax: float = 1.5,
        v_trip: float = 1.10,
        t_delay: float = 0.15,
        gfm_fraction: float = 0.0,
        X_gfm: float = 0.1,
        E_gfm: Optional[float] = None,
        v_floor: float = 0.05,
        trip_decay_rate: float = 4.0,
        trip_sharpness: float = 12.0,
        k_damp: float = 0.15,
    ):
        super().__init__(
            id, bus,
            P_set=P_set, KQ=KQ, v_ref=v_ref, Imax=Imax, v_trip=v_trip, t_delay=t_delay,
            gfm_fraction=gfm_fraction, X_gfm=X_gfm, E_gfm=(v_ref if E_gfm is None else E_gfm),
            v_floor=v_floor, trip_decay_rate=trip_decay_rate,
            trip_sharpness=trip_sharpness, k_damp=k_damp,
        )

    # ------------------------------------------------------------------
    def _v_eff(self, v_bus: float) -> float:
        v_floor = self.params["v_floor"]
        delta = 0.02
        return 0.5 * (v_bus + v_floor) + 0.5 * ((v_bus - v_floor) ** 2 + delta ** 2) ** 0.5

    def _trip_signal(self, timer: float) -> float:
        """Smooth 0->1 trip indicator: ~0 while the (bounded, [0,1]) timer is below half-charge, ~1 above it."""
        return _sigmoid(self.params["trip_sharpness"] * (timer - 0.5))

    @staticmethod
    def _clamp(i_raw: float, i_max_share: float) -> float:
        """
        Smooth symmetric current limiter for one capacity share: behaves
        like clip(i_raw, -i_max_share, i_max_share) but is C1-continuous
        everywhere, so the Jacobian used by equilibrium solving / IMS
        analysis stays well-defined at the limit. i_max_share*tanh(x/
        i_max_share) has slope EXACTLY 1 at i_raw=0 (unclamped, linear
        pass-through for currents well under the rating) and approaches
        +/-i_max_share asymptotically as |i_raw| grows. (A previous
        version of this formula additionally scaled the argument by a
        `limit_sharpness` factor and re-normalised so i_raw=Imax mapped
        to EXACTLY Imax; that construction had slope ~6, not 1, at the
        origin, so it aggressively saturated currents far below rating
        -- e.g. a genuine i_raw of only 17% of Imax came out at 76% of
        Imax. Caught by direct inspection while verifying the Imax
        slider actually does something physically sensible; removed
        rather than patched. FROZEN as of this fix -- do not reintroduce
        an extra sharpness factor without re-running the Imax
        verification sweep in tests/test_iberian_scenario.py.)
        Explicitly returns 0 (not tanh(x/~0), which would blow up) when
        this share's capacity allocation is ~0 -- the gfm_fraction=0 /
        gfm_fraction=1 edge cases.
        """
        if i_max_share <= 1e-9:
            return 0.0
        return i_max_share * np.tanh(i_raw / i_max_share)

    def current_injection(self, v_bus: float, local_x: np.ndarray, u: Optional[float] = None) -> float:
        P = self.params["P_set"] if u is None else u
        timer = float(local_x[0]) if local_x.size else 0.0
        trip = self._trip_signal(timer)

        gfm = min(max(self.params["gfm_fraction"], 0.0), 1.0)
        Imax = self.params["Imax"]
        v_eff = self._v_eff(v_bus)
        k_damp = self.params["k_damp"]

        # -- GFL share: grid-following, controlled current source with a
        # tunable reactive droop gain KQ. Structurally a MEASURED-ERROR
        # feedback law (v_ref - v_bus) times a free control gain.
        P_gfl = P * (1.0 - gfm)
        i_raw_gfl = P_gfl / v_eff + self.params["KQ"] * (self.params["v_ref"] - v_bus) - k_damp * P_gfl
        i_gfl = self._clamp(i_raw_gfl, Imax * (1.0 - gfm))

        # -- GFM share: grid-forming, Thevenin voltage source behind a
        # FIXED internal reactance X_gfm. Structurally a hardware
        # (filter-impedance) characteristic, not a control gain: the
        # restoring current i = (E_gfm - v_bus)/X_gfm exists even with
        # no measurement/control loop at all, and is numerically
        # independent of KQ.
        P_gfm_share = P * gfm
        i_raw_gfm = P_gfm_share / v_eff + (self.params["E_gfm"] - v_bus) / self.params["X_gfm"] - k_damp * P_gfm_share
        i_gfm = self._clamp(i_raw_gfm, Imax * gfm)

        return (i_gfl + i_gfm) * (1.0 - trip)

    def local_dynamics(self, v_bus: float, local_x: np.ndarray, u: Optional[float] = None) -> np.ndarray:
        timer = float(local_x[0]) if local_x.size else 0.0
        d_timer = self._timer_derivative(v_bus, timer)
        return np.array([d_timer])

    def _over_ind(self, v_bus: float) -> float:
        return _sigmoid(60.0 * (v_bus - self.params["v_trip"]) / max(self.params["v_trip"], 1e-6))

    def _timer_derivative(self, v_bus: float, timer: float) -> float:
        # Bounded overvoltage indicator (in (0,1)) rather than an
        # unbounded "excess volts" quantity: this keeps `timer` itself
        # bounded in [0,1] with a genuine, always-existing equilibrium
        # (timer* = charge*over_ind / (charge*over_ind + decay*(1-over_ind))),
        # which is required for Newton-based equilibrium solving /
        # linearisation to be well posed even at a persistently
        # over-voltage operating point (where the protection is, by
        # design, meant to eventually trip). `t_delay` sets the
        # charging time constant: at over_ind ~= 1, timer approaches
        # its half-charge (trip) point on a ~t_delay timescale.
        # Steepness chosen so over_ind is sharply binary within a ~1-2%
        # voltage band around v_trip (rather than the gentler slope
        # originally used, which left `timer` settling at a misleadingly
        # elevated value -- ~0.5 -- even a few percent BELOW v_trip,
        # conflating "safely recovered" with "still at risk"; verified
        # directly by checking the timer's own equilibrium value at
        # several post-disturbance voltages during this component's
        # tuning/validation).
        over_ind = self._over_ind(v_bus)
        charge_rate = 1.0 / max(self.params["t_delay"], 1e-3)
        decay_rate = self.params["trip_decay_rate"]
        return charge_rate * over_ind * (1.0 - timer) - decay_rate * (1.0 - over_ind) * timer

    # ------------------------------------------------------------------
    # Genuine intrinsic-manifold quantities for THIS component's own fast
    # variable (trip_timer), in the geometric-singular-perturbation sense
    # used by the IMS papers this platform is built from (fast variable
    # x_f = timer, slow variable x_s = v_bus, critical manifold M_0 =
    # {(x_f,x_s) : F_f(x_f,x_s)=0}). These are extracted DIRECTLY from
    # _timer_derivative above -- the exact same right-hand side the
    # simulator integrates -- not a separately re-derived approximation,
    # so there is no risk of the "manifold" and the "dynamics" silently
    # drifting apart.
    #
    #   Critical manifold (closed form, since _timer_derivative is AFFINE
    #   in `timer` at fixed v_bus): setting d(timer)/dt = 0 and solving
    #   for timer gives
    #       timer*(v) = charge_rate*over_ind(v) / (charge_rate*over_ind(v)
    #                    + decay_rate*(1-over_ind(v)))
    #   -- a genuine algebraic root of the fast subsystem, not a fitted
    #   or assumed curve.
    #
    #   Transverse eigenvalue: because _timer_derivative is affine in
    #   `timer`, its partial derivative w.r.t. timer is exactly
    #       lambda_perp(v) = -(charge_rate*over_ind(v) + decay_rate*(1-over_ind(v)))
    #   independent of the actual value of `timer` (not just evaluated
    #   ON the manifold -- true everywhere, by construction of this
    #   particular ODE). It is strictly negative for all v (charge_rate,
    #   decay_rate > 0), i.e. this fast subsystem is unconditionally
    #   transversally contracting -- see this module's own docstring/
    #   the funding-demo report for what that does and does not imply
    #   about overall recoverability.
    def manifold_timer_star(self, v_bus: float) -> float:
        over_ind = self._over_ind(v_bus)
        charge_rate = 1.0 / max(self.params["t_delay"], 1e-3)
        decay_rate = self.params["trip_decay_rate"]
        denom = charge_rate * over_ind + decay_rate * (1.0 - over_ind)
        return (charge_rate * over_ind) / max(denom, 1e-12)

    def transverse_eigenvalue(self, v_bus: float) -> float:
        over_ind = self._over_ind(v_bus)
        charge_rate = 1.0 / max(self.params["t_delay"], 1e-3)
        decay_rate = self.params["trip_decay_rate"]
        return -(charge_rate * over_ind + decay_rate * (1.0 - over_ind))

    def initial_state_guess(self) -> np.ndarray:
        return np.zeros(1)

    @property
    def has_input(self) -> bool:
        return True  # P_set is the natural continuation / disturbance parameter

    @property
    def input_param_name(self) -> str:
        return "P_set"


class ShuntReactor(BusComponent):
    """
    A switched shunt reactive-absorption device (e.g. a line/substation
    shunt reactor). `Q_avail` (0-1) is the *available fraction* of its
    nameplate absorption capacity `B_nom` -- representing devices that
    require manual switching or are otherwise unavailable, per the
    ENTSO-E report's description. Absorption current scales with bus
    voltage (a fixed-susceptance analogy): higher voltage draws more
    absorbing current, providing voltage support once available.
    """

    def __init__(self, id: str, bus: str, B_nom: float, Q_avail: float = 1.0):
        super().__init__(id, bus, B_nom=B_nom, Q_avail=Q_avail)

    def current_injection(self, v_bus: float, local_x: np.ndarray, u: Optional[float] = None) -> float:
        Q_avail = self.params["Q_avail"] if u is None else u
        B_eff = self.params["B_nom"] * min(max(Q_avail, 0.0), 1.0)
        return -B_eff * v_bus

    @property
    def has_input(self) -> bool:
        return True

    @property
    def input_param_name(self) -> str:
        return "Q_avail"


class LineChargingShunt(BusComponent):
    """
    Fixed capacitive shunt current representing a long transmission
    line's own charging susceptance (the "pi-model line charging Bsh"
    the funding-demo brief calls for, item 6): a constant injection

        i_inj = +B_sh * v_bus

    always present, independent of loading. Physically this captures
    the mechanism ENTSO-E's report attributes part of the pre-cascade
    voltage rise to: as active-power transfer across a line falls (here,
    following a generation trip elsewhere in the network), the
    resistive/inductive voltage DROP that transfer was causing falls
    with it, while the line's own (load-independent) capacitive
    charging current is unchanged -- so the net effect on the receiving
    bus shifts toward the higher, more lightly-loaded-line voltage.
    Deliberately a fixed shunt susceptance, not a literal distributed
    pi-section model of the line -- see the module docstring's
    positioning note.
    """

    def __init__(self, id: str, bus: str, B_sh: float, v_nom: float = 1.0):
        # Modelled as a genuine constant current injection (evaluated at
        # a fixed nominal voltage, not the live bus voltage): a v-bus-
        # proportional injection would add positive local feedback
        # (d(i_inj)/dv_bus = +B_sh > 0, a negative-conductance-like
        # term) that destabilises the bus's own voltage dynamics
        # regardless of B_sh's magnitude -- not the intended effect,
        # and not how real line charging enters an AC voltage/reactive-
        # power balance (it is mediated by the network's overall Q
        # balance, not a local per-bus runaway). Fixing the evaluation
        # point at v_nom keeps this a constant forcing term (zero
        # Jacobian contribution) while still producing the intended
        # "lighter loading -> charging is proportionally more dominant
        # -> voltage settles higher" effect through the network's own
        # KCL/KVL balance.
        super().__init__(id, bus, B_sh=B_sh, v_nom=v_nom)

    def current_injection(self, v_bus: float, local_x: np.ndarray, u: Optional[float] = None) -> float:
        B_sh = self.params["B_sh"] if u is None else u
        return B_sh * self.params["v_nom"]

    @property
    def has_input(self) -> bool:
        return True

    @property
    def input_param_name(self) -> str:
        return "B_sh"
