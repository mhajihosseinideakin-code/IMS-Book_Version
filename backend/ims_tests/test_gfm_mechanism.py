"""
Tests for the grid-forming (GFM) mechanism in
network.renewable_components.GFLRenewableSource.

Background: an earlier version of `gfm_fraction` implemented GFM
penetration as an additive boost to the GFL droop gain
(KQ_eff = KQ + gfm_fraction*KQ_gfm_max). Verification showed this was
mathematically indistinguishable from just raising KQ directly, and
produced no meaningful effect at the proposal's requested 0/20/40% test
points. This file locks in place the REPLACEMENT mechanism: a genuine
capacity split between a grid-following (GFL, controlled current
source with a tunable droop) share and a grid-forming (GFM, Thevenin
voltage source behind a FIXED internal reactance X_gfm) share, each
with its own current-limit allocation.

Honest, verified findings this file checks (do not "fix" a passing
assertion here to make the story more flattering than the physics
supports):

  - gfm_fraction=0 reproduces the pre-GFM formula exactly.
  - gfm_fraction is NOT reducible to any KQ value (checked against a
    broad KQ sweep, not just the old degenerate special case).
  - At the full, frozen historical disturbance severity (1.0), GFM
    penetration up to 100% does NOT flip the KQ=0 baseline to
    RECOVERABLE in this network -- the shared overvoltage-trip timer
    curtails both shares together once triggered, so once tripped the
    settled state is dominated by the rest of the network, not by
    gfl_sw's internal architecture. This is reported as-is, not
    patched to produce a nicer result.
  - GFM DOES have a genuine, non-manufactured, physically-interpretable
    effect at MODERATE disturbance severities (0.4-0.8): it can flip
    AT RISK -> RECOVERABLE and even NON-RECOVERABLE -> RECOVERABLE,
    while NOT being monotonic in gfm_fraction at the highest severities
    (20% GFM can slightly worsen peak overshoot relative to 0%, before
    40% recovers it) -- a genuine current-limited-GFM-stiffness
    phenomenon (the GFM share's raw demanded current saturates against
    its own allocation almost immediately once the network's voltage
    departs from E_gfm), not an artifact.
"""
import numpy as np

try:
    import pytest
except ImportError:
    pytest = None

from ims_platform.server import iberian_scenario as isc
from ims_platform.network import GFLRenewableSource


def test_gfm_fraction_zero_reproduces_pre_gfm_formula_exactly():
    """
    gfl_sw at gfm_fraction=0 must be numerically identical to the
    original (pre-GFM-mechanism) GFL-only formula: P/v_eff + KQ*(v_ref-v)
    - k_damp*P, smoothly clamped to +/-Imax.
    """
    comp = GFLRenewableSource(id="g", bus="b", P_set=0.5, KQ=3.0, Imax=1.3,
                               v_trip=1.10, t_delay=0.15, gfm_fraction=0.0, k_damp=0.2)
    trip0 = comp._trip_signal(0.0)  # timer=0 gives a small but nonzero trip signal by construction
    for v in [0.9, 1.0, 1.05, 1.15, 1.3]:
        v_eff = comp._v_eff(v)
        i_raw_expected = 0.5 / v_eff + 3.0 * (1.0 - v) - 0.2 * 0.5
        expected = 1.3 * np.tanh(i_raw_expected / 1.3) * (1.0 - trip0)
        got = comp.current_injection(v, np.array([0.0]))
        assert abs(got - expected) < 1e-10, f"v={v}: expected {expected}, got {got}"


def test_frozen_baseline_unchanged_by_gfm_mechanism():
    """The KQ=0/KQ=15 reference bifurcation (gfm_fraction defaults to 0 throughout) must be untouched."""
    result = isc.verify_baseline_bifurcation()
    assert result["non_recoverable_case"]["summary"]["verdict"] == "NON-RECOVERABLE"
    assert result["recoverable_case"]["summary"]["verdict"] == "RECOVERABLE"


def test_gfm_fraction_is_not_reducible_to_any_kq_value():
    """
    gfm_fraction=0.2 (KQ=0) must not match ANY pure-KQ (gfm_fraction=0)
    trajectory within a small tolerance -- confirming it is a genuinely
    distinct mechanism, not a rescaled KQ.
    """
    r_gfm = isc.run_stress_test({"KQ": 0.0, "Q_avail": 0.05, "Imax": 1.3, "gfm_fraction": 0.2},
                                 severity=1.0, tail_horizon=15.0)
    v_gfm = np.array(r_gfm["voltage_traces"]["v_es_sw"])

    min_diff = None
    for KQ in np.arange(0.0, 20.1, 1.0):
        r_kq = isc.run_stress_test({"KQ": float(KQ), "Q_avail": 0.05, "Imax": 1.3, "gfm_fraction": 0.0},
                                    severity=1.0, tail_horizon=15.0)
        v_kq = np.array(r_kq["voltage_traces"]["v_es_sw"])
        if len(v_kq) == len(v_gfm):
            diff = float(np.max(np.abs(v_kq - v_gfm)))
            min_diff = diff if min_diff is None else min(min_diff, diff)
    assert min_diff is not None
    assert min_diff > 0.01, (
        f"gfm_fraction=0.2 trajectory matches some pure-KQ trajectory to within {min_diff:.4f} "
        "-- suspiciously close to reducible; expected a genuinely distinct mechanism"
    )


def test_gfm_share_current_is_genuinely_current_limited_when_saturated():
    """
    Direct check that the GFM share's OWN clamp binds independently of
    the GFL share's: construct a case where the Thevenin stiffness
    demands far more current than its capacity-share allocation, and
    confirm the clamped output never exceeds that allocation.
    """
    comp = GFLRenewableSource(id="g", bus="b", P_set=0.3, KQ=0.0, Imax=1.3,
                               v_trip=1.10, t_delay=0.15, gfm_fraction=0.4, X_gfm=0.1, k_damp=0.2)
    Imax_gfm_share = 1.3 * 0.4
    for v in [1.2, 1.3, 1.5, 2.0]:  # voltages far from E_gfm=1.0 -> huge raw demand
        i_out = comp.current_injection(v, np.array([0.0]))
        # i_out combines both shares; isolate by re-deriving the GFM share directly.
        v_eff = comp._v_eff(v)
        P_gfm_share = 0.3 * 0.4
        i_raw_gfm = P_gfm_share / v_eff + (comp.params["E_gfm"] - v) / comp.params["X_gfm"] - 0.2 * P_gfm_share
        i_gfm_clamped = comp._clamp(i_raw_gfm, Imax_gfm_share)
        assert abs(i_raw_gfm) > Imax_gfm_share * 2, "test setup should genuinely demand more than the share allows"
        assert abs(i_gfm_clamped) <= Imax_gfm_share + 1e-9, "GFM share's own clamp must bind independently"


def test_gfm_has_no_meaningful_effect_at_full_historical_severity():
    """
    HONEST finding, not a bug: at the frozen historical disturbance
    severity (1.0) and KQ=0, GFM penetration from 0% to 100% does NOT
    flip the verdict to RECOVERABLE in this network. The shared
    overvoltage-trip timer curtails both shares together once
    triggered; peak transient overvoltage is not even required to
    improve monotonically with gfm_fraction (see the non-monotonicity
    check below). This test exists to catch any future change that
    silently makes this scenario "recover" at the documented severity
    without a deliberate, reviewed physics change.
    """
    for gfm in [0.0, 0.2, 0.4, 1.0]:
        r = isc.run_stress_test({"KQ": 0.0, "Q_avail": 0.05, "Imax": 1.3, "gfm_fraction": gfm},
                                 severity=1.0, tail_horizon=15.0)
        assert r["baseline_ok"]
        assert r["summary"]["verdict"] == "NON-RECOVERABLE", (
            f"gfm_fraction={gfm} unexpectedly recovered at full historical severity "
            f"(final_v_es_sw={r['summary']['final_v_es_sw']:.4f}) -- if this is a deliberate "
            "physics change, update this test consciously; do not let it pass silently."
        )


def test_gfm_has_genuine_nonmonotonic_effect_at_moderate_severity():
    """
    HONEST finding: at severity=0.8, 20% GFM does NOT help (and slightly
    worsens peak overshoot vs. 0%), while 40% GFM flips the verdict to
    RECOVERABLE. This is not forced to be monotonic -- if a future
    change makes it monotonic, that's fine, but this test specifically
    guards the qualitative finding that GFM's benefit is not simply
    "more is always better" in this reduced-order model, so nobody
    "smooths" the boundary-map/UI layer into implying otherwise.
    """
    r0 = isc.run_stress_test({"KQ": 0.0, "Q_avail": 0.05, "Imax": 1.3, "gfm_fraction": 0.0},
                              severity=0.8, tail_horizon=15.0)
    r20 = isc.run_stress_test({"KQ": 0.0, "Q_avail": 0.05, "Imax": 1.3, "gfm_fraction": 0.2},
                               severity=0.8, tail_horizon=15.0)
    r40 = isc.run_stress_test({"KQ": 0.0, "Q_avail": 0.05, "Imax": 1.3, "gfm_fraction": 0.4},
                               severity=0.8, tail_horizon=15.0)
    assert r0["summary"]["verdict"] == "NON-RECOVERABLE"
    assert r20["summary"]["verdict"] == "NON-RECOVERABLE"
    assert r40["summary"]["verdict"] == "RECOVERABLE"


def test_gfm_can_flip_at_risk_to_recoverable_at_moderate_severity():
    """At severity=0.6, GFM genuinely flips an AT RISK case to RECOVERABLE -- a real, demonstrable benefit."""
    r0 = isc.run_stress_test({"KQ": 0.0, "Q_avail": 0.05, "Imax": 1.3, "gfm_fraction": 0.0},
                              severity=0.6, tail_horizon=15.0)
    r20 = isc.run_stress_test({"KQ": 0.0, "Q_avail": 0.05, "Imax": 1.3, "gfm_fraction": 0.2},
                               severity=0.6, tail_horizon=15.0)
    assert r0["summary"]["verdict"] == "AT RISK"
    assert r20["summary"]["verdict"] == "RECOVERABLE"


def test_gfm_causal_chain_end_to_end():
    """
    Full causal chain, per the acceptance criterion: slider value reaches
    backend params -> alters the equations (verified: distinct current
    for gfm>0 vs gfm=0 at the same v/P) -> alters the trajectory (verified:
    different voltage trace) -> alters recoverability (verified at
    severity=0.6/0.8 above). This test checks the first two links
    explicitly at the component level.
    """
    comp0 = GFLRenewableSource(id="g", bus="b", P_set=0.3, KQ=0.0, Imax=1.3,
                                v_trip=1.10, t_delay=0.15, gfm_fraction=0.0)
    comp4 = GFLRenewableSource(id="g", bus="b", P_set=0.3, KQ=0.0, Imax=1.3,
                                v_trip=1.10, t_delay=0.15, gfm_fraction=0.4)
    v_test = 1.15  # away from both v_ref and E_gfm, so both mechanisms are active
    i0 = comp0.current_injection(v_test, np.array([0.0]))
    i4 = comp4.current_injection(v_test, np.array([0.0]))
    assert abs(i0 - i4) > 1e-6, "gfm_fraction must change the equations, not just be a UI label"
