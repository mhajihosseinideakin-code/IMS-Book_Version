/*
 * Regression test for the report-generation gap: buildReportBodyHtml()
 * previously only pulled 4 narrow fields (summary, equilibrium x_star,
 * controller type, recoverability index/risk/margin) and silently
 * ignored everything else the backend's stage responses actually
 * contain (network_summary, component_chain, model_order, params_used,
 * solver_diagnostics, performance_metrics, manifold info). This test
 * calls the REAL buildReportBodyHtml() function from the shipped file
 * with mock stage results matching the ACTUAL backend response shapes
 * (see server/app.py's stage_build/stage_equilibrium/stage_simulate/
 * stage_ims_analysis), and asserts the resulting report string actually
 * contains every section it's supposed to -- the same kind of check
 * that would have caught the original gap automatically.
 *
 * Run: node tests/test_explorer_report_content.js
 */
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const htmlPath = path.join(__dirname, "..", "ims_platform", "server", "static", "explorer.html");
const html = fs.readFileSync(htmlPath, "utf8");
const script = html.split("<script>")[1].split("</script>")[0];

const sandbox = {
  document: {
    addEventListener: () => {},
    getElementById: () => null,
    querySelectorAll: () => [],   // no canvases in this headless test -- Figures section is not exercised here
    createElement: () => ({ style: {}, appendChild: () => {}, classList: { toggle: () => {} } }),
  },
  window: { addEventListener: () => {}, devicePixelRatio: 1 },
  console,
  fetch: () => Promise.resolve({ json: () => Promise.resolve({}) }),
};
sandbox.window.document = sandbox.document;
vm.createContext(sandbox);
vm.runInContext(script, sandbox);

let failures = 0;
function check(name, cond, detail) {
  if (cond) { console.log("PASS " + name); }
  else { console.log("FAIL " + name + (detail ? " -> " + detail : "")); failures++; }
}

// --- Mock a fully-completed pipeline for a converter_topology project,
//     with response shapes matching the REAL backend exactly. ---
sandbox.currentProject = {
  id: "buck_converter", label: "Buck Converter", category: "converter_topology",
  description: "Averaged CCM buck converter.", state_short: ["v_out", "i_L"],
};
sandbox.stageResults = {
  build: {
    state_names: ["v_out", "conv_em_i_L"], input_names: ["conv_input"],
    params_used: { C_out: 0.02, conv_ctrl_d: 0.4, conv_em_L: 0.001, conv_em_R_L: 0.05, conv_em_v_in: 12.0, load_R: 5.0 },
    category: "converter_topology",
    network_summary: { buses: 1, lines: 0, converters: 1, loads: 1, controllers: 1,
      component_types: { Converter: 1, ConstantImpedanceLoad: 1 },
      converter_topologies: { BuckModel: 1 }, converter_control_modes: { grid_forming: 1 } },
    component_chain: ["Converter 'conv' (BuckModel, ConstantDutyController)", "ConstantImpedanceLoad 'load' (bus: out)", "Dynamic bus 'out' (C=0.02)"],
    topology_tree: ["Bus 'out' (dynamic, C=0.02)", "  |-- Converter 'conv'", "  \\-- ConstantImpedanceLoad 'load'"],
    model_order: { n_states: 2, n_inputs: 1, n_outputs: 2 },
    controller_description: "Constant Duty Cycle (open-loop reference).",
  },
  equilibrium: {
    state_names: ["v_out", "conv_em_i_L"], x_star: [4.752475, 0.950495], stable: true,
    eigenvalues: [{ re: -30.0, im: 222.71 }, { re: -30.0, im: -222.71 }],
    residual_norm: 3.5e-14, nominal_input: [0.4],
    jacobian: [[-10.0, 50.0], [-1000.0, -50.0]], jacobian_rank: 2, jacobian_spectral_radius: 224.722,
    solver_diagnostics: { function_evaluations: 7, jacobian_condition_number: 19.85, computation_time_s: 0.0032 },
    component_operating_points: { conv: { type: "BuckModel", bus: "out", operating_voltage: 4.752475, operating_current: 0.950495, operating_power: 4.517204 } },
  },
  simulate: {
    disturbed_state: [5.25, 0.75], trajectory_open_loop: { t: [0, 0.1], x: [[5.25, 4.75], [0.75, 0.95]], success: true },
    performance_metrics_open_loop: { max_deviation: [0.5, 0.2], steady_state_error: [0.002, 0.001], settling_time: [0.1, 0.08], overshoot_pct: [66.1, 10.2] },
    controller_info: { type: "None (open loop)" },
  },
  ims_analysis: {
    manifold: { points: new Array(150).fill([0, 0]), stable: new Array(150).fill(true), alpha: [] },
    manifold_stats: {
      continuation_parameter: "conv_ctrl_d", n_points: 150, dimension: 1,
      operating_interval: [0.1, 0.85], empirically_persistent: true,
      n_stable: 150, n_unstable: 0, n_singular_points: 0, singular_points: [],
      max_curvature: 6.99e-7, mean_curvature: 1.46e-7,
      arc_length: 12.34, tangent_at_nearest: [0.98, 0.20], operating_region_stable: [1.2, 10.5],
    },
    manifold_residual_trajectory: { t: [0, 0.1], residual: [0.3, 0.03] },
    residual_statistics: { max: 0.3, rms: 0.15, mean: 0.1, final: 0.03, empirical_contraction_rate: 12.5 },
    residual_projection_method: "polyline",
    projection_method_validation_summary: "The polyline projection method was validated directly against the newton_refined high-accuracy reference across representative cases: recoverability classification matched exactly in every case tested.",
    recoverability: { samples: [[5.0, 0.9]], labels: [true], index: 0.85, n_recoverable: 17, n_samples: 20, margin: 1.0, risk: "LOW",
      worst_disturbance: null, confidence_interval: [0.62, 0.96] },
    recoverability_deterministic: {
      max_residual: 5.0, rms_residual: 0.73, integral_abs_residual: 0.01, final_residual: 2.5,
      recovery_time: null, contraction_rate: 0.05, tolerance_used: 0.1, classification: "Non-Recoverable",
      ims_conditions: { full_system_local_stability: "VIOLATED", normal_hyperbolicity: "VIOLATED", transverse_contraction: "NOT_ESTABLISHED", reduced_dynamics_stable: "NOT_ESTABLISHED", empirical_finite_horizon_recovery: true, certified_regional_ims: "NOT_ESTABLISHED", overall_status: "VIOLATED" },
      limiting_state: "conv1_ctrl_e_int", limiting_state_deviation: 130.6, limiting_state_location: "component 'conv1'",
      note: "The largest single-state deviation from the nearest intrinsic-manifold point is in component 'conv1' (conv1_ctrl_e_int = +130.6 from its manifold-consistent value).",
      geometric_narrative: "The disturbed state began at a manifold residual of 5. Over the simulated horizon, the residual did not show meaningful contraction toward the intrinsic manifold, settling at a final residual of 2.5 -- exceeding the admissible tolerance of 0.1.",
    },
    manifold_residual_explanation: "The manifold residual r_m(x) is the distance from the system's current state x to the nearest point on the intrinsic manifold M. Its numerical value does not carry one universal engineering unit because the manifold is embedded in the full state space.",
    recoverability_region_summary: {
      concept_explanation: "The Recoverability Region R is the set of disturbed states whose trajectories return to, and remain within, an admissible neighborhood of the intrinsic manifold M.",
      membership_explanation: "This specific disturbed trajectory's own manifold residual answers the region-membership question directly: it settled at a residual of 2.5 against an admissible tolerance of 0.1. This means the disturbed state is classified as outside the recoverability region.",
      boundary_note: "A full directional trace of dR was not run for this analysis (this is the faster, default Basic-mode path). Enabling boundary tracing in Advanced mode computes an actual geometric approximation of R's extent.",
      inside_region: false, boundary_traced: false,
    },
    recoverability_closed_loop: { samples: [[5.0, 0.9]], labels: [true], index: 0.97, n_recoverable: 19, n_samples: 20, margin: 1.4, risk: "LOW",
      worst_disturbance: null, confidence_interval: [0.80, 1.0] },
    recoverability_boundary: {
      n_directions: 8, n_found: 7,
      points: [{ theta: 0, radius: 1.5, point: [6.25, 0.95], status: "found" }],
      min_radius: 0.77, max_radius: 3.65,
    },
  },
};
// Give simulate closed-loop data too, so the controller-comparison table has what it needs.
sandbox.stageResults.simulate.performance_metrics_closed_loop = { max_deviation: [0.3, 0.1], steady_state_error: [0.001, 0.0005], settling_time: [0.05, 0.04], overshoot_pct: [20.0, 5.0] };
sandbox.stageResults.simulate.control_effort = { rms: 0.42, peak: 1.2 };

const reportHtml = vm.runInContext("buildReportBodyHtml()", sandbox);

const expectedSections = [
  ["Network description", "network_summary and component_chain"],
  ["Mathematical model", "model_order / state variables / params_used"],
  ["Equilibrium analysis", "eigenvalues / residual norm"],
  ["Solver diagnostics", "function evaluations / Jacobian condition number / computation time"],
  ["Simulation results", "disturbed state / controller"],
  ["Performance metrics", "settling time / overshoot / steady-state error"],
  ["IMS analysis", "manifold summary"],
  ["Recoverability", "Monte Carlo statistics"],
  ["Conclusions", "auto-generated interpretation"],
];
expectedSections.forEach(function ([needle, desc]) {
  check(`report includes "${needle}" (${desc})`, reportHtml.indexOf(needle) !== -1);
});

// Spot-check actual VALUES made it through, not just section headers.
check("report includes the real Jacobian condition number value", reportHtml.indexOf("1.99e+1") !== -1 || reportHtml.indexOf("19.85") !== -1, "condition number 19.85 not found verbatim");
check("report shows state variable physical meaning and units, not just bare names", reportHtml.indexOf("Physical Meaning") !== -1 && reportHtml.indexOf("Inductor Current") !== -1 && reportHtml.indexOf(">A</td>") !== -1);
check("report shows a Jacobian summary (rank, condition number, spectral radius) separate from the full matrix, labeled as an appendix", reportHtml.indexOf("Jacobian summary") !== -1 && reportHtml.indexOf("Appendix: full Jacobian matrix") !== -1 && reportHtml.indexOf("Spectral radius") !== -1);
check("performance metrics table (max deviation, steady-state error) now shows per-state units, not bare numbers", reportHtml.indexOf("Performance metrics") !== -1 && (reportHtml.match(/0\.5000 V/) !== null || reportHtml.match(/0\.2000 A/) !== null));
check("report includes the real component chain content", reportHtml.indexOf("BuckModel") !== -1);
check("report includes the real parameter values, categorized, with units", reportHtml.indexOf("conv_em_v_in") !== -1 && reportHtml.indexOf(">12<") !== -1 && reportHtml.indexOf("Converter Parameters") !== -1 && reportHtml.indexOf(">V<") !== -1);
check("report includes the recoverability risk badge", reportHtml.indexOf("risk-LOW") !== -1);
check("report includes the deterministic recoverability classification", reportHtml.indexOf("Recoverable") !== -1 && reportHtml.indexOf("deterministic, manifold-based") !== -1);
check("report includes the IMS conditions section, correctly showing a real known violated condition", reportHtml.indexOf("Violated") !== -1 && reportHtml.indexOf("Overall IMS status") !== -1);
check("report includes the limiting-state diagnostic, explicitly labeled as supplementary and not an IMS quantity", reportHtml.indexOf("Supplementary diagnostic (not an IMS quantity)") !== -1 && reportHtml.indexOf("conv1_ctrl_e_int") !== -1);
check("report includes a geometry-grounded engineering explanation of why the trajectory is/isn't recoverable", reportHtml.indexOf("Geometric interpretation") !== -1 && reportHtml.indexOf("did not show meaningful contraction") !== -1);
check("report includes an explanation of what the manifold residual represents and why it lacks a universal unit", reportHtml.indexOf("What is the manifold residual?") !== -1 && reportHtml.indexOf("does not carry one universal engineering unit") !== -1);
check("report explicitly documents which residual projection method was used", reportHtml.indexOf("Residual projection method") !== -1 && reportHtml.indexOf("Polyline") !== -1);
check("report includes the projection-method validation benchmark summary", reportHtml.indexOf("Projection method validation") !== -1 && reportHtml.indexOf("matched exactly in every case tested") !== -1);
check("report ALWAYS includes a Recoverability Region & Boundary section, even without full boundary tracing having been run", reportHtml.indexOf("Recoverability region &amp; boundary") !== -1 && reportHtml.indexOf("Is this disturbance inside R?") !== -1 && reportHtml.indexOf("was not run for this analysis") !== -1);
check("report labels the Monte Carlo section as statistical validation, not the primary assessment", reportHtml.indexOf("Statistical validation (Monte Carlo)") !== -1);
check("report includes operating interval", reportHtml.indexOf("Operating interval") !== -1);
check("report includes empirical persistence", reportHtml.indexOf("Empirically persistent") !== -1 || reportHtml.indexOf("empirically persistent") !== -1);
check("report includes curvature statistics", reportHtml.indexOf("curvature") !== -1 || reportHtml.indexOf("Curvature") !== -1);
check("report includes the traced critical boundary section", reportHtml.indexOf("Critical boundary") !== -1);
check("report includes the real boundary radius values", reportHtml.indexOf("0.770") !== -1 || reportHtml.indexOf("0.77") !== -1);
check("report includes the confidence interval", reportHtml.indexOf("confidence interval") !== -1 || reportHtml.indexOf("Confidence") !== -1 || reportHtml.indexOf("CI") !== -1);
check("report includes the converter topology breakdown", reportHtml.indexOf("Converter topologies") !== -1 && reportHtml.indexOf("BuckModel: 1") !== -1);
check("report includes the control mode breakdown", reportHtml.indexOf("Control modes") !== -1 && reportHtml.indexOf("grid forming: 1") !== -1);
check("report includes arc length", reportHtml.indexOf("Arc length") !== -1 && reportHtml.indexOf("12.3400") !== -1);
check("report includes tangent direction", reportHtml.indexOf("Tangent at operating point") !== -1);
check("report includes recommended operating region", reportHtml.indexOf("Recommended operating region") !== -1);
check("report includes residual statistics with the real values", reportHtml.indexOf("Max / RMS / mean residual") !== -1 && reportHtml.indexOf("3.000e-1") !== -1);
check("report includes empirical contraction rate", reportHtml.indexOf("Empirical contraction rate") !== -1 && reportHtml.indexOf("12.500") !== -1);
check("report includes the controller comparison table", reportHtml.indexOf("Controller comparison") !== -1);
check("report's comparison table includes real open-loop and closed-loop index values", reportHtml.indexOf("0.85") !== -1 && reportHtml.indexOf("0.97") !== -1);
check("report's comparison table honestly notes no 3rd MRC column", reportHtml.toLowerCase().indexOf("3rd ims-native mrc column is not included") !== -1);
check("report's conclusions note the un-computed sensitivity study honestly", reportHtml.indexOf("component-level sensitivity study") !== -1);
check("report includes component operating points with the real power value and a unit", reportHtml.indexOf("Component operating points") !== -1 && reportHtml.indexOf("4.52 W") !== -1);

// --- Regression check for the "still no figures" bug: report used to
//     rely on document.querySelectorAll("#wsMain canvas") finding
//     canvases from PREVIOUS stages, which renderMain()'s el.innerHTML=""
//     destroys on every stage change -- meaning the Figures section was
//     always empty by the time "Generate Report" was reached. Figures
//     are now rendered fresh, in place, via named placeholder containers
//     populated by populateReportFigures() using stored stageResults
//     data (which persists across stage navigation, unlike the DOM). ---
const placeholderReport = vm.runInContext("buildReportBodyHtml(false)", sandbox);
[
  ["report_trajWrap", "state trajectory plot"],
  ["report_phaseWrap", "phase portrait / manifold plot"],
  ["report_residualWrap", "manifold residual plot"],
  ["report_recMapWrap", "recoverability map"],
].forEach(function ([id, desc]) {
  check(`report includes a live placeholder for the ${desc} (id="${id}")`, placeholderReport.indexOf("id='" + id + "'") !== -1);
});

// --- Regression test for a real reported bug: a manifold with only 1
// converged continuation point rendered as a bare, unexplained empty
// chart (Arc length: 0.0000, no visible curve, no explanation). A
// clear warning should now appear instead. Reuses the already-working
// full mock above, temporarily overriding just the point count.
(function () {
  const original = JSON.parse(JSON.stringify(sandbox.stageResults.ims_analysis.manifold_stats));
  sandbox.stageResults.ims_analysis.manifold_stats.n_points = 1;
  sandbox.stageResults.ims_analysis.manifold_stats.arc_length = 0.0;
  const reportHtml = vm.runInContext("buildReportBodyHtml(false)", sandbox);
  check("report warns clearly when only 1 manifold point converged, instead of leaving an unexplained empty chart",
    reportHtml.indexOf("Only 1 manifold point") !== -1 && reportHtml.indexOf("structurally isolated") !== -1);
  sandbox.stageResults.ims_analysis.manifold_stats = original;  // restore for any tests that might run after
})();

// --- Priority 2: report is now IMS-centric. Flex `order` (not source
// string order) determines the VISUAL sequence, so this test checks
// the order value assigned to each panel rather than string position
// -- confirming IMS analysis now visually precedes Equilibrium
// analysis and Simulation results, supporting the primary IMS output
// rather than following it.
(function () {
  const reportHtml = vm.runInContext("buildReportBodyHtml(false)", sandbox);
  function orderOf(panelTitle) {
    const idx = reportHtml.indexOf(">" + panelTitle + "<");
    if (idx === -1) return null;
    const before = reportHtml.lastIndexOf("order:", idx);
    const match = reportHtml.slice(before, idx).match(/order:(\d+)/);
    return match ? parseInt(match[1], 10) : null;
  }
  const orders = {
    "Project information": orderOf("Project information"),
    "Network description": orderOf("Network description"),
    "Mathematical model": orderOf("Mathematical model"),
    "IMS analysis": orderOf("IMS analysis"),
    "Simulation results": orderOf("Simulation results"),
    "Equilibrium analysis": orderOf("Equilibrium analysis"),
  };
  check("IMS analysis is ordered immediately after Mathematical model, as the primary output", orders["IMS analysis"] === orders["Mathematical model"] + 1, JSON.stringify(orders));
  check("IMS analysis is visually ordered BEFORE Equilibrium analysis (supports rather than follows)", orders["IMS analysis"] < orders["Equilibrium analysis"], JSON.stringify(orders));
  check("IMS analysis is visually ordered BEFORE Simulation results", orders["IMS analysis"] < orders["Simulation results"], JSON.stringify(orders));
  check("Project information / Network description / Mathematical model retain their original leading order", orders["Project information"] === 1 && orders["Network description"] === 2 && orders["Mathematical model"] === 3);
})();

console.log(`\n${failures === 0 ? "ALL PASSED" : failures + " FAILED"}`);
process.exit(failures === 0 ? 0 : 1);
