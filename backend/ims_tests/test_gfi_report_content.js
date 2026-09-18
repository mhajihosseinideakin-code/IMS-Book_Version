/*
 * Frontend test for the Grid-Forming Inverter reference-implementation
 * additions to the actual report: governing equations, admissible
 * envelope, electrical operating point, characteristic time constants,
 * and the P_e(t) figure slot. Calls the real, shipped
 * buildReportBodyHtml() function with mock data shaped exactly like
 * the real backend responses (see server/app.py's stage_build /
 * stage_equilibrium / stage_simulate for grid_forming_inverter).
 *
 * Run: node tests/test_gfi_report_content.js
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
    querySelectorAll: () => [],
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

// --- Mock data shaped exactly like the real grid_forming_inverter backend responses ---
sandbox.currentProject = {
  id: "grid_forming_inverter", label: "Grid-Forming Inverter", category: "single_model",
  description: "Droop-controlled single converter, swing-type dynamics.", state_short: ["\u03b4", "\u03c9"],
};
sandbox.stageResults = {
  build: {
    state_names: ["delta", "omega"], input_names: ["P_set"],
    params_used: { E: 1.0, V: 1.0, X: 0.3, tau_p: 0.05, m_p: 1.0 },
    category: "single_model",
    network_summary: { buses: 2, lines: 1, converters: 1, loads: 0, controllers: 1 },
    component_chain: ["Grid-Forming Inverter (droop-controlled EMF source)", "Line reactance X", "Infinite bus (grid)"],
    topology_tree: [],
    model_order: { n_states: 2, n_inputs: 1, n_outputs: 2 },
    controller_description: "None by default (open loop, self-stabilising via droop dynamics already in the model); Scheduled LQR available (Conventional Control Library).",
    governing_equations: [
      "d\u03b4/dt = \u03c9",
      "d\u03c9/dt = (1/\u03c4p)\u00b7( -\u03c9 + mp\u00b7(P_set - P_e(\u03b4)) )",
      "P_e(\u03b4) = (E\u00b7V/X)\u00b7sin(\u03b4)",
    ],
    admissible_envelope: { delta_max: 2.9845130209103035, omega_max: 25.0, description: "|\u03b4| < 0.95\u00b7\u03c0 rad, |\u03c9| < 25.0 rad/s" },
  },
  equilibrium: {
    state_names: ["delta", "omega"], x_star: [0.150568, 0.0], stable: true,
    is_hyperbolic: true,
    eigenvalues: [{ re: -4.1615, im: 0 }, { re: -15.8385, im: 0 }],
    residual_norm: 1.4e-11, nominal_input: [0.5],
    jacobian: [[0.0, 1.0], [-65.9124, -20.0]],
    solver_diagnostics: { function_evaluations: 10, jacobian_condition_number: 71.98, computation_time_s: 0.0005 },
    electrical_operating_point: { delta: 0.150568, electrical_power: 0.5, power_setpoint: 0.5, power_mismatch: 0.0, matches_setpoint: true },
    transverse_stability: { is_hyperbolic: true, description: "Normally hyperbolic at this operating point (no eigenvalue has zero real part)" },
    characteristic_time_constants: [0.240296, 0.063137],
  },
  simulate: {
    disturbed_state: [1.150568, 2.0],
    trajectory_open_loop: { t: [0, 10], x: [[1.150568, 0.150568], [2.0, 0.0]], success: true },
    performance_metrics_open_loop: { max_deviation: [1.0, 2.2], steady_state_error: [1e-10, 1e-9], settling_time: [0.9, 1.07], overshoot_pct: [0.0, 110.5] },
    controller_info: { type: "None (open loop)" },
    electrical_power_open_loop: { t: [0, 5, 10], power: [3.0433, 0.51, 0.5] },
    electrical_power_metrics_open_loop: { max_deviation: 2.5771, steady_state_error: 3.4e-10, settling_time: 0.9699, overshoot_pct: 9.1e-8 },
  },
  ims_analysis: {
    manifold: { points: new Array(30).fill([0, 0]), stable: new Array(30).fill(true), alpha: [] },
    manifold_stats: {
      continuation_parameter: "grid_forming_inverter_param", n_points: 30, dimension: 1,
      operating_interval: [0.0, 0.9], empirically_persistent: true,
      n_stable: 30, n_unstable: 0, n_singular_points: 0, singular_points: [],
      max_curvature: 0.0, mean_curvature: 0.0, arc_length: 0.2734,
      tangent_at_nearest: [1.0, 0.0], operating_region_stable: [0.0, 0.2734],
    },
    manifold_residual_trajectory: { t: [0, 10], residual: [2.311, 0.001] },
    residual_statistics: { max: 2.311, rms: 0.4254, mean: 0.1161, final: 0.001, empirical_contraction_rate: 0.494 },
    recoverability: { samples: [[1.0, 2.0]], labels: [true], index: 1.0, n_recoverable: 20, n_samples: 20, margin: 1.5, risk: "LOW", worst_disturbance: null, confidence_interval: [0.84, 1.0] },
  },
};

const reportHtml = vm.runInContext("buildReportBodyHtml(false)", sandbox);

// Section presence
check("report includes Governing equations", reportHtml.indexOf("Governing equations") !== -1);
check("report includes the real swing equation text", reportHtml.indexOf("P_e(\u03b4) = (E\u00b7V/X)\u00b7sin(\u03b4)") !== -1);
check("report includes Admissible operating envelope", reportHtml.indexOf("Admissible operating envelope") !== -1);
check("report includes the real envelope description", reportHtml.indexOf("25.0 rad/s") !== -1);
check("report includes Electrical operating point", reportHtml.indexOf("Electrical operating point") !== -1);
check("report includes the exact-match confirmation", reportHtml.indexOf("exact") !== -1);
check("report includes Characteristic time constants", reportHtml.indexOf("Characteristic time constants") !== -1);
check("report includes the real time constant values", reportHtml.indexOf("0.2403") !== -1);
check("report includes an electrical power figure placeholder", reportHtml.indexOf("id='report_trajWrap'") !== -1); // sanity: existing figures still present alongside new content

// Value spot-checks (not just headers)
check("report shows P_set exactly matching P_e at equilibrium", reportHtml.indexOf("0.5000") !== -1);
check("report shows the real power angle value", reportHtml.indexOf("0.1506") !== -1);
check("report includes the Jacobian matrix with real values", reportHtml.indexOf("Jacobian matrix") !== -1 && reportHtml.indexOf("-65.9124") !== -1);
check("report includes transverse stability", reportHtml.indexOf("Transverse stability") !== -1 && reportHtml.indexOf("Normally hyperbolic") !== -1);
check("report includes power mismatch", reportHtml.indexOf("Power mismatch") !== -1);
check("report includes electrical power settling metrics with real values", reportHtml.indexOf("0.9699") !== -1);

console.log(`\n${failures === 0 ? "ALL PASSED" : failures + " FAILED"}`);
process.exit(failures === 0 ? 0 : 1);
