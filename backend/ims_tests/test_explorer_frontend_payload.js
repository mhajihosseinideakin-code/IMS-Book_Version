/*
 * Regression test for the disturbance-vector-length bug (server.app's
 * network_mrc_simulate raised "operands could not be broadcast together
 * with shapes (3,) (4,)"): buildPayload() previously appended a spurious
 * extra element to an already-correctly-sized 3-element disturbance
 * array for the network_mrc category. This bug lived entirely in
 * explorer.html's JavaScript -- no Python-side (Flask test-client) test
 * could have caught it, since those construct correctly-shaped payloads
 * directly. This test extracts and exercises the real buildPayload()
 * function from the actual shipped file, with a minimal DOM stub, so a
 * regression here is caught the same way the original bug was found:
 * by checking the actual array length the frontend would send.
 *
 * Run: node tests/test_explorer_frontend_payload.js
 */
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const htmlPath = path.join(__dirname, "..", "ims_platform", "server", "static", "explorer.html");
const html = fs.readFileSync(htmlPath, "utf8");
const script = html.split("<script>")[1].split("</script>")[0];

// Minimal stubs: just enough for the script to load without executing
// any real DOM work (everything DOM-dependent is inside function
// bodies, not at top level, except the DOMContentLoaded registration).
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
  if (cond) {
    console.log("PASS " + name);
  } else {
    console.log("FAIL " + name + (detail ? " -> " + detail : ""));
    failures++;
  }
}

// --- Reproduce the exact network_mrc project shape from PROJECTS in server/app.py ---
sandbox.currentProject = {
  id: "network_auto_mrc",
  category: "network_mrc",
  params: [{ key: "R", def: 0.2 }, { key: "L", def: 1.5e-3 }, { key: "C", def: 2.5e-3 }, { key: "P", def: 10e3 }],
  default_km: 500.0,
  default_v_bus_init: 400.0,
  default_disturbance: [-5.0, 0.0, 0.0],   // 3 elements, matching this project's 3 states
  default_horizon: 0.02,
  state_short: ["v_bus", "i_L", "v_o"],
};

// Mirror openProject()'s formState initialization for this category.
sandbox.formState = { params: { R: 0.2, L: 1.5e-3, C: 2.5e-3, P: 10e3 } };
sandbox.formState.km = sandbox.currentProject.default_km;
sandbox.formState.v_bus_init = sandbox.currentProject.default_v_bus_init;
sandbox.formState.disturbance_value = sandbox.currentProject.default_disturbance.slice();
sandbox.formState.horizon = sandbox.currentProject.default_horizon;
sandbox.stageResults = { equilibrium: { x_star: [400.0, 25.0, 405.0] } };  // 3 elements

const payload = vm.runInContext("buildPayload('simulate')", sandbox);

check(
  "network_mrc simulate payload.disturbance has the SAME length as x_star (the actual crash condition)",
  payload.disturbance.length === sandbox.stageResults.equilibrium.x_star.length,
  `disturbance had length ${payload.disturbance.length}, x_star had length ${sandbox.stageResults.equilibrium.x_star.length}`
);
check(
  "network_mrc simulate payload.disturbance matches the registered default exactly",
  JSON.stringify(payload.disturbance) === JSON.stringify([-5.0, 0.0, 0.0]),
  `got ${JSON.stringify(payload.disturbance)}`
);

// Also check the ims_analysis stage, which shares the same code path.
const payload2 = vm.runInContext("buildPayload('ims_analysis')", sandbox);
check(
  "network_mrc ims_analysis payload.disturbance also correctly sized",
  payload2.disturbance.length === 3,
  `got length ${payload2.disturbance.length}`
);

// --- Basic vs Advanced mode for IMS Analysis on a custom_network:
// regression test for a real reported issue -- Basic mode was still
// running the full Monte-Carlo + boundary-tracing pipeline regardless
// of the mode toggle, and the deterministic (manifold-residual)
// assessment never actually ran here at all because no disturbance was
// ever included in the ims_analysis payload for this category.
(function () {
  sandbox.currentProject = { id: "custom", category: "custom_network", label: "Custom Network", state_short: ["v1", "v2"] };
  sandbox.builderSpec = { buses: [], lines: [], loads: [], converters: [], sources: [], input_component_id: null };
  sandbox.formState = {
    params: {}, nominal_input: [0.5], custom_state_guess: [1, 1], disturbance_value: [0.1, 0.1],
    sweep_range: [0, 1], sweep_points: 10, radius: 1.0, n_samples: 10, recovery_tol: 0.1,
    horizon: 1.0, trace_boundary: true,  // checkbox is checked, but mode is Basic
  };
  sandbox.stageResults = { equilibrium: { x_star: [1, 1] } };

  sandbox.formState.ims_advanced_mode = false;
  const basicPayload = vm.runInContext("buildPayload('ims_analysis')", sandbox);
  check("Basic mode disables Monte Carlo (recoverability_enabled=false)", basicPayload.recoverability_enabled === false);
  check("Basic mode disables boundary tracing even if the checkbox is checked", basicPayload.trace_boundary === false);
  check("Basic mode STILL sends a disturbance, so the deterministic assessment actually runs",
    basicPayload.disturbance && Array.isArray(basicPayload.disturbance.value));
  check("payload defaults residual_projection_method to auto when not explicitly set", basicPayload.residual_projection_method === "auto");

  sandbox.formState.ims_advanced_mode = true;
  const advPayload = vm.runInContext("buildPayload('ims_analysis')", sandbox);
  check("Advanced mode enables Monte Carlo", advPayload.recoverability_enabled === true);
  check("Advanced mode respects the boundary-tracing checkbox", advPayload.trace_boundary === true);

  sandbox.formState.residual_projection_method = "newton_refined";
  const explicitPayload = vm.runInContext("buildPayload('ims_analysis')", sandbox);
  check("payload respects an explicitly-selected residual projection method", explicitPayload.residual_projection_method === "newton_refined");
})();

// --- Regression test for a real reported bug: a wide (e.g. 20-state)
// Jacobian matrix rendered as a single continuous row overflowed the
// page and was cut off mid-digit in PDF export. buildMatrixTableHtml
// splits into labeled column blocks so this can't happen regardless of
// matrix size.
(function () {
  const matrix = [];
  for (let i = 0; i < 12; i++) { matrix.push(Array.from({ length: 12 }, (_, j) => i * 100 + j)); }
  const stateNames = Array.from({ length: 12 }, (_, i) => "v_bus" + (i + 1));
  const result = vm.runInContext("buildMatrixTableHtml(" + JSON.stringify(matrix) + ", " + JSON.stringify(stateNames) + ")", sandbox);
  check("a 12-column matrix splits into multiple column blocks (was one unbounded row before)", (result.match(/<table/g) || []).length === 2);
  check("block headers correctly label the column ranges", result.includes("Columns 1\u20138 of 12") && result.includes("Columns 9\u201312 of 12"));
  check("no values are lost across the split (checks the last row/column corner value)", result.includes("1111.00"));
  check("state names are used as row/column labels, not just x1, x2, ...", result.includes("v_bus12"));
})();

// --- formatEng: engineering-unit auto-scaling, per the report-quality
// request that no value should be shown without a unit or in a
// hard-to-read raw form (e.g. 0.0000008 rather than 0.8 us).
(function () {
  check("formatEng scales large voltage to kV", vm.runInContext("formatEng(3500, 'V')", sandbox) === "3.50 kV");
  check("formatEng scales large power to MW", vm.runInContext("formatEng(1250000, 'W')", sandbox) === "1.25 MW");
  check("formatEng keeps a normal-range value in base units", vm.runInContext("formatEng(211.3559, 'V')", sandbox) === "211.36 V");
  check("formatEng scales a tiny time value to microseconds", vm.runInContext("formatEng(0.000001, 's')", sandbox) === "1.00 \u03bcs");
  check("formatEng handles exactly zero without a division error", vm.runInContext("formatEng(0, 'A')", sandbox) === "0 A");
  check("formatEng handles null/undefined gracefully", vm.runInContext("formatEng(null, 'V')", sandbox) === "\u2014");
})();

// --- stateVariableInfo: physical meaning and units for state
// variables, requested to replace the previous bare "x1 = v_bus1"
// listing with real engineering information.
(function () {
  const cases = [
    ["v_bus1", "Bus 'bus1' Voltage", "V"],
    ["i_line1", "Line 'line1' Current", "A"],
    ["conv1_em_i_L", "'conv1' Inductor Current", "A"],
    ["conv1_ctrl_e_int", "'conv1' Controller Integral State", "V\u00b7s (voltage-error integral)"],
  ];
  for (const [name, expectedMeaning, expectedUnit] of cases) {
    const info = vm.runInContext("stateVariableInfo(" + JSON.stringify(name) + ")", sandbox);
    check(`stateVariableInfo(${name}) gives correct physical meaning`, info.meaning === expectedMeaning, `got ${info.meaning}`);
    check(`stateVariableInfo(${name}) gives correct unit`, info.unit === expectedUnit, `got ${info.unit}`);
  }
})();

// --- paramUnit: engineering units inferred from the platform's own
// parameter naming convention, requested to add units to the
// previously-bare "C_bus1 = 0.001" style parameter listing.
(function () {
  const cases = [
    ["C_bus1", "F"], ["L_line1", "H"], ["R_line1", "\u03a9"],
    ["conv1_ctrl_d", "(dimensionless, duty ratio)"], ["conv1_em_L", "H"],
    ["conv1_em_R_L", "\u03a9"],  // the tricky case: name ends in "_L" but means resistance
    ["conv1_em_v_in", "V"], ["load1_P", "W"], ["load1_v_floor", "V"],
    ["src1_R_source", "\u03a9"], ["src1_v_source", "V"],
    ["conv1_ctrl_Kp", "(controller gain)"], ["conv1_ctrl_Ki", "(controller gain)"],
  ];
  for (const [name, expected] of cases) {
    const got = vm.runInContext("paramUnit(" + JSON.stringify(name) + ")", sandbox);
    check(`paramUnit(${name}) === ${expected}`, got === expected, `got ${JSON.stringify(got)}`);
  }
})();

// --- categorizeParam / buildCategorizedParamsTableHtml: parameter
// grouping and unit display, requested to replace one long flat
// unlabeled parameter list.
(function () {
  const catCases = [
    ["C_bus1", "Network Parameters"], ["L_line1", "Network Parameters"], ["R_line1", "Network Parameters"],
    ["conv1_ctrl_d", "Controller Parameters"], ["conv1_em_L", "Converter Parameters"],
    ["load1_P", "Load Parameters"], ["src1_v_source", "Source Parameters"],
  ];
  for (const [name, expected] of catCases) {
    const got = vm.runInContext("categorizeParam(" + JSON.stringify(name) + ")", sandbox);
    check(`categorizeParam(${name}) === ${expected}`, got === expected, `got ${got}`);
  }
  const params = { C_bus1: 0.001, L_line1: 0.0001, R_line1: 0.1, conv1_ctrl_d: 0.5, conv1_em_L: 0.001, load1_P: 500, src1_v_source: 400 };
  const tableHtml = vm.runInContext("buildCategorizedParamsTableHtml(" + JSON.stringify(params) + ")", sandbox);
  ["Network Parameters", "Converter Parameters", "Controller Parameters", "Load Parameters", "Source Parameters"].forEach(function (cat) {
    check(`categorized parameter table includes a "${cat}" group`, tableHtml.includes(cat));
  });
  check("categorized parameter table shows the correct unit for a capacitance", tableHtml.includes(">F<"));
})();

// --- axisLabelForState / colorForState: publication-quality axis
// labels with units and consistent quantity-type color coding
// (voltage=blue, current=green, control=purple), requested to replace
// bare state names on axes and the previous mostly-orange charts.
(function () {
  check("axisLabelForState(v_bus1) gives a unit-labeled voltage axis", vm.runInContext("axisLabelForState('v_bus1')", sandbox) === "Voltage (V)");
  check("axisLabelForState(i_line1) gives a unit-labeled current axis", vm.runInContext("axisLabelForState('i_line1')", sandbox) === "Current (A)");
  check("colorForState(v_bus1) uses the voltage color", vm.runInContext("colorForState('v_bus1')", sandbox) === vm.runInContext("COLORS.voltage", sandbox));
  check("colorForState(i_line1) uses the current color", vm.runInContext("colorForState('i_line1')", sandbox) === vm.runInContext("COLORS.current", sandbox));
  check("voltage and current colors are visually distinct (not both orange)", vm.runInContext("COLORS.voltage", sandbox) !== vm.runInContext("COLORS.current", sandbox));
})();

// --- renderTrajectoryPlot: confirms the REAL function (not just the
// label-helper functions in isolation) passes unit-labeled axes to
// drawGrid. Chart text is drawn on canvas, not present in the DOM/HTML
// as searchable text, so this intercepts drawGrid's actual arguments
// rather than trying to inspect rendered output.
(function () {
  function makeCtxStub() { return new Proxy({}, { get: () => () => {} }); }
  function makeCanvasStub() { return { style: {}, getContext: () => makeCtxStub(), width: 500, height: 150 }; }
  const registry2 = {};
  function makeEl2(tag) {
    return { style: {}, appendChild: () => {}, classList: { toggle: () => {} }, clientWidth: 500, set innerHTML(v) {}, get innerHTML() { return ""; } };
  }
  const sandbox2 = {
    document: {
      addEventListener: () => {}, getElementById: (id) => registry2[id] || makeEl2("div"),
      querySelectorAll: () => [], createElement: (tag) => tag === "canvas" ? makeCanvasStub() : makeEl2(tag),
    },
    window: { addEventListener: () => {}, devicePixelRatio: 1 }, console,
  };
  sandbox2.window.document = sandbox2.document;
  vm.createContext(sandbox2);
  vm.runInContext(script, sandbox2);
  registry2["trajWrap"] = makeEl2("div");
  registry2["trajLegend"] = makeEl2("div");
  const drawGridCalls = [];
  sandbox2.__drawGridCalls = drawGridCalls;
  vm.runInContext("drawGrid = function(ctx, bounds, w, h, m, xLabel, yLabel) { __drawGridCalls.push([xLabel, yLabel]); };", sandbox2);
  vm.runInContext("drawLine = function() {}; drawMarker = function() {};", sandbox2);
  sandbox2.currentProject = { id: "p", category: "single_model", state_short: ["v_bus1", "i_line1"] };
  const data = { trajectory_open_loop: { t: [0, 0.1], x: [[1, 1], [2, 2]] } };
  vm.runInContext("renderTrajectoryPlot(" + JSON.stringify(data) + ", 'trajWrap', 'trajLegend')", sandbox2);
  check("renderTrajectoryPlot passes a unit-labeled voltage y-axis to drawGrid", drawGridCalls[0][1] === "Voltage (V)", `got ${JSON.stringify(drawGridCalls[0])}`);
  check("renderTrajectoryPlot passes a unit-labeled current y-axis to drawGrid", drawGridCalls[1][1] === "Current (A)", `got ${JSON.stringify(drawGridCalls[1])}`);
  check("renderTrajectoryPlot passes a capitalized Time (s) x-axis on the last subplot", drawGridCalls[1][0] === "Time (s)", `got ${JSON.stringify(drawGridCalls[1])}`);
})();

// --- Recent Projects tracking: a real, working feature (localStorage),
// requested to replace what would otherwise be faked/placeholder data.
(function () {
  const store = {};
  const localStorage = { getItem: (k) => store[k] || null, setItem: (k, v) => { store[k] = v; } };
  const sandbox3 = { document: { addEventListener: () => {}, getElementById: () => null }, window: {}, console, localStorage };
  sandbox3.window.document = sandbox3.document;
  vm.createContext(sandbox3);
  vm.runInContext(script, sandbox3);

  vm.runInContext("recordRecentProject('buck_converter', 'Buck Converter')", sandbox3);
  vm.runInContext("recordRecentProject('boost_converter', 'Boost Converter')", sandbox3);
  vm.runInContext("recordRecentProject('buck_converter', 'Buck Converter')", sandbox3); // re-open
  const recents = vm.runInContext("getRecentProjects()", sandbox3);
  check("recording the same project twice deduplicates rather than creating a duplicate entry", recents.length === 2, `got ${recents.length}`);
  check("re-opening a project moves it to the front (most-recent-first)", recents[0].id === "buck_converter", `got ${JSON.stringify(recents.map(r => r.id))}`);

  // Confirms the cap is enforced, not unbounded growth
  for (let i = 0; i < 10; i++) vm.runInContext(`recordRecentProject('p${i}', 'Project ${i}')`, sandbox3);
  const capped = vm.runInContext("getRecentProjects()", sandbox3);
  check("recent projects list is capped, not growing unbounded", capped.length === vm.runInContext("RECENT_PROJECTS_MAX", sandbox3));
})();

console.log(`\n${failures === 0 ? "ALL PASSED" : failures + " FAILED"}`);
process.exit(failures === 0 ? 0 : 1);
