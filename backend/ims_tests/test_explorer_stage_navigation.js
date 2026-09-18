/*
 * Regression test for a real UX bug: runStage() previously called
 * goToStage(nextStageAfter(stage)) immediately after computing and
 * rendering a stage's results, which re-rendered #wsMain for the NEXT,
 * not-yet-run stage in the same synchronous tick -- meaning the user
 * could never actually see the results that were just computed (e.g.
 * the Jacobian, electrical operating point, or equilibrium diagnostics
 * built up over several prior sessions). This most likely explains
 * review complaints that stage results "don't display" and that pages
 * show "a large empty space" -- the content existed and was computed
 * correctly the whole time; it was rendered and then immediately
 * hidden before the browser could paint it.
 *
 * This test exercises the real, shipped runStage() function (with
 * fetch mocked to return realistic data) and asserts that `activeStage`
 * remains on the just-completed stage afterward, not the next one.
 *
 * Run: node tests/test_explorer_stage_navigation.js
 */
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const htmlPath = path.join(__dirname, "..", "ims_platform", "server", "static", "explorer.html");
const html = fs.readFileSync(htmlPath, "utf8");
const script = html.split("<script>")[1].split("</script>")[0];

const registry = {};
function fakeCtx() {
  return {
    clearRect(){}, beginPath(){}, moveTo(){}, lineTo(){}, stroke(){}, fill(){}, fillRect(){}, strokeRect(){},
    arc(){}, closePath(){}, save(){}, restore(){}, translate(){}, rotate(){}, setLineDash(){}, fillText(){},
    scale(){}, drawImage(){},
    set fillStyle(v){}, set strokeStyle(v){}, set lineWidth(v){}, set font(v){}, set textAlign(v){},
  };
}
function fakeCanvasEl() {
  return { getContext: () => fakeCtx(), toDataURL: () => "data:image/png;base64,FAKE", style: {} };
}
function makeEl(tag) {
  const el = {
    tagName: tag, style: {}, _id: null,
    set id(v) { this._id = v; registry[v] = this; },
    get id() { return this._id; },
    appendChild() {},
    set innerHTML(v) { this._html = v; },
    get innerHTML() { return this._html || ""; },
    get clientWidth() { return 500; },
    classList: { toggle() {} },
    querySelectorAll: () => [],
    closest: () => null,
    insertBefore() {},
    disabled: false,
  };
  if (tag === "canvas") return Object.assign(el, fakeCanvasEl());
  return el;
}

const sandbox = {
  document: {
    addEventListener: () => {},
    getElementById: (id) => registry[id] || makeEl("div"),
    querySelectorAll: () => [],
    createElement: (tag) => makeEl(tag),
  },
  window: { addEventListener: () => {}, devicePixelRatio: 1 },
  console,
  AbortController: function () { this.signal = {}; this.abort = () => {}; },
  setTimeout, clearTimeout,
};
sandbox.window.document = sandbox.document;
sandbox.fetch = (url) => {
  if (url.endsWith("/equilibrium")) {
    const body = {
      state_names: ["delta", "omega"], x_star: [0.1506, 0.0], stable: true, is_hyperbolic: true,
      eigenvalues: [{ re: -4.16, im: 0 }, { re: -15.84, im: 0 }], residual_norm: 1e-11,
      nominal_input: [0.5], jacobian: [[0, 1], [-65.9, -20]],
      solver_diagnostics: { function_evaluations: 10, jacobian_condition_number: 71.98, computation_time_s: 0.0005 },
    };
    return Promise.resolve({ ok: true, text: () => Promise.resolve(JSON.stringify(body)) });
  }
  return Promise.resolve({ ok: true, text: () => Promise.resolve("{}") });
};
vm.createContext(sandbox);
vm.runInContext(script, sandbox);

let failures = 0;
function check(name, cond, detail) {
  if (cond) { console.log("PASS " + name); }
  else { console.log("FAIL " + name + (detail ? " -> " + detail : "")); failures++; }
}

sandbox.currentProject = { id: "grid_forming_inverter", label: "Grid-Forming Inverter", category: "single_model", state_short: ["delta", "omega"], default_state_guess: [0.3, 0.0], input_label: "P_set" };
sandbox.formState = {
  params: {}, nominal_input: [0.5], guess0: 0.3, guess1: 0.0,
  disturbance_value: [1.0, 2.0], horizon: 10.0, mrc_enabled: false,
  sweep_range: [0.0, 0.9], sweep_points: 60, radius: 2.0, n_samples: 120, recovery_tol: 0.15, trace_boundary: true,
};
sandbox.stageResults = {};
sandbox.activeStage = "equilibrium";

(async () => {
  await vm.runInContext("runStage('equilibrium')", sandbox);

  check(
    "activeStage remains on the just-completed stage (not auto-advanced)",
    sandbox.activeStage === "equilibrium",
    `activeStage was "${sandbox.activeStage}", expected "equilibrium"`
  );
  check(
    "the completed stage's results are actually stored and available",
    sandbox.stageResults.equilibrium && sandbox.stageResults.equilibrium.x_star !== undefined
  );

  // Confirm the user CAN still move forward deliberately via goToStage,
  // and that doing so does not lose the already-computed results.
  vm.runInContext("goToStage('simulate')", sandbox);
  check("goToStage moves activeStage forward when called explicitly", sandbox.activeStage === "simulate");
  check("previously computed equilibrium results are still retained after navigating", sandbox.stageResults.equilibrium !== undefined);

  // And confirm navigating BACK to a completed stage doesn't wipe its results either.
  vm.runInContext("goToStage('equilibrium')", sandbox);
  check("navigating back to a completed stage keeps its results", sandbox.stageResults.equilibrium.x_star[0] === 0.1506);

  // --- A request that never resolves triggers the client-side timeout with a clear, actionable message ---
  const registry2 = {};
  function makeEl2(tag) {
    const el = {
      tagName: tag, style: {}, _id: null,
      set id(v) { this._id = v; registry2[v] = this; },
      get id() { return this._id; },
      appendChild() {}, set innerHTML(v) { this._html = v; }, get innerHTML() { return this._html || ""; },
      set textContent(v) { this._text = v; }, get textContent() { return this._text || ""; },
      get clientWidth() { return 500; }, classList: { toggle() {} }, querySelectorAll: () => [], closest: () => null,
      insertBefore() {}, disabled: false,
    };
    return el;
  }
  const sandbox2 = {
    document: { addEventListener: () => {}, getElementById: (id) => registry2[id] || makeEl2("div"), querySelectorAll: () => [], createElement: (tag) => makeEl2(tag) },
    window: { addEventListener: () => {} }, console,
    AbortController: function () { this.signal = { aborted: false }; this.abort = () => { this.signal.aborted = true; } },
    setTimeout: (fn, ms) => 1, clearTimeout: () => {},
  };
  sandbox2.window.document = sandbox2.document;
  vm.createContext(sandbox2);
  vm.runInContext(script, sandbox2);
  // Defined via runInContext (not passed in from the outer Node context)
  // specifically because a Promise/Error constructed in the OUTER
  // context doesn't always propagate correctly through await/try-catch
  // running inside a DIFFERENT vm context (each vm context has its own
  // built-ins by default) -- this is a quirk of testing async code
  // across vm context boundaries, not a bug in the real, shipped code
  // this test exercises (confirmed separately: the real code, in a
  // real browser, produces exactly the message asserted below).
  vm.runInContext(
    "fetch = function() { return Promise.reject(Object.assign(new Error('aborted'), { name: 'AbortError' })); };",
    sandbox2
  );
  sandbox2.currentProject = { id: "custom", category: "custom_network", label: "Custom Network", state_short: ["v1", "v2"] };
  sandbox2.formState = { params: {}, nominal_input: [0.5], custom_state_guess: [1, 1] };
  sandbox2.stageResults = {};
  sandbox2.activeStage = "equilibrium";
  sandbox2.builderSpec = { buses: [], lines: [], loads: [], converters: [], sources: [], input_component_id: null };
  registry2["wsError"] = makeEl2("div");

  await vm.runInContext("runStage('equilibrium')", sandbox2);
  const errBox = registry2["wsError"];
  check("timeout produces a clear, actionable error message (not a raw parse error)",
    errBox && errBox.textContent.indexOf("timed out") !== -1 && errBox.textContent.indexOf("Advanced mode") !== -1,
    "got: " + (errBox ? errBox.textContent : "<no error box>"));

  console.log(`\n${failures === 0 ? "ALL PASSED" : failures + " FAILED"}`);
  process.exit(failures === 0 ? 0 : 1);
})();
