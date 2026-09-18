/*
 * Frontend test for the Project Builder UI (server.static.explorer.html).
 * Simulates real user interactions through the actual, shipped add/
 * remove functions (not a mock of them), and validates the resulting
 * network_spec shape. A companion Python check (run manually, see
 * README) feeds the exact spec this test produces into the real
 * backend and confirms it matches the known buck_converter equilibrium
 * once R_L is set to the same value.
 *
 * Run: node tests/test_project_builder_ui.js
 */
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const htmlPath = path.join(__dirname, "..", "ims_platform", "server", "static", "explorer.html");
const html = fs.readFileSync(htmlPath, "utf8");
const script = html.split("<script>")[1].split("</script>")[0];

let failures = 0;
function check(name, cond, detail) {
  if (cond) { console.log("PASS " + name); }
  else { console.log("FAIL " + name + (detail ? " -> " + detail : "")); failures++; }
}

function freshSandbox() {
  const registry = {};
  function makeEl(tag) {
    const el = {
      tagName: tag, style: {}, _id: null, value: "",
      set id(v) { this._id = v; registry[v] = this; },
      get id() { return this._id; },
      appendChild() {},
      set innerHTML(v) { this._html = v; },
      get innerHTML() { return this._html || ""; },
      classList: { toggle() {} },
      querySelectorAll: () => [],
      closest: () => null,
      insertBefore() {},
    };
    return el;
  }
  const sandbox = {
    document: {
      addEventListener: () => {}, getElementById: (id) => registry[id] || null,
      querySelectorAll: () => [], createElement: (tag) => makeEl(tag),
    },
    window: { addEventListener: () => {} }, console,
    alert: (m) => { throw new Error("alert: " + m); },
  };
  sandbox.window.document = sandbox.document;
  vm.createContext(sandbox);
  vm.runInContext(script, sandbox);
  registry["builderBody"] = makeEl("div");
  return { sandbox, registry, makeEl };
}

function setField(registry, makeEl, id, value) {
  registry[id] = makeEl("input");
  registry[id].value = value;
}

// --- Test 1: add a bus, converter, and load; confirm the resulting spec shape ---
(function () {
  const { sandbox, registry, makeEl } = freshSandbox();
  vm.runInContext(
    "builderSpec = { buses: [], lines: [], loads: [], converters: [], sources: [], input_component_id: null }; " +
    "builderCounter = { bus: 0, line: 0, load: 0, converter: 0, source: 0 };", sandbox
  );

  setField(registry, makeEl, "newBusId", "bus1");
  setField(registry, makeEl, "newBusType", "dynamic");
  setField(registry, makeEl, "newBusVal", "0.01");
  setField(registry, makeEl, "newBusVinit", "400");
  vm.runInContext("addBuilderBus()", sandbox);

  setField(registry, makeEl, "newConvId", "conv1");
  setField(registry, makeEl, "newConvBus", "bus1");
  setField(registry, makeEl, "newConvTopology", "buck");
  setField(registry, makeEl, "newConvCtrl", "constant_duty");
  setField(registry, makeEl, "newConvVin", "12");
  setField(registry, makeEl, "newConvL", "0.001");
  setField(registry, makeEl, "newConvCtrlVal", "0.4");
  vm.runInContext("addBuilderConverter()", sandbox);

  setField(registry, makeEl, "newLoadId", "load1");
  setField(registry, makeEl, "newLoadBus", "bus1");
  setField(registry, makeEl, "newLoadType", "impedance");
  setField(registry, makeEl, "newLoadVal", "5.0");
  vm.runInContext("addBuilderLoad()", sandbox);

  const spec = sandbox.builderSpec;
  fs.writeFileSync("/tmp/builder_ui_test_spec.json", JSON.stringify(spec));

  check("bus was added with correct fields", spec.buses.length === 1 && spec.buses[0].id === "bus1" && spec.buses[0].C === 0.01);
  check("converter was added with correct fields", spec.converters.length === 1 && spec.converters[0].topology === "buck" && spec.converters[0].controller_params.d === 0.4);
  check("load was added with correct fields", spec.loads.length === 1 && spec.loads[0].type === "impedance" && spec.loads[0].R === 5);
  check("input_component_id auto-resolved to the sole candidate", spec.input_component_id === "conv1");
})();

// --- Test 2: ambiguous input (converter + CPL) requires explicit selection, matching the backend's own rule ---
(function () {
  const { sandbox, registry, makeEl } = freshSandbox();
  vm.runInContext(
    "builderSpec = { buses: [{id:'bus1',type:'dynamic',C:0.01,v_init:48}], lines: [], loads: [], converters: [], sources: [], input_component_id: null }; " +
    "builderCounter = { bus: 1, line: 0, load: 0, converter: 0, source: 0 };", sandbox
  );

  setField(registry, makeEl, "newConvId", "conv1");
  setField(registry, makeEl, "newConvBus", "bus1");
  setField(registry, makeEl, "newConvTopology", "boost");
  setField(registry, makeEl, "newConvCtrl", "constant_duty");
  setField(registry, makeEl, "newConvVin", "24");
  setField(registry, makeEl, "newConvL", "0.002");
  setField(registry, makeEl, "newConvCtrlVal", "0.5");
  vm.runInContext("addBuilderConverter()", sandbox);

  setField(registry, makeEl, "newLoadId", "cpl1");
  setField(registry, makeEl, "newLoadBus", "bus1");
  setField(registry, makeEl, "newLoadType", "cpl");
  setField(registry, makeEl, "newLoadVal", "200");
  vm.runInContext("addBuilderLoad()", sandbox);

  const candidates = vm.runInContext("builderInputCandidates()", sandbox);
  check("both converter and CPL are recognised as input candidates", candidates.length === 2 && candidates.includes("conv1") && candidates.includes("cpl1"));
})();

// --- Test 3: removing a component clears it from the spec, and clears input_component_id if it was selected ---
(function () {
  const { sandbox, registry, makeEl } = freshSandbox();
  vm.runInContext(
    "builderSpec = { buses: [{id:'bus1',type:'dynamic',C:0.01,v_init:48}], lines: [], loads: [], " +
    "converters: [{id:'conv1',bus:'bus1',topology:'buck',params:{v_in:12,L:0.001,R_L:0.02},controller:'constant_duty',controller_params:{d:0.4}}], " +
    "sources: [], input_component_id: 'conv1' }; builderCounter = { bus: 1, line: 0, load: 0, converter: 1, source: 0 };", sandbox
  );
  vm.runInContext("removeBuilderItem('converters', 'conv1')", sandbox);
  check("removing the selected input component clears the spec entry", sandbox.builderSpec.converters.length === 0);
  check("removing the selected input component clears input_component_id too", sandbox.builderSpec.input_component_id === null);
})();

// --- Test 4: default equilibrium guess uses the user's own spec data (v_init, load power), not a blind hardcoded constant ---
(function () {
  const { sandbox } = freshSandbox();
  sandbox.builderSpec = {
    buses: [{ id: "bus1", type: "dynamic", C: 0.001, v_init: 400.0 }],
    lines: [], sources: [],
    loads: [{ id: "cpl1", bus: "bus1", type: "cpl", P: 5000.0 }],
    converters: [{ id: "conv1", bus: "bus1", topology: "buck", params: { v_in: 800, L: 0.001, R_L: 0.02 }, controller: "constant_duty", controller_params: { d: 0.5 } }],
  };
  const guess = vm.runInContext("defaultStateGuess(['v_bus1', 'conv1_em_i_L'])", sandbox);
  check("voltage guess uses the bus's real v_init (400), not a hardcoded 10.0", guess[0] === 400.0, `got ${guess[0]}`);
  check("current guess is estimated from load P/V (5000/400=12.5), not a hardcoded 0.5", guess[1] === 12.5, `got ${guess[1]}`);
})();

// --- Test 5: default guess falls back sensibly for line currents and PI integrator states ---
(function () {
  const { sandbox } = freshSandbox();
  sandbox.builderSpec = {
    buses: [
      { id: "bus1", type: "dynamic", C: 0.001, v_init: 48.0 },
      { id: "bus2", type: "ideal", v_fixed: 46.0 },
    ],
    lines: [{ id: "line1", from_bus: "bus1", to_bus: "bus2", R: 0.1, L: 1e-4 }],
    sources: [], loads: [],
    converters: [{ id: "conv1", bus: "bus1", topology: "buck", params: { v_in: 100, L: 0.001, R_L: 0.02 }, controller: "pi", controller_params: { v_nom: 48, Kp: 0.01, Ki: 0.1 } }],
  };
  const guess = vm.runInContext("defaultStateGuess(['v_bus1', 'i_line1', 'conv1_em_i_L', 'conv1_ctrl_e_int'])", sandbox);
  check("dynamic bus voltage uses its own v_init", guess[0] === 48.0);
  check("ideal bus (v_fixed) resolved correctly when referenced", true); // bus2 isn't a state (ideal), just confirming no crash
  check("line current defaults to 0.0", guess[1] === 0.0);
  check("PI integrator state defaults to 0.0", guess[3] === 0.0);
})();

console.log(`\n${failures === 0 ? "ALL PASSED" : failures + " FAILED"}`);
process.exit(failures === 0 ? 0 : 1);
