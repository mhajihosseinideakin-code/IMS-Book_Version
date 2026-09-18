import { useMemo, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ReferenceLine, ReferenceArea,
} from "recharts";
import { toast } from "sonner";
import { Toaster } from "@/components/ui/sonner";
import { Button, buttonVariants } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import {
  Sigma, Activity, ShieldCheck, Compass, CheckCircle2, XCircle, AlertTriangle,
  Play, Calculator, Download, FileText, Waves,
} from "lucide-react";
import { ApiError } from "@/lib/api";
import {
  NOMINAL_PARAMS, validateParams, getEquilibrium, getStability, runSimulation,
  csvUrl, pdfUrl,
} from "@/lib/ims";
import type {
  FourStateParams, EquilibriumResponse, StabilityResponse, SimDataset, Complex, Perturbation,
} from "@/lib/ims";

const SIG = {
  v_b: "#38BDF8", i_l: "#34D399", sigma: "#FB923C", v_o: "#A78BFA",
  e: "#F43F5E", theory: "#34D399", eq: "#64748B", dist: "#E11D48",
};

const PARAM_META: { key: keyof FourStateParams; unit: string; label: string }[] = [
  { key: "R", unit: "Ω", label: "Line resistance R" },
  { key: "L", unit: "H", label: "Line inductance L" },
  { key: "C", unit: "F", label: "Bus capacitance C" },
  { key: "P", unit: "W", label: "Nominal CPL power P" },
  { key: "v_nom", unit: "V", label: "Nominal bus voltage v_nom" },
  { key: "R_v", unit: "Ω", label: "Virtual droop R_v" },
  { key: "K_i", unit: "1/s", label: "Integral gain K_i" },
  { key: "k_m", unit: "1/s", label: "Contraction rate k_m" },
];

function fmt(x: number, p = 4): string {
  if (!isFinite(x)) return "—";
  if (x !== 0 && (Math.abs(x) < 1e-3 || Math.abs(x) >= 1e5)) return x.toExponential(p - 1);
  return Number(x.toPrecision(p)).toString();
}
function cx(z: Complex): string {
  if (Math.abs(z.im) < 1e-9) return fmt(z.re);
  return `${fmt(z.re)} ${z.im >= 0 ? "+" : "-"} ${fmt(Math.abs(z.im))}j`;
}

function Metric({ label, value, mono = true, testid }: { label: string; value: string; mono?: boolean; testid?: string }) {
  return (
    <div className="flex items-baseline justify-between gap-3 border-b border-border/50 py-1.5">
      <span className="text-[11px] uppercase tracking-wider text-muted-foreground">{label}</span>
      <span data-testid={testid} className={`${mono ? "font-mono" : ""} text-sm tabular-nums text-foreground`}>{value}</span>
    </div>
  );
}

function Panel({ title, icon, children, className = "" }: { title: string; icon?: React.ReactNode; children: React.ReactNode; className?: string }) {
  return (
    <Card className={`bg-card border-border p-4 gap-3 ${className}`}>
      <div className="flex items-center gap-2 text-foreground">
        {icon}
        <h3 className="text-[15px] font-semibold tracking-tight">{title}</h3>
      </div>
      {children}
    </Card>
  );
}

const CLAIM_TIERS = [
  { id: "exact_identity", color: "#10B981", Icon: Sigma, title: "Exact Contraction Identity",
    def: "ė_Σ = −k_m·e_Σ holds algebraically for the ideal unsaturated model while x stays in D = {v_b>0}." },
  { id: "local_stability", color: "#0284C7", Icon: ShieldCheck, title: "Local Equilibrium Stability",
    def: "Routh–Hurwitz margins + Re(λ)<0 at x* — a linearisation result, NOT a region of attraction." },
  { id: "observed_recovery", color: "#F59E0B", Icon: Activity, title: "Observed Finite-Horizon Recovery",
    def: "A single simulated trajectory returning near x* — empirical evidence, not a certified basin." },
  { id: "certified_regional", color: "#A855F7", Icon: Compass, title: "Certified Regional IMS",
    def: "Explicit invariant region + Lyapunov/contraction certificate. NOT established in this milestone." },
];

function StatePlot({ dataset, dkey, label, unit, refVal, color }: {
  dataset: SimDataset; dkey: "v_b" | "i_l" | "sigma" | "v_o"; label: string; unit: string; refVal: number; color: string;
}) {
  const data = useMemo(() => dataset.t.map((t, i) => ({ t: t * 1000, y: dataset[dkey][i] })), [dataset, dkey]);
  const events = dataset.events;
  return (
    <div data-testid={`chart-container-${dkey}`}>
      <div className="flex items-center justify-between px-1">
        <span className="text-[11px] uppercase tracking-wider text-muted-foreground">{label}</span>
        <span className="font-mono text-[11px] text-muted-foreground">ref {fmt(refVal)} {unit}</span>
      </div>
      <ResponsiveContainer width="100%" height={130}>
        <LineChart data={data} syncId="ims-mrc-sim" margin={{ top: 4, right: 12, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
          <XAxis dataKey="t" tick={{ fontSize: 10, fill: "#64748B" }} tickFormatter={(v: number) => v.toFixed(0)} stroke="#334155" />
          <YAxis tick={{ fontSize: 10, fill: "#64748B" }} width={54} stroke="#334155" tickFormatter={(v: number) => fmt(v, 3)} domain={["auto", "auto"]} />
          <Tooltip contentStyle={{ background: "#0F172A", border: "1px solid #334155", fontSize: 11 }}
            labelFormatter={(l: number) => `t = ${l.toFixed(3)} ms`} formatter={(v: number) => [fmt(v, 5), label]} />
          <ReferenceLine y={refVal} stroke={SIG.eq} strokeDasharray="5 4" />
          {events.map((e, i) => (
            <ReferenceLine key={i} x={e.t * 1000} stroke={SIG.dist} strokeDasharray="2 2" />
          ))}
          <Line type="monotone" dataKey="y" stroke={color} dot={false} strokeWidth={1.6} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function ResidualPlot({ dataset, logScale }: { dataset: SimDataset; logScale: boolean }) {
  const data = useMemo(() => dataset.t.map((t, i) => ({
    t: t * 1000,
    e: logScale ? Math.max(Math.abs(dataset.e_sigma[i]), 1e-12) : dataset.e_sigma[i],
    th: logScale ? Math.max(Math.abs(dataset.e_sigma_theory[i]), 1e-12) : dataset.e_sigma_theory[i],
  })), [dataset, logScale]);
  return (
    <div data-testid="chart-container-esigma">
      <ResponsiveContainer width="100%" height={150}>
        <LineChart data={data} syncId="ims-mrc-sim" margin={{ top: 4, right: 12, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
          <XAxis dataKey="t" tick={{ fontSize: 10, fill: "#64748B" }} tickFormatter={(v: number) => v.toFixed(0)} stroke="#334155"
            label={{ value: "t (ms)", position: "insideBottomRight", fontSize: 10, fill: "#64748B" }} />
          <YAxis tick={{ fontSize: 10, fill: "#64748B" }} width={54} stroke="#334155"
            scale={logScale ? "log" : "linear"} domain={logScale ? [1e-9, "auto"] : ["auto", "auto"]}
            tickFormatter={(v: number) => fmt(v, 2)} allowDataOverflow />
          <Tooltip contentStyle={{ background: "#0F172A", border: "1px solid #334155", fontSize: 11 }}
            labelFormatter={(l: number) => `t = ${l.toFixed(3)} ms`}
            formatter={(v: number, n: string) => [fmt(v, 5), n === "e" ? "e_Σ sim" : "e0·e^(−k_m t)"]} />
          {dataset.events.map((e, i) => (
            <ReferenceLine key={i} x={e.t * 1000} stroke={SIG.dist} strokeDasharray="2 2" />
          ))}
          <Line type="monotone" dataKey="th" stroke={SIG.theory} dot={false} strokeWidth={1.4} strokeDasharray="4 3" isAnimationActive={false} />
          <Line type="monotone" dataKey="e" stroke={SIG.e} dot={false} strokeWidth={1.6} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export default function FourState() {
  const embedded = new URLSearchParams(window.location.search).get("embedded") === "1";
  const [params, setParams] = useState<FourStateParams>({ ...NOMINAL_PARAMS });
  const [tEnd, setTEnd] = useState(0.06);
  const [pert, setPert] = useState<Perturbation>({ i_l: 0, v_b: 0, sigma: 0, v_o: 20 });
  const [dist, setDist] = useState({ P_disturbed: 30000, t_start: 0.01, t_clear: 0.03, enabled: true });
  const [logScale, setLogScale] = useState(true);

  const [validation, setValidation] = useState<string[]>([]);
  const [equilibrium, setEquilibrium] = useState<EquilibriumResponse | null>(null);
  const [stability, setStability] = useState<StabilityResponse | null>(null);
  const [dataset, setDataset] = useState<SimDataset | null>(null);

  const setP = (k: keyof FourStateParams, v: string) => setParams((p) => ({ ...p, [k]: v === "" ? NaN : Number(v) }));

  const analyze = useMutation({
    mutationFn: async () => {
      const val = await validateParams(params);
      if (!val.valid) { setValidation(val.errors); throw new ApiError(422, val.errors); }
      setValidation([]);
      const [eq, st] = await Promise.all([getEquilibrium(params), getStability(params)]);
      return { eq, st };
    },
    onSuccess: ({ eq, st }) => { setEquilibrium(eq); setStability(st); toast.success("Equilibrium & stability computed"); },
    onError: (e) => { if (e instanceof ApiError && Array.isArray(e.body)) toast.error("Invalid parameters"); else toast.error("Analysis failed"); },
  });

  const simulate = useMutation({
    mutationFn: async () => {
      const val = await validateParams(params);
      if (!val.valid) { setValidation(val.errors); throw new ApiError(422, val.errors); }
      setValidation([]);
      return runSimulation({
        params, t_end: tEnd, n_eval: 2000, rtol: 1e-8, atol: 1e-10, method: "RK45",
        initial_perturbation: pert,
        disturbance: dist.enabled ? { P_nom: params.P, P_disturbed: dist.P_disturbed, t_start: dist.t_start, t_clear: dist.t_clear } : null,
      });
    },
    onSuccess: (ds) => { setDataset(ds); toast.success(`Simulation complete — ${ds.status.recovery_outcome}`); },
    onError: () => toast.error("Simulation failed — check parameters/disturbance timing"),
  });

  const busy = analyze.isPending || simulate.isPending;

  return (
    <div className="min-h-screen bg-background text-foreground">
      <Toaster richColors position="bottom-right" />
      {/* Header — slim in embedded mode (branding hidden; the platform supplies chrome) */}
      <header className="border-b border-border bg-card/60 backdrop-blur sticky top-0 z-20">
        <div className="max-w-[1720px] mx-auto px-4 md:px-6 py-3 flex flex-wrap items-center gap-3">
          <Waves className="text-primary" size={22} />
          <div className={`flex-1 min-w-[240px] ${embedded ? "hidden" : ""}`}>
            <h1 data-testid="workflow-header-title" className="text-lg font-semibold tracking-tight leading-none">
              IMS Platform · Ideal Four-State Stabilizing MRC
            </h1>
            <p className="font-mono text-[11px] text-muted-foreground mt-0.5">x = (i_ℓ, v_b, σ, v_o) &nbsp;·&nbsp; ė_Σ = −k_m·e_Σ &nbsp;·&nbsp; D = {`{v_b > 0}`}</p>
          </div>
          <Button data-testid="btn-compute-equilibrium" onClick={() => analyze.mutate()} disabled={busy} variant="secondary" className="gap-2">
            <Calculator size={16} /> Compute Equilibrium & Stability
          </Button>
          <Button data-testid="btn-run-simulation" onClick={() => simulate.mutate()} disabled={busy} className="gap-2">
            <Play size={16} /> Run Simulation
          </Button>
        </div>
      </header>

      <main className="max-w-[1720px] mx-auto p-4 md:p-6 grid grid-cols-1 xl:grid-cols-12 gap-5">
        {/* LEFT: configuration */}
        <section className="xl:col-span-3 flex flex-col gap-4">
          <Panel title="Parameters" icon={<Sigma size={16} className="text-primary" />}>
            <div className="grid grid-cols-2 gap-2.5">
              {PARAM_META.map((m) => (
                <div key={m.key} className="flex flex-col gap-1">
                  <Label className="text-[10px] uppercase tracking-wider text-muted-foreground">{m.key} <span className="text-[9px] normal-case">({m.unit})</span></Label>
                  <Input data-testid={`input-param-${m.key}`} type="number" value={Number.isNaN(params[m.key]) ? "" : params[m.key]}
                    onChange={(e) => setP(m.key, e.target.value)} className="font-mono h-8 text-xs bg-secondary" step="any" />
                </div>
              ))}
            </div>
            <Button data-testid="btn-reset-defaults" variant="ghost" size="sm" className="mt-1 text-xs"
              onClick={() => { setParams({ ...NOMINAL_PARAMS }); toast.message("Reset to §6.8 nominal set"); }}>Reset to nominal (§6.8)</Button>
          </Panel>

          <Panel title="Initial Perturbation" icon={<Activity size={16} className="text-primary" />}>
            <p className="text-[11px] text-muted-foreground -mt-1">Added to x*. A nonzero v_o gives an off-manifold start (e_Σ(0)≠0).</p>
            <div className="grid grid-cols-2 gap-2.5">
              {(["i_l", "v_b", "sigma", "v_o"] as const).map((k) => (
                <div key={k} className="flex flex-col gap-1">
                  <Label className="text-[10px] uppercase tracking-wider text-muted-foreground">Δ{k}</Label>
                  <Input data-testid={`input-pert-${k}`} type="number" value={pert[k]} onChange={(e) => setPert((p) => ({ ...p, [k]: Number(e.target.value) }))}
                    className="font-mono h-8 text-xs bg-secondary" step="any" />
                </div>
              ))}
            </div>
          </Panel>

          <Panel title="CPL Disturbance & Horizon" icon={<AlertTriangle size={16} className="text-[#FB923C]" />}>
            <label className="flex items-center gap-2 text-xs">
              <input data-testid="toggle-disturbance" type="checkbox" checked={dist.enabled} onChange={(e) => setDist((d) => ({ ...d, enabled: e.target.checked }))} />
              Enable temporary CPL power disturbance
            </label>
            <div className="grid grid-cols-2 gap-2.5">
              <div className="flex flex-col gap-1">
                <Label className="text-[10px] uppercase tracking-wider text-muted-foreground">P disturbed (W)</Label>
                <Input data-testid="input-dist-pdist" type="number" value={dist.P_disturbed} onChange={(e) => setDist((d) => ({ ...d, P_disturbed: Number(e.target.value) }))} className="font-mono h-8 text-xs bg-secondary" step="any" />
              </div>
              <div className="flex flex-col gap-1">
                <Label className="text-[10px] uppercase tracking-wider text-muted-foreground">t_end (s)</Label>
                <Input data-testid="input-t-end" type="number" value={tEnd} onChange={(e) => setTEnd(Number(e.target.value))} className="font-mono h-8 text-xs bg-secondary" step="any" />
              </div>
              <div className="flex flex-col gap-1">
                <Label className="text-[10px] uppercase tracking-wider text-muted-foreground">t_start (s)</Label>
                <Input data-testid="input-dist-start" type="number" value={dist.t_start} onChange={(e) => setDist((d) => ({ ...d, t_start: Number(e.target.value) }))} className="font-mono h-8 text-xs bg-secondary" step="any" />
              </div>
              <div className="flex flex-col gap-1">
                <Label className="text-[10px] uppercase tracking-wider text-muted-foreground">t_clear (s)</Label>
                <Input data-testid="input-dist-clear" type="number" value={dist.t_clear} onChange={(e) => setDist((d) => ({ ...d, t_clear: Number(e.target.value) }))} className="font-mono h-8 text-xs bg-secondary" step="any" />
              </div>
            </div>
          </Panel>

          {validation.length > 0 && (
            <div data-testid="param-validation-error-banner" className="rounded-md border border-destructive/60 bg-[#450A0A] p-3">
              <div className="flex items-center gap-2 text-[#FECACA] text-sm font-medium"><XCircle size={16} /> Invalid configuration</div>
              <ul className="mt-1 text-xs text-[#FECACA] list-disc pl-5 font-mono">{validation.map((e, i) => <li key={i}>{e}</li>)}</ul>
            </div>
          )}
        </section>

        {/* MIDDLE: analytical */}
        <section className="xl:col-span-4 flex flex-col gap-4">
          <Panel title="Closed-Form Equilibrium & Residual" icon={<Sigma size={16} className="text-[#6EE7B7]" />}>
            {equilibrium ? (
              <>
                <Metric label="i_ℓ*" value={`${fmt(equilibrium.x_star.i_l)} A`} testid="equilibrium-state-il" />
                <Metric label="v_b*" value={`${fmt(equilibrium.x_star.v_b)} V`} testid="equilibrium-state-vb" />
                <Metric label="σ*" value={fmt(equilibrium.x_star.sigma)} testid="equilibrium-state-sigma" />
                <Metric label="v_o*" value={`${fmt(equilibrium.x_star.v_o)} V`} testid="equilibrium-state-vo" />
                <Metric label="g = P/v_nom²" value={fmt(equilibrium.g)} />
                <div className="mt-2 flex items-center justify-between rounded-md border border-[#059669]/50 bg-[#14291E] px-3 py-2">
                  <span className="text-[11px] uppercase tracking-wider text-[#A7F3D0]">‖f_cl(x*)‖ (numerical)</span>
                  <span data-testid="equilibrium-residual-norm" className="font-mono text-sm text-[#6EE7B7]">{fmt(equilibrium.equilibrium_residual_norm, 3)}</span>
                </div>
              </>
            ) : <p className="text-sm text-muted-foreground">Compute to populate x* and verify ‖f_cl(x*)‖ numerically.</p>}
          </Panel>

          <Panel title="Full Closed-Loop Jacobian & Local Stability" icon={<ShieldCheck size={16} className="text-[#7DD3FC]" />}>
            {stability ? (
              <>
                <div className="overflow-x-auto">
                  <table data-testid="jacobian-matrix-display" className="font-mono text-[11px] tabular-nums border border-border">
                    <tbody>
                      {stability.jacobian_analytic.map((row, r) => (
                        <tr key={r}>{row.map((v, c) => <td key={c} className="border border-border/60 px-2 py-1 text-right">{fmt(v, 4)}</td>)}</tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div className="flex items-center gap-2 text-[11px] text-muted-foreground">
                  {stability.jacobian_independent_check_ok ? <CheckCircle2 size={13} className="text-[#34D399]" /> : <XCircle size={13} className="text-destructive" />}
                  Independent finite-difference Jacobian check: max err {fmt(stability.jacobian_independent_check_max_abs_error, 2)}
                </div>
                <Metric label="Full eigenvalues (1/s)" value={stability.full_eigenvalues.map(cx).join(", ")} />
                <Metric label="Reduced eigenvalues (1/s)" value={stability.reduced_eigenvalues.map(cx).join(", ")} />
                <div data-testid="transverse-km-comparison" className="flex items-center justify-between rounded-md border border-[#0284C7]/50 bg-[#0C2340] px-3 py-2">
                  <span className="text-[11px] uppercase tracking-wider text-[#BAE6FD]">λ⊥ vs −k_m</span>
                  <span className="font-mono text-sm text-[#7DD3FC] flex items-center gap-2">
                    {cx(stability.transverse_eigenvalue)} = {fmt(stability.transverse_expected)}
                    {stability.transverse_check_ok ? <CheckCircle2 size={14} className="text-[#34D399]" /> : <XCircle size={14} className="text-destructive" />}
                  </span>
                </div>
                <Metric label="char. coeffs a2, a1, a0" value={`${fmt(stability.characteristic_coeffs.a2)}, ${fmt(stability.characteristic_coeffs.a1)}, ${fmt(stability.characteristic_coeffs.a0)}`} />
                <table data-testid="routh-hurwitz-table" className="w-full text-xs mt-1">
                  <thead><tr className="text-left text-muted-foreground uppercase text-[10px] tracking-wider"><th className="py-1">Routh–Hurwitz</th><th>Sat.</th><th className="text-right">Margin</th></tr></thead>
                  <tbody className="font-mono">
                    {Object.entries(stability.routh_hurwitz).map(([name, d]) => (
                      <tr key={name} className="border-t border-border/50">
                        <td className="py-1">{name}</td>
                        <td>{d.satisfied ? <CheckCircle2 size={13} className="text-[#34D399]" /> : <XCircle size={13} className="text-destructive" />}</td>
                        <td className="text-right tabular-nums">{fmt(d.margin, 4)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <Badge data-testid="overall-stability-badge" variant={stability.locally_exponentially_stable ? "secondary" : "destructive"}
                  className="w-fit gap-1.5 bg-[#0C2340] text-[#7DD3FC] border border-[#0284C7]/50">
                  <ShieldCheck size={13} /> {stability.locally_exponentially_stable ? "Locally exponentially stable" : "NOT locally stable"}
                </Badge>
                <p className="text-[11px] text-muted-foreground leading-relaxed">{stability.scope_note}</p>
              </>
            ) : <p className="text-sm text-muted-foreground">Compute to populate the 4×4 Jacobian, eigenvalues and RH margins.</p>}
          </Panel>
        </section>

        {/* RIGHT: simulation */}
        <section className="xl:col-span-5 flex flex-col gap-4">
          <Panel title="Synchronized State & Residual Trajectories" icon={<Activity size={16} className="text-primary" />}
            className="min-h-[200px]">
            {dataset ? (
              <div className="flex flex-col gap-1">
                <div className="flex items-center justify-between">
                  <div className="flex flex-wrap gap-3 text-[11px]">
                    {dataset.events.map((e, i) => (
                      <span key={i} className="flex items-center gap-1 text-[#FCA5A5]"><span className="w-3 border-t border-dashed border-[#E11D48]" />{e.label} @ {(e.t * 1000).toFixed(1)}ms</span>
                    ))}
                  </div>
                </div>
                <StatePlot dataset={dataset} dkey="v_b" label="Bus voltage v_b (V)" unit="V" refVal={dataset.x_star.v_b} color={SIG.v_b} />
                <StatePlot dataset={dataset} dkey="i_l" label="Line current i_ℓ (A)" unit="A" refVal={dataset.x_star.i_l} color={SIG.i_l} />
                <StatePlot dataset={dataset} dkey="sigma" label="Integral state σ" unit="" refVal={dataset.x_star.sigma} color={SIG.sigma} />
                <StatePlot dataset={dataset} dkey="v_o" label="Converter voltage v_o (V)" unit="V" refVal={dataset.x_star.v_o} color={SIG.v_o} />
                <div className="flex items-center justify-between px-1 pt-1">
                  <span className="text-[11px] uppercase tracking-wider text-[#F43F5E]">Manifold residual e_Σ &nbsp;<span className="text-[#34D399] normal-case">(dashed = e0·e^(−k_m t))</span></span>
                  <button data-testid="chart-scale-toggle-log" onClick={() => setLogScale((s) => !s)} className="font-mono text-[11px] rounded border border-border px-2 py-0.5 text-muted-foreground hover:border-border">
                    {logScale ? "log |e_Σ|" : "linear e_Σ"}
                  </button>
                </div>
                <ResidualPlot dataset={dataset} logScale={logScale} />
                <div className="mt-1 rounded-md border border-[#14291E] bg-[#14291E]/60 px-3 py-2 flex items-center justify-between">
                  <span className="text-[11px] uppercase tracking-wider text-[#A7F3D0]">max rel. error vs e0·e^(−k_m t)</span>
                  <span data-testid="residual-contraction-error" className="font-mono text-sm text-[#6EE7B7]">{fmt(dataset.residual_contraction.max_rel_error, 3)}</span>
                </div>
              </div>
            ) : <p className="text-sm text-muted-foreground">Run the simulation to render v_b, i_ℓ, σ, v_o and the manifold residual e_Σ(t).</p>}
          </Panel>

          {dataset && (
            <Panel title="Recovery Status & Export" icon={<Download size={16} className="text-primary" />}>
              <div className="grid grid-cols-2 gap-2">
                <div className="rounded-md border border-border bg-secondary px-3 py-2">
                  <div className="text-[10px] uppercase tracking-wider text-muted-foreground">Solver status</div>
                  <div className="font-mono text-sm">{dataset.status.solver_status}</div>
                </div>
                <div data-testid="recovery-outcome-card" className="rounded-md border border-[#D97706]/50 bg-[#291E0A] px-3 py-2">
                  <div className="text-[10px] uppercase tracking-wider text-[#FDE68A]">Recovery outcome</div>
                  <div className="font-mono text-sm text-[#FCD34D]">{dataset.status.recovery_outcome}</div>
                </div>
              </div>
              <p className="text-[11px] text-muted-foreground leading-relaxed">{dataset.status.disclaimer}</p>
              <div className="flex gap-2">
                <a data-testid="btn-export-csv" href={csvUrl(dataset.run_id)} className={buttonVariants({ variant: "secondary" }) + " gap-2"}>
                  <Download size={15} /> Export CSV
                </a>
                <a data-testid="btn-export-pdf" href={pdfUrl(dataset.run_id)} target="_blank" rel="noreferrer" className={buttonVariants() + " gap-2"}>
                  <FileText size={15} /> PDF Report
                </a>
              </div>
              <div className="font-mono text-[10px] text-muted-foreground">run_id: {dataset.run_id}</div>
            </Panel>
          )}
        </section>

        {/* BOTTOM: claim-level legend */}
        <section className="xl:col-span-12">
          <Panel title="Guarantee Classification — these are DIFFERENT claims and must never be conflated" icon={<Compass size={16} className="text-[#E879F9]" />}>
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-3">
              {CLAIM_TIERS.map((c) => (
                <div key={c.id} data-testid={`claim-tier-${c.id.replace("_", "-")}`} className="rounded-md border p-3" style={{ borderColor: c.color }}>
                  <div className="flex items-center gap-2 font-medium text-sm" style={{ color: c.color }}>
                    <c.Icon size={15} /> {c.title}
                  </div>
                  <p className="mt-1 text-[11px] text-muted-foreground leading-relaxed">{c.def}</p>
                </div>
              ))}
            </div>
          </Panel>
        </section>
      </main>
    </div>
  );
}
