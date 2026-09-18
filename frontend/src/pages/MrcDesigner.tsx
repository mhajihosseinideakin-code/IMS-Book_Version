import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, Legend,
} from "recharts";
import { toast } from "sonner";
import { Toaster } from "@/components/ui/sonner";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import {
  Cpu, ShieldCheck, GitCompare, CheckCircle2, XCircle, Workflow, Layers, Sigma,
} from "lucide-react";
import {
  NOMINAL_PARAMS, designerFeasibility, designerVerify, designerBeforeAfter,
} from "@/lib/ims";
import type {
  FourStateParams, FeasibilityReport, VerifyResult, BeforeAfter, Complex,
} from "@/lib/ims";

const PARAM_KEYS: (keyof FourStateParams)[] = ["R", "L", "C", "P", "v_nom", "R_v", "K_i", "k_m"];

function fmt(x: number | null | undefined, p = 4): string {
  if (x === null || x === undefined || !isFinite(x)) return "—";
  if (x !== 0 && (Math.abs(x) < 1e-3 || Math.abs(x) >= 1e5)) return x.toExponential(p - 1);
  return Number(x.toPrecision(p)).toString();
}
function cx(z: Complex): string {
  if (Math.abs(z.im) < 1e-9) return fmt(z.re);
  return `${fmt(z.re)} ${z.im >= 0 ? "+" : "-"} ${fmt(Math.abs(z.im))}j`;
}

const STAGES = ["Network Builder", "Model Assembly", "IMS Analysis", "Design MRC", "Closed-Loop Verification", "Before/After"];

function Panel({ title, icon, children }: { title: string; icon?: React.ReactNode; children: React.ReactNode }) {
  return (
    <Card className="bg-card border-border p-4 gap-3">
      <div className="flex items-center gap-2"><span>{icon}</span><h3 className="text-[15px] font-semibold tracking-tight">{title}</h3></div>
      {children}
    </Card>
  );
}

export default function MrcDesigner() {
  const embedded = new URLSearchParams(window.location.search).get("embedded") === "1";
  const [params, setParams] = useState<FourStateParams>({ ...NOMINAL_PARAMS });
  const [feas, setFeas] = useState<FeasibilityReport | null>(null);
  const [verify, setVerify] = useState<VerifyResult | null>(null);
  const [ba, setBa] = useState<BeforeAfter | null>(null);
  const setP = (k: keyof FourStateParams, v: string) => setParams((p) => ({ ...p, [k]: v === "" ? NaN : Number(v) }));

  const sim = {
    model_id: "stabilizing_mrc", params, t_end: 0.06, n_eval: 2000,
    initial_perturbation: { i_l: 0, v_b: 0, sigma: 0, v_o: 20 },
    disturbance: { P_nom: params.P, P_disturbed: 1.5 * params.P, t_start: 0.01, t_clear: 0.03 },
  };

  const design = useMutation({
    mutationFn: async () => {
      const f = await designerFeasibility(params);
      setFeas(f);
      if (!f.mrc_established) { setVerify(null); setBa(null); return { f, v: null, b: null }; }
      const [v, b] = await Promise.all([designerVerify(sim), designerBeforeAfter(sim)]);
      return { f, v, b };
    },
    onSuccess: ({ f, v, b }) => {
      setVerify(v); setBa(b);
      if (!f.mrc_established) toast.error("MRC not established for this configuration");
      else toast.success("MRC designed, verified & compared");
    },
    onError: () => toast.error("Designer run failed"),
  });

  const cmpData = ba ? ba.open_loop.t.map((t, i) => ({
    t: t * 1000,
    vb_ol: ba.open_loop.states["v_b"]?.[i],
    vb_mrc: ba.mrc.states["v_b"]?.[i],
    e_ol: ba.open_loop.e_sigma?.[i],
    e_mrc: ba.mrc.e_sigma?.[i],
  })) : [];

  return (
    <div className="min-h-screen bg-background text-foreground">
      <Toaster richColors position="bottom-right" />
      <header className="border-b border-border bg-card/60 backdrop-blur sticky top-0 z-20">
        <div className="max-w-[1720px] mx-auto px-4 md:px-6 py-3 flex flex-wrap items-center gap-3">
          <Cpu className={`text-primary ${embedded ? "hidden" : ""}`} size={22} />
          <div className={`flex-1 min-w-[240px] ${embedded ? "hidden" : ""}`}>
            <h1 className="text-lg font-semibold tracking-tight leading-none">General MRC Designer</h1>
            <p className="font-mono text-[11px] text-muted-foreground mt-0.5">A(x)=Dφ·G · u = A⁺(−k_m φ − Dφ·f) · reference: Four-State Stabilising MRC</p>
          </div>
          <Button data-testid="btn-run-designer" onClick={() => design.mutate()} disabled={design.isPending} className="gap-2">
            <Workflow size={16} /> {design.isPending ? "Running…" : "Design → Verify → Compare"}
          </Button>
        </div>
      </header>

      <main className="max-w-[1720px] mx-auto p-4 md:p-6 grid grid-cols-1 xl:grid-cols-12 gap-5">
        {/* workflow strip */}
        <section className="xl:col-span-12">
          <div className="flex flex-wrap items-center gap-2 text-[11px] font-mono">
            {STAGES.map((s, i) => (
              <span key={s} className="flex items-center gap-2">
                <span className={`px-2 py-1 rounded border ${s === "Design MRC" || s === "Closed-Loop Verification" || s === "Before/After" ? "border-primary/60 text-primary" : "border-border text-muted-foreground"}`}>{s}</span>
                {i < STAGES.length - 1 && <span className="text-muted-foreground">→</span>}
              </span>
            ))}
          </div>
        </section>

        {/* left: params */}
        <section className="xl:col-span-3 flex flex-col gap-4">
          <Panel title="Assembled Model Parameters" icon={<Sigma size={16} className="text-primary" />}>
            <p className="text-[11px] text-muted-foreground -mt-1">Reference model: Four-State Stabilising MRC (validated). One source of truth per project.</p>
            <div className="grid grid-cols-2 gap-2.5">
              {PARAM_KEYS.map((k) => (
                <div key={k} className="flex flex-col gap-1">
                  <Label className="text-[10px] uppercase tracking-wider text-muted-foreground">{k}</Label>
                  <Input data-testid={`input-designer-${k}`} type="number" value={Number.isNaN(params[k]) ? "" : params[k]}
                    onChange={(e) => setP(k, e.target.value)} className="font-mono h-8 text-xs bg-secondary" step="any" />
                </div>
              ))}
            </div>
          </Panel>
        </section>

        {/* middle: feasibility + synthesis */}
        <section className="xl:col-span-4 flex flex-col gap-4">
          <Panel title="MRC Feasibility — A(x) = Dφ(x)·G(x)" icon={<Layers size={16} className="text-[#7DD3FC]" />}>
            {feas ? (
              <>
                <div data-testid="feasibility-established-badge">
                  <Badge variant={feas.mrc_established ? "secondary" : "destructive"}
                    className={feas.mrc_established ? "bg-[#0C2340] text-[#7DD3FC] border border-[#0284C7]/50 gap-1.5" : "gap-1.5"}>
                    {feas.mrc_established ? <CheckCircle2 size={13} /> : <XCircle size={13} />}
                    {feas.mrc_established ? "MRC established (analytic)" : "MRC not established"}
                  </Badge>
                </div>
                <div className="grid grid-cols-2 gap-x-4 text-xs font-mono">
                  <div>dim φ = <b>{feas.dim_phi ?? "—"}</b></div>
                  <div>dim u = <b>{feas.dim_u}</b></div>
                  <div>rank A = <b>{feas.rank_A ?? "—"}</b></div>
                  <div>cond(A) = <b>{fmt(feas.condition_number)}</b></div>
                  <div>rel. degree 1: <b>{String(feas.relative_degree_one)}</b></div>
                  <div>G: <b>{feas.G_method}</b></div>
                  <div>φ: <b>{feas.manifold_method}</b></div>
                  <div>basis: <b>{feas.establishment_basis}</b></div>
                </div>
                {feas.A && <div className="font-mono text-[11px]">A(x) = [{feas.A.map((r) => r.map((v) => fmt(v)).join(", ")).join("; ")}]</div>}
                <p className="text-[11px] text-muted-foreground leading-relaxed">{feas.reason}</p>
              </>
            ) : <p className="text-sm text-muted-foreground">Run to evaluate control authority before any synthesis.</p>}
          </Panel>

          {verify?.design && (
            <Panel title="Synthesized Control Law" icon={<Cpu size={16} className="text-[#6EE7B7]" />}>
              <div className="text-[11px] uppercase tracking-wider text-muted-foreground">generic relation</div>
              <div className="font-mono text-[11px]">{verify.design.synthesis_relation}</div>
              <div className="text-[11px] uppercase tracking-wider text-muted-foreground mt-1">u = κ(x)</div>
              <div data-testid="synthesized-law" className="font-mono text-[11px] break-words bg-secondary rounded p-2">{verify.design.control_law_symbolic}</div>
              <div className="grid gap-2 mt-1">
                {verify.design.manifold_roles && (
                  <div className="text-[11px] grid gap-1">
                    <div><span className="text-[#94A3B8]">IMS-analysis manifold:</span> <span className="font-mono">{verify.design.manifold_roles.ims_analysis_manifold}</span></div>
                    <div><span className="text-[#94A3B8]">Candidate MRC target:</span> <span className="font-mono">{verify.design.manifold_roles.candidate_mrc_target}</span></div>
                    <div><span className="text-[#94A3B8]">Validated controlled-invariant:</span> <b className="text-[#6EE7B7]">{String(verify.design.manifold_roles.validated_controlled_invariant)}</b></div>
                  </div>
                )}
              </div>
            </Panel>
          )}
        </section>

        {/* right: verification + before/after */}
        <section className="xl:col-span-5 flex flex-col gap-4">
          {verify && (
            <Panel title="Closed-Loop Verification" icon={<ShieldCheck size={16} className="text-[#7DD3FC]" />}>
              <div className="grid grid-cols-2 gap-x-4 text-xs font-mono">
                <div>‖f_cl(x*)‖ = <b data-testid="verify-residual">{fmt(verify.equilibrium.residual_norm, 3)}</b></div>
                <div>λ⊥ = <b>{cx(verify.local_stability.transverse_eigenvalue)}</b> {verify.local_stability.transverse_check_ok ? "✓" : "✗"}</div>
              </div>
              <div className="font-mono text-[11px]">poles: {verify.local_stability.full_eigenvalues.map(cx).join(", ")}</div>
              <div className="flex flex-wrap gap-2 text-[10px]">
                <span className="px-2 py-0.5 rounded border border-[#10B981] text-[#10B981]">exact: {verify.contraction_evidence.identity} (rel err {fmt(verify.contraction_evidence.max_rel_error, 2)})</span>
                <span className="px-2 py-0.5 rounded border border-[#0284C7] text-[#7DD3FC]">local: {verify.local_stability.locally_exponentially_stable ? "stable" : "unstable"}</span>
                <span className="px-2 py-0.5 rounded border border-[#F59E0B] text-[#FCD34D]" data-testid="verify-recovery">observed: {verify.recovery_status.recovery_outcome}</span>
              </div>
              <p className="text-[11px] text-muted-foreground">{verify.local_stability.scope_note}</p>
            </Panel>
          )}

          {ba && (
            <Panel title="Before / After — Open-Loop vs MRC (identical scenario)" icon={<GitCompare size={16} className="text-[#A78BFA]" />}>
              <div className="text-[11px] uppercase tracking-wider text-muted-foreground">Bus voltage v_b (V)</div>
              <ResponsiveContainer width="100%" height={150}>
                <LineChart data={cmpData} margin={{ top: 4, right: 12, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                  <XAxis dataKey="t" tick={{ fontSize: 10, fill: "#64748B" }} tickFormatter={(v: number) => v.toFixed(0)} stroke="#334155" />
                  <YAxis tick={{ fontSize: 10, fill: "#64748B" }} width={54} stroke="#334155" tickFormatter={(v: number) => fmt(v, 3)} />
                  <Tooltip contentStyle={{ background: "#0F172A", border: "1px solid #334155", fontSize: 11 }} labelFormatter={(l: number) => `t = ${l.toFixed(3)} ms`} />
                  <Legend wrapperStyle={{ fontSize: 10 }} />
                  <ReferenceLine y={ba.x_star["v_b"]} stroke="#64748B" strokeDasharray="5 4" />
                  <Line type="monotone" dataKey="vb_ol" name="Open-loop" stroke="#F43F5E" dot={false} strokeWidth={1.4} isAnimationActive={false} />
                  <Line type="monotone" dataKey="vb_mrc" name="MRC" stroke="#34D399" dot={false} strokeWidth={1.6} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
              <table className="w-full text-xs mt-1 font-mono" data-testid="before-after-metrics">
                <thead><tr className="text-left text-muted-foreground uppercase text-[10px] tracking-wider"><th className="py-1">Metric</th><th>Open-loop</th><th>MRC</th></tr></thead>
                <tbody>
                  <tr className="border-t border-border/50"><td className="py-1">final ‖x−x*‖</td><td>{fmt(ba.metrics.final_state_deviation.open_loop)}</td><td className="text-[#6EE7B7]">{fmt(ba.metrics.final_state_deviation.mrc)}</td></tr>
                  {ba.metrics.bus_voltage.open_loop && ba.metrics.bus_voltage.mrc && (
                    <tr className="border-t border-border/50"><td className="py-1">v_b final</td><td>{fmt(ba.metrics.bus_voltage.open_loop.final)}</td><td className="text-[#6EE7B7]">{fmt(ba.metrics.bus_voltage.mrc.final)}</td></tr>
                  )}
                </tbody>
              </table>
              <p className="text-[11px] text-muted-foreground">{ba.disclaimer}</p>
            </Panel>
          )}
        </section>
      </main>
    </div>
  );
}
