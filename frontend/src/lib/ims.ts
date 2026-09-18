// TS interfaces mirroring the four-state stabilizing-MRC Pydantic/response models
// (backend/routers/four_state.py + ims_platform/stabilizing/workflow.py).
// Nothing infers across the Python boundary — keep these in sync by hand.
import { apiGet, apiPost } from "@/lib/api";

export interface FourStateParams {
  R: number;
  L: number;
  C: number;
  P: number;
  v_nom: number;
  R_v: number;
  K_i: number;
  k_m: number;
}

export interface Disturbance {
  P_nom: number;
  P_disturbed: number;
  t_start: number;
  t_clear: number;
}

export interface Perturbation {
  i_l: number;
  v_b: number;
  sigma: number;
  v_o: number;
}

export interface ValidateResponse {
  valid: boolean;
  errors: string[];
  params: FourStateParams | null;
}

export interface EquilibriumResponse {
  x_star: { i_l: number; v_b: number; sigma: number; v_o: number };
  x_star_vector: number[];
  state_order: string[];
  f_cl_at_x_star: number[];
  equilibrium_residual_norm: number;
  g: number;
  formulas: Record<string, string>;
}

export interface Complex {
  re: number;
  im: number;
}

export interface RHCondition {
  satisfied: boolean;
  margin: number;
}

export interface StabilityResponse {
  jacobian_analytic: number[][];
  jacobian_numeric: number[][];
  jacobian_independent_check_max_abs_error: number;
  jacobian_independent_check_ok: boolean;
  full_eigenvalues: Complex[];
  reduced_eigenvalues: Complex[];
  transverse_eigenvalue: Complex;
  transverse_expected: number;
  transverse_check_error: number;
  transverse_check_ok: boolean;
  characteristic_coeffs: { a2: number; a1: number; a0: number };
  routh_hurwitz: Record<string, RHCondition>;
  locally_exponentially_stable: boolean;
  scope_note: string;
  claim_levels: Record<string, string>;
}

export interface SimStatus {
  solver_status: string;
  recovery_outcome: string;
  termination_reason: string;
  constraint_status: string;
  constraint_infeasible: boolean;
  evaluated_until: number;
  criteria: Record<string, number | string>;
  disclaimer: string;
}

export interface SimDataset {
  run_id: string;
  t: number[];
  i_l: number[];
  v_b: number[];
  sigma: number[];
  v_o: number[];
  e_sigma: number[];
  e_sigma_theory: number[];
  cpl_power: number[];
  disturbance_active: number[];
  x_star: { i_l: number; v_b: number; sigma: number; v_o: number };
  e_sigma_0: number;
  residual_contraction: { max_abs_error: number; max_rel_error: number; k_m: number; note: string };
  events: { t: number; label: string }[];
  status: SimStatus;
  provenance: Record<string, unknown>;
}

export interface SimRequest {
  params: FourStateParams;
  t_end: number;
  n_eval: number;
  rtol: number;
  atol: number;
  method: string;
  initial_perturbation: Perturbation;
  disturbance: Disturbance | null;
}

const B = "/four_state";

export const validateParams = (p: FourStateParams) => apiPost<ValidateResponse>(`${B}/validate`, p);
export const getEquilibrium = (p: FourStateParams) => apiPost<EquilibriumResponse>(`${B}/equilibrium`, p);
export const getStability = (p: FourStateParams) => apiPost<StabilityResponse>(`${B}/stability`, p);
export const runSimulation = (req: SimRequest) => apiPost<SimDataset>(`${B}/simulate`, req);

export const csvUrl = (runId: string) => `/api${B}/export/csv/${runId}`;
export const pdfUrl = (runId: string) => `/api${B}/report/pdf/${runId}`;

// ---- General MRC Designer ----
const D = "/mrc_designer";
export interface FeasibilityReport {
  dim_x: number; dim_u: number; dim_phi: number | null; input_names: string[];
  G_method: string; manifold_method: string; manifold_note: string;
  A: number[][] | null; rank_A: number | null; condition_number: number | null;
  relative_degree_one: boolean | null; singular_or_illconditioned: boolean | null;
  mrc_established: boolean; establishment_basis: string; reason: string; phi_value?: number;
}
export interface DesignResult {
  mrc_established: boolean; reason?: string; title?: string;
  manifold_constraint?: string; control_law_symbolic?: string; control_law_latex?: string;
  synthesis_relation?: string; k_m?: number;
  manifold_roles?: { ims_analysis_manifold: string; candidate_mrc_target: string; validated_controlled_invariant: boolean };
  feasibility?: FeasibilityReport;
}
export interface VerifyResult {
  design: DesignResult;
  equilibrium: { x_star: Record<string, number>; residual_norm: number };
  local_stability: { full_eigenvalues: Complex[]; transverse_eigenvalue: Complex; transverse_expected: number; transverse_check_ok: boolean; locally_exponentially_stable: boolean; scope_note: string };
  residual_trajectory: { t: number[]; e_sigma: number[]; e_sigma_theory: number[] };
  contraction_evidence: { identity: string; max_rel_error: number; claim_level: string };
  recovery_status: SimStatus & { claim_level: string };
  run_id: string;
}
export interface BeforeAfter {
  design: DesignResult;
  identical_conditions: { x0: number[]; disturbance: Record<string, number> | null; solver: Record<string, unknown> };
  x_star: Record<string, number>;
  open_loop: { t: number[]; states: Record<string, number[]>; e_sigma: number[] | null };
  mrc: { t: number[]; states: Record<string, number[]>; e_sigma: number[] | null };
  metrics: { final_state_deviation: { open_loop: number; mrc: number }; bus_voltage: Record<string, { min: number; max: number; final: number; undershoot: number } | null> };
  disclaimer: string;
}
export interface DesignerSimReq { model_id: string; params: FourStateParams; k_m?: number; t_end: number; n_eval: number; initial_perturbation: Perturbation; disturbance: Disturbance | null; }
export const designerDefaultParams = () => apiGet<FourStateParams>(`${D}/default_params/stabilizing_mrc`);
export const designerFeasibility = (params: FourStateParams) => apiPost<FeasibilityReport>(`${D}/feasibility`, { model_id: "stabilizing_mrc", params });
export const designerSynthesize = (params: FourStateParams) => apiPost<DesignResult>(`${D}/synthesize`, { model_id: "stabilizing_mrc", params });
export const designerVerify = (r: DesignerSimReq) => apiPost<VerifyResult>(`${D}/verify`, r);
export const designerBeforeAfter = (r: DesignerSimReq) => apiPost<BeforeAfter>(`${D}/before_after`, r);

export const NOMINAL_PARAMS: FourStateParams = {
  R: 0.2, L: 1.5e-3, C: 2.5e-3, P: 20000, v_nom: 400, R_v: 0.5, K_i: 50, k_m: 500,
};
