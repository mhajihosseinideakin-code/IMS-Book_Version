// TS interfaces mirroring the four-state stabilizing-MRC Pydantic/response models
// (backend/routers/four_state.py + ims_platform/stabilizing/workflow.py).
// Nothing infers across the Python boundary — keep these in sync by hand.
import { apiPost } from "@/lib/api";

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

export const NOMINAL_PARAMS: FourStateParams = {
  R: 0.2, L: 1.5e-3, C: 2.5e-3, P: 20000, v_nom: 400, R_v: 0.5, K_i: 50, k_m: 500,
};
