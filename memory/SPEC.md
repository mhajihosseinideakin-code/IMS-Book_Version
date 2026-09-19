# IMS Platform — Living Spec

## What the app does
Verified end-to-end workflow for the **ideal continuous-time four-state stabilizing-MRC** model
x = (i_ℓ, v_b, σ, v_o), domain D = {v_b > 0}. Built on the vendored, pre-existing IMS Platform
Python core (`backend/ims_platform/`, formerly a Flask app), now exposed through FastAPI + a React
workspace. The legacy Flask platform is preserved and mounted at `/legacy` (WSGIMiddleware).

## Architecture (platform-integrated)
- **Primary app = the existing IMS Platform** (Flask `ims_platform/server/app.py` + `static/explorer.html`):
  hero, Projects, New Project, Case Library (5 cases), Workspace, Docs, auth. It is mounted in FastAPI
  at root `/` via WSGIMiddleware (catch-all, last) and is the public-URL experience. Its ~60 `/api/*`
  routes are served unchanged.
- **FastAPI (port 8001)** owns only `/api/four_state/*` (router `backend/routers/four_state.py`),
  `/api/status`, `/api/`, `/docs`; everything else falls through to the Flask platform.
- **Vite (port 3000, public)** proxies `/api`, `/assets`, and `/explorer.html`→Flask `/` to 8001.
  React root `/` redirects to `/explorer.html` (the platform). React route `/mrc` is the verified MRC
  analysis workspace (`src/pages/FourState.tsx`), rendered header-slim when `?embedded=1`.
- **MRC integrated natively**: explorer.html has an "Explorer → Stabilising MRC" sidebar item
  (`showStabilizingMRC()`, `#stabMrcView`) that embeds `/mrc?embedded=1` in an iframe inside the
  platform chrome (breadcrumb Project Manager › Explorer › Stabilising MRC). A monkey-patch hides the
  MRC view when navigating to any other view.
- **Math core** untouched in `backend/ims_platform/` (verified `models/stabilizing_mrc.py`); the MRC
  workflow layer `backend/ims_platform/stabilizing/` (model, workflow, report) is thin orchestration,
  no re-derivation. Electrical-constraint diagnostic model preserved (`models/converter_cpl_paper.py`).

## Data model / equations (Appendix A of the reference)
- Plant: L·di_ℓ=v_o−R·i_ℓ−v_b ; C·dv_b=i_ℓ−P/v_b ; dσ=v_nom−v_b ; dv_o=u
- Residual: e_Σ = v_o − v_nom + R_v·i_ℓ − K_i·σ
- Control law: u = −(R_v/L)(v_o−R·i_ℓ−v_b) + K_i(v_nom−v_b) − k_m·e_Σ  ⇒  ė_Σ = −k_m·e_Σ (exact, ideal)
- Equilibrium: v_b*=v_nom, i_ℓ*=P/v_nom, σ*=(R+R_v)P/(K_i·v_nom), v_o*=v_nom+RP/v_nom

## Key flows (API)
POST /api/four_state/validate | /equilibrium | /stability | /simulate
GET  /api/four_state/export/csv/{run_id} | /report/pdf/{run_id} | GET /api/four_state/model
`/simulate` caches ONE dataset per run_id; CSV + PDF reuse that exact dataset (single source).

## Nominal reference case (§6.8) — reproduces exactly
P=20kW, v_nom=400V, R=0.2Ω, L=1.5mH, C=2.5mF, R_v=0.5Ω, K_i=50/s, k_m=500/s
→ x* = (50 A, 400 V, 0.7, 410 V); ‖f_cl(x*)‖ = 0; poles −60.085, −178.291±436.028j, −500; RH stable.

## Shared analysis framework (Phase-1 engine)
- `backend/ims_platform/framework/`: `interfaces.py` (STANDARD_WORKFLOW_STAGES = Model/Network → Analysis
  → Disturbance → Simulation → IMS Results → Report; `SupportedOutputs`, `AnalysisDescriptor`, `Analysis`
  Protocol, `build_envelope`, `CLAIM_LEVELS`), `registry.py` (ANALYSIS_REGISTRY + connected Case Library
  descriptors), `analyses/stabilizing_mrc_analysis.py` (first fully-runnable analysis, delegates to the
  verified stabilizing workflow — no math re-derivation).
- FastAPI router `backend/routers/analysis.py` (`/api/analysis/*`): `registry`, `{id}/descriptor`,
  `{id}/default_config`, `{id}/validate`, `{id}/run` → standardized IMS result envelope. MRC runs reuse the
  four_state run cache so CSV/PDF export resolves framework run_ids (single dataset).
- Explorer = analysis hub: explorer.html `showAnalysisHub()` (nav "Explorer → Analysis Hub", `#analysisHubView`)
  fetches `/api/analysis/registry` and renders cards; RUNNABLE analyses open their surface, VALIDATED CASE
  studies route to their existing preserved views. Adding a new analysis = one registry entry.
- Standard envelope outputs (where supported): equilibrium, manifold/residual definition, residual trajectory,
  contraction evidence, local stability, finite-horizon recovery, solver/constraint status — each tagged with
  its claim_level; the four claim levels are never conflated.

## Auth
None. No login/PIN gating.

## MRC feasibility for user-built Network Builder models (investigation + fix)
Root cause of "every Network Builder model returns NOT ESTABLISHED" was two-fold:
- **Integration bug (fixed):** the frontend `designerModelForProject` returned `null` for `custom_network`,
  so the Design MRC stage showed a canned "not established" message WITHOUT ever inspecting the assembled
  model. Now `custom_network` calls `POST /api/network_mrc/inspect` (`network_mrc_inspect` in app.py) which
  assembles the real system and reports genuine per-model diagnostics: dim_x, dim_u, numeric control-affine
  split f(x)+G(x)u, ‖G‖ / control authority, and whether a signed controlled-target manifold exists.
- **Mathematical (not a bug):** even with f(x),G(x) available, there is NO validated signed controlled-target
  manifold φ(x) for arbitrary topologies. IMS analysis provides only a numeric distance-to-manifold residual
  (`IntrinsicManifold.residual` = distance; its gradient is a unit normal to the sampled curve, not a
  controlled-invariant transverse coordinate), so A=Dφ·G cannot justify synthesis. Correctly "not established".
Empirical trace (Network Builder→Assembly→IMS→Feasibility): boost+CPL(PI) → dim_u=1, ‖G‖>0 (control channel
present) but no φ → not established; buck+impedance(constant-duty) → control channel present too, still no φ.
The honest classification: analytic SUPPORTED only for the registered assembled Converter-CPL (Network+Auto-MRC);
Four-State remains a benchmark; all user-built topologies are "Not Established" with the specific reason.
Fix is honest — never forces SUPPORTED. New endpoint `/api/network_mrc/inspect`; frontend
`renderCustomMrcFeasibility`/`runCustomFeasibility`.

## MRC attached to the converter in the assembled Network (real controller, not just a panel)
- The Design MRC stage for `network_auto_mrc` now drives the ACTUAL assembled Network Builder model:
  feasibility+synthesis+verify via `/api/mrc_designer/*` (on ConverterCPLPaper = the assembled model's math),
  then two explicit sub-steps on the real assembled `Network`:
  - **Apply MRC to Converter** → `POST /api/network_mrc/apply` (Flask app.py `network_mrc_apply`): builds the
    assembled network with `SynthesizedMRCController` attached to converter `conv` and returns the converter's
    active-controller config (type, law, k_m, manifold e_m). UI shows "MRC ACTIVE ON CONVERTER 'conv'".
  - **Closed-Loop Simulation (baseline vs MRC)** → `POST /api/network_mrc/closed_loop`
    (`network_mrc_closed_loop`): reruns the IDENTICAL `AutomaticModelBuilder`-assembled system + same
    equilibrium + same disturbance in two configs — baseline (converter controller = `ConstantSetpointController(v_ref=0)`,
    i.e. u=v_o_dot=0 open-loop) vs MRC. Returns both trajectories (v_bus,i_L,v_o), residual e_m, final
    deviation/residual. Verified: baseline final |e_m|≈1.08 (unrecovered), MRC final |e_m|≈2e-4 (recovered).
  - Backend refactor: `_build_network_mrc_with(payload, apply_mrc)` (app.py) is the shared builder;
    `_build_network_mrc` delegates with apply_mrc=True (legacy behaviour unchanged). Routes registered via
    `_network_mrc_route`. `ConstantSetpointController` imported from `..network.controller`.
- Four-State Stabilising MRC remains a validated reference/benchmark only. Non-network projects still show an
  honest "MRC Synthesis Not Established" with the mathematical reason (no arbitrary-network claim).

## General MRC Designer (downstream stage INSIDE the project workspace)
- **No longer a standalone Explorer item.** The MRC Designer is the project workspace's downstream
  stage `design_mrc` ("Design MRC Controller"), inserted in STAGES between `ims_analysis` and `report`
  (explorer.html). It inherits the project's assembled model, params, k_m, equilibrium and IMS results
  (single source of truth) via `renderDesignMrcStage`/`runDesignMrc` — it never recreates the model or
  substitutes the four-state reference. The standalone `navMrcDesigner` sidebar item, `mrcDesignerView`
  iframe, `showMrcDesigner()` and the Analysis-Hub `mrc_designer` descriptor were removed. "Analysis Hub"
  and the "Stabilising MRC" benchmark sidebar items are kept.
- **Three honest feasibility outcomes** (per model, math reason always given): MRC Synthesis Supported
  (analytic G + analytic φ, A=Dφ·G full-row-rank & well-conditioned), Diagnostic/Feasibility Only
  (numeric-only), MRC Synthesis Not Established (no validated analytic controlled-target manifold).
  `feasStatus()` maps `establishment_basis` (analytic|numeric_only|insufficient_authority|none).
- **Supported project = "Network + Auto-MRC Demo" (network_auto_mrc, category network_mrc).** Its MRC is
  derived by the Symbolic Engine from `ConverterCPLPaper` (paper eq. 1-6/13). Backend: `converter_cpl_paper`
  registered in `_DESIGNER_MODELS` (mrc_designer/designer.py) with an anchored equilibrium
  (i_l=P/V, v_b=V=400, v_o=V+R·i_l) and a generic `verify_closed_loop` branch (finite-diff closed-loop
  Jacobian; λ⊥ found nearest −k_m; off-manifold residual contraction vs e_m(0)e^(−k_m t)). Verified: λ⊥=−k_m,
  residual rel-err ~1.5e-8, ‖f_cl(x*)‖~3e-11. HONEST: full local stability is NOT guaranteed for this
  fixture (documented structurally-positive tangential mode P/(C v_b²)); MRC guarantees residual (transverse)
  contraction only — shown with a scope note. Before/After plots the manifold residual e_m open-loop vs MRC
  (the meaningful comparison) + bus voltage; CSV export is client-side. All other projects → honest
  "Not Established" with the mathematical reason.
- Router `/api/mrc_designer/*`: feasibility (returns honest not-established, not 404, for unregistered
  models via `feasibility_for`), synthesize, verify, before_after. Registered analytic models:
  `stabilizing_mrc` (reference), `converter_cpl_paper` (assembled network).
- Reference regression unchanged: the generic path still reproduces the four-state equilibrium, λ⊥=−k_m,
  residual contraction and poles within tolerance.

## Home hero (explorer.html) — refined premium scientific landing
Logo enlarged (clamp 132-196px, full wordmark, no corner mask) and centered in the content area (after the
sidebar). Cleaner hierarchy: logo → "Trace stability. Quantify recoverability." → short description → CTAs
(Open Example Project | Create New Project) → compact connected workflow pipeline. The four bottom feature
cards (Intrinsic Manifold / Recoverability / Validated Engine / Reliable Results) were removed. Workflow strip
(`.hero-caps`) is a connected pipeline: 5 circular-icon nodes (Build Networks → Analyse Stability → Design
Control → Simulate & Validate → Generate Reports) joined by arrow separators, with hover states, subordinate to
the headline. The Disturbance→Equilibrium manifold SVG (`.hero-illustration`) is integrated via a radial
`mask-image` feathering all edges (no rectangular boundary) + a bottom fade to bg; larger height; the existing
scientific trajectory/markers are preserved. Dark identity preserved. Visual-only; no backend/nav changes.

## Claim levels (kept strictly distinct, never conflated)
ideal residual contraction (exact) | local equilibrium stability (linearisation) |
observed finite-horizon recovery (empirical) | certified regional IMS (NOT established this milestone).
