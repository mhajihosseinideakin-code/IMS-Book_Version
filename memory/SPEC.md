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

## Auth
None. No login/PIN gating.

## Claim levels (kept strictly distinct, never conflated)
ideal residual contraction (exact) | local equilibrium stability (linearisation) |
observed finite-horizon recovery (empirical) | certified regional IMS (NOT established this milestone).
