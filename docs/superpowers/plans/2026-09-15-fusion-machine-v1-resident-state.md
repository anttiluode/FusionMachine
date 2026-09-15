# FusionMachine v1 Resident State Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show that a dormant stateful computation can remain immediately ready when maintained resident, while bounded lazy replay reconstructs it with an error controlled by the computation's forgetting time.

**Architecture:** Add a small leaky-state/replay module, test it against the exact analytic omitted-history variance, then freeze a multi-seed sweep over retention and replay-window length. Extend the paper and live page only after the numerical law is measured.

**Tech Stack:** Python 3.11+, NumPy, pytest, plain HTML/CSS/JavaScript, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-15-fusion-machine-v1-resident-state-design.md`

## Global Constraints

- Same common input tapes for all policies.
- Warmup boundary state is identical for every policy.
- Full replay `K=N` must be exact; it is the privileged attacker, not hidden.
- Resident-vs-lazy claims are about readiness/latency/history storage, not free compute.
- Analytic predictions are frozen before interpretation.

---

### Task 1: Resident recurrence and bounded replay

**Files:**
- Create: `src/fusion_machine/resident.py`
- Create: `tests/test_resident.py`

**Interfaces:**
- `step(state: float, drive: float, alpha: float) -> float`
- `run_state(drives: Sequence[float], alpha: float, initial: float=0.0) -> float`
- `reconstruct_from_tail(boundary_state, hidden_drives, alpha, k) -> float`
- `analytic_replay_rmse(alpha: float, gap: int, k: int) -> float`

TDD requirements: exact recurrence check; K=gap equals full resident state; K=0 equals boundary decay with all hidden drives omitted; analytic formula equals direct squared-coefficient sum.

### Task 2: Multi-seed v1 receipt

**Files:**
- Create: `experiments/run_v1.py`
- Create: `tests/test_v1_receipt.py`
- Create: `results/v1.json`
- Create: `RESULTS_V1.md`

Run 4096 deterministic seeds for stable Monte Carlo estimates, gap=80, alphas `{0.50,0.80,0.95,0.98}`, replay K `{0,4,8,16,32,64,80}`. Generate independent Rademacher dormant drives directly so the analytic assumptions are exact; retain the direct/relational interpretation in the world description.

Receipt includes measured RMSE, analytic RMSE, relative discrepancy where prediction is nonzero, exact full-replay error, alpha=0.95 primary curve, and work/memory accounting.

Regression requirements: K=80 exact; primary measured RMSE non-increasing with K; max Monte-Carlo/analytic relative discrepancy reported and below a frozen 5% tolerance with 4096 seeds; receipt regeneration exact.

### Task 3: Interpret and expose v1

**Files:**
- Modify: `README.md`
- Modify: `PAPER.md`
- Create: `web/v1.js`
- Modify: `index.html`
- Modify: `web/style.css`
- Modify: `tests/test_web_structure.py`

Add v1 result table and the equation for omitted-history variance. The page gets a resident-vs-replay panel with alpha and replay-window controls and an analytic curve; it must state that full replay is exact and trades latency/history for continuous resident updates.

### Task 4: Verify, PR, merge

Run full pytest, regenerate both v0 and v1 receipts, compile Python, then open a PR. Require Python 3.11/3.12 CI green before merge. Preserve the v0 static deployment workflow.
