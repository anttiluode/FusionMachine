# FusionMachine v0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the smallest falsifiable FusionMachine showing that separated resident algorithm modes plus a nonlinear contextual boundary can preserve and select computations that a collapsed representation or linear boundary cannot reproduce.

**Architecture:** The Python core exposes two counterfactually distinct algorithms, an exact nonlinear selector, a best linear attacker, and a collapsed-information attacker over the complete binary truth table. A deterministic experiment writes a frozen JSON receipt and Markdown result. A dependency-light static page mirrors the same truth table so GitHub Pages acts as an inspection instrument rather than the scientific authority.

**Tech Stack:** Python 3.11+, NumPy, pytest, plain HTML/CSS/JavaScript, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-15-fusion-machine-v0-design.md`

## Global Constraints

- v0 isolates separated algorithms plus nonlinear/contextual selection; persistence, sparse communication, biological claims, growth, and general neural-network comparisons remain out of scope.
- The correlated world must make `A(x)=x0` and `B(x)=x1*x2` behaviorally identical.
- The intervention world must enumerate the complete independent binary cue/context truth table.
- Scientific output must be deterministic and frozen in a checked-in JSON receipt.
- Negative controls are retained and reported rather than tuned away.
- The browser page mirrors the mechanism but is not the scientific authority.

---

### Task 1: Core truth-table machine and exact gates

**Files:**
- Create: `pyproject.toml`
- Create: `src/fusion_machine/__init__.py`
- Create: `src/fusion_machine/core.py`
- Create: `tests/test_core.py`

**Interfaces:**
- Produces: `algorithm_a(x0: int, x1: int, x2: int) -> int`
- Produces: `algorithm_b(x0: int, x1: int, x2: int) -> int`
- Produces: `nonlinear_boundary(a: float, b: float, context: int) -> float`
- Produces: `correlated_rows() -> list[dict[str, int]]`
- Produces: `intervention_rows() -> list[dict[str, int]]`

- [ ] **Step 1: Write failing core tests**

Create tests proving: correlated rows always satisfy A=B; intervention rows contain all 16 combinations of independent A/B/context induced by the eight cue triples times two contexts; A and B disagree on half the cue triples; the nonlinear boundary returns exactly A for context -1 and B for context +1.

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `python -m pytest tests/test_core.py -v`
Expected: import failure because `fusion_machine.core` does not exist.

- [ ] **Step 3: Implement the minimal core**

`algorithm_a` returns `x0`; `algorithm_b` returns `x1*x2`; the exact selector is `0.5*(a+b)+0.5*context*(b-a)`. Enumerators use `itertools.product((-1,1), repeat=...)` and return deterministic dictionaries containing cues, A, B, context, and target.

- [ ] **Step 4: Run core tests**

Run: `python -m pytest tests/test_core.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

Commit message: `feat: add FusionMachine truth-table core`

---

### Task 2: Linear and collapsed attackers with explicit impossibility witness

**Files:**
- Modify: `src/fusion_machine/core.py`
- Create: `src/fusion_machine/attackers.py`
- Create: `tests/test_attackers.py`

**Interfaces:**
- Consumes: `intervention_rows()` from Task 1.
- Produces: `fit_linear_boundary(rows) -> dict`
- Produces: `best_collapsed_lookup(rows) -> dict`
- Produces: `collapse_witness(rows) -> tuple[dict, dict]`

- [ ] **Step 1: Write failing attacker tests**

Tests require: the best linear least-squares predictor over `[1,A,B,c]` has non-zero MSE on the complete truth table; the best deterministic lookup keyed only by `(r,c)` with `r=(A+B)/2` has non-zero classification error; `collapse_witness` returns two rows with identical `(r,c)` and opposite targets.

- [ ] **Step 2: Run attacker tests and verify failure**

Run: `python -m pytest tests/test_attackers.py -v`
Expected: import failure for `fusion_machine.attackers`.

- [ ] **Step 3: Implement attackers**

Use NumPy `lstsq` for the linear readout. For the collapsed lookup, group targets by `(r,c)` and choose the mean target as the minimum-MSE prediction; report sign classification with deterministic zero handling. Search the rows for an exact collision pair as the information witness.

- [ ] **Step 4: Run attacker and full tests**

Run: `python -m pytest tests/test_attackers.py tests/test_core.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

Commit message: `test: add linear and collapse attackers`

---

### Task 3: Deterministic scientific receipt

**Files:**
- Create: `experiments/run_v0.py`
- Create: `results/v0.json`
- Create: `RESULTS_V0.md`
- Create: `tests/test_receipt.py`

**Interfaces:**
- Consumes: core and attacker functions.
- Produces: deterministic JSON with correlated agreement, intervention disagreement fraction, nonlinear MSE/accuracy, linear MSE/accuracy, collapsed MSE/accuracy, exact fitted linear weights, and collapse witness rows.

- [ ] **Step 1: Write receipt regression test**

Test calls `build_receipt()` from `experiments.run_v0` and checks exact structural invariants plus numerical values within `1e-12`. It also compares the generated object with checked-in `results/v0.json`.

- [ ] **Step 2: Run receipt test and verify failure**

Run: `python -m pytest tests/test_receipt.py -v`
Expected: import/file failure.

- [ ] **Step 3: Implement experiment and freeze receipt**

`run_v0.py` exposes `build_receipt()` and a CLI `--out`. Generate the complete table, evaluate all policies, serialize with sorted keys and indentation, and write the canonical `results/v0.json`.

- [ ] **Step 4: Write `RESULTS_V0.md` from measured values only**

Report the exact pass/fail values, the explicit collision witness, and the claim boundary. Do not use biological language as evidence.

- [ ] **Step 5: Run full Python tests**

Run: `python -m pytest -q`
Expected: PASS.

- [ ] **Step 6: Commit**

Commit message: `science: freeze FusionMachine v0 receipt`

---

### Task 4: Paper-style README and related-work fence

**Files:**
- Create: `README.md`
- Create: `PAPER.md`
- Create: `docs/RELATED_WORK.md`

**Interfaces:**
- Consumes: frozen `RESULTS_V0.md` and design spec.
- Produces: public explanation of the mathematical object, result, limitations, and literature neighborhood.

- [ ] **Step 1: Write README around the earned claim**

Open with the machine: resident vector `[A,B]`, nonlinear boundary, sparse future direction. Include result table copied from the receipt and a lineage section linking conceptually to SighImageSuper, GAx, ThirdWay, AnttisNeuron, and NewMachine without claiming equivalence.

- [ ] **Step 2: Write `PAPER.md`**

Structure: Abstract; Motivation; Formal model; Complete truth-table experiment; Impossibility attackers; Results; Interpretation; Limitations; Next gates. The paper must explicitly distinguish “algorithm output stored as a mode” from the stronger future claim “arbitrary algorithm implemented by the mode dynamics.”

- [ ] **Step 3: Write `docs/RELATED_WORK.md`**

Anchor the work to dendritic nonlinear subunits, mixture-of-experts/conditional computation, multiplicative/context gating, recurrent/state-space models, event-triggered communication, and system observability. State what is established prior art and what FusionMachine is actually testing.

- [ ] **Step 4: Commit**

Commit message: `docs: explain FusionMachine v0 and prior-art boundary`

---

### Task 5: Live GitHub Pages instrument

**Files:**
- Create: `index.html`
- Create: `web/app.js`
- Create: `web/style.css`
- Create: `tests/test_web_structure.py`

**Interfaces:**
- Consumes: the same exact formulas as the Python core.
- Produces: static interactive truth-table explorer.

- [ ] **Step 1: Write page-structure test**

Require the root `index.html` to load `web/style.css` and `web/app.js`, contain controls for `x0`, `x1`, `x2`, and context, and contain panels for resident mode A, resident mode B, collapsed state, nonlinear publication, and attacker output.

- [ ] **Step 2: Run page test and verify failure**

Run: `python -m pytest tests/test_web_structure.py -v`
Expected: FAIL because page files do not exist.

- [ ] **Step 3: Implement the static instrument**

The page lets the user toggle cues/context, shows A and B as two resident channels, highlights agreement versus intervention disagreement, displays `r=(A+B)/2`, and contrasts the exact nonlinear publication with a frozen best-linear attacker. Include an auto-play mode that cycles the complete truth table.

- [ ] **Step 4: Run page-structure and full tests**

Run: `python -m pytest -q`
Expected: PASS.

- [ ] **Step 5: Commit**

Commit message: `feat: add live FusionMachine truth-table lab`

---

### Task 6: CI and final verification

**Files:**
- Create: `.github/workflows/ci.yml`
- Keep: `.github/workflows/static.yml`

**Interfaces:**
- CI installs editable test dependencies, runs `pytest -q`, and regenerates a temporary v0 receipt to verify determinism.

- [ ] **Step 1: Add CI workflow**

Run on pushes and pull requests for Python 3.11 and 3.12. Install with `python -m pip install -e '.[test]'`, run `pytest -q`, then run `python -m experiments.run_v0 --out /tmp/v0.json`.

- [ ] **Step 2: Open a pull request from `fusion-v0` to `main`**

PR body summarizes the exact scientific claim and negative controls.

- [ ] **Step 3: Inspect workflow runs and fix any failures**

Use the commit/PR workflow status and job logs. Do not weaken scientific assertions to make CI green; fix software or correct the receipt if the implementation disproves a planned numerical expectation.

- [ ] **Step 4: Merge only after all checks pass**

Use squash merge unless repository policy requires otherwise.
