# FusionMachine v3 Equal-Recurrent-Attacker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the frozen v3 benchmark comparing FusionMachine against a stronger dense 4-state recurrent attacker under matched state, routing, data and publication budgets, then publish deterministic results and an interactive FAIR FIGHT site section.

**Architecture:** Implement one masked tanh-RNN engine whose masks instantiate FusionMachine and whose all-ones masks instantiate the generic attacker. Train both with the same vectorized NumPy BPTT/Adam loop on the frozen two-process world, evaluate the four frozen regimes and destructive controls, write a deterministic receipt, then expose only the frozen receipt on the static site.

**Tech Stack:** Python 3.11+, NumPy >=1.26, pytest >=8, vanilla HTML/CSS/JS, GitHub Actions/Pages.

**Spec:** `docs/superpowers/specs/2026-09-15-equal-recurrent-attacker-design.md`

## Global Constraints

- Keep the frozen teacher equations, data seeds, train/test sizes, optimizer settings and model masks exactly as specified.
- Do not tune benchmark structure after seeing v3 outcomes; only bug fixes may change implementation.
- FusionMachine and generic RNN each use exactly 4 resident state scalars and 28 learned routing parameters.
- Generic RNN retains its larger 46-parameter non-routing budget; FusionMachine has 30 effective non-routing parameters.
- Both use the same 96 training episodes, 1200 optimizer steps, publication slots and 8 initialization seeds.
- No biological-neuron performance claim; v3 is a synthetic architecture comparison.
- Results must be reproducible byte-for-byte in CI on Python 3.11 and 3.12.

---

### Task 1: Masked recurrent engine and gradients

**Files:**
- Create: `src/fusion_machine/recurrent.py`
- Create: `tests/test_recurrent.py`

**Interfaces:**
- Produces `ModelMasks`, `RNNParams`, `make_fusion_masks()`, `make_generic_masks()`, `init_params(seed)`, `forward_sequence(...)`, `loss_and_grads(...)`, `AdamState`, `adam_step(...)`, and `effective_parameter_count(...)`.
- Later tasks consume these interfaces for training all model variants.

- [ ] **Step 1: Write failing mask/resource tests**

```python
from fusion_machine.recurrent import make_fusion_masks, make_generic_masks, effective_parameter_count


def test_resource_contracts():
    fusion = make_fusion_masks()
    generic = make_generic_masks()
    assert fusion.hidden_size == generic.hidden_size == 4
    assert fusion.route_parameter_count == generic.route_parameter_count == 28
    assert effective_parameter_count(fusion, include_route=False) == 30
    assert effective_parameter_count(generic, include_route=False) == 46
```

- [ ] **Step 2: Run the new tests and verify RED**

Run: `pytest tests/test_recurrent.py -q`
Expected: import failure because `fusion_machine.recurrent` does not exist.

- [ ] **Step 3: Implement parameter containers, masks and forward recurrence**

Use full tensors with masks:

```text
W_hh: (4,4)
W_x: (4,4)
b_h: (4,)
W_out: (2,4)
b_out: (2,)
R: (4,6)
r_bias: (4,)
```

`forward_sequence` returns hidden states, both output-head values, selected predictions and route logits.

- [ ] **Step 4: Add a finite-difference gradient test before implementing BPTT**

Test three representative unmasked coordinates from `W_hh`, `W_x` and `W_out` against central differences on a tiny deterministic sequence with tolerance `rtol=2e-4, atol=2e-5`. Also assert a masked Fusion cross-block coordinate has exactly zero analytic gradient.

- [ ] **Step 5: Run gradient tests and verify RED**

Expected: failure because `loss_and_grads` is absent.

- [ ] **Step 6: Implement vectorized BPTT, route cross-entropy, clipping, L2 and Adam**

Behavioral loss is mean squared error only where `query_mask=True`; routing cross-entropy weight is `0.20`. Apply masks both in forward effective weights and returned gradients. Adam uses the exact spec constants.

- [ ] **Step 7: Run Task 1 tests and full regression suite**

Run: `pytest tests/test_recurrent.py -q && pytest -q`
Expected: all green.

- [ ] **Step 8: Commit Task 1**

Commit message: `feat: add masked recurrent training engine`

---

### Task 2: Frozen nonlinear world and evaluation regimes

**Files:**
- Create: `src/fusion_machine/benchmark_v3.py`
- Create: `tests/test_v3_benchmark.py`

**Interfaces:**
- Produces `EpisodeBatch`, `generate_training_batch()`, `generate_in_distribution()`, `generate_rapid_switch()`, `generate_long_dormancy()`, `publication_slots()`, `teacher_rollout(...)`, `route_targets(...)`, and `fit_latent_probe(...)`.

- [ ] **Step 1: Write failing teacher/world tests**

Assert:

```python
train = generate_training_batch()
assert train.x.shape == (96, 64, 4)
assert train.teacher_state.shape == (96, 64, 4)
assert train.query_mask.sum(axis=1).min() >= 2
assert set(train.context.ravel()) == {0, 1}
```

Also hand-compute the first teacher update from a fixed input and compare to the frozen matrices in the spec.

- [ ] **Step 2: Run and verify RED**

Run: `pytest tests/test_v3_benchmark.py -q`
Expected: missing module/functions.

- [ ] **Step 3: Implement teacher processes, Markov input bits and block schedules**

Use only `np.random.default_rng` with explicit seeds. Training seed is `3101`; final test generators use fixed disjoint seeds recorded in code and receipt.

- [ ] **Step 4: Add failing regime-contract tests**

Assert rapid-switch block lengths are 2..6, long-dormancy switch is exactly at step 48, and publication slots are `0,4,...,60` (16 slots).

- [ ] **Step 5: Implement regimes, route targets and held-out linear probe**

Probe uses least squares with an intercept and reports multivariate variance-weighted `R^2`; no teacher latent target is exposed during model training.

- [ ] **Step 6: Run Task 2 and full tests**

Run: `pytest tests/test_v3_benchmark.py -q && pytest -q`
Expected: all green.

- [ ] **Step 7: Commit Task 2**

Commit message: `feat: freeze v3 nonlinear benchmark world`

---

### Task 3: Training loop, matched attacker and destructive controls

**Files:**
- Create: `experiments/run_v3.py`
- Create: `tests/test_v3_receipt.py`

**Interfaces:**
- Produces `train_one(seed, variant)`, `evaluate_model(...)`, `build_receipt()`, CLI `python -m experiments.run_v3 --out PATH`.

- [ ] **Step 1: Write failing receipt-structure tests**

Require receipt keys:

```text
config
resource_budget
fusion
generic
controls
paired_summary
```

and per-model metrics for `train`, `in_distribution`, `rapid_switch`, `long_dormancy`, `sparse_publication`, `routing`, and `latent_probe`.

- [ ] **Step 2: Run and verify RED**

Run: `pytest tests/test_v3_receipt.py -q`
Expected: missing experiment module.

- [ ] **Step 3: Implement fixed 1200-step full-batch training for one variant**

Variants:

```text
fusion
fusion_context_leak
fusion_dense_recurrence
generic
generic_no_context
```

All use the same training batch and optimizer settings. Initialize with seeds 0..7.

- [ ] **Step 4: Add deterministic single-seed smoke test**

Run seed 0 for 5 optimizer steps in test-only mode and assert repeated calls return bit-identical parameter arrays and losses.

- [ ] **Step 5: Implement evaluation metrics**

Compute:

- train query MSE;
- in-distribution all-step and query MSE;
- rapid-switch all-step and first-post-switch MSE;
- long-dormancy absolute error at offsets `0,1,2,4,8,16`;
- sparse-publication receiver MSE using exactly the 16 fixed slots;
- route accuracy at publication/query slots;
- latent probe held-out `R^2`;
- Fusion cross-half probe leak diagnostics.

- [ ] **Step 6: Implement paired eight-seed summary without pass/fail tuning**

For each primary metric record raw values, medians and `fusion - generic` paired differences. Classification text follows the interpretation rules in the spec and must be derived mechanically from the frozen numbers.

- [ ] **Step 7: Run v3 once and freeze `results/v3.json`**

Run: `python -m experiments.run_v3 --out results/v3.json`
Expected: deterministic JSON and printed summary. Do not alter benchmark settings based on which model wins.

- [ ] **Step 8: Add byte-level receipt regression test**

Test regenerates `build_receipt()` and compares parsed content to `results/v3.json`; CI later performs byte-for-byte `cmp`.

- [ ] **Step 9: Run v3 tests and full suite**

Run: `pytest tests/test_recurrent.py tests/test_v3_benchmark.py tests/test_v3_receipt.py -q && pytest -q`
Expected: all green.

- [ ] **Step 10: Commit Task 3**

Commit message: `experiment: add equal-state recurrent attacker gate`

---

### Task 4: Interpret results and update scientific documents

**Files:**
- Create: `RESULTS_V3.md`
- Modify: `README.md`
- Modify: `PAPER.md`
- Modify: `docs/RELATED_WORK.md`

**Interfaces:**
- Consumes only frozen `results/v3.json` and prior-art sources.

- [ ] **Step 1: Write `RESULTS_V3.md` from the receipt**

Include resource table, all eight seed values for primary metrics, medians, destructive controls, latent-probe result, and explicit negative-result interpretation if generic matches/wins.

- [ ] **Step 2: Update README/PAPER without changing the claim after the fact**

Add v3 to the project spine and replace the old “next attacker” section with the measured result and next unanswered question.

- [ ] **Step 3: Extend related-work fence**

Add:

- Koutník et al. (2014), *A Clockwork RNN*, ICML/PMLR;
- Goyal et al. (2021), *Recurrent Independent Mechanisms*, ICLR/OpenReview;
- Li et al. (2018), *Independently Recurrent Neural Network*, CVPR.

State explicitly that modular recurrence and independent recurrent mechanisms are prior art.

- [ ] **Step 4: Run documentation/result consistency tests**

Add assertions in `tests/test_v3_receipt.py` that headline numeric strings in `RESULTS_V3.md` correspond to frozen receipt medians.

- [ ] **Step 5: Run full tests**

Run: `pytest -q`
Expected: all green.

- [ ] **Step 6: Commit Task 4**

Commit message: `docs: interpret v3 matched recurrent comparison`

---

### Task 5: FAIR FIGHT live site

**Files:**
- Create: `web/v3.js`
- Create: `web/v3.css`
- Modify: `index.html`
- Modify: `tests/test_web_structure.py`

**Interfaces:**
- Site values are frozen from `results/v3.json`; browser code does not retrain or generate alternative science.

- [ ] **Step 1: Add failing web-structure test**

Require IDs/scripts for:

```text
v3-fair-fight
v3-budget
v3-scoreboard
v3-switch-trace
v3-dormancy-curve
v3-publication-strip
web/v3.js
web/v3.css
```

and visible copy saying “A generic-RNN win is a valid result.”

- [ ] **Step 2: Run and verify RED**

Run: `pytest tests/test_web_structure.py -q`
Expected: missing v3 section.

- [ ] **Step 3: Implement FAIR FIGHT section**

Show:

- 4 vs 4 state scalars;
- 28 vs 28 route params;
- 30 vs 46 effective non-routing params;
- identical train episodes/optimizer updates/publication slots;
- frozen regime metrics;
- one representative rapid-switch trace selected from the receipt;
- dormant-offset curve;
- 16 fixed publication slots.

No dynamic benchmark recomputation in JavaScript.

- [ ] **Step 4: Run web and full tests**

Run: `pytest tests/test_web_structure.py -q && pytest -q`
Expected: all green.

- [ ] **Step 5: Commit Task 5**

Commit message: `site: add v3 fair-fight recurrent benchmark`

---

### Task 6: CI, review and merge gate

**Files:**
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- CI must rebuild and compare v0, v1, v2 and v3 receipts on Python 3.11 and 3.12.

- [ ] **Step 1: Extend receipt regeneration**

Add:

```bash
python -m experiments.run_v3 --out /tmp/v3.json
cmp /tmp/v3.json results/v3.json
```

- [ ] **Step 2: Run local/static verification available in this environment**

Run the full pytest suite and v3 receipt regeneration before opening the PR.

- [ ] **Step 3: Open PR with the frozen design and actual result**

PR body must report all primary metrics whether favorable or unfavorable and note that generic has more internal parameters.

- [ ] **Step 4: Wait for both GitHub Actions matrix jobs**

Require Python 3.11 and 3.12 test + receipt jobs to conclude `success`.

- [ ] **Step 5: Review the changed-file diff for scientific/causal mistakes**

Specifically inspect BPTT masks, query masking, context leakage, publication-slot equality, route-parameter equality, data-seed separation and result-document consistency.

- [ ] **Step 6: Merge only the verified head**

After merge, verify main CI and Pages deployment complete successfully.
