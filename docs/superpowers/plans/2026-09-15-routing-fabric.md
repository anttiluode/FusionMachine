# FusionMachine v2 Routing Fabric Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and test a routed asynchronous machine in which rich resident dendritic state is published as a one-bit event, while axonal/synaptic material carries the route that selects the next dendritic computation.

**Architecture:** `routing.py` owns event-machine primitives and causal routing controls; `development.py` owns graph formation from imperfect chemistry plus activity alignment; `run_v2.py` freezes a deterministic science battery. The root static site gets a v2 animated routing panel but remains a visualization of the checked-in Python receipt.

**Tech Stack:** Python 3.11+, NumPy, pytest, plain HTML/CSS/JavaScript, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-15-routing-fabric-design.md`

## Global Constraints

- Event payload is binary; route/address identity lives in the graph rather than in event amplitude.
- Resident state, publication gating, routing, and target update are separate variables and code units.
- Chemistry/activity development is a synthetic computational analogue, not a literal molecular-development model.
- Negative controls and privileged oracles remain visible in frozen results.
- Scientific failures are reported rather than tuned away.
- Existing v0/v1 receipts remain reproducible.

---

### Task 1: Routed event-machine primitives

**Files:**
- Create: `src/fusion_machine/routing.py`
- Create: `tests/test_routing.py`

**Interfaces:**
- `DendriticCompartment(state: float, leak: float)` with `step(drive: float) -> float`
- `ais_event(state: float, threshold: float, suppressed: bool = False) -> int`
- `Route(target: int, weight: float, delay: int = 1)`
- `route_event(source: int, event: int, routes: dict[int, list[Route]]) -> list[tuple[int,int,float]]`
- `apply_writes(states: list[DendriticCompartment], writes: list[tuple[int,float]]) -> list[float]`
- `run_async_chain(...) -> dict[str, object]`

- [ ] **Step 1: Write failing routing tests**

```python
def test_same_bit_reaches_different_targets_by_axon_identity():
    routes = {0: [Route(target=1, weight=1.0)], 1: [Route(target=2, weight=1.0)]}
    assert route_event(0, 1, routes) == [(1, 1, 1.0)]
    assert route_event(1, 1, routes) == [(1, 2, 1.0)]


def test_ais_suppression_blocks_publication_not_state_update():
    d = DendriticCompartment(state=0.0, leak=0.5)
    before = d.step(1.0)
    assert before > 0
    assert ais_event(d.state, threshold=0.1, suppressed=True) == 0
    after = d.step(1.0)
    assert after > before


def test_reroute_changes_target_without_changing_event():
    event = 1
    left = route_event(0, event, {0: [Route(1, 1.0)]})
    right = route_event(0, event, {0: [Route(2, 1.0)]})
    assert left != right
```

- [ ] **Step 2: Run focused tests and verify red**

Run: `pytest tests/test_routing.py -v`
Expected: import failure because `fusion_machine.routing` does not yet exist.

- [ ] **Step 3: Implement minimal routing primitives**

Use a dataclass for `DendriticCompartment` and `Route`. `step` computes `state = leak*state + (1-leak)*drive`. `ais_event` returns `0` when suppressed, else `int(state >= threshold)`. `route_event` returns delayed `(delay,target,weighted_payload)` tuples only when `event==1`.

- [ ] **Step 4: Add asynchronous-chain test and implementation**

Construct a deterministic 3-stage chain where source 0 fires at t=0, route delays are `[1,2]`, and the event log must expose source, target, emission time, arrival time, and weight. No global layer index may be used.

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_routing.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

Commit message: `feat: add binary event routing fabric`

---

### Task 2: Chemistry + activity developmental routing

**Files:**
- Create: `src/fusion_machine/development.py`
- Create: `tests/test_development.py`

**Interfaces:**
- `make_problem(seed: int, n_sources: int = 12, n_targets: int = 12, dim: int = 6) -> dict`
- `chemical_scores(source_fp, target_fp, noise) -> np.ndarray`
- `activity_alignment(source_activity, target_activity) -> np.ndarray`
- `grow_routes(chemistry: np.ndarray, activity: np.ndarray, beta: float, top_k: int = 1) -> np.ndarray`
- `graph_accuracy(routes: np.ndarray, truth: np.ndarray) -> float`
- `routing_accuracy(routes: np.ndarray, truth: np.ndarray) -> float`

- [ ] **Step 1: Write failing deterministic tests**

Use a fixed seed problem with imperfect cues. Tests must assert that the returned problem is reproducible and that route matrices have one selected target per source for `top_k=1`.

- [ ] **Step 2: Verify red**

Run: `pytest tests/test_development.py -v`
Expected: import failure for `fusion_machine.development`.

- [ ] **Step 3: Implement synthetic development problem**

Create latent source→target permutation truth. Source and target chemical fingerprints share a noisy latent family vector, but add enough noise that chemistry alone is imperfect. Generate development episodes where source activity predicts its true target with independent corruption, so activity alone is also imperfect.

- [ ] **Step 4: Add comparison test**

For the frozen seed bank `range(32)`, compute chemistry-only (`beta=0`), activity-only (zero chemistry), combined (`beta=1` after z-score normalization), random, and shuffled-activity controls. The test should require mean combined graph accuracy to exceed both single-cue means by at least 0.05. If the frozen construction does not satisfy this, adjust only the synthetic problem parameters before freezing the experiment; do not tune against later held-out evaluation.

- [ ] **Step 5: Run development tests**

Run: `pytest tests/test_development.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

Commit message: `science: add chemistry activity route development`

---

### Task 3: Frozen v2 scientific battery

**Files:**
- Create: `experiments/run_v2.py`
- Create: `results/v2.json`
- Create: `RESULTS_V2.md`
- Create: `tests/test_v2_receipt.py`

**Interfaces:**
- `build_receipt() -> dict[str, object]`
- CLI: `python -m experiments.run_v2 --out PATH`

- [ ] **Step 1: Write failing receipt regression test**

The receipt must include top-level keys `address_in_matter`, `development`, `ais_suppression`, `reroute`, and `async_chain`. Assert exact structural invariants and compare fresh generation to checked-in JSON.

- [ ] **Step 2: Verify red**

Run: `pytest tests/test_v2_receipt.py -v`
Expected: import/file failure.

- [ ] **Step 3: Implement address-in-matter battery**

Use 8 source identities, each emitting payload `1`, with one unique intended target. Report intact, pooled-source, shuffled-route, and explicit-digital-address-oracle accuracies. The pooled attacker must operate only on the pooled bit, not source identity.

- [ ] **Step 4: Implement developmental battery**

Use separate development and held-out episode seeds. Report graph recovery and held-out routing accuracy for combined, chemistry-only, activity-only, random, and shuffled-activity policies.

- [ ] **Step 5: Implement AIS suppression and reroute batteries**

Suppression receipt must show identical resident-state trajectory with and without publication suppression over the window, zero suppressed events, and immediate post-window event when threshold is met. Reroute receipt must show bit-identical source event traces but different downstream target traces.

- [ ] **Step 6: Implement async chain receipt**

Freeze one event log with at least 4 causal hops and nonuniform delays. Verify all arrivals are generated from scheduled route delays rather than layer order.

- [ ] **Step 7: Freeze receipt and result narrative**

Write `results/v2.json` with sorted keys and `RESULTS_V2.md` from measured values only. Classify each scientific gate PASS/FAIL explicitly.

- [ ] **Step 8: Run full Python suite**

Run: `pytest -q`
Expected: PASS.

- [ ] **Step 9: Commit**

Commit message: `science: freeze FusionMachine v2 routing receipt`

---

### Task 4: Paper, README, and biology fence

**Files:**
- Modify: `README.md`
- Modify: `PAPER.md`
- Modify: `docs/RELATED_WORK.md`

**Interfaces:**
- Consumes: `RESULTS_V2.md` and `results/v2.json`.

- [ ] **Step 1: Update README spine**

Promote the machine to:

```text
resident computation
→ nonlinear AIS commit
→ one-bit event
→ material routing graph
→ local target write
→ next resident computation
```

Include the measured v2 table and keep v0/v1 intact.

- [ ] **Step 2: Extend PAPER.md**

Add a v2 section covering address-in-matter, developmental routing, suppression, reroute causal intervention, async chain, and limitations. Explicitly distinguish source/axon identity from payload information.

- [ ] **Step 3: Extend RELATED_WORK.md**

Add the 2025 Fréal & Hoogenraad Neuron review (`10.1016/j.neuron.2025.01.004`) as motivation for AIS functional separation and plasticity; add Sperry/chemoaffinity and activity-dependent refinement as historical biological neighborhoods. State that the implemented chemistry/activity score is synthetic.

- [ ] **Step 4: Commit**

Commit message: `docs: define dendrite AIS axon routing architecture`

---

### Task 5: Live v2 routing laboratory

**Files:**
- Modify: `index.html`
- Modify: `web/style.css`
- Modify: `web/app.js` or create `web/v2.js`
- Modify: `tests/test_web_structure.py`

**Interfaces:**
- Static site only; Python receipt remains authority.

- [ ] **Step 1: Write failing web-structure tests**

Require elements with ids `v2-lab`, `v2-source`, `v2-ais-gate`, `v2-pulse`, `v2-route-mode`, `v2-targets`, `v2-reroute`, and `v2-step`. Require visible copy containing `one-bit event` and `address lives in the route`.

- [ ] **Step 2: Verify red**

Run: `pytest tests/test_web_structure.py -v`
Expected: FAIL because v2 DOM is absent.

- [ ] **Step 3: Implement v2 panel**

Add three visual columns: dendritic residents, AIS commit boundary, axonal routes/target dendrites. Animate a single bright pulse along the selected wire. Controls select source, AIS suppression, learned/chemistry/activity/shuffled route mode, reroute branch, step, and autoplay.

- [ ] **Step 4: Add causal readouts**

Show payload (`1 bit`), source/axon identity, selected route, target address, current dendritic states, and a short event log. Suppression must visibly keep dendritic state moving while no pulse leaves the AIS.

- [ ] **Step 5: Run full tests**

Run: `pytest -q`
Expected: PASS.

- [ ] **Step 6: Commit**

Commit message: `feat: add live dendrite AIS axon routing lab`

---

### Task 6: CI, PR, and integration

**Files:**
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: Extend CI receipt checks**

Add:

```bash
python -m experiments.run_v2 --out /tmp/v2.json
cmp /tmp/v2.json results/v2.json
```

while retaining v0/v1 regeneration.

- [ ] **Step 2: Run fresh local verification**

Run:

```bash
pytest -q
python -m experiments.run_v0 --out /tmp/v0.json && cmp /tmp/v0.json results/v0.json
python -m experiments.run_v1 --out /tmp/v1.json && cmp /tmp/v1.json results/v1.json
python -m experiments.run_v2 --out /tmp/v2.json && cmp /tmp/v2.json results/v2.json
```

Expected: all commands exit 0.

- [ ] **Step 3: Open PR to main**

PR body must state every v2 gate result and the biological claim boundary.

- [ ] **Step 4: Inspect GitHub Actions**

Require Python 3.11 and 3.12 jobs to complete successfully. Fix software failures without weakening scientific gates.

- [ ] **Step 5: Merge after green**

Squash merge only after the final head SHA is green.