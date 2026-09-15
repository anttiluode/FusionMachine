# FusionMachine

> **Keep several computations resident. Let a nonlinear causal boundary decide which one becomes behavior.**

FusionMachine asks whether a vector/state can carry more than data: its separated modes can carry the current outputs — and then the continuing internal states — of different computations. Context arrives later, at a nonlinear causal boundary, without requiring the alternatives to have been averaged into one representation first.

This is an AI architecture experiment with a neuron-inspired lineage, **not a claim that biological dendrites literally implement these equations**.

**Live instrument:** https://anttiluode.github.io/FusionMachine/

## Current spine

```text
v0: keep computational identity separate
        ↓
v1: keep computational history resident
        ↓
next: make the resident modes genuinely different temporal algorithms
```

## v0 — same answer, different algorithm

Define two computations on binary cues in `{-1,+1}`:

```text
A(x) = x0            direct route
B(x) = x1*x2         relational route
```

In the ordinary world `x0=x1*x2`, so A and B produce exactly the same answer. Behavior alone cannot identify which route is present. Break that correlation and add a context bit:

```text
context = -1 -> publish A
context = +1 -> publish B
```

The resident representation keeps both:

```text
z = [A, B]
```

and the minimal exact boundary is

```text
y = 0.5*(A+B) + 0.5*context*(B-A).
```

### Frozen v0 result

| boundary / representation | MSE | accuracy |
|---|---:|---:|
| **nonlinear selector on separated `[A,B]`** | **0.000000** | **1.000** |
| best affine readout of `[A,B,context]` | 0.500000 | 0.750 |
| best full-data lookup after collapse to `(A+B)/2` | 0.500000 | 0.750 |
| **two-row calibrated factorized selector** | **0.000000** | **1.000** |

The collapse failure is information-theoretic. Two intervention states expose the same collapsed `(r,context)=(0,-1)` while requiring opposite outputs, so no later deterministic decoder can reconstruct the lost algorithm identity.

See [`RESULTS_V0.md`](RESULTS_V0.md) and [`results/v0.json`](results/v0.json).

## v1 — algorithmic state keeps moving while behavior looks elsewhere

v1 replaces the static route output with a persistent computation:

```text
h_t = alpha*h_(t-1) + (1-alpha)*u_t.
```

Both A and B can now have histories. During an 80-step interval context stays on A while B receives hidden drives. At the switch to B we compare:

- **resident dual** — B was updated continuously and is already current;
- **bounded replay K** — B was suspended but the last `K` dormant drives were retained;
- **full replay K=80** — privileged exact attacker that stores and replays the whole missing history.

For iid hidden ±1 drives, the switch-time error from forgetting all but the last `K` inputs has the exact variance

```text
Var[e]
 = (1-alpha)^2 * alpha^(2K)
   * (1-alpha^(2(N-K))) / (1-alpha^2).
```

Across 4,096 deterministic tapes and four persistence values, the worst measured-versus-analytic RMSE discrepancy is **1.664%**.

### Primary v1 curve — alpha=0.95, gap=80

| remembered tail K | measured switch RMSE | analytic RMSE |
|---:|---:|---:|
| 0 | 0.160939 | 0.160106 |
| 4 | 0.131933 | 0.130399 |
| 8 | 0.106385 | 0.106199 |
| 16 | 0.070331 | 0.070427 |
| 32 | 0.030810 | 0.030906 |
| 64 | 0.005413 | 0.005395 |
| **80** | **0.000000** | **0.000000** |

The more persistent the mode, the longer the history needed for accurate lazy reconstruction. The smallest `K` that reduces analytic replay RMSE to at most 10% of the zero-history value is:

| alpha | required K of 80 |
|---:|---:|
| 0.50 | **4** |
| 0.80 | **11** |
| 0.95 | **45** |
| 0.98 | **75** |

So the v1 wall sentence is:

> **Persistence is also a replay horizon.**

A slowly forgetting computation remembers distant events, but exactly for that reason those events must have been processed or retained if the computation is suspended.

### Nothing is free

For the 80-step dormant interval:

```text
resident dual
    80 dormant updates during the gap
     0 hidden drives stored
     0 replay updates at switch

lazy full replay
     0 dormant updates during the gap
    80 hidden drives stored
    80 replay updates at switch
```

Resident state is therefore a **readiness strategy**: it converts future replay latency and history storage into continuous local update cost. Full replay is exact when those other resources are allowed.

See [`RESULTS_V1.md`](RESULTS_V1.md) and [`results/v1.json`](results/v1.json).

## What the machine currently means

The emerging object is no longer just a contextual router:

```text
input
  ↓
separated continuing computations
  ↓
[h_A(t), h_B(t), ...]
  ↓
nonlinear context × mode interaction
  ↓
causal publication
```

v0 says **do not collapse computational identity before context arrives**.

v1 says **when those computations have history, preserving identity is not enough: their states must keep evolving, or the missed history must remain recoverable**.

That softens the ordinary data/program distinction. A resident coordinate can be a representation, a sufficient statistic, memory, or the partially executed state of a procedure.

## Why this connects to the earlier repos

The connection is conceptual, not a claim that the mechanisms are identical:

```text
SighImageSuper
    operators assign forgetting times to recoverable distinctions

GAx
    modes can correspond to counterfactually different algorithms

AnttisNeuron / GrowingAnttisNeuron
    branched physical structure can compile and separate dynamics

GeometricNeuronV25 / ThirdWay
    separated routes can preserve causal identity through time

NewMachine
    resident state and causal publication are different variables

FusionMachine
    preserve several computations and their histories,
    fuse/select only at a nonlinear causal boundary
```

v1 makes the Sigh connection particularly exact: the same persistence that protects old information also sets the amount of history needed to reconstruct a suspended mode.

## Prior-art fence

Dendritic nonlinear subunits, mixture-of-experts routing, multiplicative interactions, state-space filtering, streaming sufficient statistics, caching/materialized state, replay, and event-triggered communication are established ideas. [`docs/RELATED_WORK.md`](docs/RELATED_WORK.md) maps the nearest neighborhoods.

FusionMachine's narrower research question is:

> **When is it useful to keep several counterfactually distinct computations alive as resident state and delay context-dependent nonlinear selection until a causal output boundary?**

## Next gate

v2 should make the modes **genuinely different temporal algorithms**, not merely the same leaky recurrence driven by different input relations. The strongest attacker will be an equal-state-capacity generic recurrent model.

If a generic recurrent state preserves the same switching information just as efficiently, the separated-algorithm interpretation may be a useful explanatory coordinate system rather than a computational advantage. That negative would be valuable too.

Only after that should we add sparse external publication, learned mode formation, slow operator rewriting/growth, and stronger biological mappings.

## Run

```bash
python -m pip install -e ".[test]"
pytest -q
python -m experiments.run_v0 --out /tmp/v0.json
python -m experiments.run_v1 --out /tmp/v1.json
```

## Repository map

- `src/fusion_machine/core.py` — two v0 computations and exact nonlinear boundary
- `src/fusion_machine/attackers.py` — v0 linear/collapse attackers and factorized calibration
- `src/fusion_machine/resident.py` — v1 persistent state and bounded-history reconstruction
- `experiments/run_v0.py`, `experiments/run_v1.py` — deterministic scientific receipts
- `results/v0.json`, `results/v1.json` — frozen numerical results
- `RESULTS_V0.md`, `RESULTS_V1.md` — measured interpretations and claim boundaries
- `PAPER.md` — paper-style evolving argument
- `docs/RELATED_WORK.md` — literature neighborhood and novelty fence
- `index.html`, `web/` — static GitHub Pages inspection instrument
- `tests/` — mechanism, receipt, and web regressions
- `docs/superpowers/` — frozen designs and implementation plans
