# FusionMachine v1 — resident algorithmic state and bounded replay

Date: 2026-09-15

## Question

v0 showed that two counterfactually distinct computations can remain separated in one resident vector and that a nonlinear/contextual boundary can select one without first collapsing their identity.

v1 asks the stronger temporal question:

> If each mode is a computation with internal state, is there value in keeping the dormant computation running even while it is not behaviorally selected?

The expected answer must be framed as a tradeoff, not magic: eager resident computation spends work continuously; lazy computation can postpone that work, but exact recovery after a switch requires replaying the missed history or storing an equivalent sufficient state.

## Dynamical computations

At each step the world supplies binary cues

```text
x0, x1, x2 in {-1,+1}.
```

The two route-specific drives remain the v0 computations:

```text
uA_t = x0_t
uB_t = x1_t*x2_t.
```

Each route now has persistent state with the same retention `alpha`:

```text
hA_t = alpha*hA_(t-1) + (1-alpha)*uA_t
hB_t = alpha*hB_(t-1) + (1-alpha)*uB_t.
```

The published value is selected by context using the same exact boundary:

```text
y_t = 0.5*(hA_t+hB_t) + 0.5*c_t*(hB_t-hA_t)
```

with `c=-1` selecting A and `c=+1` selecting B.

## Controlled episode

Each seed contains:

1. **correlated warmup** — `uA=uB`, allowing both route states to start from the same known value;
2. **hidden divergence window** of length `N` — independent cues break the relation while context stays on A;
3. **switch to B** — the machine is suddenly asked for the computation that was behaviorally irrelevant during the hidden window.

All policies receive the exact same input tape.

The warmup boundary state is provided identically to all policies. Therefore v1 measures only what happened during the hidden divergence window, not an initialization advantage.

## Policies

### Resident dual

Updates both `hA` and `hB` every step. At the context switch, B is already current.

### Lazy zero-history

Updates only the selected computation. It retains the dormant state from the warmup boundary but does not retain hidden inputs. At the switch it can correctly apply passive decay over the missed gap, but it must treat all unremembered drives as zero.

This is stronger than simply leaving the stale state frozen.

### Bounded replay K

Updates only the selected computation during the hidden window but stores the last `K` dormant-route drives. At the switch it reconstructs the dormant state by:

- analytically decaying the known boundary state across the whole gap;
- adding the exact contributions from the last `K` stored drives.

`K=0` equals the zero-history estimator. `K=N` is exact full replay.

## Analytic replay-error law

Let the hidden divergence window contain `N` independent Rademacher dormant drives `u_j in {-1,+1}` and let only the last `K` be remembered.

The reconstruction error at the switch is exactly the omitted prefix contribution:

```text
e = (1-alpha) * sum_{r=K}^{N-1} alpha^r * u_(N-r).
```

Therefore

```text
E[e] = 0
```

and

```text
Var[e]
 = (1-alpha)^2 * sum_{r=K}^{N-1} alpha^(2r)
 = (1-alpha)^2 * alpha^(2K)
   * (1-alpha^(2(N-K))) / (1-alpha^2).
```

The predicted RMSE is the square root of this quantity.

For long gaps relative to the mode lifetime, replay error falls approximately as

```text
RMSE(K) proportional to alpha^K.
```

So the replay memory needed for a fixed relative tolerance grows with the forgetting time of the resident mode.

This is the direct bridge to SighImageSuper: persistence that makes a mode useful as memory also makes it expensive to reconstruct after it has been ignored.

## Gates

### Gate 1A — dormant computation stays current

Use `alpha=0.95`, `N=80`, and at least 256 deterministic random seeds.

Measure dormant-B switch error for:

- resident dual;
- lazy K=0;
- bounded replay K in `{4,8,16,32,64,80}`.

Required invariants:

- resident dual switch error is exactly zero relative to the full reference;
- K=N reconstruction is exact to floating-point tolerance;
- bounded replay error is non-increasing with K in aggregate;
- K=0 has nonzero aggregate error.

### Gate 1B — analytic law

Sweep

```text
alpha in {0.50, 0.80, 0.95, 0.98}
K     in {0, 4, 8, 16, 32, 64, 80}
N     = 80.
```

For each pair, compare measured Monte Carlo RMSE with the closed-form prediction above.

Primary gate: maximum relative disagreement between measured and analytic RMSE must be small enough to be explained by finite-seed Monte Carlo variation; report the exact value rather than hiding failures. The exact `K=N` zero case is tested separately.

### Gate 1C — work / memory accounting

For one hidden gap of length `N`:

- resident dual performs `N` dormant-route updates during the gap and stores one current dormant state;
- exact lazy replay performs zero dormant updates during the gap but must retain `N` dormant drives and execute `N` replay contributions at the switch.

The scientific sentence is therefore about **scheduling and readiness**, not free computation:

> resident state converts future replay latency and history storage into continuous local update cost.

This accounting should be explicit in the receipt.

## Attacks and claim boundary

v1 must not claim that resident computation beats lazy evaluation universally. Full replay is an exact attacker when complete history and switch latency are allowed.

A pass supports only:

> **A dormant dynamical computation can remain immediately available by keeping its sufficient state resident. If the computation is suspended, exact later recovery requires the missed history or an equivalent sufficient statistic; with bounded replay, reconstruction error decays according to the dormant mode's own forgetting time.**

This is closely related to ordinary streaming sufficient statistics, materialized state, state-space filtering, lazy evaluation, and replay. Novelty is not assumed.

## Next gate after v1

v2 should make the two resident modes *different dynamical algorithms*, not merely the same leaky recurrence driven by different route computations. Candidate pair:

- a temporal integrator / predictor;
- a different finite-state or relational temporal computation.

The important attacker will be an equal-state-capacity generic recurrent model. If a generic state representation preserves everything FusionMachine needs, the separated-mode interpretation is an explanatory lens rather than a computational advantage.

## Artifacts

- `src/fusion_machine/resident.py` — recurrence and replay reconstruction;
- `experiments/run_v1.py` — deterministic multi-seed gate;
- `results/v1.json` — frozen receipt;
- `RESULTS_V1.md` — measured interpretation;
- updates to `PAPER.md` and `README.md`;
- live page section visualizing a hidden dormant computation and replay window;
- tests for exact recurrence, exact K=N replay, monotonic aggregate replay error, analytic variance formula, and frozen receipt.
