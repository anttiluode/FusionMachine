# FusionMachine v3 — Equal-State Generic Recurrent Attacker

## Status

Frozen before running v3 benchmark results.

## Scientific question

Does FusionMachine's explicit separation of continuously resident computational modes provide a measurable advantage over a generic dense recurrent state when both systems receive the same observations and training episodes, have the same number of resident state scalars, use the same learned routing-parameter budget, and are evaluated through the same publication opportunities?

A negative result is valid: if the dense RNN matches FusionMachine, the dendrite/AIS/axon decomposition may be primarily an explanatory coordinate system rather than a computational advantage on this task family.

## Why this attacker

The first matched attacker is a vanilla tanh RNN with a fully dense recurrent transition. A strictly total-parameter-matched low-rank RNN would artificially constrain the attacker. A GRU/LSTM is reserved for a later gate because its additional gates add a second architectural question.

The dense RNN therefore receives **more internal trainable freedom** than FusionMachine while matching the resources named by the v2 paper:

- 4 resident state scalars;
- the same 28 learned routing parameters;
- the same training episodes and query labels;
- the same publication slots;
- the same optimizer steps and learning-rate schedule.

## Frozen synthetic world

Every episode contains two nonlinear latent processes, A and B. Both evolve at every time step regardless of which process is behaviorally relevant.

Raw input at time `t` is

```text
u_t, v_t in {-1,+1}
p_t = u_t * v_t
```

The raw bits are temporally correlated: each independently flips sign with probability `0.15` per step.

### Teacher mode A

```text
A_t = tanh(M_A A_(t-1) + U_A [u_t, v_t, p_t])

M_A = [[ 0.78,  0.22],
       [-0.18,  0.70]]

U_A = [[ 0.55,  0.10,  0.30],
       [-0.15,  0.50,  0.20]]

y_A = [0.80, -0.35] dot A_t
```

### Teacher mode B

```text
B_t = tanh(M_B B_(t-1) + U_B [u_t, v_t, p_t])

M_B = [[-0.45,  0.62],
       [ 0.30,  0.74]]

U_B = [[ 0.10,  0.60, -0.35],
       [ 0.45, -0.10,  0.40]]

y_B = [-0.40, 0.90] dot B_t
```

Context `c_t in {0,1}` chooses the behavioral target only:

```text
c_t = 0 -> target y_A
c_t = 1 -> target y_B
```

The teacher dynamics themselves never receive `c_t`.

## Training distribution

For each training episode:

- sequence length: `64`;
- initial `u` and `v`: random ±1;
- context starts randomly in A or B;
- context block lengths: integer uniform in `[16, 32]`;
- supervision occurs only at the **last step of each context block** and at the final sequence step;
- both hidden teacher processes continue evolving during every block.

This makes supervision sparse and delayed. A process can evolve for many steps while behavior is looking at the other process.

Frozen training set:

- `96` episodes;
- data seed `3101`;
- no validation-driven task redesign after results are observed.

## Model 1 — FusionMachine

Resident state has four scalars split into two two-dimensional modes:

```text
h = [h_A0, h_A1, h_B0, h_B1]
```

Transition:

```text
h_t = tanh(W_hh h_(t-1) + W_x x_t + b)
```

with structural masks:

- `W_hh` is block diagonal `2x2 + 2x2`;
- `x_t = [u_t, v_t, p_t, c_t]`;
- the `c_t` column of `W_x` is structurally zero;
- output head A reads only the A half;
- output head B reads only the B half;
- context chooses between the two output heads only at the behavioral boundary.

Thus resident computation is structurally independent of current publication relevance.

Effective non-routing trainable parameters:

```text
W_hh: 8
W_x: 12
b: 4
W_out: 4
b_out: 2
TOTAL: 30
```

## Model 2 — dense generic RNN

The attacker uses the same four hidden scalars and the same tanh recurrence, but:

- `W_hh` is fully dense `4x4`;
- all four input features including context can enter every hidden unit;
- both context-conditioned output heads may read all four hidden units.

Effective non-routing trainable parameters:

```text
W_hh: 16
W_x: 16
b: 4
W_out: 8
b_out: 2
TOTAL: 46
```

The generic model therefore has 16 more internal/readout parameters. This is intentional: it is a generous attacker rather than a parameter-starved strawman.

## Shared learned routing head

Routing is controlled rather than made the source of the architectural difference.

Each episode also has `source_id in {0,1,2,3}`. At each supervised/publication point, the destination is

```text
mode A: destination = source_id
mode B: destination = (source_id + 2) mod 4
```

Both models receive an independent routing head with exactly the same shape and training procedure:

```text
route features = one_hot(source_id, 4) || one_hot(context, 2)
route logits   = R @ route_features + r_bias
R shape        = (4,6)
r_bias shape   = (4,)
```

Routing parameter budget per model: `24 + 4 = 28`.

Routing cross-entropy is added to the behavioral loss with weight `0.20`. Because this routing problem is identical for both models, a route-accuracy difference is evidence of optimization instability, not an intended architectural effect.

## Optimizer and initialization

Both models use the same NumPy implementation of full-batch BPTT and Adam.

Frozen settings:

```text
training restarts: 8, seeds 0..7
steps: 1200
learning rate: 0.010
Adam beta1: 0.9
Adam beta2: 0.999
Adam epsilon: 1e-8
global gradient-norm clip: 5.0
L2 weight decay: 1e-4
parameter initialization: N(0, 0.15^2)
```

The masked FusionMachine tensors are stored at the same full shapes as the generic model, but masked entries contribute neither forward signal nor gradient.

No early stopping is used. Both receive exactly 1200 optimizer updates.

## Test regimes

All test data are generated from the frozen teacher and use data seeds disjoint from training.

### 1. In-distribution

- 256 episodes;
- length 64;
- context blocks `[16,32]`;
- evaluate target MSE at every time step and separately at block-end query points.

### 2. Rapid switching

- 256 episodes;
- length 64;
- context blocks `[2,6]`;
- target evaluated at every time step;
- primary statistic: MSE on the **first step after a context switch**.

This asks whether a newly relevant computation is already current.

### 3. Long dormancy

- 256 episodes;
- length 96;
- context remains on one mode for 48 steps, then switches to the other for 48;
- report absolute error at offsets `0,1,2,4,8,16` after the switch;
- the primary statistic is offset `0`.

### 4. Matched sparse publication

Use the rapid-switch episodes and predictions, but expose them to a receiver only on the same fixed publication slots for both models:

```text
publication slots: t = 0,4,8,...,60
```

At a publication slot the receiver receives the current scalar prediction and routing decision; between slots it holds the last published scalar. Both models therefore receive exactly 16 scalar publication opportunities per 64-step episode.

Report receiver MSE over all 64 time steps and route accuracy on the 16 publication slots.

This is a matched communication schedule, not a claim that biological spikes transmit scalar values.

## Hidden-world-state diagnostic

Teacher latent state is never used as a training target.

After model training, fit a linear least-squares probe from each model's four hidden scalars to the teacher's four latent scalars `[A0,A1,B0,B1]` using a probe-training dataset disjoint from both model training and final evaluation. Report held-out multivariate `R^2`.

This is a diagnostic only. High `R^2` means the world state is linearly recoverable from the learned representation; it does not prove the model discovered the same algorithm or basis.

For FusionMachine, additionally report cross-half probe leakage:

- teacher A predicted from Fusion B half;
- teacher B predicted from Fusion A half.

## Primary outcome rules

The gate is not defined as "FusionMachine must win."

Across the 8 paired initialization seeds, report medians and per-seed paired differences.

Interpretation:

- **Generic matches or wins on rapid-switch and long-dormancy metrics:** the explicit resident-mode decomposition is not computationally necessary for this benchmark.
- **Fusion wins only in-distribution:** no meaningful support for the readiness hypothesis.
- **Fusion wins on first-post-switch / long-dormancy while generic has equal or lower training error:** evidence that the structural separation acts as a useful inductive bias rather than merely improving optimization.
- **Fusion also lowers receiver error at the identical sparse-publication schedule:** stronger evidence that ready resident computation improves downstream behavior under communication constraints.

No statistical significance claim will be made from eight optimizer seeds alone. Report effect sizes, medians, raw seed values, and exact deterministic receipts.

## Required destructive controls

1. **Fusion context leak:** allow `c_t` into Fusion hidden-state updates. If this removes the switch advantage, it supports the claim that relevance-independent residence mattered.
2. **Fusion dense recurrence:** remove the block-diagonal mask while retaining the late two-head selector. This separates the value of recurrent modularity from the value of a late contextual output boundary.
3. **Generic no-context recurrence:** forbid `c_t` from the generic recurrence while leaving it dense. This checks whether context leakage, rather than dense mixing itself, is the primary failure mode.

These controls use the same state count, data and optimizer budget.

## Site update

The GitHub Pages laboratory gains a v3 section called **FAIR FIGHT** with:

- resource-budget badges;
- a table for the four evaluation regimes;
- an abrupt-switch trace comparing target, FusionMachine, and generic RNN;
- a dormant-gap error curve;
- a matched-publication timeline showing identical publication slots;
- visible wording that a generic-RNN win is a valid scientific result.

The site renders only frozen result values checked into `results/v3.json`; the Python receipt remains authoritative.

## Files

Planned additions:

- `src/fusion_machine/benchmark_v3.py` — frozen world generation and evaluation regimes;
- `src/fusion_machine/recurrent.py` — masked tanh RNN, BPTT and Adam;
- `experiments/run_v3.py` — deterministic eight-seed benchmark and receipt;
- `tests/test_recurrent.py` — masks, gradient sanity, equal-resource contracts;
- `tests/test_v3_benchmark.py` — frozen world/regime/control contracts;
- `tests/test_v3_receipt.py` — deterministic result receipt checks;
- `results/v3.json`;
- `RESULTS_V3.md`;
- `web/v3.js`, `web/v3.css`;
- updates to `README.md`, `PAPER.md`, `docs/RELATED_WORK.md`, `index.html`, CI and web structure tests.

## Prior-art fence

The comparison must explicitly acknowledge modular recurrent architectures. Clockwork RNN partitions recurrent hidden state into modules operating at different temporal rates. Recurrent Independent Mechanisms uses multiple nearly independent recurrent mechanisms with sparse communication and reports improved generalization when environmental factors change. IndRNN independently recurrent units provide another example of recurrent structural constraints improving trainability/interpretability.

FusionMachine v3 is not a novelty claim for modular recurrence. The measured question is whether this specific separation of relevance-independent resident computation from late causal selection helps under matched state, routing, data and publication budgets.
