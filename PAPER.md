# FusionMachine: Separated Computational Modes, Resident Algorithmic State, and a Nonlinear Causal Boundary

## Abstract

FusionMachine studies a simple architectural question: what changes when a vector is allowed to contain the states of several distinct computations, rather than a single blended representation, and context-dependent nonlinear selection is delayed until an output boundary?

v0 gives an exact static construction. A direct computation `A=x0` and a relational computation `B=x1*x2` are behaviorally indistinguishable in a correlated world where `x0=x1*x2`. Once that relation is broken, a context bit must select A or B. Keeping `[A,B]` separate permits exact selection through one multiplicative interaction. The best affine readout of `[A,B,context]` has MSE `0.5` and 75% sign accuracy, and collapsing to `(A+B)/2` creates an explicit information-theoretic ambiguity with the same aggregate result.

v1 makes each route stateful. A dormant leaky computation can either be kept resident or suspended and reconstructed later from retained history. For an 80-step hidden interval, bounded replay has a closed-form switch-error variance. Across 4,096 deterministic random tapes and four persistence values, measured replay RMSE matches the analytic law to within 1.664% at the worst nonzero point. The number of recent steps needed to reduce reconstruction RMSE to 10% of the zero-history value rises from 4 at `alpha=0.50` to 75 at `alpha=0.98`. Thus the same persistence that preserves old information also lengthens the history needed to reconstruct a suspended computation.

These experiments do not show that biological dendrites contain arbitrary programs, nor that FusionMachine outperforms standard recurrent or mixture-of-experts systems. They establish a narrower progression: preserve computational identity before context arrives; when those computations have history, preserve their sufficient state or the history needed to recover it.

## 1. Motivation

A conventional neural representation is usually described as data: features, activations, embeddings, or hidden state. But a state variable in a dynamical machine can also be the partially executed state of a computation. If several such computations coexist, immediate averaging may destroy a distinction that becomes important only after a later context or intervention.

This motivates three separations:

```text
resident computation    what continues to exist internally
contextual fusion       which resident distinction matters now
causal publication      what becomes behavior / downstream influence
```

The motivating biological image is a branched cell whose local states coexist before nonlinear somatic/AIS integration, but the experiments here are ordinary digital constructions. The biological analogy supplies questions, not validation.

## 2. v0 — same answer, different algorithm

### 2.1 Two computations

Let

```text
x0, x1, x2 in {-1,+1}
```

and define

```text
A(x) = x0
B(x) = x1*x2.
```

They are counterfactually distinct: A depends on one direct cue, while B depends on a relation between two other cues.

### 2.2 Correlated world

Impose

```text
x0 = x1*x2.
```

Then `A(x)=B(x)` for every observed state. A behavior-only observer cannot infer which computation generated the answer.

### 2.3 Intervention world

Remove the constraint and allow all eight input triples. Add context

```text
c in {-1,+1}
```

with target

```text
c=-1 -> y=A
c=+1 -> y=B.
```

There are 16 complete intervention rows. The separated resident representation is

```text
z = [A,B].
```

### 2.4 Exact nonlinear boundary

The selector is

```text
y = 0.5*(A+B) + 0.5*c*(B-A).
```

The first term is the context-free average. The second is a multiplicative context-by-mode interaction. It switches the causal meaning of the same resident vector without erasing either coordinate.

## 3. v0 attackers

### 3.1 Best affine readout

Consider

```text
y_hat = w0 + wA*A + wB*B + wc*c.
```

On the complete symmetric truth table the interaction `c*(B-A)` lies outside the affine feature span `{1,A,B,c}`. Least squares returns

```text
y_linear = 0.5*A + 0.5*B
```

with context coefficient zero. The residual is

```text
r = 0.5*c*(B-A).
```

Since A and B disagree on half the rows, the exact mean squared error is `0.5`; with a fixed sign tie rule, accuracy is `0.75`.

### 3.2 Early collapse

Suppose the machine destroys computational identity and keeps only

```text
r = (A+B)/2.
```

When A and B disagree, `r=0` regardless of which computation carries +1 or -1. The frozen witness is

```text
A=-1, B=+1, c=-1 -> target=-1
A=+1, B=-1, c=-1 -> target=+1.
```

Both expose `(r,c)=(0,-1)`. No deterministic downstream function of the collapsed state can solve both cases. The complete-data optimum again has MSE `0.5` and sign accuracy `0.75`.

### 3.3 Factorized calibration

If the correct nonlinear basis is supplied,

```text
y = w_sum*(A+B) + w_gate*c*(B-A),
```

one agreement row and one disagreement row identify

```text
w_sum = 0.5
w_gate = 0.5,
```

which then solves all 16 rows exactly. This is a representation-reuse diagnostic, not a general few-shot-learning result.

### 3.4 v0 result

| model | MSE | sign accuracy |
|---|---:|---:|
| nonlinear separated boundary | **0.000000** | **1.000** |
| affine separated attacker | 0.500000 | 0.750 |
| collapsed full-data attacker | 0.500000 | 0.750 |
| two-row calibrated factorized boundary | **0.000000** | **1.000** |

v0 therefore earns:

> **Preserving computational identity until context arrives can be necessary; an earlier collapse can destroy that identity irreversibly.**

## 4. v1 — from algorithm output to algorithmic state

v0 still stores only scalar outputs of hand-defined computations. v1 asks what happens when each route has temporal state.

For route drive `u_t`, define

```text
h_t = alpha*h_(t-1) + (1-alpha)*u_t.
```

The two route drives retain the v0 semantics:

```text
uA_t = x0_t
uB_t = x1_t*x2_t.
```

Now the resident vector is

```text
z_t = [hA_t, hB_t]
```

and the same causal boundary can select either continuing state:

```text
y_t = 0.5*(hA_t+hB_t) + 0.5*c_t*(hB_t-hA_t).
```

The stronger claim is not that the coordinate *names* an algorithm, but that it contains a sufficient state from which that computation can continue correctly.

## 5. Resident versus lazy computation

Consider an 80-step interval during which context selects A while hidden B-drives continue to occur. Every policy knows the same B-state at the beginning of the gap.

### Resident dual

B is updated on every hidden step. At the switch to B, its state is already current.

### Lazy zero-history

B is suspended. The known boundary state is decayed correctly across the gap, but all missed B-drives are unknown and therefore omitted.

### Bounded replay K

The machine stores only the last `K` missed B-drives. At switch time it analytically decays the known boundary state across the full gap and adds the exact recurrence contributions of those retained drives.

### Full replay

`K=N=80` stores every missed drive and exactly reproduces resident B. This is a deliberately privileged attacker and prevents the experiment from pretending that continuous resident computation creates information from nothing.

## 6. Closed-form replay error

Let the hidden dormant drives be iid Rademacher variables. If only the last `K` of `N` missed drives are retained, the omitted contribution is

```text
e = (1-alpha) * sum_(r=K)^(N-1) alpha^r u_(N-r).
```

Because the drives have zero mean and unit variance,

```text
E[e] = 0
```

and

```text
Var[e]
 = (1-alpha)^2 * sum_(r=K)^(N-1) alpha^(2r)
 = (1-alpha)^2 * alpha^(2K)
   * (1-alpha^(2(N-K))) / (1-alpha^2).
```

The predicted switch RMSE is the square root of this expression. For a gap long compared with the mode lifetime, the dominant dependence is approximately

```text
RMSE(K) proportional to alpha^K.
```

This yields a direct conceptual bridge to modal forgetting: the mode's own persistence determines how far into the missed past one must reach to reconstruct it.

## 7. v1 experiment

We sample 4,096 deterministic independent hidden tapes of length 80 and sweep

```text
alpha = 0.50, 0.80, 0.95, 0.98
K     = 0, 4, 8, 16, 32, 64, 80.
```

The Monte Carlo estimator computes the exact omitted-prefix contribution directly, avoiding catastrophic cancellation once replay error becomes tiny.

Across every nonzero sweep point, the largest relative difference between measured and analytic RMSE is **1.664%**, at `alpha=0.98, K=4`.

### 7.1 Primary alpha=0.95 curve

| K | measured RMSE | analytic RMSE |
|---:|---:|---:|
| 0 | 0.160939 | 0.160106 |
| 4 | 0.131933 | 0.130399 |
| 8 | 0.106385 | 0.106199 |
| 16 | 0.070331 | 0.070427 |
| 32 | 0.030810 | 0.030906 |
| 64 | 0.005413 | 0.005395 |
| 80 | **0.000000** | **0.000000** |

### 7.2 Persistence sets the reconstruction horizon

Define the replay horizon as the smallest `K` whose analytic RMSE is at most 10% of the zero-history RMSE.

| alpha | K required out of 80 |
|---:|---:|
| 0.50 | **4** |
| 0.80 | **11** |
| 0.95 | **45** |
| 0.98 | **75** |

This gives the compact result:

> **Persistence is also a replay horizon.**

A rapidly forgetting computation can be recovered from a short recent tail because older events no longer matter. A slowly forgetting computation preserves older distinctions, and therefore suspending it makes those same old events necessary for accurate reconstruction.

## 8. Computation has not disappeared — it moved in time

The resident and full-replay policies perform the same recurrence work at different times.

For the 80-step dormant interval:

| policy | dormant updates during gap | hidden drives stored | switch replay updates | current state scalars |
|---|---:|---:|---:|---:|
| resident dual | **80** | **0** | **0** | 1 |
| lazy full replay | **0** | **80** | **80** | 1 |

Thus v1 is not an efficiency theorem. It exposes a scheduling/storage/readiness tradeoff:

```text
resident state
    continuous local work
    compact current sufficient state
    zero switch replay latency

lazy replay
    deferred local work
    retained missed history
    switch-time reconstruction burst
```

A bounded-history lazy machine interpolates between those extremes and accepts the error given by the analytic law.

## 9. Interpretation

v0 and v1 together suggest a more precise meaning of “algorithmic mode.”

A mode has at least two parts:

```text
identity     what counterfactual computation this route represents
state        where that continuing computation currently is
```

A system that preserves identity but lets the dormant state stop evolving has not actually preserved the full computation. It has preserved only a label or checkpoint. To resume exactly, it needs the missing inputs or an equivalent sufficient statistic.

This is why the data/program distinction becomes blurry in the emerging machine. A coordinate can simultaneously be:

- a representation of past input;
- a sufficient statistic for future prediction;
- memory;
- the current state of a procedure.

## 10. Relation to established ideas

The ingredients are not new in isolation. v0 lives near multiplicative gating and conditional computation. v1 lives near recursive filtering, sufficient statistics, caching/materialized state, lazy evaluation, replay, and state-space models. Mixture-of-experts architectures provide an obvious strong alternative when computations are explicitly separated. Recurrent networks provide an obvious attacker when one generic state might encode all needed histories without named modes.

FusionMachine's scientific question is therefore narrower than “can neural networks route experts?” or “can recurrent state remember history?” It is:

> **When does preserving several counterfactually distinct computations as continuing resident state provide a useful coordinate system or computational advantage over immediately blended generic state?**

## 11. Biological interpretation boundary

A branched dendritic arbor motivates the picture of locally persistent computational states meeting a nonlinear soma/AIS boundary. Real dendritic branches are known to exhibit nonlinear integration, and neuronal structure can create different temporal filters and local states. None of the v0-v1 results establish that biological branches implement the abstract A/B modes, that the AIS acts as the exact context gate here, or that spikes are merely publication packets.

A biological mapping becomes meaningful only if later gates replace abstract state coordinates with a physically constrained branched substrate and the same computational distinctions survive matched controls.

## 12. Limitations

The current experiments remain deliberately small:

- A and B are hand-defined input computations;
- v1 gives both routes the same first-order recurrence family;
- context is supplied explicitly;
- hidden v1 drives are iid and exactly available when retained;
- full replay is allowed unlimited switch latency;
- there is no task-trained formation of routes;
- there is no equal-capacity generic recurrent attacker yet;
- there is no hardware energy or communication model;
- there is no biological validation.

The point is to make the object fail cleanly before scaling it.

## 13. Next experiment — genuinely different temporal algorithms

The critical v2 test should no longer let both resident routes be copies of the same exponential filter.

A useful pair would be computationally incompatible temporal mechanisms, for example:

```text
route A: continuous leaky/integrating predictor
route B: finite-state relational/parity/event procedure
```

Both must continue through periods when they are behaviorally irrelevant. Context switches should reveal whether the dormant procedure remains at the correct internal state.

The strongest attacker should receive the same total number of state scalars but use them as an unconstrained generic recurrent state. If that model matches or beats separated resident modes, then FusionMachine's named-mode picture may be explanatory rather than computationally privileged. That is an important possible negative result.

## 14. Reproducibility

```bash
python -m pip install -e ".[test]"
pytest -q
python -m experiments.run_v0 --out /tmp/v0.json
python -m experiments.run_v1 --out /tmp/v1.json
```

Canonical receipts live in `results/v0.json` and `results/v1.json` and are regenerated in CI.
