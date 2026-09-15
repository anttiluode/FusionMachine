# FusionMachine: Resident World Dynamics, Late Relevance, Causal Publication, and Material Routing

## Abstract

FusionMachine asks what changes when resident computation, current relevance, causal publication, and routing are treated as different variables rather than collapsed into one recurrent activation.

v0 shows that two counterfactually distinct computations can be behaviorally indistinguishable in an ordinary correlated world yet require separate identities after intervention. v1 makes those computations stateful and shows that persistence sets the amount of missed history required to reconstruct a suspended computation. v2 adds a one-bit publication boundary and a material routing graph: event amplitude can remain tiny because source/route identity carries destination information.

v3 attacks the architecture with a four-state fully dense tanh RNN. Both systems receive the same four state scalars, 28 learned routing parameters, 96 training episodes, 1,200 optimizer updates and 16 sparse publication opportunities; the generic RNN has 46 non-routing parameters versus 30 for FusionMachine. In a frozen nonlinear two-process world, eight paired seeds give median first-post-switch MSE `1.5198e-5` for FusionMachine versus `1.6539e-2` for the generic RNN, and immediate post-48-step-dormancy error `0.002981` versus `0.101834`. However, FusionMachine also attains roughly 32-fold lower training-query MSE, so the primary result is explicitly classified as confounded by optimization/training fit.

The destructive controls sharpen the result. Feeding current context into FusionMachine's resident transition increases switch error about tenfold. Removing context from the generic RNN recurrence improves its switch error about seventyfold and raises its held-out teacher-state linear-probe R² from `0.8887` to `0.9974`. Making FusionMachine recurrence dense while continuing to withhold current context preserves most of the readiness effect. The strongest v3 clue is therefore not simply modular recurrence; it is **late relevance**: state that tracks how the world changes can be more robust to abrupt relevance changes when the current task does not continuously rewrite that state.

The biological vocabulary remains motivational. The experiments establish synthetic causal decompositions and benchmarks, not a biological theory of dendrites, inhibitory interneurons, consciousness, or development.

## 1. The decomposition

A conventional recurrent unit often entangles:

```text
what is true / changing in the represented world
what currently matters
whether to emit an external effect
where that effect goes
```

FusionMachine separates these roles:

```text
WORLD INPUT
   ↓
RESIDENT DYNAMICS
   ↓
CURRENT RELEVANCE / NONLINEAR SELECTION
   ↓
PUBLICATION BOUNDARY
   ↓
LOW-DIMENSIONAL EVENT
   ↓
ROUTING FABRIC
   ↓
TARGET RESIDENT DYNAMICS
```

The guiding hypothesis is not that all context is harmful. It is narrower: a variable representing *current behavioral relevance* need not be allowed to rewrite every state variable used to predict the world.

## 2. v0 — preserve computational identity before selection

Let

```text
A(x)=x0
B(x)=x1*x2
```

and first impose the ordinary-world correlation

```text
x0=x1*x2.
```

A and B then agree behaviorally despite being counterfactually different. After breaking the correlation, context `c∈{-1,+1}` selects which answer is required.

Keeping

```text
z=[A,B]
```

permits the exact selector

```text
y = 0.5*(A+B) + 0.5*c*(B-A).
```

On the complete 16-row intervention table:

| model | MSE | accuracy |
|---|---:|---:|
| nonlinear separated boundary | **0.000000** | **1.000** |
| affine readout of separated state | 0.500000 | 0.750 |
| optimal decoder after early collapse | 0.500000 | 0.750 |

The collapse failure contains an explicit information witness: two intervention states expose the same collapsed observation and context while requiring opposite outputs.

v0 earns:

> **Do not destroy computational identity before a later variable can make that identity relevant.**

## 3. v1 — identity is insufficient when computation has history

v1 gives each route a continuing sufficient state

```text
h_t = alpha*h_(t-1) + (1-alpha)*u_t.
```

A behaviorally dormant route either keeps updating or is reconstructed later from retained missed inputs. If only the last `K` of `N` iid ±1 drives are retained, the omitted-history variance is

```text
Var[e]
 = (1-alpha)^2 * alpha^(2K)
   * (1-alpha^(2(N-K))) / (1-alpha^2).
```

For an 80-step gap, the shortest tail giving at most 10% of zero-history RMSE is 4 steps at `alpha=.50`, 11 at `.80`, 45 at `.95`, and 75 at `.98`.

Hence:

> **Persistence is also a replay horizon.**

Full replay remains exact. Resident computation buys readiness by paying continuous local work rather than storing and replaying missed history.

## 4. v2 — state, publication, and route are distinct variables

A local resident state follows

```text
h_j(t+1)=F_j(h_j(t),u_j(t)).
```

An AIS-like boundary emits

```text
e_i(t)∈{0,1}
```

and routing material applies

```text
u_j(t+d_ij) += W_ji e_i(t).
```

Eight sources emitting the identical bit route perfectly when source/material identity is intact, fall to `1/8` accuracy when identity is pooled, fail completely under a cyclic route shuffle, and return to perfect routing when a privileged digital address is appended to the event.

A separate synthetic development control combines a coarse chemical family cue with an activity-defined slot cue. Across 64 deterministic seeds, the combined route graph reaches `0.598958` accuracy versus `0.263021` activity only, `0.220052` chemistry only and `0.076823` random.

AIS-like suppression produces zero outward events while leaving the resident state trajectory exactly unchanged. A pure route intervention changes downstream target while leaving source state and source bit train fixed.

v2 earns:

> **Resident computation is not traffic, and event content is not route identity.**

## 5. v3 — equal-state recurrent attacker

### 5.1 Frozen resource contract

The v3 world contains two distinct nonlinear two-state teacher processes. Both evolve continuously from the same raw input stream. Context only determines which teacher output is currently queried; it does not alter the teacher dynamics.

The primary competitors are:

```text
FusionMachine
  4 state scalars split into two 2-state mechanisms
  block-diagonal recurrent transition
  current context excluded from resident transition
  two local output heads; context selects later

Generic attacker
  4 state scalars
  fully dense recurrent transition
  current context is available inside recurrence
  two dense output heads; context selects output
```

Both get the same learned 28-parameter routing head, training data, update count, optimizer and sparse publication schedule. The generic RNN receives **more**, not fewer, non-routing trainable parameters: `46` versus `30`.

Training consists of 96 deterministic 64-step episodes. Context remains on one teacher for 16–32 steps and supervision appears only at block ends. Test includes ordinary held-out blocks, rapid 2–6-step switching, a 48-step dormant interval followed by a switch, common sparse publication, and a held-out linear probe for the teacher's four-dimensional hidden state.

### 5.2 Primary eight-seed medians

| metric | FusionMachine | generic RNN |
|---|---:|---:|
| training-query MSE | **0.000011991** | 0.000383249 |
| ID query MSE | **0.000016380** | 0.001052879 |
| rapid first-post-switch MSE | **0.000015198** | 0.016539331 |
| immediate error after 48-step dormancy | **0.002981** | 0.101834 |
| sparse receiver MSE | **0.192786** | 0.201617 |
| route accuracy | 0.608398 | 0.608398 |
| latent teacher-state R² | **0.999822** | 0.888666 |

The raw readiness differences are large: roughly `1,088×` in first-switch MSE and `34×` in immediate post-dormancy absolute error.

But FusionMachine also has about `32×` lower training-query MSE. Therefore the frozen primary classification is:

```text
fusion_readiness_but_training_advantage_confounded
```

This is an important constraint on interpretation. v3 does not yet establish that the architectural bias wins when optimization quality is matched.

## 6. v3 destructive controls — where relevance enters matters

Three predeclared controls isolate the architecture:

| variant | rapid first-switch MSE | latent R² |
|---|---:|---:|
| Fusion, relevance excluded | **0.000015198** | **0.999822** |
| Fusion + context leak | 0.000155411 | 0.999443 |
| Fusion + dense recurrence, no context | 0.000057449 | 0.999247 |
| generic dense + context | 0.016539331 | 0.888666 |
| generic dense without context | **0.000235806** | **0.997412** |

These controls weaken the simple hypothesis that block-diagonal recurrence is the key ingredient.

First, adding context to FusionMachine's resident transition worsens switch MSE by about `10.2×` even though ordinary fit stays strong.

Second, removing context from the generic dense recurrence improves switch MSE by about `70×` and causes its latent-state probe to recover almost the complete teacher state.

Third, making FusionMachine recurrence dense while keeping relevance out raises its switch MSE only about `3.8×` relative to baseline and preserves high latent R².

The compact hypothesis suggested by this gate is:

> **State is not relevance.**

A representation whose job is to remain ready for future questions can be damaged if every change in the current question is also allowed to change the representation's dynamics.

## 7. OOD switch amplification

A useful diagnostic is the ratio between first-switch error and ordinary held-out query error.

For FusionMachine:

```text
1.5198e-5 / 1.6380e-5 ≈ 0.93
```

For the context-conditioned generic RNN:

```text
0.016539 / 0.0010529 ≈ 15.7
```

Thus the generic baseline is not simply uniformly worse; it experiences a specifically large penalty when relevance changes faster than in training. The relevance-insulated controls largely remove this switch-specific amplification.

## 8. Hidden world-state probe

Neither model receives hidden teacher-state supervision. After behavioral training, a linear map is fit from learned recurrent state to the full teacher state using separate probe data.

```text
FusionMachine                 R² = 0.999822
generic + current context     R² = 0.888666
generic without context       R² = 0.997412
```

This diagnostic is consistent with the idea that a relevance-conditioned recurrence can become a task-conditioned state, whereas relevance-insulated recurrence remains closer to a general world state. A probe is only a diagnostic; it does not prove a unique representational interpretation.

## 9. Sparse publication and routing

Both primary competitors receive exactly 16 publication opportunities in the 64-step rapid-switch test. Receiver MSE is `0.192786` versus `0.201617`, a comparatively small difference. Route accuracy is exactly tied at `0.608398` because both systems receive the same routing features and route-parameter budget.

v3 therefore does not support a dramatic sparse-communication claim. The major effect is readiness under abrupt relevance change.

## 10. Current computational interpretation

The v0→v3 progression now reads:

```text
v0  preserve identities that later context may distinguish
v1  keep stateful computations current or retain enough history to replay them
v2  separate resident state from publication and route
v3  keep current relevance from needlessly rewriting persistent world dynamics
```

The emerging primitive is:

```text
input
  ↓
resident transformations of the world
  ↓
late relevance / nonlinear selection
  ↓
causal publication
  ↓
structured routing
  ↓
next resident transformations
```

This resembles a continuously maintained computational world rather than a sequence of representations recomputed solely for the current task.

## 11. Biological neighborhood and fence

Real cortical interneuron classes target different subcellular domains of pyramidal neurons, dendrites can integrate nonlinearly, the AIS is a specialized output-initiation compartment, and axonal/synaptic structures impose highly structured connectivity. These observations motivate the decomposition.

They do **not** establish that:

- biological dendritic branches are arbitrary program slots;
- spikes are literal software commit bits;
- basket or chandelier cells implement the v3 relevance variable;
- axons contain digital destination labels;
- the synthetic development rule reproduces chemoaffinity or synaptogenesis;
- the v3 teacher world is a model of consciousness or natural cognition.

The biology should be used to generate sharper computational interventions, not as evidence for conclusions that the synthetic experiments did not test.

## 12. Prior-art boundary

The ingredients live near mature literatures: nonlinear dendritic subunits, conditional computation, multiplicative gating, recursive state estimation, state-space models, modular recurrent networks, Clockwork RNNs, Recurrent Independent Mechanisms, predictive/world-state representations, event-triggered communication, graph routing, inhibitory microcircuits and activity-dependent development.

The possible distinctive claim is therefore not “modularity,” “attention,” “recurrent state,” or “sparse spikes.” It is the experimentally testable composition:

> **Maintain multiple transformations as resident state; keep fast-changing relevance partly outside those world dynamics; let relevance select what becomes causal; let routing determine where that causal event is written next.**

## 13. Next falsification

v3 leaves a serious training-fit confound. The next gate should close it before adding more biological detail.

Useful attacks include:

- loss-matched checkpoints so both primary models have comparable training error;
- stronger gated recurrent attackers such as GRU/LSTM under explicit resource accounting;
- more random seeds and statistical uncertainty;
- tasks in which context genuinely changes hidden-world dynamics, testing when relevance *should* enter state;
- longer-horizon latent-process mixtures where keeping every process resident becomes expensive;
- learned formation, pruning and reawakening of resident mechanisms.

If late relevance survives these attacks, the computational claim becomes substantially stronger.

## 14. Reproducibility

```bash
python -m pip install -e ".[test]"
pytest -q
python -m experiments.run_v0 --out /tmp/v0.json
python -m experiments.run_v1 --out /tmp/v1.json
python -m experiments.run_v2 --out /tmp/v2.json
python -m experiments.run_v3 --out /tmp/v3-full.json
```

`results/v3.json` is the compact frozen view of the full deterministic CI-generated v3 receipt; earlier receipts remain in `results/v0.json` through `results/v2.json`.
