# FusionMachine: Resident Computational Modes, Causal Publication, and Material Routing

## Abstract

FusionMachine asks what changes when computation, publication, and routing are treated as different variables rather than collapsed into one activation. v0 shows that two counterfactually distinct computations can be behaviorally indistinguishable in an ordinary world yet require separate resident identities once context changes. v1 makes those computations stateful and shows that persistence sets the history required to reconstruct a suspended mode. v2 adds a low-dimensional causal boundary and a material routing graph: the event payload is one bit, while source/axon identity and route connectivity determine which downstream resident computation is changed next.

The v2 synthetic controls give perfect routing with intact material identity, 1/8 accuracy after pooling source identity, zero accuracy under a cyclic route shuffle, and perfect performance when a privileged digital address is appended to the pulse. A separate developmental control combines an imperfect coarse chemical cue with complementary activity alignment; across 64 deterministic seeds the combined graph reaches 0.598958 routing accuracy versus 0.220052 for chemistry alone, 0.263021 for activity alone, and 0.076823 for random wiring. AIS-like suppression blocks all outward events while leaving the resident trajectory unchanged, and a pure route intervention changes downstream targets while the source state and bit train remain fixed.

These experiments do not establish a biological theory of dendrites, axons, or the AIS. They establish a computational decomposition that can now be attacked by equal-capacity recurrent and learned-routing alternatives.

## 1. Motivation

A conventional point-neuron abstraction tends to blur several jobs together:

```text
state / computation
integration
output nonlinearity
communication
routing
```

FusionMachine separates them:

```text
resident computation
        ↓
nonlinear causal publication
        ↓
low-dimensional event
        ↓
material routing graph
        ↓
local write into another resident computation
```

The neuron analogy supplies a useful physical picture: dendritic state can be richer than the spike, the AIS can act as a publication boundary, and the axonal/synaptic structure can carry route identity. The experiments are digital and synthetic; biology motivates the questions rather than validating the answers.

## 2. v0 — computational identity before context

Let

```text
A(x)=x0
B(x)=x1*x2
```

and first restrict the observed world so that

```text
x0=x1*x2.
```

A and B then agree on every observed state even though they are counterfactually different computations. After breaking that correlation, context `c∈{-1,+1}` selects which computation should become behavior.

Keeping the resident representation

```text
z=[A,B]
```

permits the exact selector

```text
y = 0.5*(A+B) + 0.5*c*(B-A).
```

The complete 16-row intervention table gives:

| model | MSE | accuracy |
|---|---:|---:|
| nonlinear separated boundary | **0.000000** | **1.000** |
| best affine separated readout | 0.500000 | 0.750 |
| full-data decoder after collapse | 0.500000 | 0.750 |

The collapse failure has an explicit information witness: two states share the same collapsed `(A+B)/2` and the same context while requiring opposite outputs.

v0 therefore earns only the narrow statement:

> **Preserving computational identity until context arrives can matter; early blending can destroy it irreversibly.**

## 3. v1 — computational history must also survive

v1 gives a route continuing state

```text
h_t = alpha*h_(t-1) + (1-alpha)*u_t.
```

A dormant route can either remain resident or stop updating and later reconstruct itself from retained missed drives. If only the last `K` of `N` missed iid ±1 drives are retained, the omitted contribution has

```text
Var[e]
 = (1-alpha)^2 * alpha^(2K)
   * (1-alpha^(2(N-K))) / (1-alpha^2).
```

Across 4,096 deterministic tapes and `alpha∈{.50,.80,.95,.98}`, the simulated replay RMSE follows the analytic law. For an 80-step gap, the smallest `K` giving at most 10% of zero-history error is:

| alpha | K / 80 |
|---:|---:|
| .50 | 4 |
| .80 | 11 |
| .95 | 45 |
| .98 | 75 |

Hence:

> **Persistence is also a replay horizon.**

Full replay is exact. v1 is therefore a readiness/storage/scheduling result, not an efficiency theorem.

## 4. v2 — event payload and route identity are different information

v2 adds the outward half of the machine.

A resident compartment `j` has local dynamics

```text
h_j(t+1)=F_j(h_j(t),u_j(t)).
```

An AIS-like boundary emits

```text
e_i(t)=1[G_i(h_i(t),context_i(t))>theta_i]
```

with

```text
e_i(t)∈{0,1}.
```

The routing material applies

```text
u_j(t+d_ij) += W_ji e_i(t).
```

The event therefore does not need to carry a destination label in its amplitude. `W`, `d`, source identity, branch identity and local target address are state held in the machine itself.

## 5. Gate 2A — address in matter

Eight source identities emit exactly the same payload `1`. Their intended destinations differ only by route identity.

| condition | accuracy |
|---|---:|
| intact route identity | **1.000** |
| pooled source identity | 0.125 |
| cyclic route shuffle | 0.000 |
| explicit digital-address oracle | **1.000** |

The pooled control exposes the information boundary cleanly. Once source identity is erased, every event is literally the same one-bit observation; a deterministic decoder has only the balanced 1/8 target prior. The explicit-address oracle restores the missing information in the payload.

This gate is not a claim that wires as addresses are novel. It establishes the bookkeeping required by the architecture:

> **Event content and route identity are separable information channels.**

## 6. Gate 2B — synthetic developmental route formation

The route graph is not simply handed a perfect lookup. A developmental control gives 12 targets two complementary coordinates:

```text
3 coarse chemical families
×
4 activity-defined slots
```

Chemistry is ambiguous within each family. Activity is ambiguous across families. Independent noise perturbs both source and target signatures. A route score combines row-normalized cue matrices:

```text
score_ij = z(chemistry_ij) + beta*z(activity_ij)
```

with `beta=1` frozen before evaluation.

Across 64 deterministic seeds:

| developmental policy | graph / routing accuracy |
|---|---:|
| **chemistry + activity** | **0.598958** |
| activity only | 0.263021 |
| chemistry only | 0.220052 |
| shuffled activity identities | 0.190104 |
| random wiring | 0.076823 |

The combination gains 37.89 percentage points over chemistry-only and 33.59 points over activity-only.

This is deliberately a synthetic complementarity test. It should be interpreted only as:

> **A coarse identity prior and an experienced co-activation cue can jointly form a better route graph than either cue alone.**

It is not a model of real guidance molecules, growth cones, or synaptogenesis.

## 7. Gate 2C — publication permission is not resident state

Two identical resident compartments receive the same six drives. One publishes normally. The other has publication suppressed on steps 1–4.

Both resident trajectories are exactly

```text
0.500000
0.750000
0.875000
0.937500
0.968750
0.984375
```

so the maximum state difference is `0.0`. The suppressed window emits zero events, and the first post-suppression event is immediately `1`.

Thus the implementation distinguishes:

```text
what the computation currently is
```

from

```text
whether the computation is permitted to become traffic.
```

## 8. Gate 2D — route intervention is causally downstream of computation

Freeze the source pulse train:

```text
0,1,0,1,1.
```

Under one material route every emitted bit reaches target D1. Change only one route destination and the bit-identical source trace reaches D2 instead.

The source computation has not changed. The event payload has not changed. Only the graph changed.

This provides a direct causal decomposition:

> **Downstream behavior can be altered by changing the route while leaving resident computation and event generation fixed.**

## 9. Gate 2E — asynchronous causal chain

A five-compartment chain uses nonuniform material delays:

| source | target | emit | arrive |
|---:|---:|---:|---:|
| 0 | 1 | 0 | 1 |
| 1 | 2 | 1 | 3 |
| 2 | 3 | 3 | 4 |
| 3 | 4 | 4 | 7 |

There is no global layer counter. Targets update only when scheduled writes arrive; local threshold crossing then creates a new event.

The primitive is therefore closer to an asynchronous state machine than to a synchronous stack of layers.

## 10. Combined interpretation

The v0→v2 progression can now be stated compactly:

```text
v0: identity
    multiple computations can coexist without being blended

v1: history
    each computation may need continuing resident state

v2: causality
    publication and routing need not be identical to that state
```

The emerging computational unit is:

```text
DENDRITIC / RESIDENT STATE
rich local computation
        ↓
SOMA / AIS-LIKE BOUNDARY
commit / suppress
        ↓
EVENT
small public causal token
        ↓
AXONAL / SYNAPTIC MATERIAL
route, delay, gain, target address
        ↓
TARGET RESIDENT STATE
local interpretation of the write
```

This also explains how a binary pulse can participate in a rich computation: the bit is only one factor. The route graph and the receiving dynamics contain additional structured state.

## 11. Biological neighborhood and fence

Fréal & Hoogenraad (Neuron, 2025), **“The dynamic axon initial segment: From neuronal polarity to network homeostasis”**, DOI `10.1016/j.neuron.2025.01.004`, review the AIS as a specialized compartment between somatodendritic and axonal domains involved in action-potential generation/modulation and neuronal polarity. They emphasize molecular heterogeneity, activity-dependent AIS remodeling, and axo-axonic innervation at the AIS.

Those observations motivate separating resident somatodendritic computation from an output boundary. Historical chemoaffinity and modern axon-guidance/synapse-specificity work motivate asking how route identity can be embodied by developmental structure rather than repeated in every event.

FusionMachine does not establish that:

- dendritic branches are literal algorithm eigenmodes;
- an AP is semantically a software commit bit;
- axons encode explicit digital destination fields;
- chandelier cells are permission gates in the software sense;
- the synthetic chemistry/activity rule reproduces neural development.

Biological mapping remains a hypothesis generator, not the evidence for the computational claims.

## 12. Prior-art boundary

Every ingredient has mature neighbors: dendritic nonlinear subunits, conditional computation, multiplicative gating, recursive state, event-triggered communication, graph routing, neural development, chemoaffinity, activity-dependent refinement and asynchronous/event-driven systems.

A useful FusionMachine claim must therefore come from the **composition and causal separation** of those roles, not from relabeling established mechanisms.

## 13. Strong next attacker

The next gate should stop adding biological detail and compare against an equal-budget generic system:

```text
generic recurrent state
+
learned routing matrix
```

with the same number of state scalars, route parameters, training episodes and event bandwidth.

If the generic system matches FusionMachine on switching, routing and delayed-credit tasks, the dendrite/AIS/axon decomposition may be mainly an explanatory coordinate system. If the separated architecture wins under matched capacity—especially with sparse communication, abrupt context switches or delayed causal credit—then it has a stronger computational case.

## 14. Reproducibility

```bash
python -m pip install -e ".[test]"
pytest -q
python -m experiments.run_v0 --out /tmp/v0.json
python -m experiments.run_v1 --out /tmp/v1.json
python -m experiments.run_v2 --out /tmp/v2.json
```

Canonical receipts live in `results/v0.json`, `results/v1.json`, and `results/v2.json` and are regenerated in CI.
