# FusionMachine

> **Rich resident computation. Tiny causal event. Address in the route.**

FusionMachine is a research program about a machine suggested by the neuron-inspired work across SighImageSuper, AnttisNeuron, NewMachine, GAx and ThirdWay:

```text
resident computation
        ↓
nonlinear publication boundary
        ↓
one-bit event
        ↓
material routing graph
        ↓
local write into the next resident computation
```

The biological vocabulary is motivational. The experiments are ordinary digital constructions and are deliberately fenced against stronger biological claims.

**Live lab:** https://anttiluode.github.io/FusionMachine/

## Current spine

```text
v0  preserve computational identity
 ↓
v1  preserve continuing computational history
 ↓
v2  separate event payload from material routing identity
```

## v0 — same answer, different algorithm

Two computations are made behaviorally identical in an ordinary correlated world:

```text
A(x)=x0
B(x)=x1*x2
x0=x1*x2
```

After intervention, context must select A or B. Keeping `[A,B]` separate allows the exact nonlinear boundary

```text
y = 0.5*(A+B) + 0.5*c*(B-A).
```

Frozen complete-table result:

| representation / boundary | MSE | accuracy |
|---|---:|---:|
| **separated nonlinear** | **0.000** | **1.000** |
| best affine separated | 0.500 | 0.750 |
| collapsed `(A+B)/2` | 0.500 | 0.750 |

The collapse failure has an explicit information witness: two states share the same collapsed observation and context but require opposite answers.

See [`RESULTS_V0.md`](RESULTS_V0.md).

## v1 — resident algorithmic state

A route now carries temporal state

```text
h_t = alpha*h_(t-1) + (1-alpha)*u_t.
```

If a dormant computation stops running, exact resumption requires retained missed history or an equivalent sufficient statistic. For an 80-step dormant interval, bounded replay obeys an exact error law, and measured Monte Carlo results match it closely.

The shortest retained tail that reduces analytic replay RMSE to 10% of zero-history error is:

| alpha | K / 80 |
|---:|---:|
| .50 | 4 |
| .80 | 11 |
| .95 | 45 |
| .98 | 75 |

> **Persistence is also a replay horizon.**

Full replay is explicitly exact. Resident state is therefore a readiness strategy, not free computation.

See [`RESULTS_V1.md`](RESULTS_V1.md).

## v2 — dendrite → AIS → axon → dendrite

v2 adds the outward half of the machine.

A resident compartment owns rich local state. An AIS-like boundary decides whether that state becomes public. The event payload is only

```text
e_i(t) ∈ {0,1}
```

and the routing graph applies

```text
u_j(t+d_ij) += W_ji e_i(t).
```

The destination therefore need not be encoded in event amplitude. Source identity, branch connectivity, delay and synaptic weight are information stored in the machine itself.

### Gate 2A — address in matter

Eight sources emit the exact same bit `1`.

| routing condition | accuracy |
|---|---:|
| **intact source/axon identity** | **1.000** |
| pooled source identity | 0.125 |
| shuffled routes | 0.000 |
| explicit digital-address oracle | **1.000** |

The privileged oracle shows what pooling removed: destination identity can be restored by adding an explicit address to the pulse, while the intact machine gets the same information from its route graph.

### Gate 2B — development from chemistry + activity

The synthetic development control gives each target a coarse chemical family and a complementary activity-defined slot. Neither cue is sufficient alone.

Across 64 deterministic seeds:

| developmental cue | graph / routing accuracy |
|---|---:|
| **chemistry + activity** | **0.598958** |
| activity only | 0.263021 |
| chemistry only | 0.220052 |
| shuffled activity identity | 0.190104 |
| random wiring | 0.076823 |

This is deliberately a synthetic complementary-coordinate problem. It demonstrates a computational principle—an imperfect identity prior plus activity-dependent refinement can form a better transition graph—not a molecular development model.

### Gate 2C — publication is not resident state

During an AIS-suppression window, the resident trajectory is bit-for-bit identical to the unsuppressed control while **zero events** leave. When suppression is released, the then-current state immediately satisfies the event rule.

### Gate 2D — rerouting is causally distinct from computation

The source event trace

```text
0, 1, 0, 1, 1
```

is frozen. Changing only the route destination changes every downstream target from D1 to D2. Source computation and pulse train remain unchanged.

### Gate 2E — asynchronous chain

A four-hop route chain propagates at times

```text
0→1  arrive 1
1→2  arrive 3
2→3  arrive 4
3→4  arrive 7
```

There is no feed-forward layer clock. A target advances only when a delayed routed write arrives and crosses its local publication boundary.

See [`RESULTS_V2.md`](RESULTS_V2.md) and [`results/v2.json`](results/v2.json).

## What the machine means now

The computational primitive is no longer `weighted sum → activation`.

It is closer to:

```text
DENDRITIC / RESIDENT STATE
what computations are alive locally?
        ↓
SOMA / AIS BOUNDARY
should this become a causal event?
        ↓
AXONAL ROUTING FABRIC
which downstream computational territories receive it?
        ↓
SYNAPTIC WRITE
how strongly and where is the event written?
        ↓
TARGET RESIDENT STATE
what does that event mean inside the computation already running there?
```

The one-bit pulse can remain tiny because the route itself carries structured information.

## Biology fence

A 2025 review by Fréal & Hoogenraad, **“The dynamic axon initial segment: From neuronal polarity to network homeostasis”** (`10.1016/j.neuron.2025.01.004`), describes the AIS as a specialized compartment between somatodendritic and axonal domains that participates in action-potential generation/modulation and neuronal polarity, with molecular heterogeneity and activity-dependent plasticity. It also reviews axo-axonic innervation at the AIS.

That literature motivates separating resident computation, publication and routing. FusionMachine does **not** establish that:

- dendritic modes are arbitrary programs or eigenmodes in vivo;
- axons encode digital software addresses;
- the synthetic chemistry/activity score is a model of real guidance molecules;
- chandelier cells are software permission gates.

See [`docs/RELATED_WORK.md`](docs/RELATED_WORK.md).

## Next hard gate

The next experiment should stop adding metaphor and attack the architecture directly with an **equal-state-capacity generic recurrent / learned-routing system**. If generic state and generic learned routing match the same switching and routing tasks, then the named dendrite/AIS/axon decomposition may primarily be a useful coordinate system. If the separation survives under matched capacity, sparse communication, abrupt switching or delayed credit, then the architecture has a stronger computational case.

## Run

```bash
python -m pip install -e ".[test]"
pytest -q
python -m experiments.run_v0 --out /tmp/v0.json
python -m experiments.run_v1 --out /tmp/v1.json
python -m experiments.run_v2 --out /tmp/v2.json
```

## Repository map

- `src/fusion_machine/core.py` — v0 separated computations and nonlinear boundary
- `src/fusion_machine/resident.py` — v1 persistent state and bounded replay
- `src/fusion_machine/routing.py` — v2 binary publication and asynchronous material routing
- `src/fusion_machine/development.py` — v2 chemistry/activity route formation
- `experiments/run_v*.py` — deterministic scientific batteries
- `results/v*.json` — frozen receipts
- `RESULTS_V*.md` — measured interpretations and claim boundaries
- `PAPER.md` — evolving paper-style argument
- `docs/RELATED_WORK.md` — prior-art and biology fence
- `index.html`, `web/` — live static laboratory
