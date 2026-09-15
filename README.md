# FusionMachine

> **Keep the world state resident. Apply relevance later. Publish sparsely. Route through matter.**

FusionMachine is a research program about separating four jobs that ordinary recurrent/neural abstractions often blur together:

```text
resident world computation
        ↓
current relevance / nonlinear selection
        ↓
small causal event
        ↓
learned/material route
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
v2  separate event payload from routing identity
 ↓
v3  attack the architecture with an equal-state generic recurrent model
```

## v0 — same answer, different algorithm

Two computations are behaviorally identical in a correlated world:

```text
A(x)=x0
B(x)=x1*x2
x0=x1*x2
```

After intervention, context must select A or B. Keeping `[A,B]` separate allows

```text
y = 0.5*(A+B) + 0.5*c*(B-A).
```

| representation / boundary | MSE | accuracy |
|---|---:|---:|
| **separated nonlinear** | **0.000** | **1.000** |
| best affine separated | 0.500 | 0.750 |
| collapsed `(A+B)/2` | 0.500 | 0.750 |

The collapse failure has an explicit information witness: two states share the same collapsed observation and context but require opposite answers. See [`RESULTS_V0.md`](RESULTS_V0.md).

## v1 — resident algorithmic state

A route now carries temporal state

```text
h_t = alpha*h_(t-1) + (1-alpha)*u_t.
```

If a dormant computation stops running, exact resumption requires retained missed history or an equivalent sufficient statistic. For an 80-step gap, the shortest retained tail reducing analytic replay RMSE to 10% of zero-history error is:

| alpha | K / 80 |
|---:|---:|
| .50 | 4 |
| .80 | 11 |
| .95 | 45 |
| .98 | 75 |

> **Persistence is also a replay horizon.**

Full replay is explicitly exact. Resident state is a readiness strategy, not free computation. See [`RESULTS_V1.md`](RESULTS_V1.md).

## v2 — dendrite → AIS → axon → dendrite

v2 adds the outward half of the machine. A resident compartment owns rich local state. An AIS-like boundary decides whether it becomes public. The event payload is only

```text
e_i(t) ∈ {0,1}
```

and the routing graph applies

```text
u_j(t+d_ij) += W_ji e_i(t).
```

Eight sources can emit the same bit while route identity supplies the destination:

| routing condition | accuracy |
|---|---:|
| **intact source/axon identity** | **1.000** |
| pooled source identity | 0.125 |
| shuffled routes | 0.000 |
| explicit digital-address oracle | **1.000** |

A separate synthetic development gate combines complementary chemistry/activity cues: combined route recovery is **0.598958** versus `.263021` activity, `.220052` chemistry, and `.076823` random. AIS suppression leaves resident state unchanged while blocking events, and changing only the route changes downstream behavior without changing the source computation or bit train.

See [`RESULTS_V2.md`](RESULTS_V2.md).

## v3 — FAIR FIGHT: four states versus four states

v3 stops adding biology and attacks the computational decomposition.

Two nonlinear hidden teacher processes evolve continuously. Training asks about only one process at a time in long context blocks. At test time context switches rapidly or returns to a computation after a 48-step dormant interval.

The comparison is deliberately generous to the attacker:

| resource | FusionMachine | dense generic RNN |
|---|---:|---:|
| recurrent state scalars | 4 | 4 |
| learned route parameters | 28 | 28 |
| training episodes | 96 | 96 |
| optimizer updates | 1,200 | 1,200 |
| publication opportunities | 16 | 16 |
| non-routing parameters | 30 | **46** |

The generic model therefore has more internal trainable freedom, not less.

### Frozen eight-seed medians

| metric | FusionMachine | generic RNN |
|---|---:|---:|
| training-query MSE | **0.000011991** | 0.000383249 |
| held-out query MSE | **0.000016380** | 0.001052879 |
| first query after rapid switch MSE | **0.000015198** | 0.016539331 |
| immediate error after 48-step dormancy | **0.002981** | 0.101834 |
| sparse receiver MSE | **0.192786** | 0.201617 |
| route accuracy | 0.608398 | 0.608398 |
| hidden-world linear-probe R² | **0.999822** | 0.888666 |

The switch error differs by about **1,088×** and immediate post-dormancy error by about **34×**. But FusionMachine also fits the sparse training queries about **32×** better, so the frozen classification is intentionally conservative:

> **Fusion readiness advantage observed, but training advantage confounds the architectural claim.**

### The destructive controls are more informative

| variant | first-post-switch MSE | latent R² |
|---|---:|---:|
| **Fusion, relevance excluded from resident transition** | **0.000015198** | **0.999822** |
| Fusion + relevance/context leak | 0.000155411 | 0.999443 |
| Fusion + dense recurrence, no relevance | 0.000057449 | 0.999247 |
| generic dense + relevance/context | 0.016539331 | 0.888666 |
| generic dense **without relevance/context** | **0.000235806** | **0.997412** |

This gate therefore does **not** reduce to “block-diagonal recurrence beats dense recurrence.” The stronger clue is:

> **State is not relevance.**

The world does not stop evolving when behavior stops looking at one of its processes. A useful world state may therefore need to remain partly insulated from what matters *right now*, with relevance applied later at selection/publication.

See [`RESULTS_V3.md`](RESULTS_V3.md) and [`results/v3.json`](results/v3.json).

## What the machine means now

```text
WORLD / SENSORY INPUT
        ↓
RESIDENT DYNAMICS
keep multiple transformations current
        ↓
CURRENT RELEVANCE
select / fuse without rewriting the world merely because attention moved
        ↓
PUBLICATION BOUNDARY
small causal event
        ↓
ROUTING FABRIC
which downstream computational territory receives it?
        ↓
TARGET RESIDENT STATE
continue the next computation
```

That is the current synthetic object. It is closer to an asynchronous world-state machine with late relevance than to a point neuron.

## Biology fence

The neuron literature motivates several separations used here: dendritic nonlinear integration, compartment-specific inhibition, a specialized axon initial segment, sparse spikes, and structured axonal/synaptic connectivity. FusionMachine does **not** establish that:

- dendritic modes are arbitrary programs or eigenmodes in vivo;
- spikes are semantically software commit bits;
- axons encode digital software addresses;
- chandelier or basket cells implement the exact gates used here;
- the synthetic chemistry/activity score is a model of real neural development;
- v3 is evidence for consciousness or a complete biological world model.

See [`docs/RELATED_WORK.md`](docs/RELATED_WORK.md).

## Next hard gate

v3 leaves a real confound: the generic baseline trained worse. The next benchmark should close it rather than celebrate the gap. Candidate attacks are optimization/loss-matched checkpoints, a stronger gated recurrent baseline, and a world in which relevance sometimes changes the hidden dynamics and therefore legitimately belongs in state.

If the late-relevance advantage survives those attacks, it becomes a considerably stronger architectural claim.

## Run

```bash
python -m pip install -e ".[test]"
pytest -q
python -m experiments.run_v0 --out /tmp/v0.json
python -m experiments.run_v1 --out /tmp/v1.json
python -m experiments.run_v2 --out /tmp/v2.json
python -m experiments.run_v3 --out /tmp/v3-full.json
```

`results/v3.json` is the compact frozen view of the full deterministic v3 receipt produced by CI.

## Repository map

- `src/fusion_machine/core.py` — v0 separated computations and nonlinear boundary
- `src/fusion_machine/resident.py` — v1 persistent state and bounded replay
- `src/fusion_machine/routing.py` — v2 binary publication and asynchronous material routing
- `src/fusion_machine/development.py` — v2 chemistry/activity route formation
- `src/fusion_machine/recurrent.py` — v3 masked/generic recurrent models and NumPy BPTT
- `src/fusion_machine/benchmark_v3.py` — frozen nonlinear world and evaluation regimes
- `experiments/run_v*.py` — deterministic scientific batteries
- `results/v*.json` — frozen result receipts
- `RESULTS_V*.md` — measured interpretations and claim boundaries
- `PAPER.md` — evolving paper-style argument
- `docs/RELATED_WORK.md` — prior-art and biology fence
- `index.html`, `web/` — live static laboratory
