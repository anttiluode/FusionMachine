# FusionMachine v2 — routing fabric results

v2 adds the outward half of the machine. Resident state is no longer equated with traffic: a nonlinear AIS-like boundary emits a one-bit event, and source-specific axonal/synaptic material determines which downstream dendritic compartment receives the write.

The biological language is motivational. Every result below is from the synthetic deterministic Python construction in `experiments/run_v2.py` and the frozen receipt `results/v2.json`.

## Gate 2A — address in matter

Eight sources all emit the same payload: the bit `1`. The intended destination differs only by source/axon identity.

| policy | routing accuracy |
|---|---:|
| **intact material route identity** | **1.000** |
| pooled source identity | 0.125 |
| cyclically shuffled routes | 0.000 |
| explicit digital-address oracle | **1.000** |

Classification: **PASS**.

The event amplitude contains no destination label. With intact route identity, the graph supplies that information. Erasing source identity reduces the system to one indistinguishable bit and gives the balanced 1/8 ceiling. Appending an explicit address restores perfect routing, showing exactly which information was removed by pooling.

Earned statement:

> **A low-dimensional event need not carry its destination in amplitude when connection identity already supplies the address.**

This is not a novelty claim about wires carrying identity; it is the architectural fence needed for the rest of FusionMachine.

## Gate 2B — chemistry + activity development

The synthetic development world contains 12 source/target identities arranged as 3 coarse chemical families × 4 activity-defined slots. Chemistry is therefore ambiguous within a family; activity is ambiguous across families. Independent noise is added to both cues.

Across 64 frozen random seeds:

| developmental policy | mean graph / routing accuracy |
|---|---:|
| **chemistry + activity** | **0.598958** |
| activity only | 0.263021 |
| chemistry only | 0.220052 |
| shuffled activity identity | 0.190104 |
| random wiring | 0.076823 |

Classification: **PASS**.

The combined cue exceeds chemistry-only by **37.89 percentage points** and activity-only by **33.59 points**. Shuffling which source owns which activity trace removes most of the activity benefit.

The important limitation is equally strong: this is a deliberately complementary synthetic coordinate system. It demonstrates that a coarse identity prior and experienced co-activation can be jointly useful for forming a route graph; it does not establish a biological growth law.

## Gate 2C — AIS publication is not resident state

A leaky resident compartment receives the same six drives in two conditions. One condition publishes normally. In the other, publication is suppressed for steps 1–4.

```text
resident trajectory both conditions
0.500000
0.750000
0.875000
0.937500
0.968750
0.984375
```

The maximum resident-state difference is exactly **0.0**. The suppressed window emits **0 events**, while the first event immediately after release is **1**.

Classification: **PASS**.

So the implementation cleanly separates:

```text
resident computation != publication permission
```

Suppressing the causal output does not freeze or reset the dendritic state.

## Gate 2D — rerouting changes behavior without changing the source computation

The frozen source event trace is

```text
0, 1, 0, 1, 1
```

With route A, every emitted event reaches target 1. With only the material destination changed, the bit-identical trace reaches target 2 instead.

Classification: **PASS**.

This is the causal separation we wanted:

> **the same resident computation and same pulse train can have a different downstream consequence solely because the axonal route changed.**

## Gate 2E — asynchronous chain

The chain uses four causal hops with nonuniform delays:

| source | target | emit | arrive |
|---:|---:|---:|---:|
| 0 | 1 | 0 | 1 |
| 1 | 2 | 1 | 3 |
| 2 | 3 | 3 | 4 |
| 3 | 4 | 4 | 7 |

All five compartments eventually fire. There is no feed-forward layer counter; target events arise only when delayed route writes arrive and cross the local boundary.

Classification: **PASS**.

## What v2 earns

The current computational primitive is:

```text
resident dendritic computation
        ↓
AIS-like binary commit
        ↓
source-specific material route
        ↓
local synaptic write
        ↓
next resident computation
```

Or algebraically,

```text
e_i(t) ∈ {0,1}
u_j(t+d_ij) += W_ji e_i(t)
h_j(t+1) = F_j(h_j(t), u_j(t))
```

The bit `e_i` can remain tiny because `W_ji`, `d_ij`, and the source/branch identity are state held in the machine itself.

The combined v0→v2 picture is now:

- **v0:** preserve computational identity until context arrives;
- **v1:** preserve continuing algorithmic state or retain enough history to replay it;
- **v2:** when a resident computation becomes public, let the material routing graph carry the address to the next computation.

## Claim boundary

v2 does **not** show that biological dendrites are arbitrary algorithms, that axons carry software addresses, or that real developmental targeting is equivalent to `chemistry + beta*activity`. It does not establish an advantage over an equal-capacity recurrent neural network or graph neural network.

It establishes a clean synthetic architecture and its causal decomposition. The next hard gate should therefore attack the architecture with a generic recurrent system that has the same resident-state budget and explicit learned routing capacity, rather than adding more biological metaphor first.
