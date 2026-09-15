# FusionMachine v3 — equal-state recurrent attacker

v3 stops adding biological detail and attacks the architecture with a generic recurrent system under a deliberately generous resource match.

The frozen regime was specified before the result was inspected. Both systems receive the same input stream, four resident state scalars, 28 learned route parameters, 96 training episodes, 1,200 optimizer updates, 16 publication opportunities in the rapid-switch episode, and the same optimizer. The generic attacker is **not** parameter-starved: it has 46 non-routing trainable parameters versus 30 for FusionMachine.

Eight paired initialization seeds (`0..7`) are reported as medians. These eight seeds are a deterministic engineering/science gate, **not** a significance study.

## World

Two nonlinear two-state teacher processes evolve continuously from the same three raw input features. Current context determines which teacher output is behaviorally queried, but it does not alter either teacher process.

Training context changes only in long 16–32 step blocks and supervision is sparse at block ends. Test then attacks the models with:

- ordinary held-out long-block episodes;
- rapid 2–6 step context switches;
- a 48-step dormant interval followed by an immediate switch;
- an identical sparse-publication schedule;
- an offline linear probe for the full four-dimensional hidden teacher state.

FusionMachine keeps two two-state resident mechanisms and does **not** feed current context into the resident transition. The baseline attacker is a fully dense four-state tanh RNN and is allowed to use context inside its recurrence.

## Primary fair-fight result

| median over 8 paired seeds | FusionMachine | generic dense RNN |
|---|---:|---:|
| training-query MSE | **0.000011991** | 0.000383249 |
| held-out query MSE | **0.000016380** | 0.001052879 |
| rapid first-post-switch MSE | **0.000015198** | 0.016539331 |
| 48-step dormancy: immediate absolute error | **0.002981** | 0.101834 |
| sparse receiver MSE | **0.192786** | 0.201617 |
| route accuracy at common publication slots | 0.608398 | 0.608398 |
| latent teacher-state probe R² | **0.999822** | 0.888666 |

The rapid first-switch MSE is about **1,088× lower** for FusionMachine and immediate post-dormancy error is about **34× lower**.

But FusionMachine also fits the sparse training queries about **32× better**. The frozen classification is therefore deliberately conservative:

> **`fusion_readiness_but_training_advantage_confounded`**

This gate is evidence of a readiness difference in this frozen regime, not yet evidence that the architectural decomposition is intrinsically superior at equal optimization quality.

## The more revealing destructive controls

The controls isolate *where current relevance enters the computation*.

| variant | rapid first-post-switch MSE | latent R² |
|---|---:|---:|
| **Fusion: context excluded from resident dynamics** | **0.000015198** | **0.999822** |
| Fusion + context leak into resident transition | 0.000155411 | 0.999443 |
| Fusion + dense recurrence, still no context | 0.000057449 | 0.999247 |
| generic dense RNN + context | 0.016539331 | 0.888666 |
| generic dense RNN **without context in recurrence** | **0.000235806** | **0.997412** |

Letting current relevance leak into FusionMachine's resident transition increases switch error about **10.2×**.

Removing relevance/context from the generic RNN recurrence improves its switch error about **70×**.

Making FusionMachine's resident recurrence dense while still withholding current context raises its switch error only about **3.8×** and remains far below the context-conditioned generic baseline.

That means this experiment does **not** support the simple story “block-diagonal modules beat dense recurrence.” The strongest clue is instead:

> **State is not relevance.**

A process that models how the world is changing may need to continue independently of what behavior currently cares about. Relevance can be applied later, at selection/publication, without rewriting the resident world state every time attention changes.

The OOD amplification makes the same point. Relative to ordinary held-out query MSE, first-switch error is roughly:

- Fusion: **0.93×**;
- generic + context: **15.7×**.

The relevance-insulated controls largely remove that switch-specific penalty.

## Hidden world-state diagnostic

Neither system receives teacher-state supervision. After behavioral training, a held-out linear probe asks how much of the actual four-dimensional teacher state is present in the learned recurrent state.

- FusionMachine: **R² = 0.999822**
- generic + context: **R² = 0.888666**
- generic without context: **R² = 0.997412**

This is consistent with, but does not prove, the interpretation that context-conditioned recurrence learned a more task-conditioned state while relevance-insulated recurrence stayed closer to a persistent world representation.

## Sparse publication does not produce a dramatic win

Both models have exactly 16 common publication opportunities in the 64-step rapid-switch episode. Receiver MSE is `0.192786` versus `0.201617`.

That is a small difference compared with the readiness gap. v3 therefore does **not** earn a large sparse-communication advantage claim.

Route accuracy is exactly tied at `0.608398` because both systems receive the same 28-parameter routing head and route features. This is useful: routing capacity is not the source of the readiness difference.

## What v3 earns

The strongest result is not yet “FusionMachine beats RNNs.” It is a more specific architectural clue:

```text
world input
   ↓
resident world dynamics  ← keep relevance out of this path when possible
   ↓
persistent state
   ↓
current context / relevance
   ↓
selection → publication → routing
```

The next attacker should close the remaining training-fit confound. Suitable gates include optimization-matched or loss-matched checkpoints, stronger recurrent cells, and tasks where relevance itself causally changes the world and therefore *should* enter state.

## Claim boundary

- eight deterministic seeds are not a statistical significance study;
- the teacher is synthetic and hand-constructed;
- FusionMachine has the better training fit in this gate;
- a vanilla tanh RNN is not the strongest possible recurrent baseline;
- the result says nothing directly about biological consciousness, dendritic algorithms, basket cells, or chandelier cells;
- Clockwork RNNs, modular recurrent systems, Recurrent Independent Mechanisms, context gating and predictive/world-state representations are established neighboring ideas.

The compact frozen receipt is [`results/v3.json`](results/v3.json). It contains the exact medians and resource contract taken from the full deterministic CI-generated v3 receipt.
