# FusionMachine v1 — resident algorithmic state versus bounded replay

## Question

v0 kept two computations separate until a nonlinear context boundary selected one. v1 asks the temporal version:

> **If a computation is not currently behaviorally relevant, what is gained by keeping its internal state running anyway?**

Each hidden route now carries a leaky dynamical state

```text
h_t = alpha*h_(t-1) + (1-alpha)*u_t
```

and context can suddenly switch to a route that was dormant for an 80-step hidden interval.

The comparison is deliberately fair to lazy computation. Every policy starts from the same known boundary state. A lazy policy is allowed to retain the last `K` hidden drives and reconstruct the dormant state at switch time. `K=80` is full replay and is exact.

## Analytic omitted-history law

If the hidden drives are independent Rademacher values and only the last `K` of an 80-step gap are retained, switch-time reconstruction error is the omitted prefix:

```text
e = (1-alpha) * sum_(r=K)^(N-1) alpha^r u_(N-r)
```

so

```text
Var[e]
 = (1-alpha)^2 * alpha^(2K)
   * (1-alpha^(2(N-K))) / (1-alpha^2).
```

The measured experiment uses 4,096 deterministic hidden tapes. Across all nonzero points in the four-alpha sweep, the worst Monte-Carlo versus analytic RMSE discrepancy is **1.664%**, at `alpha=0.98, K=4`.

## Primary curve — alpha = 0.95, gap = 80

| remembered tail K | measured switch RMSE | analytic RMSE |
|---:|---:|---:|
| 0 | 0.160939 | 0.160106 |
| 4 | 0.131933 | 0.130399 |
| 8 | 0.106385 | 0.106199 |
| 16 | 0.070331 | 0.070427 |
| 32 | 0.030810 | 0.030906 |
| 64 | 0.005413 | 0.005395 |
| **80** | **0.000000** | **0.000000** |

The full resident computation is the reference and therefore has zero switch error: it never stopped updating. Full lazy replay also reaches zero because it has retained every missed drive.

The scientific distinction is not accuracy at unlimited resources. It is **when the work is paid and what has to be retained**.

## Persistence sets the replay horizon

For each `alpha`, we computed the smallest number of most-recent hidden drives needed to reduce analytic replay RMSE to at most **10% of the zero-history error**:

| persistence alpha | remembered steps needed | fraction of 80-step gap |
|---:|---:|---:|
| 0.50 | **4** | 5.0% |
| 0.80 | **11** | 13.8% |
| 0.95 | **45** | 56.3% |
| 0.98 | **75** | 93.8% |

This is the useful v1 result.

A rapidly forgetting computation can be reconstructed accurately from a short recent tail because distant events have almost vanished anyway. A slowly forgetting computation preserves old distinctions — but that same property makes its dormant state expensive to reconstruct if those old inputs were not processed or retained.

So persistence has a two-sided computational consequence:

```text
long memory
    -> old events still matter
    -> useful resident history
    -> expensive lazy reconstruction
```

This is the direct bridge back to SighImageSuper's forgetting-time story.

## Work / memory accounting

For one 80-step dormant interval:

| policy | dormant updates during gap | stored hidden drives | replay updates at switch | current state scalars |
|---|---:|---:|---:|---:|
| **resident dual** | **80** | **0** | **0** | 1 |
| **lazy full replay** | **0** | **80** | **80** | 1 |

Nothing is free.

Resident state pays continuous local computation and arrives at the context switch already current. Exact lazy replay postpones those updates, stores the whole missing history, and pays them as a burst when the dormant computation becomes relevant.

A bounded-history lazy system pays less storage/replay cost but accepts the analytically predictable switch error above.

## What v1 earns

> **A dormant dynamical computation can remain immediately available by keeping its sufficient state resident. Suspending it is not free: exact later recovery requires the missed history or an equivalent sufficient statistic. With bounded replay, reconstruction error falls on the dormant mode's own forgetting scale.**

An even shorter wall sentence is:

> **Persistence is also a replay horizon.**

## What v1 does not establish

- Resident computation does **not** universally beat lazy evaluation; full replay is exact.
- This is a first-order leaky state, not a complicated learned algorithm.
- Both routes use the same recurrence family; only their input computations differ.
- The hidden drives are iid and exactly known to the replay mechanism when retained.
- There is no bandwidth limit on the internal update and no hardware energy model.
- This is closely related to streaming sufficient statistics, state-space filtering, caching/materialized state, and replay; novelty is not assumed.

## Why the result matters for FusionMachine

v0 said: **do not collapse different computations before context arrives.**

v1 adds: **if those computations have history, preserving only their identities is not enough — their internal states must either keep evolving or their missed history must remain recoverable.**

That moves the object from

```text
[A(x), B(x)]
```

toward

```text
[h_A(t), h_B(t)]
```

where each coordinate is the current state of a continuing computation.

## Next gate

v2 should stop giving both routes the same leaky recurrence. The resident modes should become **genuinely different temporal algorithms** — for example, one integrator/predictor and one finite-state or relational temporal procedure — and then face an equal-state-capacity generic recurrent attacker.

If a generic recurrent state preserves everything just as efficiently, the separated-algorithm view may remain explanatory rather than computationally privileged. That is the next important attack.
