# FusionMachine v0 — separated algorithms and a nonlinear causal boundary

## Frozen result

FusionMachine v0 keeps two computations separate inside one resident vector:

```text
A(x) = x0
B(x) = x1*x2
z    = [A, B]
```

In the correlated world `x0=x1*x2`, so the two algorithms are behaviorally indistinguishable: **4/4 rows agree**.

The intervention world breaks that relation and adds a context bit. Across the complete 16-row truth table, A and B disagree on **50%** of rows. The desired output is A when `context=-1` and B when `context=+1`.

| boundary / representation | MSE | accuracy |
|---|---:|---:|
| exact nonlinear selector | **0.000000** | **1.000** |
| best affine readout of `[A,B,context]` | 0.500000 | 0.750 |
| best full-data lookup of collapsed `(A+B)/2, context` | 0.500000 | 0.750 |
| two-row calibrated factorized selector | **0.000000** | **1.000** |

The best affine readout is

```text
0.5*A + 0.5*B
```

with zero learned context weight. The missing term is the multiplicative interaction

```text
0.5 * context * (B-A)
```

which is exactly what changes which resident computation becomes behavior.

## Information-loss witness

The collapsed representation is not merely harder to optimize. It is insufficient.

These two intervention states both give

```text
collapsed state r = (A+B)/2 = 0
context = -1
```

but require opposite outputs:

```text
left:   A=-1, B=+1 -> target=-1
right:  A=+1, B=-1 -> target=+1
```

No deterministic downstream function of `(r, context)` can distinguish them. The lost algorithm identity cannot be recreated after collapse.

## Tiny calibration diagnostic

The exact boundary can be written using two mechanistic basis terms:

```text
y = w_sum  * (A+B)
  + w_gate * context*(B-A)
```

One calibration row where A and B agree identifies the sum coefficient; one row where they disagree identifies the contextual interaction. Least squares on those two rows recovers

```text
w_sum  = 0.5
w_gate = 0.5
```

and gives zero error on all 16 intervention rows. This is a representation-reuse diagnostic, not a general sample-efficiency claim: the useful nonlinear basis was supplied by design.

## What v0 earns

> **A resident vector can preserve multiple counterfactually distinct computations at once. A nonlinear context interaction can select which computation becomes behavior, while collapsing the modes can destroy information that no later boundary can recover.**

This is stronger than “nonlinearity helps” but much weaker than “a dendrite stores arbitrary algorithms.” Here A and B are tiny hand-defined computations and the correct factorized interaction is known.

## What v0 does not establish

- no biological dendrite/AIS claim;
- no learned discovery of A or B;
- no recurrent or persistent algorithm state yet;
- no sparse event communication yet;
- no advantage over an unrestricted MLP or explicit mixture-of-experts router;
- no evidence that arbitrary programs can be represented as modes.

## Next gate

v1 should make A and B **dynamical algorithms with internal state**, not static outputs. The key test is whether both internal computations can continue evolving while context changes which one is published, without forcing the dormant computation to be recomputed or reconstructed after a switch.
