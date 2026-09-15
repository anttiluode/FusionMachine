# FusionMachine

> **Keep several computations resident. Let a nonlinear causal boundary decide which one becomes behavior.**

FusionMachine is a small research program about a machine suggested by several earlier experiments: a vector/state need not represent only *data*. Its separated modes can represent the current outputs — and later the internal states — of different computations. A nonlinear/contextual boundary can then decide which resident computation becomes externally consequential without first averaging the alternatives into one representation.

This repository starts with the smallest exact case. It is **not** yet a biological-neuron model and it does not claim that arbitrary algorithms are literally stored in dendritic eigenmodes.

**Live v0 instrument:** https://anttiluode.github.io/FusionMachine/

## v0 — same answer, different algorithm

Define two computations on binary cues in `{-1,+1}`:

```text
A(x) = x0            direct route
B(x) = x1*x2         relational route
```

In the ordinary world we impose

```text
x0 = x1*x2
```

so A and B produce exactly the same answer. Behavior alone cannot tell which computation is present.

Then break that correlation and add a context bit:

```text
context = -1 -> publish A
context = +1 -> publish B
```

The resident state keeps both computations:

```text
z = [A, B]
```

and the minimal exact boundary is

```text
y = 0.5*(A+B) + 0.5*context*(B-A)
```

The second term is a multiplicative context-by-mode interaction.

### Frozen complete-truth-table result

| boundary / representation | MSE | accuracy |
|---|---:|---:|
| **nonlinear selector on separated `[A,B]`** | **0.000000** | **1.000** |
| best affine readout of `[A,B,context]` | 0.500000 | 0.750 |
| best full-data lookup after collapse to `(A+B)/2` | 0.500000 | 0.750 |
| **two-row calibrated factorized selector** | **0.000000** | **1.000** |

The best affine attacker is simply

```text
0.5*A + 0.5*B
```

because the contextual product term is outside its feature span.

The collapse attacker loses information even with the complete truth table. When A and B disagree, `(A+B)/2=0`. Two states can therefore have the same collapsed state and the same context while requiring opposite outputs. No later deterministic readout can reconstruct which algorithm had which sign.

See [`RESULTS_V0.md`](RESULTS_V0.md) and the frozen [`results/v0.json`](results/v0.json).

## What the result means

The v0 object is

```text
separated resident computations
             ↓
        [ A , B ]
             ↓
context × mode interaction
             ↓
      causal publication
```

The earned statement is narrow:

> **A resident vector can preserve multiple counterfactually distinct computations at once. A nonlinear context interaction can select which computation becomes behavior, while collapsing the modes can destroy information that no later boundary can recover.**

That is already different from treating the vector as a single blended feature representation.

## Why this connects to the earlier repos

The connection is conceptual, not a claim that the mechanisms are identical:

```text
SighImageSuper
    operators create persistent / recoverable modes

GAx
    modes can correspond to counterfactually different algorithms,
    not merely different candidate answers

AnttisNeuron / GrowingAnttisNeuron
    branched physical structure can compile and separate dynamics

GeometricNeuronV25 / ThirdWay
    separated routes can preserve causal identity through time

NewMachine
    resident state and causal publication are different variables

FusionMachine
    keep several computations resident and fuse/select them
    only at a nonlinear causal boundary
```

The important shift is that **data, memory, partial computation, and algorithmic state may all be resident dynamical state**. v0 demonstrates only the simplest static version of that idea.

## Strong prior-art fence

None of the ingredients by themselves are new fields. Dendritic nonlinear subunits, conditional computation, mixture-of-experts routing, multiplicative interactions, recurrent/state-space computation, and sparse delta communication all have substantial prior literature. [`docs/RELATED_WORK.md`](docs/RELATED_WORK.md) maps the closest neighborhoods.

FusionMachine's narrower research question is about the *combination and separation of roles*:

> When is it useful to preserve several computations as resident modes, instead of blending them immediately, and delay context-dependent nonlinear selection until a causal output boundary?

## Next gate

v1 should stop treating A and B as static outputs. Each branch should become a **dynamical algorithm with its own internal state**. Both algorithms continue to evolve every step; context only chooses which state is published.

The key attacker will be a system that keeps only the currently selected algorithm alive. If context switches, it must reconstruct the dormant computation from incomplete history. A true resident multimode machine should already have that computation at the correct state.

Only after that survives should sparse publication, learned mode formation, slow operator rewriting, and biological mappings be added.

## Run

```bash
python -m pip install -e ".[test]"
pytest -q
python -m experiments.run_v0 --out /tmp/v0.json
```

## Repository map

- `src/fusion_machine/core.py` — two computations, correlated/intervention worlds, exact nonlinear boundary
- `src/fusion_machine/attackers.py` — best affine attacker, collapse attacker, information witness, factorized calibration
- `experiments/run_v0.py` — deterministic scientific receipt
- `results/v0.json` — frozen v0 numbers
- `RESULTS_V0.md` — interpretation and claim boundary
- `PAPER.md` — paper-style v0 argument and derivation
- `docs/RELATED_WORK.md` — literature neighborhood and novelty fence
- `index.html`, `web/` — static GitHub Pages inspection instrument
- `tests/` — mechanism and receipt regressions
- `docs/superpowers/` — frozen design and implementation plan
