# FusionMachine v0 — separated algorithms, nonlinear causal boundary

Date: 2026-09-15

## Question

Can one resident vector retain two counterfactually distinct computations that are behaviorally indistinguishable in an ordinary world, while a nonlinear/contextual boundary later selects which computation becomes behavior without destroying the other?

The repo deliberately starts with the smallest exact case before adding persistence, sparse events, dendritic biology, growth, or learning rules.

## Core construction

Use three binary cues in {-1,+1}:

- direct algorithm A: `A(x) = x0`
- relational algorithm B: `B(x) = x1*x2`

The ordinary/correlated world enforces `x0 = x1*x2`, so A and B give exactly the same answer. Output behavior therefore cannot identify which algorithm is present.

The intervention world breaks that correlation. A context bit `c in {-1,+1}` defines the desired behavior:

- `c=-1` -> publish A
- `c=+1` -> publish B

A separated resident representation keeps

`z = [A(x), B(x)]`.

A collapsed representation keeps only

`r = (A(x)+B(x))/2`.

## Nonlinear boundary

The exact selector is

`y = 0.5*(A+B) + 0.5*c*(B-A)`.

The product `c*(B-A)` is the minimal nonlinear/context interaction. It behaves like a tiny causal boundary: the resident modes both continue to exist; context controls which mode becomes externally consequential.

A purely linear readout of `[A,B,c]` cannot implement this selector for all independent A/B/context combinations because it lacks the multiplicative context-by-mode interaction.

The collapsed representation is even weaker: when A and B disagree, `r=0` erases which sign belonged to which algorithm, so no downstream function of `(r,c)` can perfectly recover the selected answer.

## Gates

### Gate 0A — same answer, different algorithm

Enumerate the correlated world and verify that A and B are behaviorally identical there.

Then enumerate the intervention world and verify that they disagree on exactly half of the independent cue states.

This is a calibration gate, not a learning result.

### Gate 0B — nonlinear selection

Compare:

1. exact nonlinear selector over separated modes;
2. best least-squares linear readout over `[A,B,c]`;
3. best lookup/readout from collapsed `(r,c)`;
4. explicit oracle that receives A and B separately and context.

Primary criterion: the nonlinear selector must achieve zero error on the complete intervention truth table; the linear and collapsed attackers must have nonzero irreducible error.

### Gate 0C — information witness

Produce an explicit pair of intervention states with identical collapsed `(r,c)` but opposite required outputs. This proves the collapsed failure is information-theoretic, not an optimizer failure.

### Gate 0D — tiny calibration / representation reuse

Fit a minimal parameterized boundary only on a subset of intervention states while keeping A/B resident modes fixed. Test held-out intervention states.

Compare with a model whose representation was collapsed during the correlated phase. The point is not to claim general sample efficiency; it is to show that preserving both algorithms leaves a reusable substrate once a small amount of context evidence arrives.

## Claim boundary

A pass supports only:

> A vector can carry multiple counterfactually distinct computations simultaneously, and a nonlinear context interaction can select which computation becomes behavior. Collapsing the resident modes can destroy information that no later boundary can recover.

It does not establish that biological dendrites implement these equations, that arbitrary algorithms can be stored as vector modes, that the mechanism learns its own modes, or that it outperforms modern neural networks.

## Architecture direction after v0

If v0 passes, v1 replaces the static A/B outputs with two genuinely dynamical algorithms whose internal states evolve through time. The next question becomes whether one resident substrate can retain both dynamical computations through context switches while the boundary publishes only the currently relevant one.

Later gates may add:

- sparse publication / predictive receivers from NewMachine;
- mode persistence and purification from SighImageSuper / AnttisNeuron;
- learned route identity and temporal credit from GeometricNeuronV25 / ThirdWay;
- slow operator rewriting / growth;
- biological mappings only after the computational object survives strong controls.

## Artifacts

- deterministic Python experiment and frozen JSON receipt;
- unit tests for the truth table, exact selector, linear lower bound, and collapse witness;
- README explaining the object and claim boundary;
- static `index.html` showing the two resident algorithms, their agreement/disagreement, context selection, and the information lost by collapse;
- CI running tests and the frozen experiment.
