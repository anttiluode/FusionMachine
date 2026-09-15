# Related work and novelty fence

FusionMachine is intentionally built inside several mature research neighborhoods. This document exists to prevent the project from mistaking a useful synthesis for invention of established mechanisms.

## 1. Dendrites as nonlinear computational subunits

Poirazi, Brannon & Mel (2003), **“Pyramidal Neuron as Two-Layer Neural Network”**, modeled terminal dendritic branches as nonlinear subunits whose outputs are combined before final thresholding. DOI: `10.1016/S0896-6273(03)00149-1`.

London & Häusser (2005), **“Dendritic Computation”**, review the broad evidence that dendrites implement linear and nonlinear elementary computations rather than acting only as passive cables. DOI: `10.1146/annurev.neuro.28.061604.135703`.

Larkum, Zhu & Sakmann (1999), **“A new cellular mechanism for coupling inputs arriving at different cortical layers”**, showed a nonlinear interaction between distal dendritic input and back-propagating axonal activity (BAC firing). DOI: `10.1038/18686`.

Gidon et al. (2020), **“Dendritic action potentials and computation in human layer 2/3 cortical neurons”**, reported graded dendritic calcium action potentials in human cortical neurons and showed that the measured dendritic nonlinearity can support linearly nonseparable input classification. DOI: `10.1126/science.aax6239`.

Aizenbud et al. (2026), **“Dendritic morphology and synaptic nonlinearities enhance functional complexity in human cortical neurons”**, reported greater modeled functional complexity in human cortical pyramidal neurons and attributed important contributions to dendritic morphology and nonlinear NMDA signaling. DOI: `10.1073/pnas.2533168123`.

**Boundary:** these papers motivate thinking of a branched cell as a multi-stage nonlinear computer. FusionMachine does not establish that its abstract modes map onto specific branches, eigenmodes, ion channels, basket cells, or chandelier cells.

## 2. Conditional computation and mixture of experts

Conditional computation is an established neural-network idea: only some computation paths are activated for a given input or context. Sparsely gated mixture-of-experts systems make this explicit by maintaining many expert subnetworks and learning a router/gate that selects a sparse subset.

A canonical reference is Shazeer et al. (2017), **“Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer”**, arXiv:`1701.06538`.

Related conditional-computation work includes Bengio et al. (2015), **“Conditional Computation in Neural Networks for faster models”**, arXiv:`1511.06297`.

**Boundary:** FusionMachine is not claiming to invent expert routing. Its narrower question is whether several computations can remain *resident as state inside one shared substrate* while the causal boundary performs context-dependent selection, and whether early collapse of those modes destroys useful counterfactual identity.

A future benchmark must therefore include an explicit MoE/router attacker. If a FusionMachine gate merely rediscovers a standard MoE with worse engineering, that should be recorded as the result.

## 3. Multiplicative interactions and context gating

The exact v0 selector contains the bilinear term

```text
context * (B-A).
```

Multiplicative interactions are old and widespread. Jayakumar et al. (ICLR 2020), **“Multiplicative Interactions and Where to Find Them”**, connect multiplicative interactions to gating, attention, hypernetworks, dynamic convolutions, information-stream fusion, and conditional computation.

FusionMachine therefore does **not** claim that the product term is novel. v0 uses it because the complete truth table gives a clean algebraic demonstration of why an additive/affine boundary cannot contextually select independent resident modes.

Hypernetworks are another close neighbor: Ha, Dai & Le (2016), **“HyperNetworks”**, arXiv:`1609.09106`, use one network to generate the parameters of another, giving context a multiplicative influence over computation.

## 4. Algorithmic state and differentiable memory

Neural Turing Machines (Graves, Wayne & Danihelka, 2014, arXiv:`1410.5401`) showed that neural systems can learn algorithm-like procedures by coupling a controller to addressable external memory.

That literature makes an important fence for the phrase “algorithmic mode.” FusionMachine does not yet show program induction. Its route computations are supplied by construction. The research question is whether separated recurrent modes can preserve the *state of different computations* and switch between them without losing the history each procedure needs.

## 5. Recursive filtering, sufficient state, and predictive representations

v1 is especially close to classical state estimation. Kalman (1960), **“A New Approach to Linear Filtering and Prediction Problems”**, formulates filtering recursively through a state-transition representation rather than replaying the full observation history. DOI: `10.1115/1.3662552`.

The conceptual point is older and broader than Kalman filtering: a well-chosen current state can act as a sufficient summary of the relevant past for continuing a computation. FusionMachine v1's leaky scalar is an extremely simple example of such a recursively maintained sufficient statistic.

Littman, Sutton & Singh (NeurIPS 2001 / NIPS 14), **“Predictive Representations of State”**, make the representational idea explicit in another direction: dynamical state can be represented by predictions of future observations rather than by a privileged hidden-state label. Their work is a useful warning that there are many valid coordinates for state; FusionMachine's named resident modes are only one possible coordinate system.

**Boundary:** v1 does not invent recursive state, sufficient statistics, state-space filtering, caching, or replay. Its narrower measured result is the exact relation between one mode's persistence and the amount of omitted history that matters when that mode has been suspended.

## 6. Replay, event logs, and materialized state

In software architecture, event-sourced systems deliberately retain event histories so current state can be reconstructed by replay, while materialized/projected state trades storage and continual update work for fast current queries. The analogy to v1 is structural rather than a claim of direct technical novelty.

A useful software-systems reference is Overeem et al. (2021), **“Improving observability in Event Sourcing systems”**, *Journal of Systems and Software* 181:111015, DOI `10.1016/j.jss.2021.111015`, which discusses event logs and replay in operational systems.

FusionMachine v1's resident-versus-replay accounting should therefore be read as a tiny dynamical version of a familiar systems tradeoff: maintain current derived state continuously, or retain enough history to rebuild it later.

## 7. Resident state versus communication

Delta Networks (Neil, Lee, Delbruck & Liu, ICML 2017), **“Delta Networks for Optimized Recurrent Network Computation”**, transmit neural activations only when their change exceeds a threshold, exploiting temporal stability to reduce recurrent computation/communication.

FusionMachine inherits from the separate NewMachine/ActiveVectorNN line the idea that resident state and published traffic need not be identical. That is established territory around event-triggered and delta communication; the future question is what happens when resident state contains multiple computational modes and the publication boundary is itself context-dependent.

## 8. What FusionMachine is actually testing

The project should be judged on progressively stronger claims, not on the novelty of its ingredients.

### v0 claim — preserve computational identity

```text
preserve A and B separately
        +
nonlinear context interaction
        ->
select A or B exactly
```

while early collapse to `(A+B)/2` creates a provable information loss.

### v1 claim — preserve computational history

A dormant leaky computation may be kept current continuously, or reconstructed later from retained missed inputs. With bounded replay, switch-time error follows the mode's own persistence law. Full replay remains exact.

The useful sentence is not “resident state beats replay.” It is:

> **Persistence is also a replay horizon.**

### later target

Only if stronger gates survive recurrent, MoE, and equal-capacity attackers should the project add:

- genuinely different temporal algorithms per mode;
- sparse publication and predictive receivers;
- learned formation of computational modes;
- local credit and slow operator rewriting;
- physically branched / dendritic substrates;
- biological interpretations.

## 9. Novelty standard

A useful eventual claim would not be “neurons are two-layer networks,” “multiplicative gating works,” “experts can be routed,” “recursive state summarizes history,” or “state can be communicated sparsely.” Those are prior art.

The potentially distinctive object is instead:

> **a shared, persistent substrate that keeps several counterfactually distinct computations alive as resident modes, allows those modes to keep evolving even while behavior ignores them, and delays nonlinear/context-dependent causal selection until the output boundary.**

Whether that object provides an advantage over generic recurrent state or explicit expert routing is the scientific question of this repository.
