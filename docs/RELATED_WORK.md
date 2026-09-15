# Related work and novelty fence

FusionMachine is intentionally built inside several mature research neighborhoods. This document exists to prevent the project from mistaking a useful synthesis for invention of established mechanisms.

## 1. Dendrites as nonlinear computational subunits

Poirazi, Brannon & Mel (2003), **“Pyramidal Neuron as Two-Layer Neural Network”**, modeled terminal dendritic branches as nonlinear subunits combined before a final output stage. DOI: `10.1016/S0896-6273(03)00149-1`.

London & Häusser (2005), **“Dendritic Computation”**, review the broad evidence that dendrites implement linear and nonlinear elementary computations rather than acting only as passive cables. DOI: `10.1146/annurev.neuro.28.061604.135703`.

Larkum, Zhu & Sakmann (1999), **“A new cellular mechanism for coupling inputs arriving at different cortical layers”**, showed nonlinear interaction between distal dendritic input and back-propagating axonal activity. DOI: `10.1038/18686`.

Gidon et al. (2020), **“Dendritic action potentials and computation in human layer 2/3 cortical neurons”**, reported graded dendritic calcium action potentials in human cortical neurons and showed that the measured dendritic nonlinearity can support linearly nonseparable classification. DOI: `10.1126/science.aax6239`.

Aizenbud et al. (2026), **“Dendritic morphology and synaptic nonlinearities enhance functional complexity in human cortical neurons”**, reported greater modeled functional complexity in human cortical pyramidal neurons and important contributions from morphology and nonlinear NMDA signaling. DOI: `10.1073/pnas.2533168123`.

**Boundary:** these papers motivate a branched cell as a multi-stage nonlinear computer. FusionMachine does not establish that its abstract resident modes map onto specific branches, eigenmodes or ion-channel mechanisms.

## 2. The AIS as output boundary and plastic compartment

Fréal & Hoogenraad (Neuron, 2025), **“The dynamic axon initial segment: From neuronal polarity to network homeostasis”**, DOI `10.1016/j.neuron.2025.01.004`, review the AIS as a specialized compartment between somatodendritic and axonal domains. They emphasize two broad functions: generation/modulation of action potentials and maintenance of neuronal polarity. The review also describes molecular heterogeneity along the AIS, including distinct proximal/distal channel organization, and emphasizes that an apparently stable AIS can undergo activity-dependent structural and functional remodeling.

The review further discusses axo-axonic GABAergic innervation of the AIS, including chandelier-cell synapses in cortex and hippocampus, and notes activity-dependent plasticity of AIS structure and of chandelier-cell contacts.

**Boundary:** FusionMachine uses this literature only to motivate separating resident somatodendritic state from an output/publication boundary. It does not show that the AIS implements the software gate in v2, that chandelier cells are literal permission bits, or that the synthetic event threshold captures AIS electrophysiology.

## 3. Axon guidance, chemoaffinity, and target specificity

Sperry's chemoaffinity hypothesis historically proposed that growing axons and targets possess molecular labels that contribute to specific connectivity. Modern axon-guidance work replaces a one-barcode-per-neuron picture with combinations of gradients, receptors, adhesion systems, intermediate cues, target recognition and later synapse selection/refinement.

Activity-dependent refinement is also established biological territory: initial molecularly guided connectivity can be modified by correlated activity and competition during development and experience.

**Boundary:** FusionMachine v2's developmental rule

```text
score_ij = z(chemistry_ij) + beta*z(activity_ij)
```

is a synthetic complementary-coordinate experiment. It is not a molecular model of Eph/ephrin systems, growth cones, cell-adhesion codes, Hebbian refinement or real synaptogenesis. The result is computational: two imperfect information sources can jointly recover a better route graph than either source alone.

## 4. Conditional computation and mixture of experts

Conditional computation is established: only some computation paths are activated for a given input/context. Sparsely gated mixture-of-experts systems make this explicit with many experts and a learned router.

A canonical reference is Shazeer et al. (2017), **“Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer”**, arXiv:`1701.06538`. Related work includes Bengio et al. (2015), **“Conditional Computation in Neural Networks for faster models”**, arXiv:`1511.06297`.

**Boundary:** FusionMachine does not claim to invent routing or experts. Its question is whether several computations remain useful as continuously resident state while publication and downstream routing are separate causal stages.

## 5. Multiplicative interactions and context gating

The exact v0 selector contains

```text
context * (B-A).
```

Multiplicative interactions are old and widespread. Jayakumar et al. (ICLR 2020), **“Multiplicative Interactions and Where to Find Them”**, connect them to gating, attention, hypernetworks, dynamic convolutions, fusion and conditional computation. Hypernetworks (Ha, Dai & Le, 2016, arXiv:`1609.09106`) are another nearby mechanism.

**Boundary:** v0 uses a product term because it gives a clean algebraic separation from an affine attacker, not because multiplicative gating is novel.

## 6. Algorithmic state and differentiable memory

Neural Turing Machines (Graves, Wayne & Danihelka, 2014, arXiv:`1410.5401`) show that neural systems can learn algorithm-like procedures by coupling a controller to addressable external memory.

This literature fences the phrase **algorithmic mode**. FusionMachine does not yet show program induction; its computations and route families are supplied by construction. The question is whether the continuing state of different procedures can remain separately resident and later be selected or routed without replaying lost history.

## 7. Recursive filtering, sufficient state, and predictive representations

v1 is close to classical state estimation. Kalman (1960), **“A New Approach to Linear Filtering and Prediction Problems”**, DOI `10.1115/1.3662552`, formulates recursive state rather than replaying the full observation history.

Littman, Sutton & Singh (NeurIPS 2001), **“Predictive Representations of State”**, emphasize that dynamical state can be represented in multiple coordinates, including predictions of future observations.

**Boundary:** v1 does not invent sufficient statistics, recursive filters or replay. Its measured result is the relation between a mode's persistence and how much omitted history still matters when reconstructing a suspended computation.

## 8. Event-triggered communication and delta networks

Delta Networks (Neil, Lee, Delbruck & Liu, ICML 2017), **“Delta Networks for Optimized Recurrent Network Computation”**, exploit temporal stability by transmitting activation changes only when they exceed a threshold.

Event-triggered control, neuromorphic/event-based computation and asynchronous message-passing systems provide broader neighborhoods for the idea that resident state and traffic can differ.

**Boundary:** FusionMachine's one-bit event is not a novelty claim. v2 asks what follows when event payload, source identity, route graph and receiving state are treated as separate information-bearing objects.

## 9. Graph routing and message passing

Graph neural networks, message-passing neural networks, actor systems and asynchronous distributed systems all make source/destination structure explicit. A message can be small because graph topology and endpoint identity already carry information about where it came from and where it goes.

This is an important prior-art fence for the v2 phrase **address in matter**. The result is not that topology can route messages—that is elementary. The useful question is whether the same separation, combined with resident algorithmic state and nonlinear publication, creates a productive AI architecture under matched controls.

## 10. What FusionMachine has actually established

### v0 — preserve computational identity

Early blending can create an information-theoretic ambiguity that a later context cannot undo.

### v1 — preserve computational history

A suspended persistent computation requires retained missed history for exact reconstruction. Full replay is exact; resident state trades continuous work for readiness.

> **Persistence is also a replay horizon.**

### v2 — separate state, publication and routing

The same one-bit payload reaches different destinations under different route identities. Pooling route/source identity destroys that information; an explicit address oracle restores it. A synthetic chemistry+activity developmental rule recovers a better route graph than either cue alone. Publication suppression leaves resident state untouched, and route intervention changes downstream consequences while source state/events stay fixed.

## 11. Novelty standard

A useful eventual FusionMachine claim is **not** any of the following:

- neurons are two-layer networks;
- multiplicative gating works;
- experts can be routed;
- recursive state summarizes history;
- event-triggered communication is efficient;
- topology carries source/destination identity;
- chemoaffinity or activity-dependent refinement exists.

Those are established territories.

The potentially distinctive object is the composition:

> **a shared machine in which several computations remain live as resident state, a nonlinear boundary decides when one becomes public, the public token can be tiny because route identity is embodied in persistent structure, and local receiving dynamics determine what that routed event means next.**

Whether that composition has a computational advantage over an equal-capacity generic recurrent model with learned routing remains the next scientific question.
