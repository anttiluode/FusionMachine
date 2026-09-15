# Related work and novelty fence

FusionMachine intentionally sits inside mature research neighborhoods. This document exists to stop the project from mistaking a useful synthesis for invention of established mechanisms.

## Dendrites as nonlinear computational subunits

Poirazi, Brannon & Mel (2003), **“Pyramidal Neuron as Two-Layer Neural Network”**, modeled terminal dendritic branches as nonlinear subunits combined before a final output stage. DOI `10.1016/S0896-6273(03)00149-1`.

London & Häusser (2005), **“Dendritic Computation”**, review evidence that dendrites perform linear and nonlinear elementary computations rather than acting only as passive cables. DOI `10.1146/annurev.neuro.28.061604.135703`.

Larkum, Zhu & Sakmann (1999), Gidon et al. (2020), and later morphology/nonlinearity work further establish active dendrites as a rich computational substrate.

**Boundary:** this motivates branched local dynamics. FusionMachine does not show that an abstract resident mode is a literal biological branch, eigenmode, token, or arbitrary program.

## Compartment-specific inhibition and the AIS

Cortical inhibitory interneurons target different pyramidal-cell compartments. Somatostatin/Martinotti populations preferentially target dendrites; basket cells preferentially target somatic/perisomatic regions; chandelier/axo-axonic cells target the axon initial segment.

Fréal & Hoogenraad (Neuron, 2025), **“The dynamic axon initial segment: From neuronal polarity to network homeostasis”**, DOI `10.1016/j.neuron.2025.01.004`, review the AIS as a specialized boundary between somatodendritic and axonal domains involved in action-potential generation/modulation and neuronal polarity, with molecular heterogeneity and activity-dependent remodeling.

**Boundary:** this motivates separating dendritic integration, perisomatic control and output initiation. It does not establish FusionMachine's exact software roles for basket or chandelier cells.

## Axon guidance, target specificity and activity-dependent refinement

Sperry's chemoaffinity work and modern guidance literature motivate the idea that route identity can be partly embodied in persistent biological structure. Modern development is not a one-neuron/one-barcode lookup: gradients, receptors, adhesion systems, intermediate cues, target recognition, synapse specificity and activity-dependent refinement all contribute.

FusionMachine v2's rule

```text
score_ij = z(chemistry_ij) + beta*z(activity_ij)
```

is only a synthetic complementary-cue experiment. It is not a molecular model of growth cones, Eph/ephrin systems, adhesion codes, Hebbian refinement or synaptogenesis.

## Conditional computation, multiplicative gating and expert routing

Conditional computation and mixture-of-experts architectures already route inputs to subsets of computation. Shazeer et al. (2017), **“Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer”**, arXiv `1701.06538`, is a canonical reference. Bengio et al. (2015), **“Conditional Computation in Neural Networks for faster models”**, is another.

FusionMachine v0's exact selector uses the bilinear term `context*(B-A)`. Multiplicative interactions are established; see Jayakumar et al. (ICLR 2020), **“Multiplicative Interactions and Where to Find Them.”**

**Boundary:** neither routing nor multiplicative context gating is novel. v0 only establishes an exact information-loss witness for collapsing counterfactually different computations too early.

## Recurrent state, sufficient statistics and replay

Kalman filtering, state-space models, predictive-state representations and recurrent neural networks all maintain compact state instead of replaying raw history. Littman, Sutton & Singh (NeurIPS 2001), **“Predictive Representations of State”**, is a useful warning that there is no unique privileged coordinate system for dynamical state.

FusionMachine v1 therefore does not invent resident state or sufficient statistics. Its measured result is narrower: for one persistent recurrence, the mode's own forgetting constant determines how far back missed history matters when reconstructing a suspended computation.

## Modular recurrence is prior art

This fence matters directly for v3.

Koutník et al. (ICML 2014), **“A Clockwork RNN”**, partition a recurrent hidden layer into modules operating at different temporal granularities. PMLR 32(2):1863–1871; arXiv `1402.3511`.

Goyal et al. (ICLR 2021), **“Recurrent Independent Mechanisms”**, arXiv `1909.10893`, use multiple groups of recurrent cells with nearly independent transition dynamics and sparse attention-mediated communication. The stated motivation includes modular dynamics, specialization and improved generalization when underlying factors change out of distribution.

These systems are very close conceptual neighbors to “different computations remain resident.”

**Boundary:** FusionMachine cannot claim that modular recurrent mechanisms, independent dynamics, sparse communication, attention between mechanisms or OOD benefits are new. The v3 question is more specific: what happens when *current relevance* is withheld from the resident world dynamics and applied later at selection/publication?

## v3 and the state-versus-relevance distinction

The primary v3 attacker is intentionally strong in freedom: a four-state dense tanh RNN receives the same four state scalars, route budget, training episodes, optimizer updates and publication opportunities as FusionMachine, while having more non-routing trainable parameters.

The primary result strongly favors FusionMachine on abrupt switching and dormant-process readiness, but FusionMachine also fits the training queries substantially better. The primary comparison is therefore classified as confounded rather than as a clean architecture win.

The destructive controls are the more informative part:

- allowing context/relevance into FusionMachine's resident transition worsens abrupt-switch performance;
- removing context/relevance from the generic dense recurrence improves abrupt-switch performance dramatically and produces an almost complete linear representation of the hidden teacher world;
- making FusionMachine recurrence dense while continuing to exclude context preserves most of the readiness behavior.

This does not establish a universal theorem that context should stay out of recurrent state. It identifies a testable inductive bias for worlds where latent processes continue whether or not they are currently behaviorally relevant.

## Event-triggered communication and graph routing

Delta Networks and broader event-triggered/neuromorphic control establish that resident state need not be communicated continuously. Graph/message-passing systems establish that endpoint identity and topology can carry routing information independently of message amplitude.

FusionMachine's phrases **state is not traffic** and **address in the route** should therefore be read as architectural bookkeeping, not novelty claims.

## Neural population codes and the token analogy

Neuroscience generally describes sensory and cognitive information as distributed population activity, often with mixed selectivity and state dependence, rather than as discrete universal token IDs. Different stages and areas can nevertheless transform incoming signals into representations whose geometry makes particular downstream computations easier.

For FusionMachine this suggests using terms such as **local latent code**, **population state**, **event alphabet**, or **resident basis** when mapping the machine back toward biology. “Token” can be a useful analogy, but should not silently imply word-like discrete symbols.

## What the gates have actually established

```text
v0  early collapse can irreversibly destroy computational identity
v1  stateful computation must remain current or missed history must remain recoverable
v2  resident state, publication and routing can be causally separated
v3  in one frozen world, relevance-insulated recurrence is far more ready for abrupt relevance changes;
    the primary architecture comparison remains confounded by better Fusion training fit
```

The v3 compact wall sentence is:

> **State is not relevance.**

It is a hypothesis about role separation, not a claim that relevance never belongs in state. In worlds where goals/actions alter hidden dynamics, context may correctly be part of the state transition.

## Novelty standard

A useful eventual claim is **not**:

- dendrites are nonlinear;
- inhibitory cell classes target different compartments;
- experts or recurrent modules can be routed;
- multiplicative gating works;
- recursive state summarizes history;
- event-triggered communication saves traffic;
- topology carries address information;
- molecular identity and activity shape connectivity.

The potentially distinctive object is the tested composition:

> **Maintain multiple transformations as persistent world state; keep fast-changing relevance partly outside those dynamics; let relevance decide what becomes causal; let persistent routing structure determine where the event is written next.**

The next standard is stricter than v3: close the training-fit confound with loss-matched/optimization-matched checkpoints and stronger gated recurrent attackers before claiming a computational advantage.
