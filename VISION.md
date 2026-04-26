# Vision — an ever-evolving ecology of small models

## The thesis

We are not building a model. We are building an **ecology of small
models that route among themselves**, establish those routes over
time, and grow hierarchy organically. A given task is solved by
whichever subgraph of the ecology happens to fit it; new tasks recruit
new subgraphs; new subgraphs recruit new specialists; the system
expands by extending at the edges, not by adding mass at the center.

The goal is not a trained model and a release. The goal is a living
set of small models that becomes more capable over time as more peers
join, more synapses wire up, and more routes get reinforced.

## The primitives

Four operations make the ecology compose:

- **One-to-one**: a model invokes a single specialist via a synapse.
  This is what the AttendBridge experiments validated: a tiny router
  reaches into a frozen specialist and harvests its hidden state.
- **One-to-many**: a router invokes multiple specialists at once,
  each with its own gate, each contributing additively to the router's
  state. The dual_task experiment exercised this — limited not by the
  primitive but by specialist input-distribution sensitivity.
- **Many-to-many**: a graph of routers and specialists, each able to
  attend to several others. This is the cluster picture — Firebase as
  the connective tissue, every node both publishing competences and
  pulling them.
- **Recursion**: a *router* is itself just a model with a particular
  shape, so it can be used as the specialist for a higher-order
  router. A network can plug into another network that plugs into
  yet others. Hierarchy is whatever stabilizes by training, not
  something we predeclare.

These four together are sufficient: any computational arrangement we
might design by hand can be expressed as some routing graph among
small models. The interesting research question is which
arrangements *self-organize* under realistic training pressure.

## Why small models

A monolithic large model concentrates competence in one place.
Reaching new behaviors costs new parameters; updating one behavior
risks all the others. An ecology of small models concentrates
competence at many places. Reaching new behaviors costs a synapse
(one W_recv + gate, ~1.1k params on our scale) — far cheaper than
adding capacity to a monolith. Updating one specialist doesn't
disturb the others; their routes are decoupled.

The synapse experiments showed this empirically: at router d=16 the
synapse gives +30 points; at router d=32+ the router alone solves
the task and the synapse is near-redundant. The right interpretation
is that **synapses are the cheap way to extend small organisms** —
not the magic bullet for any size.

## Inference-time extension

This is the property the trained-model paradigm cannot offer. In a
trained-model world, capabilities ship in the weights. To add a
capability you re-train and re-release. In an ecology:

1. A new specialist emerges (someone trained a `XOR` solver, or a
   `chess_endgame` solver, or a `reverse_string` solver).
2. It registers itself by uploading its weights and capabilities to
   the shared substrate (here, Firebase).
3. Existing routers, on next invocation, can discover the new peer
   and decide — via a learned routing pass — whether to wire a
   synapse to it. No retraining of the routers' base weights is
   required for the *act of considering* the new peer; only the
   routing parameters need to update if the peer turns out to help.

The cluster's reachable phenotype grows with every new peer. Unlike
a monolithic model, no retraining cycle gates the expansion.

## Hierarchy is emergent, not declared

We don't designate "this model is a router, that model is a
specialist." A router is simply a model whose loss has been shaped
by the presence of synapses to other models. Some models will end up
serving mostly as routers (high gate activations into others, low
direct competence on raw input); others will end up as specialists
(low routing, high direct competence). Most will be a mix.

A hierarchy is whatever subgraph happens to perform well on a class
of tasks. As the population grows and tasks shift, the effective
hierarchy reshapes. Some models gravitate up, others down. The
cluster does the architectural search; we don't.

## What an experiment day looks like in this paradigm

Not "train model X to do task Y." Rather:

- A new task arrives. The cluster's existing routers each try a
  small training pass on it, with all current peers available as
  potential synapses.
- The router that lands the best loss (among many) is kept. Its
  open synapses define which existing peers contributed.
- Optionally, the new router is itself published as a peer for
  future tasks.
- Tasks that no current router can solve well even with synapses
  trigger a new specialist to be trained from scratch on the
  bottleneck sub-skill, then published.

Compute is spent where the cluster's reach is thinnest. Capabilities
accumulate. Releases don't happen.

## What's already standing

- Synapse v2 (AttendBridge): one-to-one and one-to-many primitives
  validated on simple tasks.
- Firebase as the substrate: nodes register, publish teacher blobs,
  download teacher blobs, all via plain HTTPS. No SSH required.
- Hardened save/load/teacher paths (NaN guards × 3) so corrupted
  models can't propagate through the cluster.
- 21 mastered specialists currently live, retrain in flight.

## What's next

In rough order of architectural reach:

- **Recursion test**: train a router on compose_logic_gate, then use
  *that router* as the specialist for a higher-order router on
  compose_logic_gate_3. If the higher-order router opens its gate
  into the lower router, the hierarchy primitive holds.
- **Token slicing**: per-specialist input slicing so multi-specialist
  composition isn't capped by frozen specialists' input-distribution
  brittleness. This unlocks routers that compose specialists that
  weren't co-designed.
- **Inference-time hot-plug**: a router trained with K specialists
  must be able to attend to a K+1th specialist that arrives
  post-training, without retraining the base. Tests the "ever-
  evolving set" property in its strongest form.
- **Self-organized peer discovery**: routers query Firebase at
  training start to find their available specialists, rather than
  having paths hardcoded by a human. Closes the loop on autonomous
  cluster growth.
- **Distillation from the synapse-router into a solo model**: if
  this works, synapses become a *teaching* primitive — the cluster
  composes a capability, then distills it into a single specialist
  that absorbs the composition. The library compounds.
