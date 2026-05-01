---
title: Vision
chapter: Prologue
status: living
open_sections: 1
summary: "The north star behind the Lab: an ecology of small routed models, currently represented by {{lab.models}} model records and {{lab.streams}} archive streams."
---

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

## How the ecology actually grows (the empirical answer)

The intuitive picture was: a new specialist arrives, an existing
router learns to attend to it via a brief bridge fine-tune, no
retraining of the base weights required.

**The empirical answer is no.** Three configurations were tested:

| Setup | Δ acc |
|---|---|
| Solo router, post-hoc graft new bridge (frozen base) | 70.7 → 70.3 (no gain) |
| Solo router, graft + base fine-tune at lr=1e-4 | 67.2 → 69.1 (no gain) |
| Solo router with reserved placeholder slot, swap placeholder for real specialist (frozen base) | 70.3 → 70.7 (no gain) |
| Same but with base fine-tune at lr=1e-4 | 67.2 → 68.0 (no gain) |

In every regime the new bridge's gate closes during fine-tuning. The
base's frozen representation simply doesn't have the structure to be
*improved* by signal at the synapse layer; it has to learn that
structure during base training, which means it has to be there from
step 1.

So the ecology's growth pattern is closer to biological evolution
than runtime adaptation:

1. A new specialist emerges (someone trained a `XOR` solver, or a
   `chess_endgame` solver, or a `reverse_string` solver).
2. It registers itself by publishing weights to the shared substrate
   (here, Firebase teacher_blobs).
3. **Existing routers stay as-is** — they've already shaped their
   representations around the peers they were born with. Their
   competence on their original tasks remains.
4. **New routers** are trained from scratch with the expanded peer
   set available. Each new task spawns a fresh router that can
   include any subset of current peers as synapses, including
   peers added since the last router was built.

The cluster's reachable phenotype grows by *adding new routers*,
not by upgrading deployed ones. Old routers persist as legacy
phenotypes for their tasks; new routers compose new behaviors over
the union of all current specialists.

This is consistent with how brains develop: connection growth and
pruning happen during plastic phases, not on frozen circuits. A
mature region doesn't gain a new input by having one wired to it
post-hoc; a new region forms with the inputs it needs from the
start.

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
- Recursion: a saved router can be loaded and used as a specialist
  for a higher-order router. The hierarchy primitive holds —
  routers and specialists are interchangeable at runtime.
- **Distillation-via-synapse: closes the loop.** A synapse-router
  (e.g. 97% on compose_logic_gate via composition of a router + the
  logic_gate leaf) can be distilled into a *solo* ProgressiveModel
  that captures the composed capability in a single set of weights.
  The student matched the teacher (96% vs 97%). The student is then
  publishable as a new leaf specialist; future routers can compose
  over an expanded library that now includes it. The ecology
  literally compounds.
- Firebase as the substrate: nodes register, publish teacher blobs,
  download teacher blobs, all via plain HTTPS. No SSH required.
- Hardened save/load/teacher paths (NaN guards × 3) so corrupted
  models can't propagate through the cluster.
- 22 mastered specialists currently live, plus a brand-new
  compose_logic_gate specialist born via the distillation cycle.

## The growth cycle

1. **Compose.** A new task arrives. A small router is trained
   from scratch, with the current specialist library available
   as potential synapses. The router learns which peers to attend
   to and produces the task's first solver.
2. **Distill.** The router is distilled into a solo
   ProgressiveModel that captures the composed capability in a
   single set of weights. The router itself can be retired or
   kept around; what matters is the new solo .pt.
3. **Publish.** The solo .pt becomes a new specialist. It joins
   the Firebase library. Now any future router can use it as a
   leaf — no composition cost at runtime.
4. **Repeat.** The next task gets a fresh router. Its synapse
   options now include this newly-distilled specialist.
   Compositions get progressively deeper because each cycle
   adds new primitives that themselves contain prior
   compositions.

The cluster's reach grows by *layers of crystallization*. Each
synapse-composition is, in time, frozen into a leaf. The leaves
themselves had previously been compositions. Six layers down,
"it" is a tower of distillations.

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
