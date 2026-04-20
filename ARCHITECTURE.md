# The Two-Brain Problem: Language × Computation in Mamba-3

## The Problem

We have demonstrated two separate capabilities of Mamba-3:

**Brain 1 (Language/Orator):** A character-level bilingual model that generates
plausible English and Spanish text. Trained on Tatoeba sentences. The model
learns to produce tokens autoregressively — each generated token is a "thought"
made visible. This is what all LLMs do. Like a Greek philosopher who reasons
by speaking aloud.

**Brain 2 (Computation/Mathematician):** The parity experiment. The SSM's
recurrent state learned to track XOR accumulation via a phase rotation of
exactly -π. No tokens generated. No "thinking out loud." The answer emerged
from 16 steps of silent state evolution. Like a mathematician doing algebra
on a blackboard in their head.

**The gap:** Brain 1 can *describe* algorithms ("sort by comparing adjacent
pairs...") but can't *execute* them beyond what fits in its generation budget.
Brain 2 can *execute* algorithms silently and exactly, but only the one
algorithm that was baked into its weights during training (parity).

**The question:** How do we make Brain 2 general-purpose? How does language
(Brain 1) tell the SSM (Brain 2) *what* to compute at inference time?

---

## The FPGA Analogy

The SSM is best understood as a **Field-Programmable Gate Array** (FPGA):

| FPGA concept | SSM equivalent |
|---|---|
| Flip-flops (memory elements) | State vector `h` — (B, H, headdim, d_state) |
| Look-up tables (configurable logic) | Data-dependent gates: A (decay), B (write), C (read), angles (rotation) |
| Configuration bitstream | The token sequence after in_proj — each token "programs" what the state does at that timestep |
| Clock cycle | One step of the recurrent scan |
| Routing fabric | Projection matrices (in_proj, out_proj) that map between token space and state space |
| Circuit output | The readout C·h at each step |

### Why this analogy is precise

In an FPGA, the silicon is fixed. What changes is the **bitstream** — a
binary configuration that tells each logic cell what function to compute,
and how cells connect to each other. The same chip can be a video encoder,
a network switch, or a cryptocurrency miner depending on the bitstream.

In Mamba-3, the architecture is fixed. What changes is the **input sequence**.
Each token, after passing through `in_proj`, produces:
- `A` and `DT`: how much state decays (the "clock" of the circuit)
- `B` and `x`: what gets written into state (the "data input")
- `C`: what gets read from state (the "data output")  
- `angles`: how the state rotates (the "phase logic")
- `trap`: how current and previous inputs blend (the "pipeline register")

**Every token is an instruction that programs the state machine for one cycle.**

### What parity proved

Training on parity was like **synthesizing a 1-bit counter circuit** and
loading it onto the FPGA. The weights learned a fixed bitstream:

```
When input = 1: rotate phase component 0 by -π (flip the counter)
When input = 0: rotate by ~0 (no-op)
```

This circuit is hardcoded in the weights. It runs at inference time but
only computes parity. We didn't build a general FPGA — we built an ASIC
(application-specific integrated circuit).

### What we want

A **general FPGA** where:
1. The language model (Brain 1) acts as the **synthesis tool** — it reads a
   problem description and generates a "bitstream" (sequence of control tokens)
2. The SSM layers (Brain 2) act as the **programmable fabric** — they execute
   the bitstream, evolving state according to the programmed logic
3. The result is read out from the final state

---

## The Training Challenge

### Stage 1: Teach the FPGA its primitives

Train the SSM on multiple raw algorithmic tasks **simultaneously**:
- Parity (XOR accumulation)
- Counting (mod-N accumulation)
- Sorting (comparison + swap)
- Stack operations (push/pop)
- Sequence reversal
- Pattern matching

**No language.** Raw input → raw output. The SSM must learn to multiplex
its state vector to support all these operations at once, discriminated
by the input pattern itself.

**Key experiment:** Can one model hold multiple algorithms without
destructive interference? If parity lives in phase component 0 and
counting lives in component 3, they can coexist. If they fight for the
same components, we need more state dimensions.

### Stage 2: Teach composition

Train on tasks that require **chaining** primitives:
- "Count to N, then check parity of the count"
- "Sort, then find the median"
- "Push items onto a stack, then pop in reverse"

This is like FPGA **routing** — connecting the output of one sub-circuit
to the input of another. The SSM must learn to pipeline state through
multiple operations.

### Stage 3: Teach compilation (the bridge)

Train on tasks where the input is a **natural language description** and
the output requires **silent computation**:
- "What is the parity of: 1 0 1 1 0 1 1 0 1 0?" → 0
  (Brain 1 parses the language, Brain 2 computes the parity)
- "Sort these numbers: 5 2 8 1" → 1 2 5 8
  (Brain 1 identifies the task, Brain 2 executes sorting in state)
- "Hanoi with 3 disks" → sequence of moves
  (Brain 1 sets up the problem, Brain 2 executes the recursive algorithm)

The training signal teaches the model: for this class of problem, don't
generate step-by-step tokens — trust your internal state to compute the
answer.

### Stage 4: Self-aware computation

The model learns to **decide** when to think aloud (Brain 1) vs compute
silently (Brain 2):
- "Explain how parity works" → Brain 1 (generate text)
- "Compute the parity of this 100-element sequence" → Brain 2 (silent)
- "Solve Hanoi with 3 disks and explain each step" → Brain 2 computes,
  Brain 1 narrates

---

## Open Questions

1. **State capacity.** The SSM state is a fixed-size continuous vector.
   How many simultaneous "circuits" can it hold? Is `d_state=16` enough
   for parity + sorting + counting, or do we need to scale?

2. **Interference.** When we train parity and sorting together, do the
   learned programs interfere? Does parity accuracy drop when sorting
   is added? This is the FPGA "routing congestion" problem.

3. **Unbounded computation.** A real FPGA can run indefinitely. Our SSM
   has a fixed sequence length. Can the model learn to "checkpoint" its
   state and continue across multiple forward passes? This is the
   difference between a combinational circuit and a sequential one.

4. **Programmability depth.** In the parity experiment, the "program"
   was trivial: one rotation per input bit. Hanoi requires recursion
   (a stack). Can the SSM state represent a stack? How deep?

5. **The compilation gap.** Even if Brain 2 can execute multiple
   algorithms, how does Brain 1 learn to select and configure the right
   one? This is the hardest question — it's essentially program synthesis.

---

## Experimental Plan

### Experiment A: Multi-task interference test
Train one Mamba-3 model on parity + sorting + counting simultaneously
(raw sequences, no language). Measure accuracy on each task individually.
Compare to single-task baselines.

### Experiment B: State capacity scaling
Run Experiment A with d_state = 8, 16, 32, 64. Plot task accuracy vs
state dimension. Find the minimum state size that holds all tasks without
interference.

### Experiment C: Composition
Train on chained tasks (count-then-parity). Does the model generalize to
chains it hasn't seen?

### Experiment D: The bridge
Train a model on mixed data: raw algorithmic tasks + natural language
descriptions that require computation. Does the model learn when to
compute silently vs when to generate reasoning tokens?

---

## Connection to the Curriculum

This architecture document describes the **internal** training — what
happens inside the SSM. The CURRICULUM.md describes the **external**
training — the language, reasoning, and structured output capabilities.

The full model needs both:
- CURRICULUM.md phases 1-4 train Brain 1 (the orator, the philosopher)
- This document's stages 1-4 train Brain 2 (the mathematician, the FPGA)
- The final model is the synthesis: a system that can both reason in
  language AND compute silently, choosing the right mode for each problem.
