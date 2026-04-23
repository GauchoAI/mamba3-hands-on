# Phase 1→4 Plan: Fast Specialists → Distillation → Generalization → Formal Math

## Overview

```
Phase 1: Fast Specialists          Phase 2: Distillation
┌──────────────────────┐          ┌──────────────────────┐
│ 15 tiny models        │          │ 1 byte-level student  │
│ optimal repr per task │───────→  │ learns from all 15    │
│ minutes to mastery    │ dark     │ byte parsing + reason │
│ raw bits/nums/bools   │ knowledge│ progressive growing   │
└──────────────────────┘          └──────────┬───────────┘
                                             │
Phase 4: Formal Math              Phase 3: Generalization
┌──────────────────────┐          ┌──────────────────────┐
│ truth tables → symbols│          │ 18 boss tasks         │
│ XOR(a,b) = (a+b)%2   │←─────── │ zero-shot + few-shot  │
│ grounded notation     │ if pass  │ iterate if fail       │
└──────────────────────┘          └──────────────────────┘
```

---

## Phase 1: Fast Specialists

### Goal
Master all 15 tasks with the smallest possible model and fewest
possible steps, by giving each task its optimal input representation.

### How it differs from current approach
Currently: ALL tasks use byte encoding (vocab=260, string input).
Parity sees `"0 1 1 0 1"` as 9 ASCII tokens. 60,000 steps to 63%.

New: Each task gets the representation that makes it TRIVIAL for the
SSM. Parity sees `[0, 1, 1, 0, 1]` as 5 raw bits. 400 steps to 100%.

### Task representations

**Binary tasks (vocab=2):**
- parity: input=bit sequence, output=0/1 (even/odd)
- binary_pattern_next: input=bit sequence, output=0/1 (next bit)
- mirror_detection: input=bit sequence, output=0/1 (mirror/not)
- logic_gate: input=2 bits + gate type, output=0/1

**Comparison tasks (vocab=small numbers):**
- same_different: input=2 numbers, output=0/1 (same/diff)
- odd_one_out: input=N numbers, output=index of odd one

**Sequence tasks (vocab=numbers):**
- sequence_completion: input=number sequence, output=next number
- pattern_period: input=number sequence, output=period length
- run_length_next: input=number sequence, output=next value
- repeat_count: input=sequence, output=count
- arithmetic_next: input=number sequence, output=next
- geometric_next: input=number sequence, output=next
- alternating_next: input=number sequence, output=next

**Logic tasks (vocab=propositions):**
- logic_chain: input=chain of implications, output=0/1
- modus_ponens: input=premise + rule, output=0/1

### Architecture per task type

```python
# Binary tasks: tiny, 1 layer, vocab=2
{"vocab_in": 2, "vocab_out": 2, "d_model": 32, "layers": 1, "d_state": 16}
# Expected: 8K params, <500 steps

# Comparison tasks: small, 1-2 layers, vocab=max_number
{"vocab_in": 21, "vocab_out": 2, "d_model": 32, "layers": 1, "d_state": 16}
# Expected: 10K params, <1000 steps

# Sequence tasks: medium, 2-3 layers, vocab=max_number
{"vocab_in": 100, "vocab_out": 100, "d_model": 48, "layers": 2, "d_state": 16}
# Expected: 30K params, <2000 steps

# Logic tasks: small, 1 layer, vocab=propositions
{"vocab_in": 10, "vocab_out": 2, "d_model": 32, "layers": 1, "d_state": 16}
# Expected: 8K params, <500 steps
```

### What we build

**`task_model_spec.py`** — declarative spec per task:
```python
TASK_SPECS = {
    "parity": {
        "vocab_in": 2, "vocab_out": 2,
        "d_model": 32, "layers": 1, "d_state": 16,
        "make_batch": make_parity_batch,  # returns (input_tensor, target_tensor)
        "max_steps": 1000,
        "target_acc": 0.99,
    },
    "same_different": {
        "vocab_in": 21, "vocab_out": 2,
        "d_model": 32, "layers": 1, "d_state": 16,
        "make_batch": make_same_different_batch,
        "max_steps": 2000,
        "target_acc": 0.99,
    },
    # ... all 15
}
```

**`fast_specialist.py`** — generic trainer:
```python
def train_fast(task_name, spec, device):
    model = build_model(spec)  # from declarative spec
    for step in range(spec["max_steps"]):
        inputs, targets = spec["make_batch"](batch=64, device=device)
        loss = F.cross_entropy(model(inputs), targets)
        # ... optimize ...
        if eval_acc >= spec["target_acc"]:
            save_specialist(task_name, model, spec)
            return model
    return model
```

**`batch_generators.py`** — raw tensor generators per task:
```python
def make_parity_batch(batch, L=16, device="cuda"):
    bits = torch.randint(0, 2, (batch, L), device=device)
    parity = torch.cumsum(bits, dim=1) % 2
    return bits, parity

def make_same_different_batch(batch, max_val=20, device="cuda"):
    a = torch.randint(0, max_val+1, (batch,), device=device)
    b = torch.randint(0, max_val+1, (batch,), device=device)
    # 50% chance same
    same_mask = torch.rand(batch, device=device) < 0.5
    b[same_mask] = a[same_mask]
    inputs = torch.stack([a, b], dim=1)  # (B, 2)
    targets = (a != b).long()  # 0=same, 1=different
    return inputs, targets
```

### Integration with existing system
- Specialists save to `checkpoints/specialists/{task}_fast.pt`
- Each checkpoint includes the spec (for reproducibility)
- Specialists register in the StateDB as teachers
- The existing byte-level specialists continue running in parallel
- When a fast specialist masters a task, it becomes available as a
  teacher for distillation immediately

### GA evolution
The GA can mutate the spec: d_model, layers, d_state, batch_size, lr.
But NOT the representation — that's fixed per task (the optimal one).
This is a much smaller search space than before.

---

## Phase 2: Distillation

### Goal
Train one byte-level student (vocab=260) that learns all 15 tasks
from the fast specialists' dark knowledge.

### Why byte-level
The student MUST use bytes because:
- Bytes are compatible with natural language (Phase 5)
- Bytes are compatible with formal math notation (Phase 4)
- Bytes are the universal representation — any data can be bytes
- The student learns byte parsing as a SIDE EFFECT of distillation

### How distillation works across representations

The teachers speak different languages. The student speaks bytes.
Translation is at the ANSWER level:

```
Parity teacher (vocab=2):
  input: [0, 1, 1, 0, 1] → output distribution: [0.96, 0.04]
  meaning: 96% chance "odd"

Student (vocab=260):
  input: "0 1 1 0 1" (bytes) → must output "D" (byte 68)
  teacher signal: "the answer is odd with 96% confidence"
  
Translation: teacher's class 1 ("odd") maps to student's byte 68 ("D")
```

### Architecture

```python
student = ProgressiveModel(d_model=64, d_state=16)
student.add_kernel_layer()  # starts small
student.add_kernel_layer()
student.add_kernel_layer()
student.set_mode("kernel")
```

The existing `ProgressiveModel` with byte tokenization. Same as the
current specialists, but trained via distillation instead of from scratch.

### Training loop

```python
for cycle in range(max_cycles):
    for task in ALL_TASKS:
        # Generate examples in BOTH representations
        raw_input, raw_target = task_spec["make_batch"](batch=64)
        byte_input, byte_sep = tokenize_for_student(task, raw_input)
        
        # Get teacher's output distribution (in teacher's vocab)
        with torch.no_grad():
            teacher_logits = teacher_models[task](raw_input)
        
        # Get student's output distribution (in byte vocab)
        student_logits = student(byte_input)
        
        # Distillation loss: translate teacher → student answer space
        # teacher says "class 1 with 96% conf" → student should say "D" with 96% conf
        soft_targets = translate_distribution(teacher_logits, task)
        loss = kl_div(student_logits[output_positions], soft_targets)
        
        # Also hard label loss for grounding
        hard_loss = cross_entropy(student_logits[output_positions], byte_targets)
        
        total_loss = 0.7 * loss + 0.3 * hard_loss
```

### Translation layer per task

```python
ANSWER_MAPS = {
    "parity": {0: ord("S"), 1: ord("D")},           # even→S, odd→D
    "same_different": {0: ord("S"), 1: ord("D")},    # same→S, diff→D
    "logic_gate": {0: ord("0"), 1: ord("1")},        # false→0, true→1
    "mirror_detection": {0: ord("N"), 1: ord("M")},  # no→N, mirror→M
    # For number outputs: teacher's class N → byte sequence for str(N)
    "arithmetic_next": "number_to_bytes",
    "sequence_completion": "number_to_bytes",
}
```

### What makes this faster than training from scratch
- The student doesn't discover the answers — teachers provide them
- The student's B matrix learns byte parsing while getting correct
  answers (simultaneous learning, not sequential)
- The teachers' confidence on edge cases (dark knowledge) gives the
  student richer gradients than hard labels
- PCGrad resolves multi-task gradient conflicts

### Progressive curriculum
1. Start with the easiest teachers (parity, logic_gate — binary answers)
2. Add comparison teachers (same_different, mirror_detection)
3. Add sequence teachers (arithmetic_next, pattern_period)
4. Add logic teachers (modus_ponens, logic_chain)

Each stage adds more complexity. The student builds byte parsing
skills incrementally.

### Success criteria
- Student achieves >90% on all 15 tasks using byte encoding
- Student trains faster than the current byte-level specialists
  (target: 10x faster, since teachers provide the answers)

---

## Phase 3: Generalization Test

### Goal
Test whether the student has learned REASONING or just MEMORIZED
15 specific tasks. This determines if we proceed or iterate.

### Test suite: 18 boss tasks (already defined)

From `generators/boss_tasks.py`:
- set_union, set_intersection, set_difference
- sort_ascending, sort_descending
- find_min, find_max, find_second_largest
- sum_sequence, product_sequence
- modular_arithmetic
- reverse_sequence, rotate_sequence
- deduplicate
- xor_sequence
- count_unique
- majority_element
- median

### Test protocol

**Zero-shot (no training):**
Present boss task examples to the student. Can it solve them
without ANY training on these tasks?
- If >50% accuracy: strong generalization signal
- If >30%: some transfer, worth investigating what transfers
- If <10%: no generalization, the student memorized

**Few-shot (10 examples, then test):**
Give the student 10 examples of a boss task, train for 50 steps,
then test on fresh examples.
- If >80% accuracy: fast learning = good primitives
- Compare to: untrained model (no specialist knowledge)
- The delta tells us how much the specialist knowledge helps

**Transfer analysis:**
For each boss task, measure: which of the 15 specialist skills
correlates with success? If the student solves sort_ascending
and sort_ascending correlates with sequence_completion skill,
we know WHY it transferred.

### Decision tree

```
Zero-shot > 50% on most boss tasks?
├─ YES → Proceed to Phase 4 (formal math)
│
└─ NO → Which boss tasks fail?
         ├─ Identify missing reasoning primitives
         ├─ Design new specialist tasks that teach those primitives
         ├─ Loop back to Phase 1 (add tasks, train specialists)
         └─ Distill again (Phase 2)
         └─ Test again (Phase 3)
```

### What we build

**`boss_eval.py`** — evaluator for all 18 boss tasks:
```python
def evaluate_boss_tasks(student_model, mode="zero_shot"):
    results = {}
    for task_name, generator in BOSS_TASKS.items():
        if mode == "zero_shot":
            acc = eval_zero_shot(student_model, generator)
        elif mode == "few_shot":
            acc = eval_few_shot(student_model, generator, n_examples=10)
        results[task_name] = acc
    return results
```

**`transfer_analysis.py`** — correlation between specialist skills
and boss task performance:
```python
def analyze_transfer(student_model, specialist_accs, boss_accs):
    # For each boss task, which specialist skills predict success?
    correlations = {}
    for boss_task in boss_accs:
        for specialist_task in specialist_accs:
            corr = compute_correlation(specialist_accs[specialist_task],
                                       boss_accs[boss_task])
            correlations[(boss_task, specialist_task)] = corr
    return correlations
```

### Success criteria
- >50% zero-shot on at least 10/18 boss tasks
- >80% few-shot on at least 14/18 boss tasks
- Clear transfer patterns (specialist→boss correlations)

---

## Phase 4: Formal Math

### Goal
Teach the student to express its reasoning in mathematical notation.
Ground formal symbols in actual computation.

### Why this matters
The student can compute `[1,0,1,1] → odd`. But it doesn't know
WHY. Teaching it `XOR(1, XOR(0, XOR(1,1))) = 1 → odd` connects
the computation to a symbolic framework. Now the model can:
- Verify its own reasoning ("let me check: XOR step by step...")
- Generalize to new expressions ("what about AND(1, OR(0,1))?")
- Communicate its reasoning in a language humans understand

### Curriculum

**Level 1: Truth tables → notation**
```
Input:  [1, 0] → 1     paired with    "XOR(1, 0) = 1"
Input:  [1, 1] → 0     paired with    "XOR(1, 1) = 0"
Input:  [0, 0] → 0     paired with    "XOR(0, 0) = 0"
```
The model already knows the LEFT side (from parity training).
It learns the RIGHT side is a different way of saying the same thing.

**Level 2: Composition**
```
"XOR(1, XOR(0, 1)) = XOR(1, 1) = 0"
"AND(1, OR(0, 1)) = AND(1, 1) = 1"
```
Multi-step symbolic evaluation. The model traces through the
expression, applying operations it already knows.

**Level 3: Variables and quantifiers**
```
"∀ a,b: XOR(a,b) = (a + b) mod 2"
"parity(s) = reduce(XOR, s)"
```
Abstract reasoning about operations. The model connects concrete
computations to universal rules.

**Level 4: Proofs**
```
"Theorem: parity(s ++ [1]) = NOT(parity(s))"
"Proof: parity(s ++ [1]) = XOR(parity(s), 1) = NOT(parity(s)) ∎"
```
The model generates valid proofs grounded in its computational
experience. Every step is backed by an operation it can actually
perform.

### What we build

**`math_generators.py`** — generates paired examples:
```python
def gen_math_parity():
    bits = [random.randint(0,1) for _ in range(random.randint(2,6))]
    # Concrete
    concrete = f"parity({','.join(str(b) for b in bits)}) = {'even' if sum(bits)%2==0 else 'odd'}"
    # Symbolic
    expr = str(bits[0])
    for b in bits[1:]:
        expr = f"XOR({expr}, {b})"
    result = reduce(lambda a,b: a^b, bits)
    symbolic = f"{expr} = {result}"
    return {"input": concrete, "output": symbolic}
```

**Training approach:**
- Use the SAME student model from Phase 2
- Add cortex layers (language capability) while freezing kernel layers
- The kernel computes; the cortex expresses
- Train on paired (concrete answer, symbolic expression) examples

### Success criteria
- Student can evaluate symbolic expressions it hasn't seen before
- Student can generate step-by-step traces for multi-step operations
- Student's symbolic outputs are CORRECT (verified by SymPy)
- Student can handle novel compositions of known operations

---

## Project structure

```
mamba3-hands-on/
├── phase1/
│   ├── task_model_spec.py      # declarative specs per task
│   ├── batch_generators.py     # raw tensor generators
│   ├── fast_specialist.py      # generic fast trainer
│   └── train_all.py            # train all 15 fast specialists
├── phase2/
│   ├── distill_student.py      # byte-level student distillation
│   ├── translation.py          # teacher→student answer mapping
│   └── progressive_curriculum.py
├── phase3/
│   ├── boss_eval.py            # 18 boss task evaluator
│   ├── transfer_analysis.py    # which skills transfer to what
│   └── iterate.py              # loop back logic
├── phase4/
│   ├── math_generators.py      # paired concrete→symbolic examples
│   ├── math_curriculum.py      # truth tables → composition → proofs
│   └── verify.py               # SymPy verification of outputs
├── state/                      # SQLite DB, checkpoints (sacred)
├── docs/                       # plans, findings, vision
└── three_populations.py        # orchestrator (stateless)
```

## Integration with existing infrastructure

Everything we built today carries forward:
- **StateDB** — tracks specialists, lineage, diagnostics for ALL phases
- **Firebase push** — workers push progress for ALL phases
- **Diagnostician** — detects plateaus in ANY phase's training
- **Champion-challenger** — protects progress in ANY phase
- **Register inspector** — observes internals in ANY phase
- **GA evolution** — evolves configs in Phase 1 (spec mutation)

The orchestrator becomes phase-aware:
```python
phase = db.get_config("current_phase", 1)
if phase == 1:
    train_fast_specialists()
elif phase == 2:
    distill_student()
elif phase == 3:
    evaluate_boss_tasks()
elif phase == 4:
    train_formal_math()
```

Phase transitions stored in DB. No state lost between phases.
Hot-reload: change phase via `sqlite3 training.db "UPDATE runtime_config
SET value='2' WHERE key='current_phase'"` — no restart.
