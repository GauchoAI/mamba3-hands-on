# Bootstrap Training Plan

## Core Principle

The human brain doesn't learn sorting by seeing 10,000 sorted lists. It
learns pattern recognition first, then comparison, then ordering, then
algorithms. Each level is built FROM the previous one. The model at Level N
becomes the foundation for Level N+1.

We have the hardware (M4 locally, H100 planned). The bottleneck is the
curriculum. This document defines the bootstrap.

---

## The Six Levels

### Level 0 — Pattern Recognition (the substrate)

**What:** Identify what repeats, what's the same, what's different, what
comes next. This is the most fundamental cognitive operation — everything
else is built on it.

**Why first:** The SSM's recurrent state naturally tracks repeating patterns.
This is what the architecture was literally designed for. Pattern recognition
is to the SSM what floating-point arithmetic is to a GPU — the native operation.

**Training data (code-generated, pure synthetic):**

```
Sequence completion:
  A B A B A B ?  →  A
  1 2 3 1 2 3 ?  →  1
  X X Y X X Y X X ?  →  Y

Same/different:
  3 3 → SAME
  3 5 → DIFF

Odd one out:
  2 2 2 7 2 → 7

Group membership:
  A: [1,3,5]  B: [2,4,6]  Input: 5 → A

Count occurrences:
  3 1 4 1 5 1 → count(1) = 3

Find the pattern rule:
  2 4 6 8 → rule: +2
  1 2 4 8 → rule: ×2
```

**Validation:** The model must score >95% on:
- Sequence completion (variable length, variable alphabet)
- Same/different classification
- Odd-one-out detection
- Pattern rule identification

**Data generator:** Pure Python. Random alphabets, random pattern lengths,
random repetition counts. Millions of examples trivially generated.

**Hardware:** M4 is sufficient. Small model (d_model=64, d_state=16).

---

### Level 1 — Comparison and Relation

**What:** Is A bigger than B? Is A equal to B? Is A a member of set S?
Order two items. These are the atomic operations that everything above
depends on.

**Prerequisite:** Level 0 (pattern recognition). The model must already
understand "same" and "different" to learn "bigger" and "smaller."

**Training data (code-generated):**

```
Comparison:
  7 3 CMP → GT       (7 > 3)
  2 8 CMP → LT       (2 < 8)
  5 5 CMP → EQ       (5 = 5)

Ordering pair:
  7 3 ORDER → 3 7
  2 8 ORDER → 2 8    (already ordered)

Min/Max of pair:
  7 3 MIN → 3
  7 3 MAX → 7

Between:
  5 BETWEEN 3 8 → YES
  2 BETWEEN 3 8 → NO
```

**Key design:** These operations work on ARBITRARY values, not just 0-9.
Use values up to 100, 1000, eventually unbounded. The model must learn
the RELATION, not memorize which of 10 values is bigger.

**Validation:** >95% on comparison with values never seen in training
(e.g., train on 0-100, test on 100-200). This proves genuine comparison,
not lookup table.

**Bootstrap from Level 0:** The checkpoint from Level 0 is the starting
point. We don't train from scratch.

**Hardware:** M4.

---

### Level 2 — Accumulation and Counting

**What:** Running sums, counting, parity (XOR accumulation), modular
arithmetic. The model tracks a value that evolves with each input.

**Prerequisite:** Level 1 (comparison). The model knows "bigger/smaller"
and can now learn "more/fewer" and "increasing/decreasing."

**Training data (code-generated):**

```
Counting:
  A B A C A → count(A) = 3

Running sum:
  3 1 4 → running: 3 4 8

Parity (already proven to work):
  1 0 1 1 0 → 1

Modular accumulation:
  2 1 2 → (mod 3): 2 0 2

Threshold detection:
  3 1 4 1 5 THRESHOLD 10 → position 4 (sum exceeds 10 at position 4)
```

**Bootstrap from Level 1:** The model already knows comparison. Now it
learns to track state across time — which is the SSM's core strength.

**Validation:** >95% on parity (any length up to 64), counting, running
sums. Length generalization: train on L≤16, test on L=32.

**Hardware:** M4.

---

### Level 3 — Composition (micro-operation chains)

**What:** Chain Level 1 and Level 2 operations into algorithms. This is
where sorting, searching, and reversing live — NOT as memorized patterns,
but as compositions of COMPARE + SWAP + ITERATE.

**Prerequisite:** Level 2 (accumulation). The model can track state and
compare values.

**Training data (code-generated execution traces):**

```
Bubble sort trace:
  INPUT 5 3 1
  CMP 0 1 → GT     (5 > 3)
  SWAP 0 1 → 3 5 1
  CMP 1 2 → GT     (5 > 1)
  SWAP 1 2 → 3 1 5
  CMP 0 1 → GT     (3 > 1)
  SWAP 0 1 → 1 3 5
  CMP 1 2 → LT     (3 < 5)
  DONE → 1 3 5

Linear search trace:
  INPUT 3 7 2 9 TARGET 7
  CMP arr[0] 7 → NEQ (3 ≠ 7)
  CMP arr[1] 7 → EQ  (7 = 7)
  FOUND at index 1

Insertion trace:
  SORTED: 1 4 7
  INSERT 5
  CMP 5 7 → LT     (5 < 7, shift 7 right)
  CMP 5 4 → GT     (5 > 4, insert here)
  RESULT: 1 4 5 7
```

**This is the Von Neumann level.** The training data is a sequence of
instructions (CMP, SWAP, FOUND, DONE) interleaved with data. The step
function processes both through the same recurrence. The instructions
program the state.

**Critical test:** Train on vocab=0-20, test on vocab=0-100. If sorting
still works, the model learned COMPARISON-BASED sorting, not counting sort.
This is the test that separates genuine algorithms from lookup tables.

**Bootstrap from Level 2:** The model already knows CMP and accumulation.
Now it learns to chain them. The code generator composes Level 1 operations
into traces.

**Code generator design:**
```python
def generate_bubble_sort_trace(arr):
    """Generate execution trace using Level 1 primitives."""
    trace = [("INPUT", *arr)]
    for i in range(len(arr)):
        for j in range(len(arr) - 1 - i):
            cmp_result = "GT" if arr[j] > arr[j+1] else "LT"
            trace.append(("CMP", j, j+1, cmp_result))
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
                trace.append(("SWAP", j, j+1, *arr))
    trace.append(("DONE", *arr))
    return trace
```

**Hardware:** M4 for small scale. H100 for scaling to longer sequences
and larger vocabularies.

---

### Level 4 — Recursion and Abstraction

**What:** Solve a problem by breaking it into smaller instances of itself.
Hanoi, merge sort, tree traversal, divide-and-conquer.

**Prerequisite:** Level 3 (composition). The model can execute chains of
operations. Now it learns that an operation can CALL ITSELF.

**Training data (code-generated recursive traces):**

```
Hanoi trace:
  HANOI 3 A C B
    RECURSE HANOI 2 A B C
      RECURSE HANOI 1 A C B
        MOVE 1 A→C
      RETURN
      MOVE 2 A→B
      RECURSE HANOI 1 C B A
        MOVE 1 C→B
      RETURN
    RETURN
    MOVE 3 A→C
    RECURSE HANOI 2 B C A
      RECURSE HANOI 1 B A C
        MOVE 1 B→A
      RETURN
      MOVE 2 B→C
      RECURSE HANOI 1 A C B
        MOVE 1 A→C
      RETURN
    RETURN
  DONE

Merge sort trace:
  MERGESORT [5, 2, 8, 1]
    SPLIT → [5, 2] [8, 1]
    RECURSE LEFT: MERGESORT [5, 2]
      SPLIT → [5] [8]
      MERGE [5] [2] → CMP 5 2 → GT → [2, 5]
    RECURSE RIGHT: MERGESORT [8, 1]
      SPLIT → [8] [1]
      MERGE [8] [1] → CMP 8 1 → GT → [1, 8]
    MERGE [2, 5] [1, 8]
      CMP 2 1 → GT → take 1
      CMP 2 8 → LT → take 2
      CMP 5 8 → LT → take 5
      take 8
      → [1, 2, 5, 8]
  DONE
```

**The stack is explicit in the trace.** RECURSE pushes, RETURN pops. The
model sees the stack operations happening and learns to track the call
depth in its state vector.

**Bootstrap from Level 3:** The model already knows CMP, SWAP, MERGE as
operations. Now it learns that these operations can be nested and called
recursively. The traces show the recursive structure explicitly.

**Validation:** Solve Hanoi-N for N not seen in training. If trained on
N=2,3,4 and it solves N=5, it learned recursion. If it fails, it
memorized specific move sequences.

**Hardware:** H100 likely needed. Recursive traces are long (Hanoi-5 = 31
moves + stack frames). The sequential scan becomes expensive.

---

### Level 5 — Symbolic Manipulation

**What:** Apply rewrite rules to expressions. Algebra, simplification,
equation solving, formal logic proofs. This is pattern recognition (Level 0)
applied to mathematical expressions, with computation (Levels 1-4) providing
the engine.

**Prerequisite:** Level 0 (pattern recognition — to match expression
patterns) + Level 4 (recursion — to apply rules to sub-expressions).

**Training data (generated by SymPy / custom CAS):**

```
Algebraic simplification:
  SIMPLIFY 3x + 2x
  MATCH: Ax + Bx where A=3, B=2
  RULE: combine_like_terms → (A+B)x
  COMPUTE: 3+2=5
  RESULT: 5x

Equation solving:
  SOLVE 2x + 3 = 7
  RULE: subtract_from_both_sides(3)
  COMPUTE: 7-3=4
  STATE: 2x = 4
  RULE: divide_both_sides(2)
  COMPUTE: 4/2=2
  STATE: x = 2
  RESULT: x = 2

Factoring:
  FACTOR x² + 5x + 6
  MATCH: x² + bx + c where b=5, c=6
  FIND: two numbers that multiply to 6 and add to 5
  SEARCH: 1×6=6, 1+6=7 ✗ | 2×3=6, 2+3=5 ✓
  RULE: (x + p)(x + q) where p=2, q=3
  RESULT: (x + 2)(x + 3)
```

**Bootstrap from Level 0 + Level 4:** Pattern matching (Level 0) finds
the rewrite rule to apply. Recursion (Level 4) applies rules to
sub-expressions. Computation (Levels 1-3) evaluates the arithmetic.

**Code generator:** SymPy can generate millions of algebraic manipulation
traces. We express each step as: MATCH pattern → APPLY rule → COMPUTE
arithmetic → RESULT.

**Hardware:** H100.

---

### Level 6 — Language as Interface

**What:** Natural language grounded in computation. The model can describe
what it's doing, explain why, and accept instructions in language.

**Prerequisite:** All previous levels. Language without computation is
poetry. Language with computation is reasoning.

**Training data:** The curriculum we already built (CURRICULUM.md), BUT
now the model has Levels 0-5 as a foundation. When it generates a
```thinking``` block, the "thinking" is grounded in actual operations it
can perform.

```
User: "Sort these numbers: 8, 3, 1, 5"

```thinking
I need to sort 4 numbers. I'll compare adjacent pairs and swap.
CMP 8 3 → GT, SWAP → 3 8 1 5
CMP 8 1 → GT, SWAP → 3 1 8 5
CMP 8 5 → GT, SWAP → 3 1 5 8
CMP 3 1 → GT, SWAP → 1 3 5 8
CMP 3 5 → LT, done
```

Sorted: 1, 3, 5, 8

```python
sorted([8, 3, 1, 5])  # [1, 3, 5, 8]
```
```

The thinking block is no longer "pretending to think." It's the model
actually executing its micro-operations and showing its work.

**Hardware:** H100 for the full model. M4 for prototyping.

---

## Implementation Plan

### Phase 1: Data Generators (do now, on M4)

Build Python code generators for each level's training data. Each generator
produces JSONL files that can be fed to the training harness.

```
generators/
├── level0_patterns.py       # sequence completion, same/diff, odd-one-out
├── level1_comparison.py     # CMP, ORDER, MIN, MAX, BETWEEN
├── level2_accumulation.py   # counting, running sum, parity, modular
├── level3_composition.py    # sorting traces, search traces, insertion
├── level4_recursion.py      # Hanoi traces, merge sort traces
├── level5_symbolic.py       # SymPy algebraic traces
└── common.py                # shared utilities, tokenization, formatting
```

Each generator:
- Takes parameters (difficulty, vocab range, sequence length)
- Produces JSONL with `input` and `output` fields
- Can be amplified by Cerebras for variety in phrasing (Levels 5-6)
- Includes a `--validate` mode that checks correctness of generated data

### Phase 2: Training Harness (do now, on M4)

Extend `mamba3_lm.py` or build fresh:
- **Checkpointing:** Save/load model + optimizer + scheduler + level info
- **Level-aware training:** Load data for current level only
- **Validation gate:** Before advancing to Level N+1, run Level N test suite
- **Config-driven:** YAML file specifies model size, current level, data paths

```yaml
model:
  d_model: 64
  d_state: 16
  n_layers: 1
  expand: 2

training:
  current_level: 0
  checkpoint: null           # or path to Level N-1 checkpoint
  steps_per_level: 5000
  validation_threshold: 0.95
  batch_size: 128
  lr: 3e-3

data:
  level_0: data/level0/
  level_1: data/level1/
  level_2: data/level2/
```

### Phase 3: Bootstrap Loop (execute level by level)

```
for level in 0, 1, 2, 3, 4, 5, 6:
    1. Generate data:  python generators/level{N}.py --count 10000
    2. If level > 0:   load checkpoint from level N-1
    3. Train:          python train.py --config bootstrap.yaml --level N
    4. Validate:       python validate.py --level N --threshold 0.95
    5. If validation passes: save checkpoint, proceed to N+1
    6. If validation fails:  generate more data, adjust hyperparams, retry
```

### Phase 4: Scale to H100 (later)

Once the bootstrap loop works on M4 at small scale:
- Scale model: d_model=256→1024, n_layers=1→8, d_state=16→64
- Scale data: 10K→1M examples per level
- Use Cerebras for data amplification at Levels 5-6
- Run on H100 with the official Mamba-3 CUDA kernels (chunked parallel scan)

---

## Execution Order (what to do right now)

1. **Build Level 0 data generator** (`generators/level0_patterns.py`)
2. **Generate 10K Level 0 examples**
3. **Train on Level 0**, validate to >95%
4. **Build Level 1 data generator**, generate 10K examples
5. **Train Level 1 from Level 0 checkpoint**, validate
6. Continue the loop

Start with Level 0. Everything else follows.

---

## Success Criteria

The bootstrap succeeds if:

1. **Level 3 sorts with unseen vocab.** Train on values 0-20, test on
   values 50-100. If accuracy >90%, the model learned comparison-based
   sorting, not a lookup table.

2. **Level 4 solves unseen Hanoi sizes.** Train on N=2,3,4. Test on N=5.
   If correct, the model learned recursion.

3. **Level 5 solves unseen equations.** Train on linear equations. Test
   on quadratics. If it applies the right rules, symbolic manipulation
   generalizes.

4. **Level 6 explains its work.** Given a problem in natural language,
   the model produces a thinking block that shows actual micro-operations,
   then the answer, then (optionally) code. The thinking is grounded in
   computation, not statistical text generation.

If all four hold, we have a system that genuinely reasons — not because
it memorized answers, but because it bootstrapped from pattern recognition
through six levels of increasingly powerful computation.
