# Mamba-3 Training Curriculum

A reasoning-first approach to training a bilingual (EN/ES) language model
from scratch using the Mamba-3 architecture.

**Core principle:** Logic first, language second. Fluency without reasoning
is autocomplete. We build from formal logic upward.

---

## Phase 1 — Pure Logic (synthetic, no natural language)

The foundation. These tasks require no language understanding — just state
tracking, recursion, and rule application. We already proved Mamba-3 can
do this (parity, selective copy, mod counting). Now we formalize it.

### 1.1 Tower of Hanoi
- Input: number of disks, initial state
- Output: complete move sequence
- Difficulty: 2 → 3 → 4 → ... disks
- Tests: recursion, planning, state tracking across many steps

### 1.2 Propositional Logic
- Modus ponens: `A → B, A ∴ B`
- Chain reasoning: `A → B, B → C, A ∴ C`
- Contrapositive: `A → B, ¬B ∴ ¬A`
- Disjunctive syllogism: `A ∨ B, ¬A ∴ B`
- Tests: rule application, negation handling

### 1.3 Variable Tracing
- Simple programs: `x=3; y=x+1; z=x*y; what is z?`
- Conditionals: `if x > 5 then y=1 else y=0`
- Loops: `x=0; repeat 3 times: x=x+2; what is x?`
- Tests: sequential state updates (directly maps to SSM state)

### 1.4 Set and Sequence Operations
- Set membership, union, intersection
- Sequence patterns: "what comes next: 2, 4, 8, ?"
- Sorting: given a list, produce sorted output
- Tests: structured reasoning over collections

---

## Phase 2 — Logic Meets Language

Bridge formal logic into natural language. Bilingual from this phase onward.

### 2.1 Syllogisms (EN + ES)
- Classical: "All men are mortal. Socrates is a man. Therefore, Socrates is mortal."
- "Todos los hombres son mortales. Sócrates es un hombre. Por lo tanto, Sócrates es mortal."
- Invalid syllogisms (model must identify the error)
- Tests: logical reasoning expressed in natural language

### 2.2 Greek Philosophy / Logical Argumentation
- Aristotelian logic applied to everyday statements
- Identifying premises and conclusions
- Distinguishing valid from invalid arguments
- Tests: natural language understanding in service of logic

### 2.3 Logic Puzzles
- Einstein/Zebra puzzles: "The baker lives next to the teacher..."
- Knights and Knaves: "A says B is a liar..."
- Constraint satisfaction in prose
- Tests: multi-step deduction from natural language premises

### 2.4 Mathematical Word Problems
- Arithmetic in context: "María tiene 5 manzanas, le da 2 a Juan..."
- Rate/distance/time problems
- Tests: extracting formal operations from natural language

---

## Phase 3 — Comprehension and Compression

Prove the model can process, retain, and compress information.

### 3.1 Summarization
- Paragraph → one-sentence summary
- Multi-paragraph → bullet points
- Bilingual: summarize EN text in ES and vice versa
- Tests: selective retention (maps to selective copy primitive)

### 3.2 Question Answering
- Extractive: answer is in the passage
- Inferential: answer requires combining facts
- Tests: retrieval + reasoning

### 3.3 Code Understanding
- "What does this function return?"
- "Find the bug in this code"
- Tests: variable tracing at scale (Phase 1.3 in real code)

---

## Phase 4 — Structured Output and Self-Awareness

Teach the model to produce structured, actionable output.

### 4.1 Tool Calling
- Format: `<tool name="search">{"query": "..."}</tool>`
- The model learns when and how to invoke tools
- Tests: structured output generation, knowing its own limits

### 4.2 Markdown Generation
- Tables, lists, headers, code blocks
- Tests: format awareness, structural output

### 4.3 Self-Annotation / Memory
- `<thinking>...</thinking>` blocks for chain-of-thought
- `<memory>The user mentioned X, which relates to Y</memory>`
- The model annotates what it's tracking and why
- Tests: metacognition, explicit state management

---

## Data Strategy

**Seed + Amplify workflow:**
1. We craft ~10 high-quality seed examples per lesson
2. Cerebras (500 tok/s) amplifies each seed set to ~1000 examples
3. Store as parquet shards on TB4 (4TB available)
4. Training harness mixes shards by phase/lesson with configurable weights

**Tokenization:** Byte-level (proven to work with Mamba-3) with option to
add BPE later for comparison.

**Checkpointing:** Save every N steps. Resume with different data mix.
Add new phases/lessons without retraining from scratch.

---

## Evaluation

Each phase has its own eval suite:

| Phase | Eval | Pass criteria |
|---|---|---|
| 1.1 | Hanoi-3 solve rate | >95% |
| 1.2 | Propositional logic accuracy | >95% |
| 1.3 | Variable trace accuracy | >95% |
| 2.1 | Syllogism validity classification | >90% |
| 2.4 | Word problem accuracy | >80% |
| 3.1 | ROUGE-L on summaries | >0.3 |
| 4.1 | Tool call format correctness | >95% |

---

## Infrastructure

- **Local (M4 Mac mini):** iteration, seed generation, small-scale training
- **TB4 SSD (4TB):** datasets in parquet, checkpoints
- **Cerebras:** data amplification (500 tok/s)
- **H100 (planned):** full-scale training runs
