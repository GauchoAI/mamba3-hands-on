# Tower of Hanoi reasoning arc — findings journal

A focused journal of all the entries about teaching small Mamba-3 models
to do Tower of Hanoi. Originally lived inline in the root `findings.md`;
moved here as part of the structuring pass (2026-04-30) so the root file
stays an index, not a multi-thousand-line document.

The arc spans the `n=40 cliff` discovery → debugging it as a decoder
bug → ruling out output-decoder hypothesis → ruling out rightmost-byte
attention → discovering it's a *bounded counter*, not a missing
algorithm → the GRU fix → Hanoi step function (1574-param MLP →
byte-perfect at n=12 / 4095 moves) → Hanoi-exec (model executes the
algorithm via register bank).

All entries below are verbatim relocations from the root file. Same
content, same dates, same numbering.

---

## Entry — Hanoi true invariance: the role-MLP plateau and the GRU fix (2026-04-28)

User question that opened the thread:

> "what is that 'we never trained for' that seems like we are not
>  correctly setting up invariants, again an index problem?"

The lab notebook for this session: we'd just shipped a mixed-K MLP
ensemble that hit **100% on n=16, 17 held-out** (196,606 canonical states)
trained on n=2..15, plus a saved checkpoint and a composite-task demo
(`hanoi_solve.py`, `hanoi_composite_demo.py`). We reported that as
"true 100%" and pivoted toward solver-mode at higher n. The user pushed
back: if the encoding were truly invariant, n shouldn't matter at all.

This entry is the diagnosis chain that vindicated their pushback and
ended with a 45,318-param GRU that gets 100% on canonical traces from
n=15 (training) up to n=23 — 8.4 million states — and, with off-trace
augmentation, solves arbitrary reachable starts at n=5..18 OPTIMALLY in
90 seconds wall-clock.

**1. Probe the invariance claim instead of speculating**

Stream-eval the mixed-K ensemble (`probe_invariance.py`) on canonical
traces in 64k chunks (the user's OS nearly OOM'd when an earlier version
materialized the full n=23 trace — 8.4M states × 24 int64 + 7-model
feature replication; saved as a feedback memory):

```
n=15 (trained):  100.0000%
n=16 (held-out): 100.0000%
n=17 (held-out): 100.0000%
n=18:             99.9989%   (   3 / 262k wrong)
n=19:             99.9424%   ( 302 / 524k wrong)
n=20:             99.7671%   (2442 / 1M wrong)
```

The "100%" reported earlier was a coincidence of the small held-out
range (n=16, 17). Accuracy decays cleanly with n — even on canonical-
trace states, not solver-mode where 1 wrong move snowballs.

**2. The features are invariant. The MLP isn't.**

Looking at `role_features_K`:
- `smallest[i]` = peg of disk i (i-th smallest) — same role at every n.
- `largest[k]` = peg of disk n_disks-1-k — same role at every n.
- `parity = n_disks % 2` — 0/1, invariant in form.
- `cmp_*` = top-disk comparisons (0/1/2 outputs).

Nothing in this is index-dependent in form. So why does it fail?

`probe_novel_fingerprints.py` settles it. For each error at n=18..20,
check whether its role-fingerprint per K appeared in any n=2..15
canonical trace:

```
n=18:    3 errors → K=10:   3 novel,    0 seen   K=12:   3 novel,    0 seen
n=19:  302 errors → K=10: 302 novel,    0 seen   K=12: 302 novel,    0 seen
n=20: 2442 errors → K=10:2442 novel,    0 seen   K=12:2442 novel,    0 seen
```

(K=8 has a handful of "seen" — those are the K=8 blind-spot collisions
where the small-K view is degenerate.)

So the leak isn't an index leak. It's one level up: the *features* are
invariant, but the **MLP only learned the fingerprint set produced by
n=2..15**. At n≥18 the canonical trace visits combinatorially novel
role-feature combinations and the MLP has no learned response.

A clean way to see why off-trace augmentation can't reach those
fingerprints: take an n=18 error fingerprint and try to construct a
consistent n=15 state with the same combined (smallest_K, largest_K,
parity, cmp_*) vector. The "K=8 largest" at n=15 is disks 14..7; at
n=18 it's disks 17..10. For both to produce the same fingerprint
vector, the overlapping disks (3..11 etc. depending on K) must agree
on their pegs in both views, and at high n the overlap forces
contradictions. The fingerprint space genuinely *grows* with n.

**3. Empirical confirmation that off-trace augmentation alone fails**

`discover_hanoi_offtrace.py`. Add 200,529 random reachable states from
n=2..15, labeled with `optimal_move_from_state` (recursive O(n) oracle:
find largest disk not on target, decide to move it now or recurse on
the smaller subproblem). Train a single K=12 MLP with role features:

```
n=15 (training): 100.00%   (perfect fit on training)
n=17:             93.89%   ← vs ~95% for canonical-only K=12 (no help)
n=18:             84.45%
n=19:             83.87%
n=20:             77.04%
```

Worse, not better. Confirms: the bottleneck is the *architecture*, not
the data coverage.

**4. The fix: structural invariance**

`discover_hanoi_invariant.py`. A 45,318-param GRU that processes the
disk-peg sequence largest→smallest with shared weights per position.
The function it learns is defined for any sequence length, not just
the lengths whose fingerprints showed up in training:

```python
class HanoiInvariantGRU(nn.Module):
    def __init__(self, d_emb=16, d_hidden=64, n_layers=2):
        super().__init__()
        self.peg_emb = nn.Embedding(4, d_emb)  # 0, 1, 2, ABSENT
        self.gru = nn.GRU(d_emb, d_hidden, num_layers=n_layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(d_hidden, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, N_ACTIONS))

    def forward(self, pegs):
        pegs_clean = torch.where(pegs == -1, torch.full_like(pegs, ABSENT), pegs)
        x = self.peg_emb(pegs_clean.flip(-1))      # largest first
        h, _ = self.gru(x)
        return self.head(h[:, -1])                  # readout at smallest
```

Trained on the same n=2..15 canonical traces. 3,000 steps, ~100s on CPU.

```
n |     states |    correct |         acc | verdict
----------------------------------------------------------
15 |      32767 |      32767 |   100.0000% | ✓
16 |      65535 |      65535 |   100.0000% | ✓
17 |     131071 |     131071 |   100.0000% | ✓
18 |     262143 |     262143 |   100.0000% | ✓
19 |     524287 |     524287 |   100.0000% | ✓
20 |    1048575 |    1048575 |   100.0000% | ✓
21 |    2097151 |    2097151 |   100.0000% | ✓
22 |    4194303 |    4194303 |   100.0000% | ✓
23 |    8388607 |    8388607 |   100.0000% | ✓
```

100% from n=15 (trained) all the way to n=23 (8,388,607 states).
At 500 training steps it was already 100% on n=18; the structural
invariance kicks in immediately.

**5. Length invariance ≠ start invariance**

A 50-parallel batched-lockstep solver (`hanoi_parallel_solve.py`)
revealed the next layer: the bare GRU **fails** on random off-canonical
starts even at n=5 (2 of 3 random-start runs hit the step cap). Length
invariance is one axis; *start* invariance — handle any reachable
configuration, not just the canonical-trace states — is a different one.

The fix: **structural invariance + off-trace augmentation, together.**
The GRU gives the structure, off-trace augmentation gives the function
coverage:

```
canonical prediction (length invariance, with off-trace included):
  n=15..22 : 100.0000%
  n=23     :  99.6262%   (slight regression vs canonical-only GRU)

off-canonical solver (start invariance):
  n=5..18 random starts:  50/50 OPTIMAL in 90s wall-clock
  n=18..22 random starts: 28/30 OPTIMAL, zero failures
                          (2 killed at 57 min for runtime budget; pattern
                          was clearly converging — same per-tick rate, no
                          cycles, just very long n=22 instances)
```

Total: 78 / 80 OPTIMAL across the random-start probes, **zero
non-optimal solutions**, just two budget-cap timeouts at the largest n.

**6. Two checkpoints, two purposes**

  - `checkpoints/hanoi_invariant_gru.pt` — bare GRU, canonical-only
    training. Pure length invariance up to n=23 at 100%. Use when you
    only need a canonical-start solver.
  - `checkpoints/hanoi_invariant_gru_offtrace.pt` — GRU + 200k random
    reachable states from n=2..15. Both invariances. Slight 99.63%
    canonical regression at n=23 in exchange for full start invariance.

**7. The lesson worth saving**

For a recursive task (Hanoi, GCD, Fibonacci, …), feature engineering
roles into a fixed-K MLP gives *almost* invariant generalization — but
the MLP's learned function is a lookup over the fingerprint set it
saw, and that set genuinely grows with n. The diagnosis tool is
fingerprint novelty (`probe_novel_fingerprints.py`): if every error is
on a feature-vector never seen in training, more training data can't
fix it. The fix is structural — a network whose weights don't depend
on n. A small GRU over the disk-peg sequence is enough; we didn't need
attention or Mamba, just shared-per-position recurrence.

The user's pushback ("indices are anti-invariant") was directionally
right, but the precise diagnosis turned out to be one level of
abstraction up: *features* invariant ✓, *function space* not. Two
invariances are needed: length and start. The architecture gives the
first; data augmentation gives the second; both are required.

Files this session: `probe_invariance.py`, `probe_novel_fingerprints.py`,
`discover_hanoi_offtrace.py`, `discover_hanoi_invariant.py`,
`hanoi_solve_gru.py`, `hanoi_parallel_solve.py`. Commits `9ceefa4` and
`dc0a45a`.

---

## Entry — Memorization vs computation: the Tower-of-Hanoi cliff (2026-04-26)

**Setup.** `tower_of_hanoi` task: input `HANOI n` → output `2^n − 1` as bytes.
Curriculum trained the model on `n ∈ [1, 8]` with `d_model=64, L=2, 74,658
params`. Final training accuracy: 100%.

**Question.** Did it learn the recurrence `2^n − 1`, or did it memorize the
eight input-output pairs?

**Test.** Held the .pt file fixed (no retraining, no resizing — *zero* model
modification). Evaluated on `n ∈ [1, 25]`, 100 trials per value of n. Script:
`length_gen_hanoi.py`.

**Result.** A perfect cliff at the curriculum boundary:

```
 n   target    pred   acc
 1   1         1      100%   ┐
 2   3         3      100%   │
 3   7         7      100%   │  in distribution
 ...                         │  (curriculum: n ≤ 8)
 8   255       255    100%   ┘
 9   511       1        0%   ┐
10   1023      1        0%   │
11   2047      1        0%   │
12   4095      3        0%   │
13   8191      7        0%   │
14   16383     15       0%   │  out of distribution
15   32767     31       0%   │
16   65535     63       0%   │
17   131071    127      0%   │
18   262143    255      0%   ┘
19   524287    1        0%
22   4194303   3        0%
25   33554431  31       0%
```

100% accuracy inside `[1, 8]`, **0% everywhere outside**, with zero ambiguity.
But the *wrong predictions* tell the actual story.

**The trick the model found.** Look at n=12 → predicts 3. n=13 → 7. n=14 → 15.
n=18 → 255. The model is reading the **last digit** of n and outputting
`2^(last_digit) − 1`. Because every training input was a single digit, "last
digit" and "n" were the same thing in the training distribution. The model
optimized for the laziest pattern that fit the data.

This is exactly the signature of memorization-via-shortcut: it didn't learn
to compute `2^n`, it learned to *attend to a single byte position* and use
that as a lookup index. Inside the trained range the shortcut is correct;
outside, it surfaces.

**Why this matters.** The Mamba-3 SSM has 2,048 register slots per layer
(8 heads × 16 hd × 16 d_state). That's *plenty* of capacity to encode the
recurrence `2^n − 1`. The model didn't fail because of capacity — it
succeeded at the wrong objective because the training distribution let
it. The same architecture, same registers, would learn the algorithm if
the curriculum forced it.

**Next.** Same model size, extended curriculum past where the digit-shortcut
breaks (multi-digit n). If the same `d=64, L=2` reaches n=20 accuracy with
no architectural change, that's the recurrence learned — proof that "small
fixed-capacity, growing data" is enough to push from memory to computation.

---

## Entry — The cliff moves but doesn't disappear (2026-04-26)

**Setup.** Same `d=64, L=2, 74,658 params`. Curriculum extended to add
stages at `n_disks ∈ {12, 16, 20}`. 40 training cycles total. Final
training accuracy: 100%. Re-ran `length_gen_hanoi.py --n-max 30`.

**Result.** The cliff moved from n=9 to **n=21** — exactly one past the new
curriculum boundary. Inside `[1, 20]` the model is 100% accurate, including
multi-digit outputs up to 7 characters (`2^20 − 1 = 1048575`). It is doing
*real digit-by-digit synthesis* in that range — predictions like `'1048575'`
are not the kind of shortcut you can hit by reading a single byte.

But outside `[1, 20]` the failures rhyme with the first experiment:

```
n=21 → '1'                target 2097151
n=22 → '3'                target 4194303
n=24 → '15'               target 16777215
n=28 → '2143'             target 268435455   (gibberish)
n=30 → '10048575'         target 1073741823  (corrupted 2^20-1)
```

The last-digit shortcut returns the moment we leave the trained range. At
n=30 the prediction is "10048575" — the model literally pasted a corrupted
version of *the largest answer it had seen during training* (1048575),
because that was the most-recently-rehearsed long output.

**Interpretation.** Within the curriculum the model *is* computing — the
digit-by-digit output structure is real. Beyond the curriculum it has no
incentive to extrapolate, so it doesn't. The architecture isn't the
bottleneck; the *training distribution is*. Same registers, same params,
same dynamics — the model will compute as far as you push it and no
further.

This is consistent with how Mamba-3's recurrent state should behave: 2,048
register slots are more than enough to hold a counter + a doubling
operation, so the algorithm fits. The model just needs the curriculum to
demand it.

**The plant/fungus framing again.** A small organism doesn't grow by
adding mass; it grows by extending its reach into more nutrients. The
nutrient here is the curriculum span. Each stage we add forces the model
to rewire the same 75k-param scaffold to handle more.

**Next experiment.** Push to `n_disks ∈ {30, 50, 100}`. At n=100 the answer
is a 31-digit number. Memorization at that scale costs ~20× the parameter
budget; computation is asymptotically free. If we see the cliff keep
tracking the curriculum boundary into 50+ disks — same model size — that's
strong evidence the architecture supports general computation and the only
gating factor is the data.

---

## Entry — The "n=40 cliff" was a decoder bug. Bounded program found. (2026-04-27)

After running scheduled-sampling Hanoi (curriculum out to n=100) we saw
the cliff sit at n=40 and called it the "self-conditioned trust horizon" —
hypothesizing the model couldn't sustain self-emission past 12 tokens.

**That hypothesis was wrong. The cliff was a bug in the test harness.**

`length_gen_hanoi.py` had `max_new=12` hard-coded in the autoregressive
decoder. Outputs longer than 12 tokens were truncated, making it look
like "the model emits EOS at position 13." It was actually the test
harness terminating the decoder loop before the model had a chance to
emit more. With `max_new=64`:

```
n=1..100   →  100% accuracy  (31-digit answers correct, e.g. n=100 → 1267650600228229401496703205375)
n=110+     →  fails, cliff sits exactly at the training boundary
```

**The model has program-like behavior across the entire trained range,
including 31-digit autoregressive emission with correct EOS placement at
every length 1..31.** That's not what a memorization model does.

**But** it doesn't truly extrapolate. n=110 (20 disks past the curriculum
max) already fails, with predictions like `''` or single digits. So:

- The user's strict pressure test ("a true program is unbounded")
  still rules — `2^n − 1` is unbounded; this isn't.
- The middle ground between "memorization" and "true program" is real:
  a learned continuous-state procedure that produces correct
  multi-digit outputs over the trained range. Calling that "bounded
  program" is more accurate than "memorization."

**Architectural follow-up.** Tried output-history attention as a
copy/lookup primitive on top of the SSM (smallest change in the
spectrum from "minimal" to "Neural Turing Machine"). Initial runs
unstable; tuning ongoing. Even if it fixes the n>100 extrapolation,
unbounded program-shape probably needs a discrete-register primitive
(design #2) — the SSM's continuous-state blending fundamentally
limits how a counter can be tracked across many output tokens.

**Methodology lesson.** *Always look at what the model actually emits
before drawing conclusions about what it learned.* The 12-cap turned
out to be the bug; the real story (program-shape over the trained
range) was hiding in plain sight. Several rounds of "memorization"
diagnoses were over-claiming based on a buggy decoder.

---

## Entry — The shortcut is *last-digit attention*, not output decoding (2026-04-27)

**Setup.** The decimal-Hanoi failure mode (`HANOI 25` → `255` =
2^8 − 1) was originally attributed to two possible bottlenecks:
the binary→decimal output head, or the recurrence itself. To
disambiguate I added a unary-output variant — `gen_tower_of_hanoi_binary`
— where the answer is just `'1' * n`. No arithmetic, no
carries, no decimal. The model only has to count to n and emit n
ones. Trained the same baseline architecture (d=64, L=3, ~104k
params, no oracle, no registers, no noise) on the staged
curriculum; n_disks=12 hit 100% in 4 cycles, then stage 5 (n=16)
NaN'd. The cycle-4 checkpoint (clean, n≤12) was used for the test.

**Result.** 12/12 on n∈[1,12]. **0/88 on n∈[13,100]**, and the
failure pattern is the diagnostic:

```
n=21 → '1'         n=31 → '1'         n=61 → '1'
n=22 → '11'        n=32 → '11'        n=62 → '11'
n=23 → '111'       n=33 → '111'       n=63 → '111'
n=24 → '1111'      n=34 → '1111'      ...
n=25 → '11111'     n=35 → '11111'     n=70 → '1111111111'
n=29 → '111111111' n=39 → '111111111' n=99 → '111111111'
n=100 → '1111111111'
```

The model is reading the **last digit of n** and emitting that
many ones. It treats `"21"` as "1", `"35"` as "5", `"100"` as "0"
(re-rolled via length norm to "10"). The first digit of a
multi-digit input is invisible to it.

**Why this matters.** This rules out the output-decoder hypothesis.
It was never the binary→decimal converter — even with that path
removed, the model picks the same shortcut. The architecture is
not building a counter; it's running a token-level lookup with
strong rightmost-token bias.

**Mechanistic guess.** The SSM's local kernel mixes adjacent
tokens, but in the byte-tokenizer setup `"HANOI 21"` ends with
the byte for `'1'` immediately before SEP. Whatever feature gets
written to the post-SEP register at the start of generation is
dominated by that last byte. There's no architectural pressure to
*combine* the digits into a place-valued integer before
generation begins, so the model never learns to.

**Implication for the next ship.** Three things would each force
the model to attend to all input digits:

1. **Length-modulated output supervision** — pad the answer with
   the input's length encoding so the loss credits "pred_len ==
   input-derived-n", not just per-digit CE. Cheap.
2. **Bidirectional / multi-pass input encoding** — let the SSM
   re-read the input from both sides before answer generation.
   Architectural but contained.
3. **Discrete loop-counter primitive** — an explicit integer
   register the model can decrement, with hard `--` and `==0`
   semantics, supervised by the trajectory oracle.

The failure here is sharper than the trajectory-distillation
result because the task was unary — there's no plausible
alternative explanation involving output-side computation. The
representation of `n` itself is the bottleneck.

---

## Entry — Bidir input breaks rightmost-byte shortcut, exposes the bounded counter (2026-04-27)

**Setup.** Following the HANOIBIN diagnostic showing the model
reads only the rightmost input byte, I added `--bidir-input`: append
the byte-reverse of the input after itself, separated by a space.
For HANOIBIN the input becomes `"HANOIBIN 21 12 NIBIONAH"`. The
rightmost byte is now always `'H'` regardless of n — the
rightmost-byte shortcut is mechanically impossible. Trained the
same architecture (d=64, L=3, ~104k params) at lr=1e-4 (lr=3e-4
had ~50% stage-3 NaN rate). Reached stage 6 (n=20) at 100%
teacher-forced byte accuracy, NaN at cycle 13 — saved cycle 12,
loss=0.94.

**Result, autoregressive eval on n=1..30:**

```
n=1  → '111111111111' (12 ones)        n=18 → 18 ones ✓
n=2  → '111111111111111111111' (21)    n=19 → 19 ones ✓
n=3  → 20 ones                         n=20 → 21 ones
n=11 → 11 ones ✓                       n=21 → 19 ones
n=27 → 27 ones ✓                       n=30 → 21 ones
```

3/20 in-distribution, 1/10 out (n=27, accidentally landing on
the model's natural cycle). Mean prediction length: ~17-25.

**The shortcut shape is now revealed.** Per-position teacher-forced
probe: feeding a long input (n=50, all 50 ones in answer span),
the model's EOS probability oscillates with period ~20 starting
at position sep+22:

```
pos  0..19 → predict '1'  (p≈1.0)
pos 20-21 → transition
pos 22-33 → predict EOS   (p≈0.99)
pos 34-39 → predict '1'   (next cycle)
pos 40-46 → predict EOS
pos 47-50 → predict '1'
pos 51    → predict EOS
```

The model has learned **a cyclic pattern with period ~20** in its
hidden state — close to the training maximum. EOS prediction is
position-driven, not input-driven. The bidir input *did* break
the rightmost-byte shortcut: the resulting failure mode is not
"emit last_digit(n) ones," it's "emit ~training_max ones."

**What this rules in and out.**

- ✗ Output-decoder bottleneck (HANOIBIN diagnostic, prior entry)
- ✗ Rightmost-byte attention (bidir input, this entry)
- ✓ The architecture cannot extract n from its input and use it
  as a counter. The recurrence learns a **bounded periodic
  pattern** whose period is set by the training distribution.

**Implication.** Soft attention over a continuous register bank
will not learn unbounded counting at this scale. The next ships
need to give the model a *discrete* loop primitive — an integer
register with hard `--` and `==0` semantics — supervised by an
oracle that ties the register's initial value to the parsed input.
The current trajectory oracle supervises the *value to write* but
not the *iteration to perform*; a counter primitive supervises
both.

Saved checkpoint: `tower_of_hanoi_binary_bidir.pt` (~104k params,
lr=1e-4, 12 clean cycles, NaN at 13).

---

## Entry — Hanoi-exec: model executes Tower of Hanoi via register bank (2026-04-28)

User's framing: "I am hoping that the model will actually be able
to execute the algorithm. That is why I chose Tower of Hanoi
because it's very simple algorithm that very small human babies
can learn."

Built two new architectural primitives and composed them with the
existing LoopCounter:

  - **RegisterBank** (progressive_model.py): 16 discrete integer
    registers, value range [0, 16). Three output heads (read_addr,
    write_addr, write_val) plus a value-embedding for read-feedback.
    Hard discrete I/O; no max_count limits in the way LoopCounter's
    final form has none.
  - **gen_exec_trace** (hanoi_exec_oracle.py): per-byte ground-truth
    trace for Hanoi(n). Initial register state has reg[0]=n;
    reg[1..n] = 0 (peg A); rest 0. Each move emission spans 6 bytes;
    READ peg-of-disk-k at the first byte; WRITE peg-of-disk-k :=
    dst at the last byte.
  - **LoopCounter for termination**: oracle places counter trajectory
    = total trace bytes, decrementing per output position. With
    iteration_token=None the LoopCounter contributes only the EOS-
    gating signal, leaving byte choice fully to the model+RegisterBank.

Multi-head loss (token CE + read_addr CE + write_addr CE +
write_val CE) supervises the four heads jointly during teacher-
forced training. AR validation runs the model autoregressively
with its own register state and own emitted tokens; the LoopCounter
trajectory is fed at each step since termination is oracle-gated.

**Result, 1-minute training run:**
  - d=32, L=2, 27,626 params
  - batch=32, 30 steps/cycle, lr=5e-4
  - 25 cycles × 3.2s = 80s wall
  - Cycle 25: token=100%, read=100%, write=100%, val=100% on
    n=2,3,4 in-distribution

  AR validation byte-for-byte vs Python's recursive Hanoi:
  - n=2 (3 moves, 18 bytes):  ✓
  - n=3 (7 moves, 42 bytes):  ✓
  - n=4 (15 moves, 90 bytes): ✓

The model uses its register bank to track which disk is on which
peg through the entire trace, makes the right read/write decisions
at every position, and emits the correct move sequence — entirely
autoregressively. This is genuine execution, not memorization: at
each timestep the model's choice is conditioned on its register
state (not on any oracle-supplied content signal).

**OOD limitation acknowledged.** For n=5+ the model emits correct-
LENGTH traces (LoopCounter works) but content drifts as small
write-addr errors compound over 30+ moves. This is a 27k-param
capacity ceiling, not architectural: write_addr head plateaus at
93-99% with this model size. Bigger model or broader curriculum
should extend the iron-solid range; both are out of the 1-minute
budget on M4 Pro.

**Composition of three primitives works.** LoopCounter (parameter-
free, unbounded c) + RegisterBank (16 discrete registers, hard I/O)
+ Mamba-3 SSM (the loop body) execute Tower of Hanoi at small n.
Same external-primitive pattern as HANOIBIN/FIB-decimal, with the
addition of state primitives that make multi-step state tracking
work.

Saved: `tower_of_hanoi_binary_paramfree.pt` (HANOIBIN, n=100k),
`fib_decimal.pt` (FIBD, 200/200 to n=200),
`hanoi_exec.pt` (this — n=2,3,4 byte-perfect AR).

---

## Entry — Hanoi step function: perfect extension (2026-04-28)

User's bar: "100% accuracy = function correctly defined. If we
can run a few steps and not others, it's not the right primitive."

**Met it.** A 1,574-parameter MLP trained on n=2..6 for **1.9
seconds** runs Hanoi(n) byte-perfect at n=12 (4,095 moves).
Train sees 119 (state, action) pairs → AR-correct at every
out-of-distribution n we tested (n=7,8,9,10,12).

**The architectural insight that closed it: ROLE encoding.**

We had been encoding the state as per-disk pegs — "disk 1 on peg
A, disk 2 on peg B, …" — which scales with n. Disks 7+ never
appeared in training; their embeddings were random; OOD failed.

The fix: encode each peg's TOP DISK as a *role*, not a disk id:

```
role[peg] in {empty, smallest_visible, middle, largest}
```

State becomes `(n_parity, move_parity, role_A, role_B, role_C)` —
5 small ints, **n-invariant**. Across all reachable Hanoi
configurations at every n, only **36 distinct states** exist.
Training on n=2..6 visits every one of them. Inference at
n=20 hits the same 36 states. There is no out-of-distribution
state to memorize against, because the state space is closed
under the algorithm.

**The step function itself**: one forward pass = one structured
action. No byte rendering. f: state₅ → action₆ where the action
enumerates `{A→B, A→C, B→A, B→C, C→A, C→B}`.

**Architecture**: 5 feature embeddings (d=8) → concat → linear
(40 → 32) → ReLU → linear (32 → 6). 1,574 params total.

**Training**: 119 pairs, 2000 SGD steps batch=64, lr=3e-3, 1.9s
on M4 Pro. Final loss 0.0001.

**Autoregressive validation (model uses its OWN previous action
to advance state, not teacher-forced)**:

| n | reach | result |
|---|---|---|
| 2..6 | training | 3+7+15+31+63 actions perfect ✓ |
| 7 | OOD | 127/127 ✓ |
| 8 | OOD | 255/255 ✓ |
| 9 | OOD | 511/511 ✓ |
| 10 | OOD | 1023/1023 ✓ |
| 12 | OOD | 4095/4095 ✓ |

**Lessons captured.**

1. **State must be closed under the algorithm**, not parameterised
   by problem size. Encode invariants (roles, ranks, comparisons)
   not literal values (disk ids, integers).
2. **One forward pass = one structured action.** Byte rendering
   forces the model to jointly learn presentation and algorithm,
   which gates generalisation.
3. **Tool tracks state in plain Python** (n-invariant or n-aware,
   doesn't matter; tool can encode roles freely). Model is a pure
   step function. Both primitives compose: same shape will fit
   any deterministic puzzle (Tetris, GCD, bubble sort, Sokoban …).
4. **The Lego is now small.** 1.6k params, 2 seconds to train.
   The base substrate for "playing puzzles" is genuinely tiny.

Code: `hanoi_step_function.py`, `train_step_function.py`. Saved:
`checkpoints/specialists/hanoi_step_fn.pt`.

This is the foundational primitive for the user's "Lego pieces
composed at random" vision. The Hanoi step is no longer a
trace-memorising language model — it's a function over a closed
finite state space that generalises by construction.

---

## Entry — Hanoi cliff at n=40 (algorithm learned, EOS broken) (2026-04-26)

Pushed the Hanoi curriculum to `n_disks ∈ {30, 50}` (in addition to
the prior {12, 16, 20}). Same `d=64, L=2, 74,658 params`. Re-ran
`length_gen_hanoi.py --n-max 70`.

**The cliff moved from n=21 to n=40.**

```
 n=1..39   100% accuracy        (correct multi-digit synthesis through 12 digits)
 n=40+     0% with characteristic failure mode
```

The failure mode at the new cliff is *qualitatively different* from
the n=9 cliff:

```
n=40 → predicted '109951162777'    target '1099511627775'   (12 digits, missing trailing 5)
n=41 → '219902325555'               '2199023255551'          (missing trailing 1)
n=42 → '439804651110'               '4398046511103'          (missing trailing 3)
n=43 → '879609302220'               '8796093022207'          (missing trailing 7)
...
n=50 → '112589990684'               '1125899906842623'        (truncated 12 / 16 digits)
```

The model is producing **correct-prefix multi-digit answers** that
are missing a digit (or several) at the end. It's not memorizing —
it's computing the algorithm and the autoregressive decoder is
predicting EOS too early at outputs longer than ~12 digits.

This is the cleanest possible separation between "the algorithm"
(learned and working) and "knowing when to stop" (a separate skill
that wasn't pressured by the curriculum). Same architecture, same
registers, same 74,658 params — just expanded data.

**The rolling story.**

| Curriculum max n | Where the cliff lands | Failure flavor |
|---|---|---|
| n_disks=8 | n=9 | last-digit shortcut (memorization) |
| n_disks=20 | n=21 | last-digit shortcut returns at boundary |
| n_disks=50 | n=40 | correct-prefix outputs, EOS broken (computation) |

The cliff *tracks the curriculum boundary*. The architecture is not
the bottleneck. With each round of curriculum extension the model
shifts further from memorization toward computation.

**Next.** A length-aware terminal heuristic — either explicit token
budget, or curriculum stages that *vary* the answer-digit count
within a single n-range so EOS prediction gets supervised at every
length. Either should remove the EOS cliff entirely.

---

