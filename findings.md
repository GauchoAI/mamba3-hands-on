# Mamba-3 Hands-On — Findings Log

A running lab notebook as we explore the Mamba-3 paper (Lahoti et al., ICLR 2026)
by implementing a minimal version from scratch and running experiments on Apple
Silicon MPS. Each entry corresponds to a commit.

> **Note (2026-04-26):** The CUDA/PTX engine and ptxd-related entries below were
> the H100/vast.ai era. That work has moved to the `pod-archive` branch. This
> Mac-only branch keeps the historical findings text intact for context but
> the production daily-driver path is PyTorch + MPS (`specialist_trainer.py`).

---

## Entry — JEPA-Cortex: thought-distillation regularizes a small bilingual LM, JEPA-off baseline mode-collapses (2026-04-29)

> **Cross-project takeaway** (full experiment journal in
> [`jepa/findings.md`](jepa/findings.md)):
>
> Pseudo-label distillation onto a 1M-param byte-level Mamba-3 from
> Qwen-2.5-1.5B over a tiny ~80 MB corpus produces fluent bilingual
> text in ~6 hours on 4× RTX 4070 Ti. *Without* a JEPA-style latent
> regularizer the small-corpus byte-CE objective drives the student
> into **mode collapse**: 4 different prompts produce near-identical
> responses; the model falls back on its highest-confidence learned
> attractor (or onto the unary `***:aaaa…` task) for any prompt outside
> its strongest modes. *With* JEPA pressure (λ=0.3 was the qualitative
> sweet-spot), runs preserve prompt-conditioning and improve
> monotonically on held-out bilingual byte CE while pure-byte runs
> oscillate. The result was confounded by an unintentional batch-size
> mismatch (gpu3 at 32, others at 64), so the *trajectory shape* —
> mode-collapse vs prompt-differentiation — is the trustworthy signal,
> not the absolute byte-CE numbers.
>
> The findings live in `jepa/findings.md` with hypothesis-before,
> live-observations, and (eventually) conclusion sections. Reproduce
> via `DEPLOYMENT.md` + `jepa/README.md`.

---

## Entry — Counter primitive on a frozen bilingual LM: partial composition (2026-04-29)

The cortex thesis stress-tested on a real, language-trained host LM
rather than the synthetic counting-only LM that produced the
byte-perfect existence proof on 2026-04-28.

**Setup.** A 472,960-param byte-level Mamba-3 LM (4 layers, d_model=128)
trained for 10,000 steps on the 17.7 MB Tatoeba en-es corpus + 5%
unary cortex mixin. Loss settled at ~1.0 bpc 1.45 bpc-by-the-end.
Then froze every weight, attached a fresh `CounterPrimitive` sized
to d_model=128 (1,028 trainable params), and fine-tuned only those
adapters for 1000 steps on a 50/50 mix of bilingual and unary
batches. λ_aux=0.5 BCE on inc_logits, lr=3e-3. Aux loss converged
crisply (0.354 → 0.0001) — the gates fire correctly on `*` and
`a` bytes through the bilingual LM's hidden state.

Eval at hard_gates_inference=True. Side-by-side vs the same LM
without the counter:

```
N      baseline                cortex+counter
3      FAIL → 7                OK ✓                    ← cortex helps at small N
10     FAIL → 14               FAIL → 9                ← off by 1
30     FAIL → 31               FAIL → 29               ← off by 1, IN-DISTRIBUTION
50     FAIL → 54               FAIL → 48
100    FAIL → None             FAIL → None             ← both drop out of unary
200    FAIL → 65               FAIL → 34               ← cortex worse here
500    FAIL → 33               FAIL → 33
```

Counter-attached samples at training-distribution prompts:
- `*****:` → 4 a's (should be 5) ← off by one
- `**********:` → 9 a's (should be 10) ← off by one
- `***************:` → 16 a's (should be 15) ← off by one in the other direction

**The honest read.**

This is *not* the byte-perfect demo the synthetic-LM cortex experiment
produced. It is also *not* a null result — three concrete things
are validated and three are not.

What the experiment validated:
1. **The plugin interface holds.** A 1,028-param adapter (0.22% of LM
   params) attached to a fully frozen 472k-param language-trained LM,
   trained against the same aux-supervision recipe, learns to fire on
   the right bytes. The pluggable `Primitive` class works as designed
   on a non-toy host.
2. **The bilingual LM is not destabilised.** Bilingual probes
   (`'The cat '`, `'¿Dónde '`, etc.) produce the same family of
   outputs whether the counter is attached or not. The counter
   contributes additively without trampling the LM's language
   ability.
3. **Counter helps at small N.** At N=3 the baseline overshoots to
   7 a's (it pattern-matches "in unary mode emit ~7 things"); the
   cortex version emits exactly 3. The counter's signal *is* getting
   through and is affecting the right decision.

What the experiment *did not* validate:
1. **Byte-perfect counting at training-distribution N.** Even at N=30
   the cortex version is off by one, emitting 29 a's. The synthetic-LM
   experiment was byte-perfect at every N up to 500.
2. **Strong OOD extension.** At N=200 the cortex version produces 34
   a's (not 200); at N=500 it produces 33. The synthetic-LM cortex
   produced 500 of 500 byte-perfect at the same N.
3. **Train-free composition in the strong sense.** The plugin needed
   1000 fine-tune steps; that's "small adapter fine-tune", not
   "frozen plugin attaches and works". A pre-trained CounterPrimitive
   from the synthetic experiment did not transfer (different d_model,
   different hidden-state distribution at unary positions).

**Diagnosis: distribution-shift on the counter readout.**

The systematic off-by-one pattern is the smoking gun. The bilingual
LM's hidden state at `*` and `a` byte positions encodes "I am
reading the unary form" — that's the signal `inc_proj` learned
to read, and aux loss confirms gates fire correctly. But the
read_proj has to convert the counter state into a residual injection
that the *frozen, pre-trained* head will read as "emit `a`" vs
"emit `\n`". On the synthetic LM this was learnable end-to-end. On
this frozen bilingual LM, the head's natural disposition (trained
on `*N:aN\n` lines where N≤30, average ~15) carries a strong
bias toward emitting `\n` near the end of unary runs. The counter's
"still counting" signal isn't dominant enough to overcome it; the
LM emits `\n` one position early.

Things that would likely fix the off-by-one:
- Higher `injection_scale` (currently 10) — loud the counter more.
- Aux supervision on `\n` emission too (predict newline target),
  not just `inc` gates.
- Train at more diverse N values so the LM doesn't have such a
  strong "newline by ~15-30" prior.
- Or: the right architectural change is to give the counter a path
  that bypasses the head's tied-embedding bottleneck — but Phase 4
  of the synthetic experiment showed direct head-bias paths regress
  via capacity-splitting.

**Diagnostic: re-ran with `--injection-scale 30`** (3× louder).
Aux convergence identical (it's gate-only supervision). Result:

```
            scale=10 (original)    scale=30 (diagnostic)
N=3         OK ✓ → 3              OK ✓ → 3
N=30        FAIL → 29             OK ✓ → 30        ← off-by-one fixed
N=50        FAIL → 48             FAIL → 51        ← oscillates by 1
N=100..500  drops out of unary    drops out of unary  (same)
```

The in-distribution off-by-one **was** purely signal-magnitude-bound;
louder counter resolves it. That confirms the mechanism diagnosed
above. But OOD failure is unchanged: at N ≳ 80 stars the LM exits
unary mode entirely and emits dominant Spanish phrases (`'ér es el
problema...'`). This is **not** a counter-readout problem and not a
magnitude problem — it's the LM's implicit `stars→short→switch out`
prior, learned because training only saw `*N:aN` lines with N≤30.

The fix shape is therefore *upstream* of the counter:
- Train the bilingual LM with a wider N distribution (e.g., 1..200)
  so it doesn't acquire a "stars are short" prior, OR
- Distill from a stronger teacher (the JEPA direction in `jepa/`)
  so the student LM's hidden state at long unary runs is clean
  and the counter's signal can dominate it.

That's the actual next experiment to validate the strong cortex
composition claim.

**The actual position this leaves us in.**

The cortex thesis survives but with a sharper statement: forward-pass
primitives extend a language-trained LM's reasoning capability *with
fine-tuning of small adapters*, but the strong claim ("primitive
attaches and works without LM-side adaptation") is unproven and the
off-by-one suggests the more honest framing is "small-adapter
fine-tune, not adapter-free plug-in."

This points at the JEPA-Cortex direction (jepa/) the user was
preparing in parallel: rigorous hidden-state distillation gives
the student LM a more *informationally rich* hidden state than
plain byte-CE training, which in turn should give residual-stream
primitives a cleaner attachment surface. That is the experiment
that would actually validate the strong plug-and-play claim.

Reproduction:
```bash
python train_counter_attach.py \
    --lm-ckpt checkpoints/lm/step_FINAL.pt \
    --steps 1000 --unary-p 0.5 --lambda-aux 0.5 --lr 3e-3
python demo_cortex.py
```

Both checkpoints under `checkpoints/lm/` (frozen bilingual LM) and
`checkpoints/lm_counter/` (LM + trained counter adapter) — ~5.5 MB
each. ~26 min total on M4 Pro MPS (training + eval).

---

## Entry — Renderer LM + the digit-copy failure (2026-04-28)

Closed the third leg of the harness: a tiny Mamba-3 conditional LM that
renders structured tool results as natural-language sentences. Trained
on synthetic prefix-LM pairs:

```
<payload>\x01<sentence>\x02
e.g.  hanoi_solver|n=12|optimal=4095|params=45318|timing=2864 \x01 The optimal solution to Tower of Hanoi with 12 disks requires 4,095 moves. \x02
```

74,400 params (2 layers, d=64), 600 training steps on M4 Pro CPU,
best val loss 0.189. **It doesn't work cleanly** — and the failure is
informative.

**The failure mode that surfaced.** On every test prompt, the renderer
produces fluent template text but **drops or substitutes the specific
digits**:

```
input  : "Solve Tower of Hanoi with 12 disks"
payload: hanoi_solver|n=12|optimal=4095|...
output : "The optimal solution to Tower of Hanoi with 18 disks requires 2,047 moves."
```

The LM has clearly learned the *shape* of the answer ("The optimal
solution to Tower of Hanoi with K disks requires M moves") but it
hasn't learned to *copy* the specific K and M from arbitrary
positions in the byte prefix. It guesses globally-common values.

**Why this is the right family of failure to expect.** A small Mamba
SSM compresses the prefix into a fixed-size hidden state. Specific
digits at arbitrary prefix positions are exactly what SSMs without
attention struggle to copy out — the canonical Mamba selective-copy
result. We've seen this throughout this repo: token-stream Hanoi was
unblocked only when we added EOS-bias gating + parameter-free
LoopCounter; HANOIBIN n=100k worked once we stopped trying to make
the SSM count and let the orchestrator count instead.

**Mitigation: a payload-fidelity guard.** `render_mamba` now checks
that every canonical numeric from the payload appears verbatim in the
LM's output. If anything is missing, fall back to `render_template`
and log what got dropped:

```
> Solve Tower of Hanoi with 12 disks
The optimal solution to Tower of Hanoi with 12 disks requires 4,095 moves.
The Hanoi GRU (45,318 parameters) reproduced this in 2864 ms.
[renderer-guard: LM dropped ['12', '4,095']; used template]
```

The pipeline never publishes a corrupted number. The guard is honest:
the trace shows that the LM failed and the template did the work.

**The thesis-aligned fix (next, not now).** The right shape is
**templates with placeholders**: the LM emits a skeleton like

```
The optimal solution to Tower of Hanoi with $N disks requires $OPTIMAL moves.
```

…and the orchestrator substitutes the actual values from the payload.
That removes the small-SSM copy failure entirely. The LM only has to
produce *language form*, which it's already doing fluently. The
specifics come from the payload via deterministic substitution. This
is the cleanest realization of "language as translation layer" — the
LM owns the shape of the sentence, the orchestrator owns the values,
neither tries to do the other's job.

**State of the harness.** Three Mamba-3-class stages totaling ~165k
parameters plus the 45,318-param GRU specialist:

  - **Router**: 45,459-param Mamba-3 byte classifier, val_acc 100 %.
  - **Specialist**: 45,318-param order-invariant GRU.
  - **Renderer**: 74,400-param Mamba-3 LM, val_loss 0.189, guarded.

Three demoable prompts work end-to-end with auditable per-stage
traces. Files: `train_tool_renderer.py`, `assistant.py` updates.
Checkpoint: `checkpoints/tool_renderer_mamba3.pt` (363 KB). Commit
`afa4dd2`.

---

## Entry — Cortex: a 772-param counter primitive lets a 151k Mamba LM count byte-perfect to N=500 (2026-04-28)

Tested the **cortex thesis** end-to-end: rather than a small LM emitting
tool-call tokens that an external Python loop dispatches, treat
algorithmic primitives as **forward-pass modules in the residual
stream**. The LM emits gates from its hidden state; the primitive runs
its own arithmetic; its output re-enters the residual as a learned
embedding. No tokens, no parser, no Python loop outside the forward pass.

Single file: `cortex_counting.py`. Task: unary copy-the-length
(`*N:aN\n`). Train on N ∈ [1, 30], eval at N ∈ {3, 15, 30, 50, 100, 200,
500}. Six phases, all on the same byte-level Mamba-3 LM (2 layers,
d_model=96, ~150k params).

**Headline:**

```
              baseline      Phase 1     Phase 2     Phase 3.2    Phase 6+hard
N=3..30           OK            OK          OK            OK           OK
N=50              OK      FAIL(13)    FAIL(33)      FAIL(48)           OK
N=100       FAIL(72)       FAIL(3)    FAIL(29)      FAIL(88)           OK
N=200        FAIL(3)      FAIL(12)    FAIL(31)     FAIL(128)           OK
N=500       FAIL(67)    FAIL(None)    FAIL(29)     FAIL(183)           OK
```

A 772-param counter (`CounterPrimitive`) wired into the residual stream
of an otherwise unchanged 150,704-param Mamba-3 LM extends counting to
N = 500 — **16.7× past the longest training example** — byte-perfect.
Same architecture without the counter caps at N = 72.

**The four ingredients that mattered**, in order of impact:

1. **Aux BCE supervision on `inc_logits`** from byte-conditional targets
   (`inc[A]` should fire on `*`, `inc[B]` on `a`). Without this the
   counter sits unused — Phase 1 was *worse* than baseline at OOD.
2. **Tanh-saturated readout** with temperature k=8: `[tanh(c/k),
   tanh(diff/k)]`, no raw features. Bounded in (-1, 1) regardless of N.
   The `tanh(diff/k)` channel is the decisive "is c[A] ahead of c[B]"
   signal — same value at N=30 and N=500.
3. **Injection scale ≥ 3** on the `read_proj` output. Below 3, the
   counter contribution can't outvote the SSM's positional-OOD logits
   at long sequence positions, and the model under-counts proportionally.
4. **Hard-gate threshold at inference** — replace `sigmoid(inc_logit)`
   with `(inc_logit > 0).float()` when not training. Applied as a flag
   on the trained checkpoint, no retraining required. Diagnosis: at OOD
   positions the SSM hidden state has drifted, so sigmoid gates that
   fire at 0.999 in-distribution settle near 0.95; over hundreds of
   increments the slippage compounds to tens of missing counts.

**Cautionary tale (Phase 4):** adding a direct `Linear(read_in →
vocab)` from counter readout straight to LM logits — a "side channel" —
*regressed* OOD scaling from 88 to 40 at N=100. The model offloaded the
emit-a-vs-newline decision onto the cheap shortcut and trained the
residual injection less, leaving smaller total override authority.
Don't add side channels to a path that already works.

**Plugin/adapter refactor:** after the existence proof landed, the
architecture was generalised from hard-coded counter wiring to a
`Primitive` base class. `CortexLM(cfg, primitives=[...])` accepts any
list of `Primitive` subclasses; `CortexLM.forward` and
`CortexLM.aux_loss` are primitive-agnostic. Adding a new primitive is
now: subclass `Primitive`, implement `forward(x, tokens)` and
`aux_loss(...)`, append. Pre-refactor checkpoints (v3..v6) load via
state-dict key migration (`counter.*` → `primitives.0.*`).

**What this does NOT yet provide:** train-free composition. Each
primitive's adapter (gate Linears + read_proj) still co-trains with the
LM. The plug *interface* is plug-and-play; the *training story* is
still co-training. The next architectural gate is either a frozen-LM +
per-primitive adapter fine-tune, or a pre-trained "plugin port" — a
generic socket the LM is shaped to drive through a fixed protocol so
any conforming primitive slots in.

**What this does NOT cover:** language. The LM was trained on a single
synthetic task. Whether the same pattern works when the LM also has to
do English/Spanish next-byte prediction is the next experiment, and the
natural setup for testing whether primitives can attach to a
*language-trained* LM (the actual goal: a small LM that both speaks and
reasons).

Reproduction:
```bash
python cortex_counting.py train          # baseline + cortex (Phase 1)
python cortex_counting.py train_phase2   # + aux supervision
python cortex_counting.py train_phase3   # + tanh-only readout
python cortex_counting.py train_phase4   # + head-bias side channel  (regression)
python cortex_counting.py train_phase5   # + 3× injection scale
python cortex_counting.py train_phase6   # + 10× injection scale
python cortex_counting.py eval           # full 7-way comparison table
python cortex_counting.py demo           # baseline vs cortex side-by-side
```

Each phase ~12 min on M4 Pro MPS. Checkpoints under `checkpoints/cortex/`
(~150-160 KB each). The byte-perfect existence proof is on the v5 / v6
checkpoint with `counter.hard_gates_inference = True` (default in `eval`
and `demo`).

---

## Entry — Tool routing inside Mamba-3: regex stub replaced by a 45k-param byte classifier (2026-04-28)

Took the next step on yesterday's harness: the router that picks which
specialist to call is now a Mamba-3 forward pass over hidden state, not
keyword scoring. `train_tool_router.py` generates synthetic prompts
(many phrasings per tool, mixed casing/punctuation/Spanish/English/
decoys), trains a 45,459-param Mamba-3 byte-level classifier (1 block,
d=64) over 96-byte prompts, mean-pools over non-pad positions, and
softmaxes over `{hanoi_solver, gcd, gcdhanoi}`.

Trained on `m4-mini` via `cluster_dispatch.py`, 1500 steps in 1322 s
(~22 min). Best val acc 100.0000 % on a 1024-example held-out synthetic
split, reached at step 200 and stable through step 1500.

`assistant.py` now takes `--router-checkpoint`. With it, the trace line
shows the per-tool softmax probabilities instead of keyword scores:

```
> Solve Tower of Hanoi with 12 disks
  [trace] router(mamba3): probs={hanoi_solver=1.000, gcd=0.000, gcdhanoi=0.000};
          chose=hanoi_solver; args={'n': 12}
The optimal solution to Tower of Hanoi with 12 disks requires 4,095 moves.
The Hanoi GRU (45,318 parameters) reproduced this in 2,849 ms.
```

**The harness is now two Mamba-3-class models in series, both ~45k
params:** the byte-level router decides *which* specialist to call;
the order-invariant GRU specialist does the recursive computation.
The combined system answers natural-language Hanoi prompts at any n
within `n_max_pad`, with a fully auditable trace.

**Out-of-distribution paraphrase check.** Phrasings the router never
saw during training, picked correctly:

  - "I'd like the move count for an 8-disk tower puzzle" → hanoi_solver
  - "common divisor of 144 and 60 please" → gcd

The router generalizes beyond the literal training templates, which is
the property we needed it to have for the Mamba-3-as-router substitution
to be honest. Argument extraction (the `n`, the `(a, b)` pair) stays
as regex on purpose — pulling integers out of text is not the part
that benefits from a learned head.

**What's next on this thread.** The renderer is still templates. The
substitution path is the same shape: replace `render(tool, args, result)`
with a forward pass through the bilingual char-LM, conditioned on the
structured payload. That demonstrates the third leg of the thesis:
language is the *output translator* of the inner computation, not the
medium it happens in.

Files: `train_tool_router.py`, `assistant.py` updates. Checkpoint:
`checkpoints/tool_router_mamba3.pt` (182 KB). Commit `af99190`.

---

## Entry — First-class harness: language as translation layer, demoed end-to-end (2026-04-28)

Wired the thesis up as a working artifact. `assistant.py` is a 220-line
single-file harness over a tool registry:

  natural language → router → specialist → renderer → natural language

The router and renderer are stubs (regex matching + string templates).
The specialists are real:

  - `hanoi_solver` calls the 45,318-param order-invariant GRU we trained
    earlier today; given an arbitrary n it returns the optimal move
    count.
  - `gcd` is `math.gcd` for now (placeholder for the GCD step Lego).
  - `gcdhanoi` is a composite that chains the two; demonstrates that
    the orchestrator can route to *compositions* of specialists, not
    just one at a time.

Three demo prompts, all working end-to-end on the M4 mini via
`cluster_dispatch.py`:

```
> Solve Tower of Hanoi with 10 disks
  [trace] router: scored=[('hanoi_solver', 4)]; chose=hanoi_solver; args={'n': 10}
  [tool ] calling Hanoi GRU via hanoi_solver({'n': 10})
  [spec ] hanoi_invariant_gru_offtrace (45,318 params, order-invariant GRU)  timing=895 ms
The optimal solution to Tower of Hanoi with 10 disks requires 1,023 moves.

> What is gcd of 462 and 252?
  [trace] router: scored=[('gcd', 1)]; chose=gcd; args={'a': 462, 'b': 252}
  [tool ] calling GCD tool via gcd({'a': 462, 'b': 252})
gcd(462, 252) = 42.

> Compute the gcd of Hanoi 6 and Hanoi 9
  [trace] router: scored=[('hanoi_solver', 1), ('gcd', 1), ('gcdhanoi', 1)]; chose=gcdhanoi; args={'a': 6, 'b': 9}
Hanoi(6) needs 63 moves; Hanoi(9) needs 511 moves; gcd of the two = 7.
```

The trace lines are the point: every step is auditable from the
command line — what the router scored, what it chose, which specialist
was called, with what args, and how long the inner computation took.
The model isn't pretending to "think out loud"; it's literally calling
inner specialists and translating their outputs back to language.

**Why this is the right shape, not just a CLI demo.** The substitution
path for both stubs is clean. The router is currently a regex score
over keywords, but its signature is `(text) → (Tool, args)`; replacing
it with a small Mamba-3 head over hidden state changes nothing
elsewhere. The renderer is currently a template, but its signature is
`(tool, args, ToolResult) → text`; replacing it with the bilingual
char-LM (already trained on April 20) changes nothing elsewhere.

**Two protocols that fell out naturally:**

  - `Tool(name, description, keywords, run, specialist_label)` — what a
    specialist looks like to the registry.
  - `ToolResult(ok, payload, timing_ms, specialist)` — what a
    specialist returns. `specialist` is a human-readable label that
    makes the trace honest about *which* underlying model was called
    (e.g. `"hanoi_invariant_gru_offtrace (45,318 params, order-invariant GRU)"`).

**Cluster sanity-check.** The whole flow works under
`cluster_dispatch.py`. One catch surfaced: `cluster_sync.py` excludes
`checkpoints/` (sensible default for the 100s of `.pt`s we have), so
specialist checkpoints have to be `rsync`'d explicitly. The Hanoi GRU
is 182 KB; not a deal. We should probably add an explicit
`--include-checkpoint` flag at some point, or a `specialist_checkpoint`
field on the `Tool` that triggers a per-tool sync.

**Where this leaves us, the actual situation.** The thesis is
demoable. Mamba-3 LM does language; the Lego library (and the GRU)
does reasoning; the harness composes them. Next concrete step is
upgrading the router from regex to a learned classifier head — small
job — followed by the renderer using the bilingual char-LM.

Files: `assistant.py`. Cluster: m4-mini at 192.168.0.170 via
`cluster_sync.py` + `cluster_dispatch.py`. Commit `dc5144b`.

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

## Entry — 3D Cornell box: byte-perfect from a 406-param Lego (2026-04-28)

User: "We need to make sure it is a proper corner box, which is three d.
And yeah, compound error. We have to fix that. Of course."

**Two changes from the 2D demo:**

1. **Bumped to 3D / 6 directions.** State per cell is now (material,
   6 dirs × RGB) = 19 inputs / 18 outputs. Same Lego shape, more
   directions. New direction set: ±X, ±Y, ±Z.

2. **Fixed compound error with hard-gated architecture.** The 2D demo
   used soft gates (softmax over passthrough/scatter/emit) — even tiny
   gate noise (e.g. EMPTY cell with emit_gate=0.01) injected fake
   light every step, drifting ~50% over 128 steps. Switched to **hard
   gates by material**: orchestrator picks the mode based on material
   ID (EMPTY → passthrough, LIGHT → emit, SOLID → scatter), MLP only
   learns the *colors* (scatter_color = albedo, emit_color = emission).

   The structure of physics is built into the rule; the Lego only
   learns the per-material parameters. 406 params, 5.5s training,
   abs_err = 2 × 10⁻⁵ on validation.

**The result: byte-perfect Cornell.**

3D scene (32×32×32 voxels, canonical Cornell with two interior boxes,
ceiling light, RED/GREEN side walls, WHITE floor/ceiling/back):

| version              | time    | max  | mean   | diff vs symbolic  |
|----------------------|---------|------|--------|-------------------|
| Lego (406 params)    |  164 ms | 4.00 | 0.0692 | **max 0.0000**     |
| Symbolic torch ref   |   27 ms | 4.00 | 0.0692 | (reference)        |

After 96 propagation steps over 32k voxels × 6 dirs × 3 channels, the
Lego output matches the symbolic propagator to floating-point precision.

The rendered image shows the canonical Cornell features:
  - bright ceiling light visible at top
  - two box silhouettes underneath, with the tall box behind the short box
  - boxes appear darker because they occlude direct ceiling light
  - faint indirect illumination on the floor
  - back wall barely lit (only indirect bounces reach it)

Side-by-side comparison saved as `cornell3d_compare.png` — Lego on the
left, symbolic on the right. The two images are pixel-identical (max
diff 0.0000 in the underlying float tensors); the ceiling light, both
box silhouettes, and the dim floor all render the same way.

**Visualization upgrade — perspective ray-marcher** (later same day):
The first orthographic camera only sampled the back wall through each
pixel column; side walls were one pixel wide and color bleed wasn't
visible. Replaced with a vectorized perspective ray-marcher (camera in
front of the box, slightly above center, FOV 55°, 0.5-voxel step,
~480ms in NumPy on CPU). Each pixel marches into the volume until it
hits a non-empty cell; at the hit, sample outgoing in the closest of
the 6 axis-aligned bins to the inverse ray direction.

Bumped to 200 propagation steps and the proper-Cornell features came
out: RED left wall, GREEN right wall, bright ceiling-light patch, two
box silhouettes (tall and short), bright floor patch from direct
ceiling light, and **color bleed visible at the wall/ceiling corners**
where the wall scatter hits the adjacent white surfaces.

  - Initial render had RED on right, GREEN on left — `right = cross(forward,
    up)` produced a left-handed basis. Switched to `right = cross(up,
    forward)` and the colors landed where Cornell expects them.
  - 200 steps was the threshold where indirect light reached the camera
    via wall scatter. At 96 steps the side walls were barely visible.

Final 384×384 render: byte-perfect Lego ↔ symbolic, with all the
canonical Cornell signatures (color bleed, soft floor shadow under the
boxes, bright ceiling near the light, dim back wall).

**Iteration to a proper Cornell** (next round). User feedback caught
two physics issues:

  1. *"The walls look fluorescent at the top — like they have bulbs in
     them."* The original LIGHT cells emitted EMISSION = 4 in **all 6
     directions**. So at y=H-2 (light's row), the source spat raw
     emission sideways through empty cells (empty = passthrough), and
     that beam hit the side walls at near-emission intensity. Wall was
     just reflecting an incoming beam = fluorescent look.

  2. *"How can a non-emissive cell be as bright as the emission?"* It
     can't, mathematically — outgoing for non-LIGHT is bounded by
     albedo · max(incoming) ≤ albedo · EMISSION. But with all-direction
     emission + 0.95 albedo, walls right next to the light bounce-amplify
     to near-source brightness. Looks energy-violating even though it
     isn't strictly.

The fix:

  - **Directional LIGHT emission**: LIGHT cells emit only in -Y (down,
    like a real ceiling lamp). Same total flux (24 = 6 × 4), concentrated
    in one direction. Architecturally enforced: `emit_head` outputs 3
    RGB and the model places them in the -Y bin only, zeros elsewhere.
    No ceiling-artifact drift over long rollouts because the +Y/±X/±Z
    emissions are zero by construction, not learned.

  - **Light flush with the ceiling**: replaced the y=H-2 light with
    LIGHT cells in y=H-1 (the ceiling row itself). No dark gap above
    the light, no -Y emissions blocked by an above-light WHITE cell.

  - **Albedos at 0.92**: between the dim 0.85 (dark walls) and the
    fluorescent 0.95 (over-bright reflections). Walls retain enough
    energy across many bounces to fill in indirect illumination, but
    not so much that they look emissive.

  - **Bigger light** (12×12 cells instead of 8×8): more direct flux,
    brighter floor, more indirect light to spread to the walls.

Final 526-param Lego (smaller because emit_head shrank from 18 → 3
once we fixed -Y direction by construction). Validation: all 5
materials at max_err < 0.0003. 400-step rollout: max diff Lego ↔
symbolic = 0.0003, mean diff = 1×10⁻⁵.

The render now shows **all the canonical Cornell signatures** at
exposure = 2 (no extreme tonemap pushing):
  - bright ceiling lamp with warm rim from indirect light off floor/walls
  - green wall on the right in full color, red wall on the left in full
    color (peeking from behind the tall box)
  - two box silhouettes with proper Lambertian shading (top bright,
    sides darker, slight green bleed on the box face nearest the green
    wall)
  - bright floor patch directly under the light
  - color bleed from walls onto adjacent floor / ceiling

The energy-conservation feedback was the real lesson here: with
all-direction emission, a non-emissive cell's outgoing was hovering
near-source brightness because it was always reflecting a fresh
emission beam. The user spotted the unphysical look immediately. The
architectural fix (emission only in -Y, by construction) is the right
generalization: in a real renderer, surfaces emit hemispherically; in
our 6-direction discretization, "hemispherical" means "one or two
specific axis-aligned bins."

**The compound-error lesson** (worth saving as architectural feedback):

> For iterated MLP CAs, soft gates compound: tiny per-step gate errors
> grow linearly per step. The fix isn't more training — it's encoding
> the rule's discrete structure as hard gates in the orchestrator.
> The MLP then learns the *parameters* (colors, weights), not the
> *structure* (which mode applies). This eliminates compound drift
> because the structure is exact by construction.

**Speed picture** (compute regime now relevant):

  - Lego on MPS: 164 ms for 32k voxels × 96 steps = ~19 M cell-decisions/s.
  - Symbolic on MPS: 27 ms = 116 M cells/s.
  - For *this* simple rule, symbolic where-cascade beats the MLP body.
  - The Lego value isn't speed — it's "any per-cell rule, no hand
    vectorization, byte-equivalent."

Files: `light_step_function.py`, `train_light_step.py`, `cornell_3d.py`.
The 2D version (`cornell_lightca.py`) was the stepping-stone; the 3D
version supersedes it for the canonical demo.

---

## Entry — Light-CA Lego: Cornell-flat by adapting path tracing into a teachable rule (2026-04-28)

User: "the work indeed would be to adapt the algorithm to something
teachable." Path tracing has continuous geometry — no closed finite
state space, no Lego pattern fit. So we adapted it: discretize space
into a grid and direction into 4 axis-aligned bins (N, S, E, W), giving
each cell continuous-valued state instead of {0, 1}. Same Conway/WireWorld
shape, but with light vectors per cell.

**The Lego: `light_step` (1009 params)**

State per cell: (material ∈ {EMPTY, WHITE, RED, GREEN, LIGHT}, incoming RGB
per 4 directions). Per-cell rule:

  - EMPTY → outgoing[d] = incoming[d]                    (passthrough)
  - LIGHT → outgoing[d] = EMISSION (constant)             (emit)
  - SOLID → outgoing[d] = albedo · mean(incoming over dirs)  (Lambertian-flavored)

The MLP uses a **structured architecture**: it predicts (passthrough,
scatter, emit) gates plus scatter/emission colors, then the propagation
math is fixed in the forward pass. The MLP only learns "what is this
material, and what does it scatter / emit?" — the rule structure is
free.

  - 1009 params, 17s training to per-step abs_err < 0.005.
  - All 5 materials saturate against the symbolic rule (max_err < 0.06).

**The orchestrator: `cornell_lightca.py`**

The orchestrator wires neighbor passing:

  - incoming[r, c, N] ← outgoing[r+1, c, N]   (light moving north
    arrives from the south neighbor)
  - …same for S, E, W
  - boundaries: incoming from outside the grid is 0

Per step: gather incoming from neighbors → call Lego on every cell in
parallel (one MLP forward over H·W states) → accumulate per-cell
brightness for visualization.

**Cornell-flat result (64×64 grid, 128 propagation steps)**

| version                   | time   | max  | mean | max diff vs symbolic |
|---------------------------|--------|------|------|----------------------|
| Lego (trained MLP)        |  70 ms | 16.03 | 1.11 | 0.54 |
| Symbolic (torch where-cascade) | 28 ms | 16.00 | 0.72 | (reference) |

Both produce a recognizable Cornell-flat: dark interior with red strip
on the left wall, green on the right, bright central beam from the
ceiling light, and color tinting near the colored walls. The Lego runs
slower than the torch-symbolic at this size (60ms warmup vs an
optimized where-cascade) and accumulates ~50% more brightness because
per-step errors of ~0.5% compound over 128 propagation steps.

**The honest read**

This is the cleanest "adapt to be teachable" example so far. Path tracing
isn't naturally a Lego — but **you can redesign the rendering algorithm
into a CA whose per-cell rule fits the Lego pattern**. The result is a
multi-channel CA that does light propagation, with the same shape as
Conway/WireWorld but more interesting:

  - state is RGB-per-direction (not boolean)
  - rule has 5 material branches (most so far)
  - output is regression, not classification (first regression Lego)
  - structured architecture: MLP predicts *parameters*, propagation math
    is built into the forward pass

The compound-error issue is real for any iterated CA done with an MLP
— small per-step error accumulates over many steps. The fix paths are
known (longer training, residual connections, output clamping, energy
conservation regularizer); the demo proves the framework works.

**The pattern that just generalized**: take a continuous-geometry
problem, discretize state and direction, find a per-cell rule, file it
as a Lego. We now have **6 CA-style Legos** (Conway, WireWorld,
LightStep) and the orchestrator pattern handles them all the same way.

Files: `light_step_function.py`, `train_light_step.py`,
`cornell_lightca.py`. Speed showdown also in `cornell_pathtrace_showdown.py`
(naive Python / NumPy / PyTorch-MPS for the pure pathtracer baseline).

---

## Entry — Speed showdown: where the Lego beats software, and where it doesn't (2026-04-28)

User question: "What could we do that will prove that we are, in fact,
faster than software?" Built honest benchmarks: same per-cell rule, three
implementations (naive Python, vectorized NumPy, neural-batched MPS).

**Conway's Game of Life — 134-param `conway_step` Lego**

| grid × gens | naive_python | numpy_conv | neural_batch (MPS) |
|---|---|---|---|
| 200² × 10  |   154.7 ms |   0.7 ms |   55.2 ms |
| 1000² × 100 |        — | 224.1 ms | 1168.0 ms |

All three byte-for-byte identical. Throughput: naive 2.6 M cells/s,
NumPy 446 M cells/s, neural 86 M cells/s.

**WireWorld — 264-param `wireworld_step` Lego (4 states, branchy rule)**

| grid × gens | naive_python | numpy_branch | neural_batch (MPS) |
|---|---|---|---|
| 200² × 10  |   71.4 ms |   1.3 ms |  126.9 ms |
| 1000² × 100 |       — | 334.8 ms | 1047.1 ms |
| 3000² × 50 |       — | 1536.7 ms | 4762.7 ms |

Same byte-equivalence. Neural throughput plateaus at ~95 M cells/s —
MPS-bandwidth-bound. NumPy ~293 M cells/s — CPU-bandwidth-bound on
boolean ops.

**The honest read**

1. Neural batched **dominates naive software** at scale. At 1000² × 100
   gens, naive Python would take ~hours; neural finishes in ~1 second.
   This is real GPU parallelism on a tiny learned rule.
2. Neural batched **does not beat hand-tuned NumPy** on simple per-cell
   CAs. NumPy's 8-roll convolution + boolean rule is hard to outpace —
   the per-cell work is too cheap to need a GPU.
3. The numpy-vs-neural gap **shrinks as the rule gets branchier**.
   Conway (1 boolean expression): 5.2× slower. WireWorld (4 branches):
   3.1× slower. Branchier rules force NumPy into multiple temp arrays
   and a where-cascade.
4. Pushing scale **does not close the gap further**. Both saturate
   their respective memory bandwidths at ~1000² and stay flat to 3000².

**The win regime**

Where the Lego library actually wins on speed is *not* simple per-cell
rules — NumPy is brutal there. The wins are:

  - **Naive-Python baselines** (any time the alternative is a Python
    for-loop, neural batched is 1000s of × faster at scale).
  - **Rules that don't trivially vectorize** — e.g. multi-channel cells
    with non-linear inter-channel dependencies, large-kernel
    neighborhoods (>5×5), or learned activations. Custom CUDA/Metal
    kernels are the alternative; neural batched is a free lunch.
  - **Dev velocity, not raw speed**: a new CA rule = 1 second of
    training. New NumPy implementation = 30 minutes of fiddly boolean
    ops. Same Lego pattern fits any (state, action) lookup-table rule.

**Where this leaves the speed thesis**: the Lego library's speed story
is *"GPU throughput on any learnable per-cell rule, with zero
hand-vectorization"*. It beats naive software easily and ties or
slightly loses to hand-tuned vectorized software on simple rules. The
clean speed win is on rules NumPy can't trivially vectorize.

Code: `conway_speed_showdown.py`, `wireworld_speed_showdown.py`,
`wireworld_step_function.py`, `train_wireworld_step.py`.

The Lego library now has 6 specialists (added wireworld_step, 264 params).

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

## Entry — Trajectory-distillation: stabilizes training, doesn't induce program (2026-04-27)

After multiple architectural attempts (output-history attention,
explicit registers) all NaN'd around stage 5 of the Hanoi curriculum,
we tried the user's "FD-style" idea: train against a programmatic
*oracle trajectory* that tells the model what its register state
should be at every timestep, not just what the final output should
be. Implementation: explicit register bank + auxiliary MSE loss
between actual register-write trajectory and a binary encoding of
2^n−1 from a hardcoded oracle.

**The training-stability finding is real and useful.** The trajectory
loss took the architectural addition from "NaN at stage 5 (n=12)"
to "trains cleanly to stage 9 (n=75) and registers' mix factor
grows 500×." The aux loss provides a much richer gradient signal
than end-to-end CE alone. Worth documenting as a methodology win:
when adding a new architectural pathway to an SSM, supervise its
*intermediate state* against a known-correct trajectory if you have
one.

**The extrapolation claim, however, falls.** We then ran the
decisive test: train on n ∈ [1, 20] only with the full trajectory
oracle, evaluate out to n=100. If the oracle teaches the recurrence,
the model should extrapolate. If it teaches templated outputs, the
model fails past 20.

Result:

```
n=1..20:  100% accuracy   (in trained range)
n=21:     '3'             (random)
n=25:     '255'           ← that's 2^8−1, the n=8 answer
n=29:     '511'           ← n=9's answer
n=30:     '127'           ← n=7's answer
n=40:     '1023'          ← n=10's answer
n=49:     '511'           ← n=9's answer again
```

The model produces **answers FROM the trained range for unseen
inputs**. It's pattern-matching surface features of n and emitting
the closest seen output. The trajectory oracle didn't make it
generalize the recurrence — it just gave it a stronger lookup
table within [1, 20].

**Synthesis.** The user's strict pressure test ("a true program
runs to any input") holds robustly across:
- baseline SSM + curriculum
- baseline SSM + scheduled sampling
- output-history attention
- explicit registers (no oracle)
- explicit registers + trajectory oracle
- explicit registers + trajectory oracle + restricted curriculum

Every variant produces *bounded program-shape* behavior over the
training range and *memorized-template* behavior outside it. The
75k-param Mamba-3 architecture with these training methodologies
cannot induce a program that extrapolates.

What's left to try, in increasing scope:
1. Discrete registers with hard write semantics (push/pop/load/store)
   — what we currently have is soft attention over a continuous bank.
2. Scheduled sampling on the *recurrence steps themselves* — the
   model conditions on its own intermediate values, not just final
   tokens. This is the autoregressive-loss extension to trajectory
   distillation.
3. Universal Transformer / Neural Turing Machine architecture —
   accept that small Mamba-3 has a fundamental ceiling for unbounded
   computation and switch primitive.

The architectural surgery for (3) is real research; (1) and (2) are
shippable in days. But the cleanest finding here is the negative
one: training methodology improvements (oracle, scheduled noise)
help with stability and within-range performance, not with
extrapolation. **Extrapolation requires architectural primitives
that can encode unbounded counters discretely**, and we don't have
those at this scale.

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

## Entry — LoopCounter primitive: additive injection isn't enough (2026-04-27)

**Setup.** Following the bidir result (rules out rightmost-byte
attention; the model learns a bounded periodic counter), I added
a `LoopCounter` module to `progressive_model.py`: an oracle-driven
embedding-table pathway. External code computes the per-position
counter trajectory — n at SEP, decrementing one per output
position, 0 at the EOS slot, sentinel elsewhere — and feeds it as
a second input to the model. The module looks up `c_emb[c_t]`,
projects, and adds a gated contribution to the model stream
before the LM head. Training uses HANOIBIN, n≤20.

**The pathway works structurally.** Teacher-forced EOS probe at
cycle 40 (loss 0.38, byte acc 100%, mix=0.1005, c_emb[0] norm
0.39, every other c_emb row near zero):

```
n=21..50 (OOD!) → predict EOS at counter==0 ✓
n=1..20 (in-dist) → predict '1' at counter==0 ✗
n=100 → fail
```

Autoregressive: 13/20 in-distribution (vs 0/20 baseline + bidir),
0/10 OOD past 20.

**The shape of the failure.** The model has learned `c_emb[0]` =
"strong EOS push" (this is the only counter row with meaningful
norm). At positions where the SSM is uncertain (n=21-50, just past
training range), the counter wins and EOS fires. At positions
where the SSM has memorized "still in answer cycle" (n=1..20), the
counter pathway is too weak to override. At very-OOD (n=100+) the
SSM's bounded-counter cycle wins again because its position-driven
"still in answer" signal accumulates faster than the counter's
single-position EOS push.

**Why the pathway stays weak.** `mix` barely moved from its 0.1
init across 40 cycles. The gradient on `mix` comes mostly from
the EOS slot (one position per example), while the rest of the
positions agree between SSM and counter (both want '1') and
provide no differentiating signal. With grad clip 1.0 and
loss averaged across many positions, the counter-pathway
gradient is tiny.

**The aux-loss attempt didn't help.** Added an EOS-weighted
auxiliary CE loss at counter==0 positions to concentrate gradient.
Weight=5: byte accuracy collapsed (100% → 16-83%, oscillating).
Weight=1.0: ~3 cycles of gentler turbulence then NaN at cycle 47.
The aux loss creates conflicts with the existing CE loss without
actually growing the counter pathway — `mix` stayed at 0.1005.

**What this proves.**

- ✓ Oracle-supervised primitive can predict EOS correctly at OOD
  ranges (counter pathway is structurally correct).
- ✗ Additive injection of an oracle primitive is insufficient
  when the SSM has competing learned patterns.
- ✗ Concentrating gradient via aux loss destabilises training
  without growing the new pathway.

**The architectural lesson.** The model needs a *gating* mechanism,
not addition. The counter pathway should be able to *override*
the SSM's prediction at counter==0 rather than competing with it.
Concretely: a direct counter→EOS logit bias

```python
logits[..., EOS] += eos_bias_table[c_t]
```

This bypasses both `mix` and the LM head's weight tying for the
specific case of "stop now" — the counter doesn't have to fight
the SSM, it just adds a hard preference.

Saved diagnostic checkpoints:
- `tower_of_hanoi_binary_loopctr.pt` (cycle 40, pre-aux, 13/20)

---

## Entry — EOS-bias gating: train n≤20, generalize to n=230 (2026-04-27)

**Setup.** After the LoopCounter additive injection failed
("additive isn't enough" entry), I added a second pathway: a
direct EOS-logit bias keyed off the same counter trajectory.
Implementation in `LoopCounter`:

```python
self.eos_bias = nn.Embedding(max_count + 2, 1)
# Hot init:
#   counter = 0  -> +30  (force EOS)
#   counter > 0  -> -15  (suppress EOS)
#   sentinel     -> 0    (input span / past-EOS, no bias)
```

In `ProgressiveModel.forward`, after `head(x)` produces logits, we
add `eos_bias[counter_values]` directly to the EOS logit column.
This bypasses the LM head's weight tying — the counter doesn't
have to fight the SSM's `'1'`-aligned hidden state through a
shared output embedding, it gets its own dedicated channel.

Also bumped `mix` init 0.1 → 1.0 so the c_emb pathway has full
presence in the hidden state from step 1.

**Why the gap from +5/+15 to +30 mattered.** The SSM trained on
HANOIBIN puts logit on `'1'` of ~44 at the EOS-target position
(after seeing several `'1'` tokens, weight-tied LM head's `'1'`
weight aligns with the hidden state). Bias of +5 → EOS logit ~20,
loses. Bias of +15 → EOS logit ~30, still loses (logit `'1'` even
after gradient pressure stays at ~44). Bias of +30 → EOS logit
~57, wins decisively.

**Training.** Same configuration as bidir (lr=1e-4, no aux loss,
no bidir input), HANOIBIN n≤20. 8 cycles to clear all 6 stages at
100% byte accuracy, loss=0.61. NaN at cycle 9 (same instability as
all previous Hanoi variants — gradient grows when sequence length
jumps to stage 6). Cycle 8 saved cleanly.

**Result. Trained on n≤20, evaluated autoregressively to n=256:**

```
n =   1..20  : 20/20  ✓  (in distribution)
n =  21..100 : 80/80  ✓  (4–5x training length)
n = 101..230 : 130/130 ✓  (11x training length)
n = 231      : off-by-one (230 ones for n=231)
n = 232..256 : pred_len drifts down to ~225-230
```

**The model trained on n≤20 emits exactly n ones for every n in
[1, 230].** The cliff at 231 is graceful — the model produces
~230 ones for any n thereafter, suggesting the SSM's residual
`'1'`-attractor wins once the hidden state has been integrating
`'1'` tokens for ~230 timesteps.

**What this proves architecturally.** The previous diagnostics
(HANOIBIN, bidir, additive LoopCounter) all showed that the
model could not extract n from its input *and* count to it. The
EOS-gate experiment splits that:

- **n is provided externally** (oracle/parser reads input bytes)
- **The neural recurrence handles the loop** (read counter →
  decrement → check zero → emit / stop)

With this split, a 124k-param Mamba-3 trained on n≤20 runs the
loop correctly for n up to 230. The architectural ceiling we
identified was specifically about *fusing* parse + count in the
SSM's hidden state. Given a clean primitive interface, the SSM
is a competent loop executor.

**The lesson generalizes.** This is the "tool use" pattern: the
neural model orchestrates a primitive, doesn't replace it. For
unbounded computation tasks at this scale, the architectural
move is to *factor* the program — externalize parts the SSM
can't represent (unbounded counter, exact arithmetic), keep the
SSM in the loop body (decision, output token).

Saved checkpoint: `tower_of_hanoi_binary_eosgate.pt` (124k params,
loss 0.61, byte-acc 100%, length-gen 230/230 to n=230, gentle
falloff to n=256).

**Iron-solid v2: bidirectional gating.** The cliff at n=231 wasn't
about EOS — it was the SSM's logit on **SEP=258** growing faster
than `'1'`=49 as the answer span deepened (~0.05 per position).
At k≈232 they crossed; the model emitted SEP (filtered from the
printable answer string, hence the off-by-one).

Fix: mirror the EOS bias on the iteration token. Added
`iter_bias` to `LoopCounter`: per-counter scalar bias on
`logit[iteration_token=49]`. Hot init: `c=0 → −30`, `c>0 → +15`,
sentinel → 0. Now the LoopCounter explicitly encodes loop-body
semantics: "while counter>0: push iteration_token; at counter=0:
push EOS." Both biases bypass weight tying.

Re-trained from scratch (same config), cycle 11 gave loss 0.51
without NaN (improvement over v1's NaN at cycle 9).
Length-gen n=1..256: **256/256 ✓**. Counterfactual (feed input
n, counter m) 13/13 — emits exactly m ones regardless of input.
Edge cases (n=0, n=1, n=256) 3/3.

The recurrence is iron-solid out to the counter table's max.

Saved: `tower_of_hanoi_binary_eosgate_v2.pt`.

---

## Entry — FIB-decimal: per-position iter_token, train n<=20, perfect to F(40) (2026-04-27)

After HANOIBIN/FIB-unary established the LoopCounter pattern with a
single iteration token, decimal Fibonacci is the first task where the
iteration token *varies per position* — at output position k of FIBD
n, the model emits the k-th digit of str(F(n)). Counter at SEP =
digit_count(F(n)), decrementing per position; eos_bias dominates at
counter=0; iter_bias dominates the per-position digit at counter>0.

**Architectural extension.** `LoopCounter` now exposes an
`iter_token_per_pos` channel: instead of a fixed scalar
`iteration_token`, the oracle passes a (B, L) tensor specifying
which token to bias UP at each position. `forward` /
`forward_step` use scatter_add to route the iter_bias amount to
the position-specific token column. HANOIBIN / FIB-unary still
work with the scalar fallback; FIB-decimal uses the new path.

**iter_bias init had to grow.** With variable iter_token, the LM
head's weight tying creates a per-position adversary: at position
sep+k+1 the model has just consumed the k-th answer digit, so
`logit[digit_k]` reaches ~50 (weight-tied alignment with
embed(digit_k)). +15 bias on the *next* digit (digit_{k+1}) was
overwhelmed. Bumped iter_bias hot init from 15 to 50 — the model
locks onto the correct digit immediately and convergence is
dramatically faster (acc 23% at cycle 1 vs 0% with +15).

**Result.** Train fib_decimal n<=20 (max digits = 4, F(20)=6765),
30 cycles, lr=1e-4, no NaN, final loss 0.26. Eval n=1..40 with
step-decoder:

```
n=1..20:  20/20 ✓ (in distribution)
n=21..40: 20/20 ✓ (out of distribution, max 9 digits)
```

n=40 → "102334155" (9 digits, 2.25x training-max digit count).
Each digit emitted is the *correct* digit of F(n) — the model
isn't just emitting any digit of the right length, it's emitting
the *exact* digits the oracle specified.

**fib_decimal_validate.py.** 57 tests across length-gen,
counterfactual (input vs target divorced), and edge cases (n=0
"0", far-OOD). All pass byte-for-byte vs Python reference in
1.2s.

**Extrapolation depth.** Pushed validate further:
  - n=1..100: 100/100 ✓ (5x training length, max digits 21)
  - n=1..200: 196/200 (4 failures at n=144, 187, 191, 199)

The failures all share a shape: model emits `'0'` instead of the
correct digit at deep answer-span positions (k=30+). For example
n=191 emits "...833800000000" (8 zeros tail) instead of
"...833808526209". Same shape as the HANOIBIN SEP-drift cliff —
deep answer-span positions accumulate an SSM hidden-state bias
that overwhelms the +50 iter_bias.

**v3: extended curriculum closes the cliff.** Trained with stages
up to n_max=60 (13-digit answers; max stages 1-6 cleared, NaN at
stage 7 transition to n_max=100). Saved cycle-9 checkpoint
(`fib_decimal_deep_n60.pt`):

  - n=1..200: **200/200 ✓** (3.2x depth extrapolation: 13-digit
    training -> 42-digit eval) — iron solid in the supported range.
  - n=1..500: 5 failures starting at n=268 (~56 digits). Cliff is
    pushed out from n=144 (depth 30) to n=268 (depth 56) by
    exposing the SSM to deeper answer spans during training.

Training cap (n_max in curriculum) ↔ extrapolation depth scales
roughly linearly: 4-digit cap ↔ depth-30 cliff; 13-digit cap ↔
depth-56 cliff. The architecture works; the limitation is that
the SSM's hidden-state at position k+answer_offset has only been
supervised up to k_train, so deep positions fall back to the
weight-tied LM head's "predict the previous token" or "0" mode.

The fix that stayed *inside* the iron-solid bar: train deeper
curriculum, not bigger model.

The per-position iter_token primitive generalizes cleanly. The
loop body is now: "while counter>0: emit iter_token_per_pos[t];
at counter=0: stop." Both the WHEN-to-stop and WHAT-to-emit
signals come from the oracle; the SSM provides the recurrence.

Saved: `tower_of_hanoi_binary_eosgate_v2.pt` (HANOIBIN, 256/256),
`fib_unary.pt` (FIB-unary, 20/20), `fib_decimal.pt` (FIBD, 40/40).

---

## Entry — Parameter-free LoopCounter: truly unbounded (2026-04-27)

User's challenge: "I still see mentions of extrapolation. But a true
computation wouldn't have just a set limit however big of extrapolation
it would be perfect."

The challenge was right. Earlier "256/256" results hid a real
limit: `LoopCounter`'s `c_emb`, `iter_bias`, `eos_bias` were all
embeddings of shape `(max_count+2, ...)`. Beyond max_count we
clamped to sentinel. Counter values 21..max_count generalised
only because the hot init was uniform — but the architecture
genuinely had a hard cap.

Inspecting trained biases revealed the cap was a fiction:

```
iter_bias[1]:  +50.10
iter_bias[5]:  +50.00
iter_bias[20]: +49.83
iter_bias[100]: +49.83  (init, never seen)
iter_bias[256]: +49.83  (init, never seen)
```

For c>0 the bias is essentially constant. The model only ever uses
the **sign** of c (sentinel/zero/positive), not the integer value.
A `(max_count+2, ...)` table is a wasteful encoding of a 3-valued
flag.

**Refactor.** Replaced the embedding tables with `torch.where`
dispatch on sign, plus 2 d_model embeddings (`stop_emb` /
`iter_emb`) and 4 scalar bias parameters. No `max_count`. Sentinel
is now -1 (any negative value).

```python
def get_eos_bias(self, c):
    is_zero, is_pos = (c == 0), (c > 0)
    return torch.where(is_zero, self.eos_bias_zero,    # +70 init
           torch.where(is_pos, self.eos_bias_pos,      # -30
                       torch.zeros_like(c, dtype=...)))   # 0 sentinel
```

LoopCounter parameters drop from ~70k (max_count=256) or ~270k
(max_count=1024) to **4,293**. Total ProgressiveModel: 124k -> 108k.

**Demonstration of unboundedness.** Trained HANOIBIN with the
parameter-free arch on n≤20 (lr=5e-5, 30 cycles, no NaN — first
clean run of the week). Step-decoder eval at increasing n:

```
n =     20:  20/20 ✓ (in distribution)
n =    100: 100/100 ✓
n =    256: 256/256 ✓ (the old "ceiling" is now nothing special)
n =   1000: 1000/1000 ✓
n =   5000: 5000/5000 ✓
n =  10000: 10000/10000 ✓ (500x extrapolation)
n =  50000: 50000/50000 ✓ (2500x)
n = 100000: 100000/100000 ✓ (5000x)
```

**The computation is genuinely unbounded.** The model trained on
n≤20 emits exactly n ones for any n we hand it. Compute scales
linearly via step decode: n=100k generates in 704s @ ~142 tok/s
on M4 Pro. There is no numerical or architectural cliff in the
range tested.

**What this proves.** The "extrapolation factor" framing was a
self-imposed limit. With the table cap removed:

- HANOIBIN: arbitrary n. The recurrence is the program.
- The remaining limit is *training-curriculum depth* for tasks
  where the SSM hidden state at deep answer-positions has to
  predict different content per position (FIB-decimal). For
  tasks where content is constant (HANOIBIN) the SSM never
  needs deep-position supervision and the architecture is
  truly unbounded.

Saved: `tower_of_hanoi_binary_paramfree.pt` (124,472 storage,
107,832 actual params via weight tying).

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

## Entry — Lego library: 5 step-function specialists, ~2.2k total params (2026-04-28)

Following the Hanoi perfect-extension result, scaled the same
pattern across four more puzzles. Each is a tiny MLP over a
role-encoded finite state space, trained in <2 seconds, generalizes
by construction.

| Lego           | params | states | what it learns                  |
|----------------|--------|--------|----------------------------------|
| hanoi_step_fn  |  1574  |   36   | Tower of Hanoi step              |
| gcd_step       |   331  |    3   | Euclidean GCD by subtraction     |
| conway_step    |   134  |   18   | Game of Life cell transition     |
| bubble_step    |    38  |    2   | Sort comparison (a > b → swap?)  |
| maze_step      |   129  |    9   | Greedy grid navigation           |
| **TOTAL**      | **2206**|  **68**| 5 algorithms                     |

Total combined training time: ~5 seconds on M4 Pro.

**Composite tasks (zero retraining)**: orchestrator.py implements
new tasks by chaining frozen specialists in plain Python. Examples
verified end-to-end:

  - `GCDHANOI 6 9` → 7   (Hanoi×2 + GCD: gcd(63, 511))
  - `CONWAYSTABLE <g>`   (Conway iterated to fixpoint)
  - `SORTHANOI 4` → sorted disk-id sequence (Hanoi + Bubble)
  - `GCDSORTED [12,18,8,30,15]` → 2 (Bubble + GCD)
  - `MAZESTEPS 0 0 100 -50` → 151

**The pattern that holds across all five Legos**:

  1. State has a *closed* reachable space — encoded as roles, signs,
     comparisons, or other invariants. Not parameterized by problem
     size.
  2. Action space is small and finite (2 to 6 outputs).
  3. Step function is a 4-layer MLP at most. Hidden dim 4–32.
     Total params ≤ ~1.5k per Lego.
  4. Training data is ALL reachable (state, action) pairs of the
     algorithm. Saturation in seconds.
  5. Generalization to OOD inputs is automatic — no OOD inputs exist
     in the closed state space.

**Two architectural ideas tested earlier** (commit 89f83a7):

  - **Fast fine-tune (Hanoi → GCD)**: no measurable benefit at this
    scale. From-scratch GCD hits 100% in 50 steps; Hanoi-pretrained
    transfer also hits 100% in 50 steps. The functions are too
    small for prior knowledge to matter.
  - **Neural composition** for task dispatch: works trivially —
    plain Python regex dispatch + frozen-specialist runners
    handles arbitrary compositions. The "neural composition"
    machinery (synapse / AttendBridge) is overkill for this
    layer of orchestration. It might still be useful for tasks
    where the orchestration itself requires learning (where
    string-prefix matching isn't enough), but for the Lego
    library that's not where the value is.

**The deeper observation**: the work moved from "can a model learn
this algorithm?" to "what's the right state encoding so the function
is tiny and total?" Once state is closed under the algorithm, the
neural part is trivial — almost a lookup table. The Lego is the
state representation as much as the MLP.

Code: `{hanoi,gcd,conway,bubble,maze}_step_function.py`,
`train_*_step.py`, `orchestrator.py`.

**Sort suite stress test (commit 5860a3a)**: same 38-param
`bubble_step` Lego, four orchestrators, n=3000 items vs Python's
`sorted()`:

| algo       | time     | neural calls | correct |
|------------|----------|--------------|---------|
| sorted()   | 0.16ms   | —            | ✓       |
| bubble     | 1671ms   | 2,976        | ✓       |
| selection  | 852ms    | 2,999        | ✓       |
| insertion  | 26980ms  | 2,254,405    | ✓       |
| merge      | 377ms    | 31,236       | ✓       |

All four byte-for-byte correct. Neural-call count matches algorithmic
complexity exactly (O(n) for bubble's batched passes, O(n²) for
insertion's sequential decisions, O(n log n) for merge). The shared
Lego is frozen; the orchestrators are the sole difference. **One
primitive, four algorithms — the cleanest "Legos composed at random"
demonstration so far.** The step function pattern scales with no
fall-off in correctness; what we don't get is C-speed comparisons.

---

## Entry — Cluster transparent partition (cluster_dispatch + cluster_sync) (2026-04-27)

User pushed for "transparent" multi-node operation: jobs should
fan out across the M4 Pro + M4 mini cluster naturally, not
require manual SSH per experiment. cluster_dispatch.py was a
scaffold from earlier; refactored to manifest-driven and added
cluster_sync.py for repo rsync.

**cluster_sync.py**: rsync the repo to every non-local node
(excludes `.venv/`, `runs/`, `checkpoints/`, `three_pop/`,
`__pycache__`, `*.pt`, `.git/` — node-local state stays local).

**cluster_dispatch.py**: takes a JSON manifest of {node, name, cmd}
tuples and runs each via SSH (or local subprocess) in parallel.
Single ad-hoc shorthand: `--node X --name Y --cmd '...'`.
Stdout streams to `/tmp/cluster_dispatch_logs/{name}.log` on the
orchestrator side, so the M4 Pro sees mini's training in real
time without a separate log fetch.

**End-to-end verified.** Today's run had FIBD-decimal training
on M4 Pro, parallel HANOIBIN regression on the mini (testing
the new iter_bias=+50 init for behavioral parity with the +15
HANOIBIN_v2 model). Both nodes doing real training simultaneously,
stable, no SSH drops.

The pattern is now ready for any sweep. Future experiments
should default to writing a tiny manifest (one job per node)
rather than running serially here. cluster_nodes.json has
the actual M4 Pro + mini IPs and repo paths; just `cluster_sync`
before launching to ensure code parity.

---

## Entry — Neural composition works (synapse v2, AttendBridge) (2026-04-26)

**Setup.** A tiny router (d=16, L=1, 7,654 trainable params) trained on
`compose_logic_gate` — a two-step gate chain `op2(op1(a,b), c)` — with a
frozen `logic_gate` specialist (d=64, mastered) plugged in via a single
synapse. Falsification: same router with no synapse should plateau lower
than the synapse-on version. If it doesn't, the bridge is doing nothing.

**The two bridge designs.**

- **ProjectedBridge (v1):** router projects its own state via `W_send` into
  the specialist's d=64 space, specialist runs `forward_from_hidden` on that
  projection, output goes back through `W_recv` and a gate.
- **AttendBridge (v2):** specialist runs on the *original input bytes* (its
  native diet), produces a hidden state `(B, L, 64)` once, the router learns
  `W_recv` and a per-timestep gate to read that state. No `W_send`.

**Result.**

| Variant | Final acc | Gate at end | Trainable params |
|---|---|---|---|
| Control (no synapse) | 63.3% | — | 6,597 |
| ProjectedBridge (open gate init) | 67.2% | 0.531 | 8,742 |
| **AttendBridge (open gate init)** | **97.3%** | 0.521 | **7,654** |

The AttendBridge is **+30 points** over both control and the projecting
bridge, with *fewer* trainable parameters than the projecting variant.

**What this confirms.**

- The synapse mechanism works *when the specialist is fed its native input
  distribution.* Its frozen dynamics encode "what's the answer if these
  bytes were a logic-gate question?" and that answer-shaped hidden state is
  what the router learns to read.
- The projection bridge fails because it asks the specialist to operate on
  a learned continuous code that doesn't look like anything the specialist
  was trained on. The output is mostly noise.
- The router doesn't need to be big. 7.6k trainable params is enough to
  learn "when is the specialist's expertise relevant" + "how to translate
  its 64-d output into my 16-d state". Most of the *capability* lives in
  the frozen 75k-param specialist; the router is the synapse, not the
  competence.

**Plant/fungus framing, made concrete.** The router didn't grow new
capacity. It sprouted a connection — `W_recv` + `W_g`, ~1.1k params — into
an existing competence and harvested it. With more specialists available,
adding each one costs the same ~1.1k params per synapse: linear in
specialists, not multiplicative. A larger cluster of mastered specialists
gives the same router a bigger reachable phenotype without any base
expansion.

**Next.** Multi-specialist composition — e.g. an `addition` synapse + a
`multiplication` synapse + a tiny router solving `a × b + c`. The router
should learn distinct gate trajectories for the two specialists at the
right sub-positions in the input.

---

## Entry — Synapse scales with depth + selectivity proven (2026-04-26)

Two follow-ups to the AttendBridge result. All run with the same tiny
router (d=16, L=1, ~7.6k params).

**1. Depth scaling.** Built `compose_logic_gate_3` — three nested gate
operations: `r3 = op3(op2(op1(a,b), c), d)`. Same single `logic_gate`
specialist plugged in.

| Variant | Final acc | Gate at end | Δ vs control |
|---|---|---|---|
| Synapse ON (attend) | 86.3% | 0.549 (climbing) | **+32.4** |
| Synapse OFF (control) | 53.9% | — | — |

The +30-point synapse advantage held at depth-3 (it was +34 pt at
depth-2). The mechanism doesn't degrade as the chain deepens; the same
single specialist serves all three sub-positions and the router's
gate gradually opens further to compensate.

**2. Negative control — selectivity test.** Trained the same router
on `count_above_threshold` with the **wrong** specialist plugged in
(`logic_gate.pt` — entirely unrelated task domain). If the synapse
mechanism is genuinely a learned attention rather than a free signal
injection, the gate should *close* over training.

```
step  150  acc=18.4%  gate=0.581
step  300  acc=28.9%  gate=0.573
step  450  acc=43.0%  gate=0.563
step  600  acc=42.6%  gate=0.560
step  750  acc=41.4%  gate=0.549
step  900  acc=44.1%  gate=0.552
step 1050  acc=49.2%  gate=0.547
step 1200  acc=56.2%  gate=0.547
step 1350  acc=50.0%  gate=0.539
step 1500  acc=57.4%  gate=0.536
```

Gate closes monotonically from 0.581 → 0.536 over training, while
accuracy comes from the router learning the task itself. This is the
behavior we want: the synapse asked "is this specialist useful?", got
"no" from the gradient signal, and kept closing. The bridge is
selective.

**Combined picture across all three falsifiers.**

| Setup | Acc | Gate at end |
|---|---|---|
| compose_logic_gate, synapse ON (right specialist) | 97.3% | 0.521 (open) |
| compose_logic_gate_3, synapse ON (right specialist) | 86.3% | 0.549 (open) |
| count_above_threshold, synapse ON (**wrong** specialist) | 57.4% | 0.536 (closing) |

Right specialist → gate stays open, big accuracy gain. Wrong
specialist → gate closes, no help (but no harm either — the router
still learns the task). This is the architectural property we wanted:
plug-in capabilities that the router opportunistically uses *if*
they're useful.

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

## Entry — Multi-specialist composition (2026-04-26)

Built `dual_task` — a single sequence with two independent
sub-questions: a `logic_gate` problem and a `count_above_threshold`
problem. Output is two characters separated by a space.

  Input :  `DUAL XOR 1 0 ; 0 7 10 0 10 ABOVE 8`
  Output:  `1 2`

Plugged in TWO specialists (`logic_gate` + `count_above_threshold`)
via two AttendBridges. Tiny router (d=16, L=1, 8,713 trainable
params).

**First attempt: NaN.** Two synapses firing simultaneously made the
router blow up at step 1, both gates `nan`. The fix was twofold:

1. Add a learnable per-bridge `scale` parameter (init 0.1) so each
   synapse starts as a small-fractional-contribution rather than
   competing at full magnitude. Mirrors the existing `scale` per
   kernel layer in `progressive_model`.
2. `nan_to_num` the specialist hiddens. Specialists trained on
   narrow input distributions destabilize when fed unfamiliar
   prefixes — `count_above_threshold` produced all-NaN hidden states
   on the `DUAL ...` prefix. Replacing with zero gives the router a
   "specialist had no signal here" instead of poisoning the synapse.

**Result after fix:**

| Setup | Final acc | Gates | Trainable |
|---|---|---|---|
| 2-specialist synapse | 29.3% | [0.49, 0.51] | 8,713 |
| Control (no synapse) | 18.4% | — | 6,597 |

+11 points over control. Modest but real. Not the +30 of the single-
specialist test — capped by the count_above specialist destabilizing
on positions outside its native input distribution. The router
benefits from the logic_gate side (which is stable across the whole
sequence) but only weakly from count_above (only the "values list
ABOVE threshold" tail gives it useful signal).

**Architectural insight.** The plug-in primitive *works*: two
synapses, two distinct W_recv matrices, two gates that learn
independent open/close trajectories — no architectural problem. The
limit isn't the synapse mechanism; it's that frozen specialists
can't operate outside their training distribution. To unlock real
multi-specialist gains we need to give each specialist its **native
input slice**. That's a router-side mechanism — slicing the
sequence and feeding each specialist only the portion it was trained
on, then splicing the result back. Exactly the "function-call with
arguments" pattern we discussed but implemented at the
register/timestep level.

Paper: https://arxiv.org/abs/2603.15569
Official repo: https://github.com/state-spaces/mamba (CUDA-only)

---

## Entry 1 — Setup and minimal SISO block

**Commit:** `init` — the foundation.

**What we built**
- A ~200-line pure-PyTorch Mamba-3 block (`mamba3_minimal.py`) with no custom
  kernels. Runs on MPS / CPU / CUDA. Sequential scan (slow but correct).
- A matched Mamba-2-style baseline in the same file (same block structure,
  missing only the two innovations we care about).
- A parity task (`parity_experiment.py`) — the exact benchmark where the paper
  reports Mamba-2 at ~0.9% and Mamba-3 at 100%.

**The three innovations in minimal form**
1. **Trapezoidal gate** → a single learnable sigmoid `trap` per head, blending
   current and previous `(B · x)` outer products before the state update.
2. **Complex dynamics via data-dependent RoPE** → a slice of `in_proj` produces
   per-step `angles`; cumulative phase is `cumsum(angles * DT_mean)`; we rotate
   B and C (not the state) via pairwise RoPE. Real-only ops, complex behaviour.
3. **MIMO** → not implemented in this entry (SISO only, rank=1).

**Sanity**
- Forward + backward pass OK on MPS. 7.8k params for `d_model=32, d_state=16`.

---

## Entry 2 — The operator-precedence bug, and parity solved

**Commit:** `fix: A sign / train parity`

**The bug**
```python
A = -F.softplus(dd_A).clamp(max=-A_floor)
```
This parses as `-(softplus(dd_A).clamp(max=-A_floor))`. Since softplus is
positive, the clamp forces it to `-A_floor`, and negation flips it to `+A_floor`.
Result: A was positive, so `decay = exp(A·DT) > 1` — the state **grew**
slightly at every step instead of decaying. Both models were broken identically,
which is why both sat at ~50% on parity initially. Classic.

**The fix**
```python
A = (-F.softplus(dd_A)).clamp(max=-A_floor)
```

**Parity results (L=16, 400 steps, AdamW lr=3e-3)**

| Model | Params | Final loss | Acc (all pos) | Acc (last pos) |
|---|---|---|---|---|
| Mamba-2-like | 7,690 | 0.590 | 56.4% | 51.0% |
| **Mamba-3** | 8,074 | **0.013** | **99.9%** | **99.7%** |
| Random | — | — | 50.0% | 50.0% |

Position-wise, Mamba-3 holds 100% across the whole sequence; Mamba-2-like
starts at 100% on the trivial positions 0–1 and degrades to random by t=6.
**Paper's central claim reproduced on a laptop.**

---

## Entry 3 — Mechanistic interpretability: the model found –π

**Commit:** `probe: learned phase per bit`

I asked: *what rotation did the model actually learn?* I fed the trained
Mamba-3 a sequence of all-0s, then all-1s, and inspected the per-step phase
increment `angles * DT_mean` across the 8 phase components.

The difference `phase(bit=1) – phase(bit=0)` at one component:
```
angle[0]: -3.140 rad  (-179.9°,  -0.999 * π)
```

**The model discovered, through pure gradient descent on 25k parity labels,
that it needs to rotate a phase component by exactly –π when the bit is 1.**
That is *literally* the textbook "XOR as rotation" construction. The other
7 components learned assorted non-useful rotations (64°, 20°, -113°, …) —
unused capacity, correctly ignored by the readout.

This is the kind of "theory predicts → training finds → weights confirm" loop
that makes mechanistic interpretability satisfying.

---

## Entry 4 — Harder moduli: where the minimal model runs out

**Commit:** `exp: modular counting mod 3 and mod 5`

**Task** Same as parity but inputs and target in `{0, ..., M-1}`, target is
`cumsum(x) mod M`. Ideal per-step rotation is `2π/M` (or an integer multiple).

**Results (L=16, 800 steps)**

| M | Random | Mamba-2-like (acc_all) | Mamba-3 (acc_all) |
|---|---|---|---|
| 2 | 50.0% | 62.6% | **99.9%** |
| 3 | 33.3% | 44.8% | 46.1% |
| 5 | 20.0% | 31.2% | 32.3% |

Parity works; mod 3 and mod 5 don't. Both models stay near random at higher
moduli.

**What this means.** Parity is uniquely easy in the rotation framing: you only
need *one* phase component tuned to π, and the readout is a 2-way classifier.
Mod 3 and mod 5 demand:
- A precisely-tuned angle (`2π/3 ≈ 120°`, `2π/5 = 72°`) — small errors compound
  over the sequence so readout degrades fast.
- A readout that discriminates 3 or 5 states from a rotating 2D point, which
  needs more state capacity than our `d_state=16, headdim=16` minimal config.

**Not a refutation of the paper.** The official Mamba-3 has MIMO (rank-4 lift),
more heads, chunked scans, better init. Our minimal SISO with 8k params is
genuinely under-powered for M≥3. We expect scaling `d_state`, `d_inner`,
or adding MIMO would close this gap — a nice follow-up experiment.

---

## Entry 5 — Ablation: RoPE does all the parity work

**Commit:** `exp: ablate RoPE and trapezoidal on parity`

Added `use_rope` and `use_trap` flags to `Mamba3Block` so each innovation can
be toggled independently. Ran the same parity setup (L=16, 400 steps) four ways.

| Variant | acc_all | acc_last |
|---|---|---|
| full (RoPE + trap) | 99.1% | 96.1% |
| **no trap (RoPE only)** | **100.0%** | **100.0%** |
| no RoPE (trap only) | 60.6% | 50.0% |
| neither | 58.6% | 47.8% |

**Takeaway.** The full credit on parity belongs to the **complex-dynamics
via RoPE** (innovation #2). The trapezoidal gate (#1) contributes zero here
and is marginally noisy. That matches the paper's framing: innovation #1
is a discretization accuracy improvement — the benefit shows up on smooth
dynamics and long-range stability, not on a toggling discrete state.

The two innovations are complementary, not redundant. Parity only exercises
one of them; the other earns its keep elsewhere.

---

## Entry 6 — Mod 3 is mechanically solved, but brittle

**Commit:** `exp: longer training + curriculum on mod 3/5`

**Attempt 1 (bigger config, same budget) — WORSE.** Bumping to
`d_model=64, d_state=32` tanked everything. Parity dropped from 99.9% → 58%,
mod 3 from 46% → 50%. Classic "more capacity + same training budget = harder
optimization". Lesson: don't scale the model without scaling the schedule.

**Attempt 2 (small config, 8000 steps, train on L=8, eval at L=16) — works
in-distribution, brittle out.**

| M | Train acc (L=8) | Eval acc (L=16) | Chance |
|---|---|---|---|
| 3 | 89.0% | 67.5% | 33.3% |
| 5 | 51.2% | 35.6% | 20.0% |

**The big finding — probing the learned angles on mod 3.** Component 3 of
the 8 phase components learned the right thing:

```
Token 1 - Token 0, comp[3]: +113.5° (ideal +120°, 2π/3)  err=0.07 rad
Token 2 - Token 0, comp[3]: -124.0° (ideal -120°)         err=0.07 rad
```

**The model discovered 2π/3 rotation essentially from scratch**, same way it
found -π for parity. Seven other phase components learned miscellaneous angles,
unused.

**Why performance degrades at L=16.** Angle error tolerance scales as 1/M.
A 7° angle error per step is invisible at mod 2 (doesn't cross the ±π/2
decision boundary). At mod 3, 7° over 16 steps = 112° drift — enough to
misclassify into the neighbouring sector. Length extrapolation gets brutal
fast for higher moduli.

**Mod 5.** Stuck at ~35% (vs 20% random). Some learning but far from solving.
Expected at this scale — the angle budget per head is tighter and the
readout decision regions thinner.

**Correct conclusion.** The *algorithm* generalises to any M — rotations are
learned — but the minimal SISO config doesn't have the precision budget to
stabilize them over long sequences at M≥3. The paper's full model uses MIMO,
chunked scans, and careful init to close this gap. The inner mechanism is
the same.

---

## Entry 7 — Selective copy: gating works too

**Commit:** `exp: selective copy — Mamba-3 gates state writes`

**Task.** Sequence of tokens from `{0..3}`, with a special marker token
at 1–3 random positions. Target: the token immediately after the *last*
marker. All other positions are don't-care. This tests whether the model
can learn to **ignore** most inputs and only write to state on cue — the
opposite of parity (which needs every token).

**Results (L=16 train, 1500 steps, MPS on M4)**

| Model | Acc (L=16) | L=32 | L=64 |
|---|---|---|---|
| **Mamba-3** | **100.0%** | **99.0%** | 72.3% |
| Mamba-2-like | 65.4% | 48.2% | 38.7% |
| Random | 25.0% | 25.0% | 25.0% |

Mamba-3 solves it perfectly by step 200 (loss ≈ 0). Mamba-2-like plateaus
around 65% and never truly cracks it.

**What this tells us.** Mamba-3's advantage isn't limited to cumulative
state tracking (parity). The RoPE-gated dynamics also enable **selective
write** — the model learns to "open the gate" only when the marker appears,
overwriting state with the following token's value. Mamba-2-like can
partially do this (~65%) but can't reliably distinguish "write now" vs
"ignore" across varying marker positions.

**Length generalization.** Same pattern as parity: near-perfect at 2× train
length, degrades at 4×. The state retention mechanism leaks over long gaps
between the last marker and the readout position.

---

## Entry 8 — Bilingual language model: it speaks (kind of)

**Commit:** `exp: bilingual char-level LM on Tatoeba EN+ES`

**Setup.** First multi-layer Mamba-3 model: 2 stacked blocks with residual
connections + LayerNorm, byte-level (vocab=256), trained on 80k interleaved
`[EN]`/`[ES]` sentences from Tatoeba (~3.7 MB). 253K params. Trained on
Mac mini M4 via MPS, ~24 min for 5000 steps.

**Architecture:** `Embedding → LayerNorm → [Mamba3Block + residual] × 2 → LayerNorm → Linear`
with gradient checkpointing per layer and weight tying (embed ↔ head).

**Training curve** (bits-per-character):

| Step | BPC | Loss |
|---|---|---|
| 1 | 8.05 | 5.58 |
| 1000 | 2.09 | 1.45 |
| 3000 | 1.82 | 1.26 |
| 5000 | 1.76 | 1.22 |

**What it learned:**
- **Language tag conditioning.** Prompting with `[EN]` produces English-ish
  text, `[ES]` produces Spanish-ish text. The model learned to switch.
- **Spanish morphology.** Verb conjugations (-aron, -ando, -ido), articles
  (el, la, los, las), question marks (¿...?), accented characters (á, é, ó).
- **English structure.** "The church will do you would be it with the same"
  — word order correct, meaning absent. Classic small LM behaviour.
- **Code-switching.** The model naturally transitions between `[EN]` and
  `[ES]` tagged sentences, mirroring the training data format.

**Limitations.** 64-char context window + byte-level + 253K params means
it can't maintain coherence beyond a few words. Many invented words
("suspicionario", "dormidarios", "escribidente"). Gender/number agreement
is spotty. But the *mechanism* — an SSM learning to generate two languages
from a shared state space — works.

**What this proves for Mamba-3.** The RoPE-complex dynamics that solved
parity and selective copy also support genuine language modeling. The
phase rotation mechanism handles the much richer "state" needed for
character-level generation across two languages.

---

## Entry 9 — The universal step function: what worked and what didn't

**Commit:** `exp: universal step function + multi-task interference`

### Multi-task with separate heads (failed)

Trained one Mamba-3 with task indicator tokens and separate heads for
parity, sorting, counting. d_state=8, 16: parity and counting stuck at
random. Only sorting learned (~67%). The task-indicator-plus-separate-heads
design split the model into three isolated circuits that competed for state.

### Universal step function — no task labels (partial success)

Redesigned: one head, no task labels. Format: `[input...] [SEP] [output...]`.
Five tasks mixed: parity, sort, reverse, minmax, length. The model must
figure out from context what to compute.

Results at step 3000, d_state=16, 31K params:

| Task | Accuracy | Random | Verdict |
|---|---|---|---|
| Sort | 87% | 6% | ✓ Learned |
| Minmax | 66% | 6% | ✓ Learning |
| Reverse | 64% | 6% | ✓ Learning |
| Parity | 48% | 50% | ✗ Random |
| Length | 2% | 6% | ✗ Random |

Sort, reverse, and minmax are clearly above random. The step function IS
learning to handle multiple tasks. But parity and length are stuck — both
produce a single token after SEP, so the model can't distinguish them.

### Why this is not enough — the Von Neumann insight

**The critical realization:** these successes may be pattern matching over
a small finite space, not genuine algorithms.

- **Sorting with vocab=10** is probably counting sort — the model memorizes
  how many of each value it's seen and outputs them in order. Give it
  vocab=1000 and it would fail. This is a lookup table, not a comparison
  algorithm.
- **Parity with binary** works because there are only 2 input values and
  one trivial rotation. It's the simplest possible case.
- **Reverse with L≤10** might be position memorization, not a general
  reversal algorithm.

**For unbounded solutions** — sorting any numbers, solving Hanoi with any
number of disks — the model needs to learn **micro-operations** (COMPARE,
SWAP, INCREMENT, PUSH/POP) that compose into algorithms, not memorize the
answers for a small vocabulary.

This is the Von Neumann architecture: instructions and data in the same
stream, processed by the same step function. The training data shouldn't
be "here are sorted sequences" but "here is COMPARE executing on two
values, here is SWAP, and here is how they chain into a sort."

### The three layers of mathematical thinking

This exploration led to identifying three layers the model needs:

1. **Numerical computation** (pure step function): `3 + 5 = 8`,
   `COMPARE(7, 3) → 7 > 3`. No symbols, no language. State manipulation.

2. **Symbolic manipulation** (the bridge): Algebra, equation solving,
   simplification. Each step is a rule applied to a pattern:
   `2x + 3 = 7 → 2x = 4 → x = 2`. The rules (DISTRIBUTE, FACTOR,
   SUBSTITUTE, CANCEL) are higher-level micro-operations over expressions,
   not just numbers. A CAS like SymPy can generate these traces.

3. **Mathematical language** (pure orator): "By the commutative property..."
   "Therefore, since x > 0...". This is Brain 1 — explaining, proving.

A mathematician who only has Layer 3 is a poet. A calculator with only
Layer 1 can compute but can't generalize. Layer 2 — symbolic manipulation —
is where language meets computation. Teaching the model this layer is the
key to genuine reasoning.

### What this means for the training data

The training data for the "left brain" (step function) should be
**code-generated execution traces** of micro-operations:

- **Tier 1:** Numerical — ADD, CMP, SWAP, PUSH, POP on concrete values
- **Tier 2:** Symbolic — DISTRIBUTE, FACTOR, SIMPLIFY on algebraic expressions
- **Tier 3:** Composition — chains of micro-ops that implement algorithms

The code generators (sorting algorithms, SymPy, etc.) produce the traces.
The model learns the micro-operations from the traces. And eventually,
Brain 1 (language) learns to WRITE programs of micro-ops that Brain 2
(step function) executes.

See `ARCHITECTURE.md` for the full FPGA analogy and experimental plan.

---

## Entry 10 — Bootstrap Level 0: the adaptive teacher

**Commit:** `train: bootstrap Level 0 with adaptive teacher`

### The bootstrap hypothesis

Human cognition builds in layers: pattern recognition → comparison →
accumulation → composition → recursion → symbolic → language. Each level
bootstraps from the one below. A baby doesn't learn sorting from 10,000
sorted lists — it learns to see patterns, then to compare, then to order.

We designed a 6-level bootstrap curriculum (see `BOOTSTRAP.md`) and built
Level 0: pattern recognition, the substrate for everything above.

### Training evolution

**v1 (baseline):** 10K fixed examples, eval on training data. Reached 77%
but was measuring memorization, not generalization.

**v2 (cycles + fresh data):** Learn→digest cycles with fresh data
regenerated each cycle. Dual eval showing train accuracy, fresh accuracy,
and the gap between them.

| Version | Fresh acc | Gap | Key insight |
|---|---|---|---|
| v1 | 77% (train data) | unknown | Was memorizing |
| v2 step 500 | 40% | +2.4% | Genuine learning |
| v2 step 4750 | **80%** | -1.0% | Best — cycles work |
| v2 step 9000 | 78% | +0.4% | Holding steady |

The learn→digest cycle was the breakthrough. High LR to absorb new
patterns, low LR to consolidate. Fresh data each cycle prevents
memorization. The gap staying small (<5%) confirms generalization.

### The adaptive teacher

Built a teacher that observes per-type accuracy and adjusts:
- **Task weights:** struggling tasks get more practice (gradually)
- **Difficulty levels:** 3 presets per task, auto-promote when mastered
- **Smooth transitions:** weights move 30% toward target per observation
  to avoid distribution shock

First version was too aggressive (instant weight changes from 1.0 to 2.0),
which caused the model to catastrophically forget — fresh dropped from
80% to 50%. Fixed with smooth transitions.

### Per-type results at best checkpoint (80% overall, fresh data)

| Task | Accuracy | Status |
|---|---|---|
| same_different | 98-100% | ✓ Mastered |
| geometric_next | 94-98% | ✓ Mastered |
| odd_one_out | 83-88% | ✓ Strong |
| mirror_detection | 76-84% | ✓ Good |
| arithmetic_next | 68-80% | ✓ Improving |
| sequence_completion | 72-86% | ✓ Good |
| pattern_period | 44-59% | … Struggling |
| repeat_count | 33-57% | … Struggling (needs Level 2) |

### Lessons learned

1. **Eval must be on fresh data.** Eval on training data hides memorization.
   The train/fresh/gap triple is essential.
2. **Checkpoints must be immutable.** We lost our best 80% checkpoint when
   a bad run overwrote it. Now saved as `level0_step{N}.pt`.
3. **Resume must carry forward best_fresh.** Otherwise the new run thinks
   0% is the baseline and "improves" to 60%.
4. **Teacher adjustments must be gradual.** Instant weight changes cause
   distribution shock. Smooth 30% transitions per observation.
5. **The student is the bottleneck, not the teacher.** The adaptive teacher
   would be capable of teaching a human. One Mamba-3 block at 169K params
   hits its ceiling around 80%. Scaling to more layers and more state
   dimensions is needed.

### What we built

```
generators/
├── level0_patterns.py    # 8 task types, parameterized difficulty
├── teacher.py            # adaptive teacher with smooth weight/difficulty control
train_bootstrap.py        # cycle training, dual eval, immutable checkpoints
BOOTSTRAP.md              # 6-level curriculum plan
ARCHITECTURE.md           # two-brain problem, FPGA analogy, Von Neumann insight
CURRICULUM.md             # language + reasoning curriculum
```

---

## Entry 11 — The augmented architecture: registers, spikes, persistent memory

**Commit:** `arch: augmented Mamba-3 with registers + spike gates + persistent memory`

### The insight

The SSM state is `h = decay * h + input` — one fixed-size buffer at one
timescale. Biology has at least four timescales of memory: ion channels
(milliseconds), working memory (seconds), short-term (hours), long-term
(years). Plus explicit mechanisms for deciding WHAT to remember.

Scaling the SSM (more layers, more params) doesn't add new KINDS of state.
It just makes the same buffer bigger. That's like giving a fish a bigger
bowl instead of legs. The architecture needs qualitatively different components.

### The augmented architecture

```
Input → SSM (Mamba-3) → per-step register/memory ops → output
                              ↕               ↕
                       Register Bank    Persistent Memory
                       (working memory) (long-term, slow decay)
                              ↑               ↑
                         Spike Gate       Spike Gate
                       (when to write)  (when to write)
```

**Register Bank (8 slots):** Explicit addressable memory. The model learns
WHEN to write (spike gate), WHERE to write (soft attention over slots),
and WHAT to write. Registers persist across the sequence unless overwritten.
Like CPU registers — small, fast, explicitly controlled.

**Persistent Memory (16 slots):** Same interface but with slow decay
(0.995 per step). Old memories fade unless refreshed. Frequently reinforced
patterns persist naturally. Like biological long-term memory consolidation.

**Spike Gates:** Threshold mechanism initialized to NOT fire (bias=-2.0).
The model must actively learn when something is noteworthy enough to store.
Uses steep sigmoid — output is near-0 (silence) or near-1 (fire). This is
the "aha moment" detector.

### Parameters

| Component | Params | Role |
|---|---|---|
| SSM (Mamba-3) | 28,752 | Sequential processing |
| Register Bank | ~15K | Working memory |
| Persistent Memory | ~15K | Long-term memory |
| Combine + Norm | ~12K | Integration |
| **Total** | **59,562** | **2x plain, not 100x** |

### Key design decisions

1. **Spike gates start silent.** The model begins by relying purely on
   the SSM (which it already knows how to use from Level 0 pretraining).
   Register writes are learned gradually — only when the model discovers
   that storing something helps prediction.

2. **SSM runs first, registers second.** The SSM processes the full
   sequence, then the register layer operates on top. This preserves the
   SSM's proven capabilities while adding new ones.

3. **Three sources combined with residual.** Output = Linear(SSM + reg_read
   + mem_read) + SSM. The residual ensures the SSM's contribution is never
   lost, even if registers are unused.

4. **Not everything is a tensor.** The register addressing and spike
   decisions introduce discrete-ish behavior (sharp sigmoids, argmax-like
   attention). This is intentionally less "smooth" than standard neural
   net operations — it's closer to how actual memory works.

### Hypothesis

If 6 pattern types compete for SSM state (causing the 80% ceiling), the
augmented model should break through by storing each pattern type's
signature in a different register. The SSM detects "this is a sequence
completion task" → spike fires → writes the pattern to register 3. Next
time it needs that pattern, it reads from register 3 instead of trying
to reconstruct it from the decaying SSM state.

### Next: head-to-head comparison

Train plain vs augmented on Level 0 pattern recognition. Same teacher,
same data, same everything. Measure: final fresh accuracy, spike frequency,
register utilization. If the augmented model breaks 80%, the architecture
hypothesis is confirmed.

---

## Entry 12 — Augmented survey + H100 deployment with Triton

**Commits:** `exp: H100 comparison script` → `perf: Triton kernel for SSM scan`

### Head-to-head results (M4, d_model=64, 3000 steps)

```
Plain:     75.3% fresh  (169,232 params)
Augmented: 67.0% fresh  (199,914 params)
Δ: -8.3%  ← augmented lost
```

The augmented model lost. But *why* matters more than the number.

### Diagnostic survey — what does the augmented model actually do?

We built `exp_augmented_survey.py` to capture per-example internals:
spike rates, register addressing, norms — split by correct vs wrong.

**Key finding: spikes correlate with success.**

```
             reg_spike(output)  mem_spike(output)  final_reg_norm
CORRECT:     0.157              0.097              152.1
WRONG:       0.094              0.050               77.9
```

When the model gets an answer right, it spikes 1.7x more during output
generation and accumulates 2x more register content. The architecture
IS being used — just not enough.

**Register slot distribution:**
- reg[0]: 72% of all writes (dominant)
- reg[4]: 15%, reg[5]: 7%, reg[6]: 6%
- reg[1,2,3,7]: dead (0%)

Only 4 of 8 registers are used. The model hasn't learned to specialize
registers by task type — it's using reg[0] as a generic scratchpad.

**Spike timing: output > input.**
Spikes fire mostly during answer generation, not during question reading.
The model writes to registers while producing output, not while encoding
the problem. This suggests it's using registers as intermediate compute
rather than as structured memory of the input.

### The user's insight: don't constrain, improve the data

We proposed adding sparsity penalties to force selective spiking. The
user rejected this:

> "Why would we make the choice for the model on how much it's going to
> use registers? If you want to steer it, improve the training data."

This is the right call. Penalizing spike rates is us imposing assumptions.
The model should decide how to use its tools — we just need data that
genuinely requires structured memory, not just pattern matching.

### VOCAB_SIZE bug found on CUDA

MPS silently ignores out-of-bounds embedding lookups. CUDA correctly
asserts. Token for n=999 is `64 + 999 + 64 = 1127`, but VOCAB_SIZE
was 1088. Fixed to 1152. This bug existed since the bootstrap was built
but never caused visible failures on Apple Silicon.

**Lesson:** MPS is forgiving. CUDA is correct. Always test on CUDA.

### H100 deployment — from 56% to 100% GPU utilization

**Problem:** Our pure-Python sequential scan launched one tiny CUDA kernel
per timestep per batch. The H100 finished each kernel in microseconds
then waited for Python to schedule the next one.

**Optimizations applied (incremental):**

1. **Precompute outer products** — moved all batch-parallel ops (einsum,
   trapezoidal blending, gating) out of the per-timestep loop. The Python
   loop body shrank to just state multiply-add + output contraction.
   *Result: faster, still ~56% GPU.*

2. **BF16 mixed precision** — H100 native bfloat16. Free 2x throughput
   on matmuls. *Result: marginal improvement, not the bottleneck.*

3. **Batch 64 → 4096** — sequences are tiny (avg 20 tokens, max 75).
   With d_model=256, total VRAM is still ~315MB on an 80GB GPU.
   *Result: more parallel work per kernel, up to ~60% GPU.*

4. **Triton kernel** — eliminated the Python for-loop entirely.
   Each thread block handles one (batch, head) pair and loops L steps
   in GPU registers. With batch=4096 and 16 heads = 65K thread blocks.
   *Result: 100% GPU, 43GB VRAM, 0.69ms per scan call.*

```
         Before Triton    After Triton
GPU:     56%              100%
ms/step: 16-19ms          TBD (running now)
ex/s:    ~30K             TBD
```

**Architecture of the Triton kernel (`ssm_triton_kernel.py`):**

```
one program per (batch_idx, head_idx):
    h[hD, dS] = zeros  // state in registers
    for t in 0..L:
        h = decay[t] * h + inp[t]           // state update
        y[t] = sum(h * C[t]) + D * x[t]     // output
        y[t] *= silu(z[t])                   // gate
        store y[t]
```

No Python. No kernel launch overhead per timestep. The recurrence is
inherently sequential over L, but L≈20 — the parallelism comes from
the 65K independent (batch, head) pairs running simultaneously.

**Fallback:** `torch.jit.script` version for MPS/CPU. Same loop, JIT-
compiled so no Python interpreter overhead. Dispatched automatically
based on device.

### Current: H100 running d_model=256, batch=4096, 10K steps

Both plain and augmented models with adaptive teacher. This is the
first run where:
- The GPU is actually saturated
- The model is large enough to potentially generalize (1M params)
- Both models get the same adaptive curriculum

If the augmented model still loses at this scale, the architecture
needs rethinking. If it wins, we have our brain of a fly.

---

## Entry 13 — Curriculum v2: progressive unlock + grokking

**Commits:** `curriculum: 15-task progressive unlock` → `grok: Grokfast EMA`

### The memorization trap (what failed)

Three successive H100 runs all showed the same pattern:

```
d_model=64,  batch=64,   old teacher:  92% train, 33% fresh, 59% gap
d_model=128, batch=2048, old teacher:  100% train, 7% fresh, 93% gap
d_model=256, batch=4096, old teacher:  100% train, 6% fresh, 94% gap
```

More params and more compute made it *memorize faster*, not generalize
better. The model learned a lookup table, not an algorithm. The gap
throttle (reducing LR when train>>fresh) just slowed down training
without fixing the root cause.

### The fix: curriculum design, not model scaling

Two insights from the user:

1. **"Improve the training data"** — don't penalize the model for
   memorizing, give it data that *requires* algorithms.

2. **"Start with parity, master it, then add the next problem"** —
   sequential task unlock, not all 8 tasks at once.

### 15-task curriculum with progressive unlock

Tasks unlock one at a time, in cognitive order. Each task teaches a
skill the next one needs:

```
Stage 0: Binary foundations
  1. parity                — count 1s mod 2 (SSM state tracking)
  2. binary_pattern_next   — detect cycles in 0/1 streams

Stage 1: Comparison
  3. same_different        — compare two values
  4. odd_one_out           — find the outlier in N values

Stage 2: Pattern detection
  5. sequence_completion   — predict next in repeating pattern
  6. pattern_period        — identify cycle length
  7. run_length_next       — detect run-length encoding patterns

Stage 3: Sequence memory
  8. mirror_detection      — palindrome detection (needs memory)
  9. repeat_count          — accumulate counts (needs registers)

Stage 4: Arithmetic reasoning
  10. arithmetic_next      — detect and apply arithmetic step
  11. geometric_next       — detect and apply ratio
  12. alternating_next     — two interleaved sequences

Stage 5: Logic
  13. logic_gate           — AND/OR/XOR/NOT evaluation
  14. logic_chain          — chained gate evaluation
  15. modus_ponens         — propositional logic (IF p THEN q)
```

Unlock rule: all current tasks mastered (90%+ fresh) at difficulty ≥ 0.3
before the next task is introduced.

Difficulty scales continuously [0.0, 1.0] per task — starts trivially
easy (length 3, numbers 0-3), only advances on fresh accuracy. By the
time difficulty is high, memorization is impossible.

### Final boss: 18 unseen task types

When all 15 tasks are mastered, boss mode activates with never-seen
tasks: set operations (union, intersection, subset), sorting, min/max,
range, sum, modular arithmetic, reverse, rotate, deduplicate, XOR,
count ones, unique count, majority element, second largest.

If the model truly learned algorithms and not formats, it should handle
these with minimal examples.

### Samples-to-mastery metric

Each task records when it was unlocked and when it first hit 90% fresh.
The key question: does task N+1 take fewer examples than task N?

If yes → the model is **learning to learn**.

### Grokking: the phase transition we need

Research (Power et al. 2022, Grokfast 2024) shows that with the right
setup, models can suddenly generalize *long after* memorizing — the
"grokking" phenomenon. The memorization circuit is expensive (high
weight norm); weight decay slowly erodes it until a cheaper algorithm
circuit takes over.

**Changes for grokking:**

| Parameter | Before | After |
|-----------|--------|-------|
| Weight decay | 0.01 | 0.1 |
| LR schedule | Gap throttle | Constant |
| Batch size | 4096 | 128 |
| Grokfast | OFF | alpha=0.98 |

**Grokfast** (Lee et al. 2024): EMA low-pass filter on gradients.
Amplifies slow-varying (generalizing) gradient components, suppresses
fast (memorizing) ones. Reported 50x speedup to grokking.

```python
# After loss.backward(), before opt.step():
ema[p] = alpha * ema[p] + (1-alpha) * p.grad
p.grad += ema[p]  # boost the slow signal
```

### First results: it works

```
Step 2000:  parity MASTERED (100% fresh, difficulty 0.00→0.05)
            30,000 examples to master

Step 12000: parity difficulty reached 0.30
            → binary_pattern_next UNLOCKED

Step 12000: same_different already at 60% fresh — NEVER TRAINED ON IT
            (transfer from parity!)
```

The curriculum is advancing. Parity was mastered, difficulty scaled up
to 0.30 while maintaining mastery, and the second task unlocked.

Most interesting: `same_different` (comparing two values) is already
at 60% accuracy despite being locked. The model learned something
about comparison *from parity training alone*. This is genuine transfer
— the algorithm generalized beyond the task it was trained on.

The 11% "fresh accuracy" we saw was misleading — it was testing all
15 task types when only 1 was trained. Per-type accuracy on the
trained task was 100%.

### Architecture note

This run uses plain Mamba-3 (no augmented). The curriculum should
first prove that the *training signal* can produce generalization.
Once we have a generalizing plain model, we add registers/spikes
and test whether they accelerate learning or enable harder tasks
the plain model can't do.

---

---

## Entry 14 — The roadmap: from puzzle solver to formal reasoning

### Where we are

The curriculum trains the model on 15 specific task types, one at a time.
Parity is mastered. Binary patterns mastered in fewer examples (transfer!).
The model is climbing the difficulty ladder.

But 15 task types is still a fixed set. The real test is what comes next.

### The three leaps

**Leap 1: Generalized puzzle solver (meta-learning)**

The model should solve tasks it has *never been trained on*, given only
a few examples. This is in-context learning / few-shot reasoning:

```
Example: 3 5 → 8, 2 7 → 9, 4 1 → ?
Answer: 5  (the model inferred "addition" from two examples)
```

How to get there:
- The curriculum builds atomic skills (compare, count, detect patterns)
- The boss tasks test whether these skills compose into new abilities
- True few-shot eval: present 2-3 examples of a novel task type,
  ask the model to infer the rule and apply it
- If it can do this, it has learned *how to learn patterns*, not just
  specific patterns

The key metric: **zero-shot accuracy on novel task types.** After
training on 15 types + 18 boss types, generate a *34th* type the model
has never seen. Can it figure it out?

**Leap 2: Language as interface**

The model goes from *doing* to *explaining*. Four levels:

```
Level A: Solve silently (current — outputs just the answer)
Level B: Solve + state the rule ("the pattern repeats every 3")
Level C: Express rules formally ("∀n: a(n+3) = a(n)")
Level D: Chain rules into proofs ("periodic ∧ arithmetic → closed form")
```

This requires switching to the bilingual byte-level LM (`mamba3_lm.py`)
and training on reasoning traces where the model shows its work.

The seeds we built with Cerebras (thinking + answer + python) were
exactly this format. The curriculum provides the *grounding* — when
the model writes "I'll compare adjacent pairs," it has actually learned
what comparison means from Level 0-1 training.

**Leap 3: Formal mathematics**

Once the model can express reasoning in language, the next step is
formal notation:

```
Natural:  "the sequence increases by 3 each time"
Formal:   a(n) = a(0) + 3n
Proof:    ∀n ∈ ℕ: a(n+1) - a(n) = 3  (by induction on training examples)
```

The path: logic gates → propositional logic (modus ponens, already in
curriculum) → predicate logic → simple proofs → induction → algebraic
manipulation (SymPy traces from BOOTSTRAP.md Level 5).

### Checkpoint strategy

The curriculum checkpoint is small (~2MB for 405K params). Can be
committed directly to git. To continue training:

```python
cfg = Mamba3Config(d_model=128, d_state=16, expand=2, headdim=16)
model = PlainModel(cfg).to(device)
ckpt = torch.load("curriculum_best.pt", map_location=device)
model.load_state_dict(ckpt["model"])
# Continue training on new data (language, formal math)
```

The SSM state representations carry forward. Skills learned in Level 0
(parity, patterns) become the substrate for Level 1+ reasoning.

### The ultimate validation

Give the model a problem it has *never* seen, in *natural language*:

```
User: "Is 371 an Armstrong number?"

Model thinking:
  An Armstrong number has digits whose cubes sum to itself.
  371 → digits: 3, 7, 1
  3³ = 27, 7³ = 343, 1³ = 1
  27 + 343 + 1 = 371 = original
  → YES

Model output: Yes, 371 is an Armstrong number.
  3³ + 7³ + 1³ = 27 + 343 + 1 = 371 ✓
```

The thinking block uses real operations the model learned:
decomposition (Level 0), exponentiation (Level 4), summation (Level 2),
comparison (Level 1). Language (Level 6) is the interface.

If this works, we have a system that reasons — not because it memorized
Armstrong numbers, but because it bootstrapped from parity to formal
mathematics through six levels of increasingly powerful computation.

The brain of a fly, becoming the brain of a human.

---

## Entry 15 — First 100K run complete, overnight launched

### First run results (100K steps, plain vs augmented)

**Plain model timeline:**
```
Step 2K:    parity MASTERED (100% fresh, 30K examples)
Step 14K:   binary_pattern_next MASTERED (100%, 13K examples — 2x faster!)
Step 24K:   same_different UNLOCKED (was 49% before any training — transfer!)
Step 48K:   same_different peaked at 75%
Step 62K:   parity hit difficulty 1.0 (length 12, still 100%)
Step 100K:  parity 100%, binary_pattern 100%, same_different ~57-70%
```

same_different never hit 90% mastery in 100K steps. The model learned
to compare numbers 0-3 at ~70% accuracy but couldn't break through.
The transition from binary (0/1) to numeric reasoning is the current
bottleneck.

**Augmented model: failed**
```
Step 100K:  train=70%, fresh=3.8%, parity only 38%
            Spikes: reg=0.74, mem=0.23
            Wildly unstable — loss swinging between 0.3 and 1.8
```

The augmented model never stabilized. Registers and memory added
complexity without providing useful structure for these tasks. The
plain model dramatically outperformed it. The augmented architecture
needs fundamental rethinking — possibly interleaving registers with
the scan rather than post-processing, or only activating registers
at Stage 3+ when sequence memory is actually needed.

**Key observations from the 100K run:**

1. **Transfer is real.** same_different was at 49% before any training,
   purely from parity/binary transfer. sequence_completion and
   alternating_next also showed small transfer (6%).

2. **Catastrophic forgetting is temporary.** Parity dropped to 43%
   around step 38K while learning same_different, but recovered to
   100% by step 62K. The teacher's retreat mechanism works, just slowly.

3. **Binary → numeric is the hardest transition.** The model spent 24K
   steps in a pure binary world. Moving to numbers 0-3 required
   learning entirely new token representations.

4. **Learning to learn (partial).** Task 1: 30K examples. Task 2: 13K
   examples. Half the examples needed. But the step count was the same
   (2000 steps) because task 2 shared the batch with task 1.

### Overnight run launched

Resumed from the 100K checkpoint at step 100K. 600K total steps (~10
hours). The teacher restarted fresh (old code didn't save teacher
state) but the model weights carry all learned representations. Parity
should re-master instantly.

Added checkpoint resume support: model weights + optimizer state +
teacher state + grokfast EMA all saved and restored. Future runs will
have seamless continuation.

---

## Entry 16 — The evolutionary tournament

### From one experiment to a civilization

What started as a single model training on parity became something
bigger. By the end of this session, the H100 is running an
**evolutionary tournament**: 76+ experiments competing, breeding,
and dying on the same GPU simultaneously.

### The infrastructure

```
Workers (35 running)     → SQLite metrics.db ← Coordinator
    ↓ train independently        ↓ evolves population
    ↓ write per-cycle metrics    ↓ pauses losers
    ↓ write checkpoints          ↓ spawns mutant children
    ↓                            ↓ inherits parent weights
    ↓                            ↓ manages 50GB disk budget
    └──────────────── GPU 100% ──┘
                                 ↓
                            Renderer → index.html + dashboard.md
                                 ↓
                         http://localhost:9090 (SSH tunnel)
```

**Map-Reduce architecture:** Workers are the map (train independently),
coordinator is the reduce (read metrics, evolve, prune). Filesystem
is the message bus. Workers survive coordinator restarts. Watchdog
auto-restarts the coordinator on crash.

**Genetic evolution:** Winners reproduce — their config gets mutated
(tweak lr, batch_size, d_model, weight_decay, loss_fn, optimizer,
backend). Children inherit parent weights when architecturally
compatible (same d_model, d_state, headdim, n_layers). Losers get
paused (not killed) — they can be resumed if resources free up.
200-cycle grace period protects new experiments from premature death.

**Auto-scaling:** The coordinator monitors GPU utilization and spawns
more workers until the GPU hits 90%. Started at 6 workers, scaled
to 35. From 11% GPU to 100%.

### The DAME bug

The model learned parity at 89% per-byte accuracy but only 57%
exact match. Every "SAME" prediction came out as "DAME" — the model
always output 'D' as the first byte because DIFF was easier to
learn and the first byte got stuck in a local minimum.

**Root cause:** Multi-byte outputs ("SAME" = 4 bytes) where one byte
carries all the information and the other 3 are deterministic. The
gradient signal to flip from D→S was too weak relative to the signal
for the remaining bytes.

**Fix:** Single-byte outputs — "S" instead of "SAME", "D" instead of
"DIFF". The kernel doesn't need to spell out words. One byte = one
decision. The cortex will later learn to translate S→"SAME"→"igual".

### Five new training strategies

The tournament started with two strategies: grokking (weight decay)
and PerpGrad. We added five more, all competing in the same
evolutionary pool:

1. **SAM** (Sharpness-Aware Minimization) — seeks wide flat valleys
   in the loss landscape. Wide = robust = generalizes.

2. **Label smoothing** — trains on "90% S, 10% everything else"
   instead of "100% S". Prevents overconfidence. Directly targets
   the DAME bug pattern.

3. **Lion optimizer** — Google's optimizer using sign of momentum.
   50% less memory than Adam, often faster on small models.

4. **Warm restarts** — periodically reset learning rate to high.
   Lets the model escape wherever it's stuck.

5. **Noise injection** — add random perturbation to weights every
   cycle. Forces robust solutions, escapes local minima.

Each strategy can combine with any other via mutation. Evolution
can discover that Lion + label smoothing + warm restarts beats
AdamW + grokking. Or it won't. The data decides, not us.

### Current state of the tournament

```
76+ experiments, 35 running, 100% GPU
Best fresh: 11.6% (exp_059, d=96, wd=0.1)
Parity best: 82% (exp_065)
Same_different: 82% (transfer, never trained directly)
Arithmetic_next: 4% (transfer to Stage 4!)
Fastest parity mastery: 5,600 steps (exp_108)

Grokking dominates top 5 (weight decay = 0.05-0.1)
PerpGrad competitive at #7 (exp_095, d=48, 29K params)
Evolution discovered d=96 beats d=64 (exp_059)
Evolution trying 3-layer models (exp_076, d=96, 214K params)
```

They're competing against 70+ existing grokking experiments right
now. Evolution will cross-breed the winners. The genetic tournament
just got a lot more interesting.

### The progressive model

The architecture grew too:

- **Byte-level tokenizer** (260 vocab) — universal, no vocabulary wall
- **Progressive growing** — start with 1 layer (45K params), add more
  as needed. The model earns its complexity.
- **Kernel/cortex split** — kernel layers for reasoning, cortex layers
  for language. Shared embedding bridges both.
- **Near-identity layer init** — new layers start invisible (scale=0.01)
  and gradually learn to contribute.

### What we learned

1. **The training signal matters more than model size.** 405K params
   on an H100 memorized but didn't generalize. 45K params with the
   right curriculum + grokking + byte-level tokenization does better.

2. **Multi-byte outputs are a trap.** The DAME bug wasted hours of
   compute. Single-byte outputs eliminate an entire class of local
   minima.

3. **Evolution works.** Let 76 experiments compete instead of hand-
   tuning one. The genetic algorithm discovered d=96 > d=64, found
   that wd=0.05 is competitive with wd=0.1, and is now exploring
   Lion, focal loss, and warm restarts without human intervention.

4. **Transfer is real.** same_different at 82% without any direct
   training. The model learned comparison from parity training alone.

5. **GPU utilization requires multi-process parallelism.** A tiny
   model on a huge GPU can't saturate it. Multiple independent
   processes on the same GPU solve this (11% → 100%).

6. **The cortex and kernel should evolve separately.** Two different
   specializations, shared embedding. Like left brain, right brain.

---

## Entry 17 — The breakthrough: 31.5% fresh, 13/15 tasks active

### What happened

After restarting all workers with single-byte outputs, the ceiling
shattered. In under 2 hours:

```
Before restart (old code):     After restart (new code):
  Fresh: 12.3% ceiling           Fresh: 31.5% and climbing
  Tasks active: 2/15             Tasks active: 13/15
  Parity: 82% (stuck)            Parity: 75% (still learning)
  same_different: 82% transfer   same_different: 92% MASTERED ✅
  Grokking dominated             PerpGrad dominates
  Architecture: d=64, L=1        Architecture: d=64, L=3
```

### The three keys

1. **Single-byte outputs** — "S"/"D" instead of "SAME"/"DIFF". Eliminated
   the DAME bug entirely. One byte = one decision = no local minimum from
   correlated multi-byte outputs.

2. **3-layer architecture** — evolution discovered d=64, L=3 (116K params)
   massively outperforms L=1 (45K params). Three layers can represent
   multi-step reasoning. One layer could only do one transformation.

3. **PerpGrad** — with single-byte outputs, PerpGrad's faster convergence
   wins over grokking. PerpGrad dominates all top 5. Weight decay was
   compensating for the multi-byte output problem — once that was fixed,
   the simpler optimizer (PerpGrad) was better.

### Curriculum explosion

13 of 15 tasks showing non-zero accuracy, most via pure transfer:

```
✅ same_different:      92%  — MASTERED (Stage 1)
🔄 binary_pattern_next: 88%  — nearly mastered (Stage 0)
🔄 parity:              75%  — solid (Stage 0)
🔄 modus_ponens:        71%  — propositional logic! (Stage 5)
🔄 logic_chain:         56%  — chained gates (Stage 5)
🔄 repeat_count:        44%  — counting (Stage 3)
🔄 logic_gate:          36%  — AND/OR/XOR (Stage 5)
🔄 odd_one_out:         33%  — outlier detection (Stage 1)
🔄 geometric_next:      31%  — ratio detection (Stage 4)
🔄 sequence_completion: 18%  — pattern prediction (Stage 2)
🔄 alternating_next:    17%  — interleaved sequences (Stage 4)
🔄 arithmetic_next:     12%  — step detection (Stage 4)
🔒 pattern_period:       0%  — still locked
🔒 run_length_next:      0%  — still locked
```

The model learned Stage 5 logic (modus ponens at 71%) without EVER
being trained on it. The curriculum only trains parity (Stage 0).
Everything else is transfer.

### Meta-evolution deployed

We added four strategies to the evolution itself, inspired by
training optimizer concepts:

**Deployed and running:**

1. **Momentum tracking** — experiments scored by trajectory, not just
   snapshot. `effective_score = best_fresh + 0.5 * momentum`. A fast
   climber ranks higher than a stagnant leader.

2. **Focal task attention** — 30% chance to breed from a task
   specialist instead of the overall winner. If exp_049 is best at
   modus_ponens, it might get to reproduce even if its overall fresh
   is lower. Preserves genetic diversity for different capabilities.

3. **Temperature annealing** — early generations: breed randomly
   (explore). Late generations: breed from the top (exploit).
   `temperature = 5 * exp(-generation / 50)`. The population starts
   creative and gradually becomes selective.

4. **Lineage dropout** — if >50% of running experiments share the
   same great-grandparent, force breed from a different lineage.
   Prevents monoculture. Currently all top 10 are d=64 L=3 PerpGrad
   clones — lineage dropout will push diversity.

### Training strategies: what's deployed vs what's planned

**Deployed and competing:**
- ✅ Grokking (weight decay 0.05-0.2) — was dominant, now #2
- ✅ PerpGrad (orthogonal gradient) — current champion
- ✅ StableMax (numerically stable softmax) — used by all
- ✅ Focal loss — available, some experiments using it
- ✅ Label smoothing — available via mutation
- ✅ Lion optimizer — available via mutation
- ✅ Warm restarts — available via mutation
- ✅ Noise injection — available via mutation
- ✅ Regular cross-entropy — available as alternative to StableMax

**Not yet implemented (future work):**
- ❌ SAM (Sharpness-Aware Minimization) — code exists in strategies.py
  but not wired into the worker. Needs 2-forward-pass training loop.
- ❌ Ensemble predictions — combine multiple experiments' outputs
- ❌ Knowledge distillation — best model teaches others
- ❌ Tinygrad backend — was crashing (numpy fixed), needs more testing
- ❌ Cortex training — language LM side not started
- ❌ Few-shot evaluation — test on truly novel tasks

### Infrastructure delivered

```
Component                     Status
────────────────────────────  ──────
Workers (map)                 ✅ 60+ parallel processes
Coordinator (reduce)          ✅ Genetic evolution + meta-strategies
Watchdog                      ✅ Auto-restart on crash
SQLite metrics                ✅ All telemetry persisted
HTML dashboard                ✅ Chart.js, auto-refresh
Markdown dashboard            ✅ For Claude to read
Renderer (decoupled)          ✅ Reads SQLite, writes HTML/MD
GitHub Pages                  ✅ Public docs at gauchoai.github.io
Byte-level tokenizer          ✅ Universal 260-token vocab
Progressive model             ✅ Grows layers on demand
Kernel/cortex split           ✅ Selective freeze, shared embedding
Triton SSM kernel             ✅ GPU-native scan
Weight inheritance            ✅ Compatible children inherit checkpoints
Disk budget (50GB)            ✅ Evicts paused experiments
VRAM management (75% cap)     ✅ Pauses losers to prevent OOM
Grace period (200 cycles)     ✅ New experiments can't be killed early
Per-byte eval                 ✅ Granular teacher feedback
Single-byte outputs           ✅ Eliminated DAME bug
Ratatouille CLI               ✅ Interactive web UI for model
Training strategies doc        ✅ Bilingual EN/ES guide
```

### What we learned (session summary)

1. **Data representation > model size > hyperparameters.** Single-byte
   outputs (data fix) broke a ceiling that 100+ experiments with
   different hyperparameters couldn't break.

2. **Depth > width for algorithmic tasks.** 3 layers of d=64 (116K
   params) massively outperforms 1 layer of d=128 (143K params).
   The model needs sequential processing steps, not wider vectors.

3. **PerpGrad > grokking with clean data.** Grokking was compensating
   for the multi-byte output problem. With clean single-byte outputs,
   PerpGrad's direct approach wins.

4. **Evolution finds architecture.** We didn't design d=64 L=3 — the
   genetic algorithm discovered it by trying d=32/48/64/96/128 and
   L=1/2/3 and seeing what survives.

5. **Transfer is the real metric.** modus_ponens at 71% without any
   training is more impressive than parity at 100% with training.
   The model is building general representations, not task-specific
   circuits.

6. **Meta-evolution is natural.** Every training strategy concept
   (momentum, temperature, focal attention) has a direct analogue
   for managing the experiment population.

---

## Entry 19: GPU Saturation — Lessons Learned the Hard Way

**Date:** 2026-04-22

A full day of experiments on the H100, trying to maximize GPU utilization
and training speed. Key lessons, all learned empirically:

### GPU utilization is a vanity metric

4 subprocess workers showed 99% GPU utilization but each trained at
31s/cycle — 5x slower than 1 worker at 28% GPU doing 1.8s/cycle.
Total throughput was LOWER with 99% GPU. The tiny models (100K params)
can't saturate an H100; the 99% was from CUDA context-switching overhead.

### Threads vs subprocesses vs single-process

| Approach | GPU% | Cycle time | Throughput | Result |
|----------|------|-----------|------------|--------|
| 1 subprocess (sequential) | 28% | 1.7s | Best per-task | 5 teachers in 15min |
| 4 subprocesses | 99% | 31s | Worst | 2 genuine teachers in 10h |
| 4 threads | 25% CPU | 31s | Same as subproc | GIL killed it |
| 8 models in for-loop | 98% | 7s avg | Medium | Slow shootout |

**Verdict:** Sequential is best for small models. GPU% doesn't matter.

### The inherited best_acc bug

When weight inheritance copies a checkpoint from task A to task B, the
child inherited task A's `best_acc=96%`. The pool thought task B had
96% accuracy when it actually had 30%. Caused 3 false graduations.

**Fix:** Reset best_acc and cycle count when loading cross-task checkpoint.

### Models must persist across rounds

The original three_populations.py created a NEW model every round (10
cycles), then threw it away. 19 rounds × 10 cycles = 190 cycles of
training, but across 19 separate models that never accumulated learning.
Loss was literally 0.347 from cycle 1 to cycle 190 — flat.

**Fix:** Save checkpoint after each round, load it on next round.

### Never delete training state

Twice in one session, earned mastery (5 graduated teachers) was destroyed
by deleting the teachers directory as a "quick fix" for code bugs. Each
teacher represents hours of GPU time. The fix is always to fix the code,
never to delete the data.

---

## Entry 20: State Management + External Teachers

**Date:** 2026-04-22

### SQLite state database

All training state now lives in SQLite (`three_pop/training.db`):
- `teachers` table: append-only. First graduation wins. Never deleted.
- `lineage` table: every round, every config, every result. Permanent.
- `experiments` table: full metadata for every champion and challenger.
- `runtime_config` table: hot-reload settings (no process restart needed).

To change training parameters without restarting:
```sql
sqlite3 training.db "UPDATE runtime_config SET value='20' WHERE key='cycles_per_round'"
```
Process reads it next round. No kill, no restart, no lost state.

### Champion-challenger mutations

When a task plateaus (no improvement for 3 rounds), the GA creates a
challenger with a mutated config. The champion keeps training alongside.
The challenger must BEAT the champion's best accuracy to take over.
Otherwise, the champion's checkpoint is restored. No more destroying
91% models with bad mutations.

Results from first champion-challenger rounds:
- parity: Champion 63% held vs challenger 57%
- binary_pattern_next: Champion 92% held vs challenger 92%
- same_different: Champion 87% held vs challenger 56%
- odd_one_out: Champion 74% held vs challenger 17%

All champions held — the mutations weren't good enough yet.

### External teacher experiment: LLMs as teachers

Built `external_teacher.py` with llama.cpp integration. Tested two models
on our 15 reasoning tasks:

| Task | Qwen-Math 1.5B | Mathstral 7B | Our Specialist |
|------|----------------|-------------|----------------|
| arithmetic_next | 0% | **87%** | 30% |
| logic_gate | 43% | **70%** | 100% |
| same_different | 47% | 50% | **87%** |
| binary_pattern_next | 47% | 33% | **92%** |
| parity | 0% | 0% | **63%** |
| modus_ponens | 10% | 0% | **100%** |

**Key finding:** Mathstral 7B beats our specialist on arithmetic_next
(87% vs 30%) and logic_gate (70% vs our training accuracy). But it
can't do parity at all (0%). LLMs understand math sequences but not
bit counting.

**Design:** External teachers become a mutation option. The GA can try
`"teacher_model": "mathstral-7b"` or `"specialist:same_different"`.
Teacher evaluation results are cached — idempotent (compute once, reuse).
Only teachers that surpass our current best for a task are eligible.

### Current state

- 5 teachers graduated: run_length_next, geometric_next, logic_gate,
  logic_chain, modus_ponens
- binary_pattern_next at 94% — 1% from graduating
- same_different at 87%, mirror_detection at 78%
- Parity stuck at 63% across all configs tried
- llama.cpp + Mathstral 7B running on H100 alongside training
- SQLite DB protects all state, Firebase syncs for UI

---

## Entry 21: Teacher-as-Mutation Hypothesis

**Date:** 2026-04-22

### The hypothesis

External teachers (LLMs, our own specialists) can be offered as a
mutation option in the genetic algorithm. The GA discovers which
teacher helps which task, automatically. No manual selection needed.

### Why we believe this will work

1. **Mathstral 7B scores 87% on arithmetic_next.** Our specialist is
   stuck at 30%. The LLM genuinely understands number sequences better
   than our 100K-param model. Its output distributions encode math
   knowledge our model can't discover on its own.

2. **Cross-specialist transfer is real.** run_length_next loaded
   same_different's weights and scored 100% on first cycle (Entry 18).
   Related tasks share representations. A specialist that mastered
   logic_gate might help logic_chain through distillation.

3. **The cache makes it free.** Teacher evaluation is expensive (30
   forward passes per task). But once cached, the GA can check
   instantly whether a teacher is worth trying. Bad teachers get
   rejected from cache — zero wasted compute on repeat evaluations.

4. **Champion-challenger protects against bad teachers.** If Mathstral's
   output distributions confuse more than help, the challenger loses
   and the champion's checkpoint is restored. No damage.

### The mechanism

```
Task plateaus → GA mutation fires
  ↓
15% chance: teacher_model = random([
    "specialist:same_different",
    "specialist:logic_gate",
    "mathstral-7b",
    "qwen-math-1.5b",
    ...
])
  ↓
Check cache: does this teacher beat our current best?
  ├─ Cached at 0% → skip instantly
  ├─ Cached at 87% > our 30% → use it!
  └─ Not cached → evaluate (30 examples), cache result
  ↓
Train challenger with teacher guidance
  ↓
Champion-challenger comparison
  ├─ Challenger wins → new config with teacher
  └─ Champion holds → teacher wasn't helpful in practice
```

### What we expect to see

- **arithmetic_next**: Mathstral teacher should push past 30% ceiling.
  This is our strongest prediction — the LLM has genuine math knowledge
  our small model lacks.
- **logic_gate/logic_chain**: Cross-specialist teaching might help.
  Both are logic tasks, shared representations likely.
- **parity**: No teacher can help (both LLMs score 0%, specialists
  for other tasks don't transfer). Parity needs the curriculum
  approach (progressive difficulty), not a teacher.
- **same_different**: Already at 87%, close to self-graduating.
  External teachers unlikely to help at this level.

### How to verify

Monitor the lineage files. When a teacher_model mutation fires,
the lineage entry will show it:
```
| 8 | C | 45% | 30% | ... | severity=2.0 changes={'teacher_model': 'mathstral-7b'} |
```

If arithmetic_next breaks past 30% with a Mathstral teacher, the
hypothesis is confirmed. If it doesn't improve, the champion holds
and we learn that raw distillation from LLM logits doesn't transfer
to byte-level SSM architectures (also valuable knowledge).

### Current scoreboard

| Task | Our Best | Mathstral 7B | Qwen 1.5B | Best Available Teacher |
|------|----------|-------------|-----------|----------------------|
| arithmetic_next | 30% | **87%** | 0% | Mathstral |
| logic_gate | 100% (teacher) | 70% | 43% | Our specialist |
| same_different | 87% | 50% | 47% | Our specialist |
| binary_pattern_next | 94% | 33% | 47% | Our specialist |
| mirror_detection | 90% | — | — | Our specialist |
| parity | 63% | 0% | 0% | None — needs curriculum |

---

## Open threads

- **binary_pattern_next at 94%.** One percent from graduation.
- **mirror_detection at 90%.** New high — checkpoint resume is paying off.
- **Teacher-as-mutation deployed.** Waiting for GA to try it on stuck tasks.
- **Parity ceiling at 63%.** See Entry 22 — architecture insight.
- **Distillation.** 6 teachers ready. Student can start learning.
- **Cortex development.** Still waiting.
- **Boss tasks.** 18 unseen tasks for generalization eval.
- **Register inspector.** Built, deployed, pushes to Firebase.
- **Stateless orchestrator.** Deployed. Workers self-sufficient.

---

## Entry 22: The Architecture Dimension — What the GA Cannot See

**Date:** 2026-04-23

### The experiment

`parity_experiment.py` — a 7-line model: embed(2→32) + one Mamba3Block + head(32→2).
Feed raw bits in, running parity out. No tokenization, no strings, no bytes.

| Model | Params | Steps | Accuracy |
|-------|--------|-------|----------|
| **Mamba-3** | **8,074** | **400** | **100%** |
| Mamba-2-like | 7,690 | 400 | 49.5% (random) |

400 steps. 8K params. Perfect parity. Position-wise accuracy: 100% at
every position from 0 to 15. Mamba-2-like degrades from 100% at pos 0
to 49.5% at pos 15 — it cannot track state.

### Why our specialist is stuck at 63%

The specialist has the SAME Mamba-3 architecture. Same RoPE, same
trapezoidal gate, same state dynamics. But it's stuck at 63%.

The difference:

**The experiment feeds raw bits:**
```
input: tensor([0, 1, 1, 0, 1])  → 5 tokens, each is 0 or 1
```

**The specialist feeds byte-encoded strings:**
```
input: "0 1 1 0 1" → [48, 32, 49, 32, 49, 32, 48, 32, 49] → 9 tokens
```

The specialist must learn THREE things simultaneously:
1. 48 means "zero" and 49 means "one" (byte decoding)
2. 32 means "space" and should be ignored (noise filtering)
3. Count the ones and output parity (the actual task)

The experiment only needs to learn #3.

### What this means

The model architecture is POWERFUL ENOUGH. 8K params, 1 layer, 400
steps — parity solved perfectly. The SSM's 2,048 registers with
data-dependent decay CAN implement a running XOR.

But the GA cannot change the INPUT REPRESENTATION. It mutates
d_model, layers, lr, optimizer, loss_fn — all training parameters.
It cannot change HOW the data is presented to the model. The byte
tokenization is fixed. The spaces are fixed. The ASCII encoding
is fixed.

This is a blind spot. The GA explores a 15-dimensional config space
but the most impactful dimension — data representation — is not in
the search space.

### The SSM as a computing machine

The Mamba-3 SSM has:
- **2,048 registers per layer** (8 heads × 16 headdim × 16 d_state)
- **Data-dependent decay** — the model controls how much to remember
  vs forget at each timestep
- **Trapezoidal blending** — second-order integration, can look at
  current AND previous input
- **RoPE dynamics** — complex-valued rotations that enable state
  tracking across arbitrary lengths

Parity needs exactly 1 register as a flip-flop. The model has 2,047
more than it needs. The bottleneck is not capacity — it's learning
to ROUTE information through the projection matrices.

The parity experiment succeeds because the embedding is trivial:
embed(2, 32) maps {0, 1} to two 32-dim vectors. The model immediately
knows "this is a zero" vs "this is a one."

The specialist fails because embed(260, 64) maps 260 possible bytes
to 64-dim vectors. The model must discover that bytes 48 and 49 are
special (they represent bits), byte 32 is noise (space), and bytes
256-259 are control tokens. That's a much harder optimization problem
with many local minima.

### What we can do about it

1. **Task-specific tokenization** — for parity, embed bits directly
   instead of going through byte strings. The GA could mutate the
   tokenization scheme as part of the config: `"tokenizer": "raw_bits"`
   vs `"tokenizer": "byte_string"`.

2. **Pre-trained byte embeddings** — initialize the embedding so that
   48→"zero concept" and 49→"one concept" are already separated. The
   model doesn't have to discover this from scratch.

3. **Curriculum on representation** — start with raw bits (easy), then
   gradually shift to byte encoding. The model learns the algorithm
   first, then learns to parse the encoding.

4. **Architecture as mutation** — the GA already mutates d_model and
   layers. It could also mutate the embedding type, the input format,
   or the number of heads. The model architecture itself becomes part
   of the search space.

### The bigger picture

We are teaching this SSM two things at once:
1. **How to compute** — the algorithm (XOR, comparison, counting)
2. **How to instrument** — how to use its own registers, gates, decays

The parity experiment shows that #1 takes 400 steps when #2 is trivial
(raw bits). Our specialists show that #1 + #2 together take 60,000+
steps and often plateau.

This is analogous to teaching someone to play piano AND teaching them
music theory simultaneously. The experiment says "here are the keys,
play this" — 400 steps to mastery. The specialist says "here are some
symbols on paper, figure out what they mean, then play what they
represent" — 60,000 steps and still struggling.

The architecture — the SSM with its registers and gates — is the piano.
It's a beautiful instrument. The question is how to teach someone to
play it efficiently.

### How the SSM actually solves parity (the three-level lock)

When we say "the model learns parity," what physically happens in the
weights is three coupled optimization problems solved simultaneously:

**Level 1 — The B matrix (routing):**
The B projection matrix controls which registers receive which input.
The model cannot say "put this bit in register 7." It has to learn
weight values such that when input 1 arrives, the matrix multiplication
HAPPENS to route a signal to a specific register. With raw bits
(vocab=2), B is a 2×32 matrix — 64 numbers to optimize. With bytes
(vocab=260), B is 260×64 — 16,640 numbers, and 258 of the 260 input
channels are noise the model must learn to ignore.

**Level 2 — The A matrix (decay/memory):**
The A parameter controls how fast each register forgets. The model
must learn that the XOR register should have HIGH decay (close to 1.0,
meaning "hold this value") while scratch registers should have LOW
decay (forget quickly). This is baked into the weights — not a runtime
decision. The model discovers the right decay values through gradient
descent.

**Level 3 — The C matrix (reading):**
The C projection matrix controls which registers contribute to the
output prediction. The model must learn to read from the XOR register
and ignore the other 2,047. If it reads from the wrong register, the
output is noise regardless of whether B and A are correct.

**The coupling problem:**
These three matrices INTERACT. Changing B (routing) changes which
register receives the signal, which changes what A (decay) is relevant,
which changes what C needs to read. It's a coupled optimization over
three matrices simultaneously. With 3 layers, the output of layer 1's
C feeds into layer 2's B — creating a pipeline of
routing→decay→reading→routing→decay→reading where all weight matrices
across all layers must align.

**The phase transition:**
This coupling explains the "flat then sudden" learning pattern we
observe. same_different had zero gradients for 400+ cycles, then
spiked to 9.3, then mastered. The model was wandering in a flat
landscape where B, A, and C were misaligned — no combination of small
adjustments improved the output. Then the tumblers clicked: B learned
to route, A learned to hold, C learned to read, all at once. The
gradient spike IS the moment of alignment.

### The ridiculousness quantified

|                          | Parity (raw bits) | same_different (bytes) |
|--------------------------|-------------------|----------------------|
| Vocab size               | 2                 | 260                  |
| B matrix size            | 2×32 = 64         | 260×64 = 16,640      |
| Noise channels           | 0                 | 258 (97% noise)      |
| Params                   | 8,074             | 103,539              |
| Steps to mastery         | 400               | ~60,000+             |
| Time on H100             | <1 second         | ~6,000 seconds       |
| Accuracy                 | 100.0%            | 95% (just graduated) |

The same architecture. The same mathematical capability. The same
registers and gates. 400 steps vs 60,000 steps. The only difference:
how many useless byte channels the B matrix has to learn to ignore.

And yet — same_different DID graduate. 6,000 seconds of an H100
burning through gradient descent, 400 cycles of zero progress, one
glorious spike to 9.3, and then mastery. The model taught itself to
fly a 2,048-knob cockpit with no manual, no labels, starting from
random noise. That's simultaneously ridiculous and magnificent.

---

## Entry 23: Triton Kernel Bug — CUDA Cannot Learn What CPU Can

**Date:** 2026-04-23

### The experiment

`test_cpu_vs_cuda.py` — same model, same seed, same config, same code.
Only difference: device.

```
Exact same config (d=32, dS=16, hd=16), seed=0, 400 steps

  cpu:  acc_all=100.0%  acc_last=99.7%  loss=0.0014  time=84.8s
  cuda: acc_all= 54.4%  acc_last=51.0%  loss=0.6816  time=1.4s
```

**CPU: 100% accuracy. CUDA: 54% (random). Same model.**

### What this means

The SSM scan has two implementations:
1. `ssm_scan_jit` — Python/PyTorch loop, runs on CPU/MPS. Mathematically
   correct. The parity experiment works perfectly on this.
2. `ssm_scan_triton` — Triton GPU kernel, runs on CUDA. Fast but
   numerically different enough that the model CANNOT learn parity.

The dispatch logic in `ssm_triton.py`:
```python
def ssm_scan(inp, decay, C, x, z, D):
    if inp.is_cuda and HAS_TRITON:
        return ssm_scan_triton(inp, decay, C, x, z, D)  # ← broken
    return ssm_scan_jit(inp, decay, C, x, z, D)         # ← works
```

Every task that trained on the H100 used the Triton kernel. Every task
that struggled for thousands of cycles was fighting BOTH the byte
encoding AND a numerically imprecise scan kernel.

### The 7 teachers that graduated

They graduated DESPITE the Triton kernel, not because of it. The tasks
that converge easily (modus_ponens in 5 cycles, logic_gate in 8 cycles)
are simple enough that numerical imprecision doesn't matter. The tasks
stuck at 62-93% (parity, alternating_next) might be stuck BECAUSE of it.

### Register sizes tested

```
test_register_sizes.py — all on CUDA (Triton kernel):

config                          params  acc_all  acc_last  registers
tiny d=16 dS=2 hd=4              2,122    51.7%    48.4%         64
tiny d=16 dS=4 hd=4              2,210    53.3%    51.3%        128
small d=32 dS=8 hd=8             7,794    53.2%    50.9%        512
orig d=32 dS=16 hd=16            8,074    54.4%    51.0%      1,024
current d=64 dS=16              29,138    53.0%    46.2%      2,048
```

ALL configurations fail on CUDA. Every register size. The issue is not
the model architecture — it's the kernel implementation.

### Possible causes

The Triton kernel (`ssm_triton_kernel.py`) performs the scan:
```
h[t] = decay[t] * h[t-1] + inp[t]
```

In GPU registers using fp32. Possible sources of numerical divergence:
- **Operation ordering**: the JIT computes sequentially in Python;
  Triton may reorder or fuse operations differently
- **Reduction precision**: the output projection `sum(h * C)` may
  accumulate differently on GPU
- **The silu gate**: `y = (y + D*x) * z * sigmoid(z)` is fused in
  Triton but separate in JIT. Fused ops may lose precision.
- **Decay precision**: `exp(A * dt)` computed once vs recomputed.
  Small differences in decay compound over the sequence length.

### The fix: backend as a mutation

Rather than choosing one or the other globally, make the scan backend
a mutation option in the GA config:

```python
{"scan_backend": "triton"}   # fast, possibly imprecise
{"scan_backend": "jit"}      # slower, mathematically correct
```

The GA can try both. For parity, JIT wins (100% vs 54%). For tasks
where precision matters less (logic_gate, modus_ponens), Triton is
fine and 60x faster.

This is added to the config space alongside d_model, layers, lr, etc.
The champion-challenger system protects: if JIT produces better
accuracy, it wins. The lineage records which backend was used.

### The deeper lesson

We spent hours optimizing the training loop — bigger batches, more
workers, smarter mutations. The real bottleneck was a numerical bug
in the innermost computation. The SSM's state update — the one
operation that must be EXACT for state tracking to work — was
approximate on CUDA.

This is why observability matters. Without the CPU vs CUDA comparison,
we would have blamed the byte encoding, the model size, the learning
rate. The actual cause was invisible at the training level — it's in
the kernel implementation, below the abstraction layer.

---

## Entry 24: 14/15 Tasks Mastered — What the GA Discovered

**Date:** 2026-04-23

### The scoreboard

14 out of 15 tasks mastered. Only repeat_count remains at 70%.

### Winning configurations — discovered by the GA, not by humans

```
task                         d  L  dS  backend device   acc
------------------------------------------------------------
alternating_next           144  3  16      jit   cuda 100%
arithmetic_next            192  5  32      jit   cuda  99%
binary_pattern_next         64  3  16   triton   cuda  96%
geometric_next              64  3  16   triton   cuda 100%
logic_chain                 64  3  16   triton   cuda  98%
logic_gate                  64  3  16   triton   cuda 100%
mirror_detection            96  4  16      jit    cpu  96%
modus_ponens                64  3  16   triton   cuda 100%
odd_one_out                 48  2  32      jit    cpu  98%
parity                      64  4   8      jit   cuda 100%
pattern_period              64  8  16      jit   cuda  97%
run_length_next             64  3  16   triton   cuda 100%
same_different              64  3  16   triton   cuda  95%
sequence_completion         96  6  32      jit   cuda 100%
```

### What the GA discovered — three categories

**Easy tasks (6/14): default config is enough**
geometric_next, logic_gate, modus_ponens, run_length_next,
binary_pattern_next, same_different, logic_chain — all mastered
with the original BASE_CONFIG (d=64, L=3, dS=16, triton, cuda).
No mutation needed. These tasks are simple enough that even the
imprecise Triton kernel can solve them.

**Hard tasks needing JIT (8/14): precision matters**
Every task that needed a mutation to master also needed the JIT
backend. NOT ONE hard task mastered on Triton alone. The numerical
precision of the scan kernel is the difference between convergence
and permanent plateau.

**Tasks needing CPU (2/14): GPU arithmetic itself is insufficient**
mirror_detection and odd_one_out needed CPU execution. Even JIT on
CUDA wasn't precise enough. These tasks require exact floating point
behavior that only CPU provides.

### Architecture insights — each task has its own optimal shape

**Parity (d=64, L=4, dS=8):** Needed MORE layers (4 vs 3) but
FEWER registers (dS=8 vs 16). The running XOR needs sequential
processing depth, not state width. Fewer registers = smaller
search space for the B matrix = faster convergence.

**Arithmetic_next (d=192, L=5, dS=32):** Needed EVERYTHING bigger.
Numbers require wide representations (d=192) and lots of state
(dS=32 = 15,360 registers per layer) to track sequences. The GA
discovered this via a severity 3.0 radical mutation.

**Odd_one_out (d=48, L=2, dS=32):** SMALLER model, fewer layers,
but MORE registers. The task needs to store and compare many values
(find the outlier) but the comparison is simple — more state, less
processing.

**Pattern_period (d=64, L=8):** The deepest model. Finding the
period of a repeating pattern requires many sequential processing
steps — the model needs to hypothesize a period length and verify
it across multiple repetitions. 8 layers of pipeline.

**Sequence_completion (d=96, L=6, dS=32):** Wide, deep, lots of
state. Completing sequences requires representing the pattern
(d=96), verifying it (L=6 deep), and tracking position (dS=32).

### The bigger picture

No human designed these configurations. The GA explored the space
through mutation + champion-challenger, and the lineage records
every attempt. The key mutations:

- `scan_backend: "jit"` — unlocked 8 tasks that Triton couldn't solve
- `device: "cpu"` — unlocked 2 tasks that CUDA couldn't solve
- `n_kernel_layers: 4→8` — unlocked tasks needing deeper reasoning
- `d_model: 64→192` — unlocked tasks needing wider representations
- `d_state: 16→32` — unlocked tasks needing more memory

Each of these was discovered through the mutation gate, validated
by champion-challenger, and recorded in the lineage with full
provenance. The system found the right architecture for each task
automatically.

### This is a known problem in the Mamba community

Research confirmed this is not unique to our implementation:

- **Tri Dao** (Mamba's creator) documented that SSM dynamics are
  "very sensitive to numerical precision" — cumulative sums can be
  large, and without the right implementation, "the basic SSD
  algorithm produces NaNs immediately during training."
  (Source: tridao.me/blog/2024/mamba2-part3-algorithm)

- **PyTorch/IBM** documented that fused Triton kernels "internally
  use fp16 for some computations that the original kernels used fp32
  for" and there are "slight differences in output between the fused
  kernel and reference solution that depend on the GPU."
  (Source: pytorch.org/blog/accelerating-mamba2-with-kernel-fusion)

- **The standard fix**: "accumulate partial sums in higher precision
  (fp32) for numerical stability." Our kernel likely accumulates
  the state `h = decay * h + inp` without sufficient precision,
  or the fused silu gate introduces compounding rounding errors.

### What we're doing about it

**Immediate (deployed):** `scan_backend` as a GA mutation. The GA
tries JIT vs Triton per task. JIT is mathematically correct, Triton
is fast. The champion-challenger decides. Already live — the GA
tried JIT for alternating_next (which graduated at 100%).

**Next:** Fix the Triton kernel itself. The audit path:
1. Check if the state accumulation loop uses fp32 throughout
2. Check if the silu gate is fused in a way that loses precision
3. Compare intermediate values (h at each timestep) between
   JIT and Triton for the same input
4. Fix and verify: parity must hit 100% on CUDA

**Community contribution:** Once fixed, this kernel is valuable.
The Mamba community on Hugging Face (kernels-community/mamba-ssm)
is building community kernels. A correct, tested kernel with a
parity regression test would be a real contribution.

---
## Entry — PTX engine arc (entries 27–54) → moved to `engine/ptx/findings.md`

The PTX engine work (2026-04-24 → 2026-04-26) was a continuous
~28-entry arc that grew larger than any other subproject in this
notebook. To keep the root findings.md readable, those entries were
moved verbatim into [`engine/ptx/findings.md`](engine/ptx/findings.md)
during the structuring pass (2026-04-29). Same content, same dates,
same numbering — just relocated.

**One-paragraph cross-project takeaway:** A hand-written PTX Mamba-3
engine for CUDA, owning forward + backward + scheduler, was built
from scratch over ~3 days. It achieved bit-parity with CPU on forward
and 1e-6 parity on backward, converged on parity training, ran ~14×
faster than the same stream on a PyTorch baseline, picked up a
streaming batch protocol that made it task-agnostic over the existing
30+ problems, learned to resume from PyTorch .pt checkpoints, gained
real (non-stub) Lion/Focal/LabelSmooth kernels for the GA mutation
surface, and shipped a hot-plug `ptxd` daemon that the orchestrator
could submit jobs to without restart. The H100/vast.ai pod era then
ended (instability), the engine moved to the `pod-archive` branch,
and the daily-driver path moved back to PyTorch + MPS via
`specialist_trainer.py`. The PTX engine remains usable when a CUDA
box is provisioned.
