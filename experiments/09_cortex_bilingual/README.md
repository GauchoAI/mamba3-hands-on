# Chapter 09 — Cortex bilingual

**Status:** **closed** by decision 2026-04-30. See
[`findings.md`](findings.md) for the closure entry and the claim
hierarchy this experiment line tested.

# cortex_bilingual/ — counter primitive on a language-trained LM

Stress-test of the cortex thesis: does a small algorithmic primitive
("forward-pass module in the residual stream") still compose when
attached to a *language-trained* small LM, instead of the synthetic
counting-only LM that produced the byte-perfect proof in
`cortex_counting.py`?

**Status:** partial validation. See `findings.md` entry
`"Counter primitive on a frozen bilingual LM"` for the full read.

## What's in here

### Bilingual LM training
- `train_bilingual_cortex_lm.py` — PyTorch trainer for a `CortexLM`
  with `primitives=[]` (= a plain Mamba-3 byte LM, plug-ready).
  Trains on `data/bilingual.txt` (Tatoeba en-es + 5% unary cortex
  mixin built by `mamba_platform.make_bilingual_corpus`).
- `train_bilingual_mlx.py` — MLX equivalent (uses the local
  `mamba3_mlx.py`). 2.87× faster than the PyTorch version on M4 Pro
  with `--dtype bfloat16`.

### MLX port (the framework speedup)
- `mamba3_mlx.py` — Mamba-3 block + `CortexLM` + `Primitive` base
  + `CounterPrimitive` written natively in MLX. Avoids PyTorch MPS's
  `F.pad` >3D fallback.
- `parity_mlx.py` — numerical parity test between the PyTorch
  reference (`mamba_platform.cortex_counting`) and the MLX port.
  Max-abs-diff 3.58e-7 at fp32, well under 1e-3 tolerance.

### Counter-attach experiment
- `train_counter_attach.py` — loads a frozen bilingual `CortexLM`,
  attaches a fresh `CounterPrimitive` sized to the LM's `d_model`,
  freezes the LM's 472,960 params, fine-tunes only the counter's
  ~1,028 adapter parameters on a 50/50 mix of bilingual + cortex
  unary batches.
- `demo_cortex.py` — side-by-side baseline (LM only) vs cortex
  (LM + counter) at `N ∈ {3, 10, 30, 50, 100, 200, 500}` plus
  bilingual probes.

## Reproduction

All commands run from the repo root.

```bash
# 1. Build the bilingual corpus
PYTHONPATH=src python -m mamba_platform.make_bilingual_corpus        # -> data/bilingual.txt

# 2. Train the bilingual LM (10k steps, ~6h on M4 Pro MPS)
PYTHONPATH=src python experiments/09_cortex_bilingual/train_bilingual_cortex_lm.py \
    --steps 10000 --ckpt-every 500 --seq-len 128

# 2'. Or use the MLX trainer (2.87× faster):
PYTHONPATH=src python experiments/09_cortex_bilingual/train_bilingual_mlx.py \
    --steps 10000 --ckpt-every 500 --seq-len 128 --dtype bfloat16

# 3. Verify MLX parity vs PyTorch (sanity check)
PYTHONPATH=src python experiments/09_cortex_bilingual/parity_mlx.py

# 4. Attach a fresh counter and fine-tune only the adapter
PYTHONPATH=src python experiments/09_cortex_bilingual/train_counter_attach.py \
    --lm-ckpt checkpoints/lm/step_FINAL.pt \
    --steps 1000 --injection-scale 30.0

# 5. Side-by-side demo
PYTHONPATH=src python experiments/09_cortex_bilingual/demo_cortex.py
```

## How files import each other

Each script has a small `sys.path` preamble:

```python
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# -> repo root, so `from mamba_platform.cortex_counting import ...` works
```

Sibling-file imports (`from mamba3_mlx import ...` inside
`train_bilingual_mlx.py`, etc.) work via `os.path.dirname(__file__)`
inserted into `sys.path` first.

The shared cortex foundation (`mamba_platform.cortex_counting`) lives in
`src/mamba_platform/` and is *not* duplicated here. That contrasts with the `jepa/`
folder, which deliberately ships its own copy because it patches
`CortexLM.forward` for hidden-state distillation. We don't patch the
cortex code from here — we only attach `Primitive` subclasses through
the existing plugin interface — so a single shared copy is correct.

## Result summary

```
            scale=10           scale=30 (diagnostic)
N=3         OK ✓              OK ✓
N=30        FAIL → 29         OK ✓ (in-distribution byte-perfect)
N=50        FAIL → 48         FAIL → 51 (off by one, oscillates)
N=100..500  drops out of unary mode (LM mode-switch, not a magnitude issue)
```

The off-by-one at training-distribution N is **signal-magnitude-bound**
— a louder counter (scale=30) fixes it. The OOD ceiling is *not*
magnitude-bound; the LM was trained on `*N:aN` lines with N≤30 and
acquired an implicit "stars are short → switch out of unary" prior
that no residual injection can override at this scale.

## What this points at next

The fix shape for the OOD ceiling is **upstream of the counter**:
- Train the bilingual LM on a wider N distribution (e.g. 1..200), or
- Distill a richer hidden state from a stronger teacher (the
  `jepa/` direction) so the student LM has informationally cleaner
  representations at long unary runs, giving primitives more to read.
