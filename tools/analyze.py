"""Analyze what the model actually predicts — per-byte breakdown."""
import torch
import json
from pathlib import Path
from progressive_model import ProgressiveModel, ByteTokenizer, VOCAB_SIZE, PAD
from generators.level0_patterns import gen_parity, gen_same_different

# Find best checkpoint
runs = Path("runs")
best_exp = None
best_fresh = 0
for p in sorted(runs.iterdir()):
    metrics_path = p / "metrics.json"
    if metrics_path.exists():
        m = json.load(open(metrics_path))
        if m.get("best_fresh", 0) > best_fresh:
            best_fresh = m["best_fresh"]
            best_exp = p.name

print(f"Analyzing {best_exp} (best_fresh={best_fresh:.1%})")

ckpt_path = runs / best_exp / "checkpoint.pt"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
cfg = ckpt["config"]
model = ProgressiveModel(d_model=cfg["d_model"], d_state=cfg["d_state"], headdim=cfg["headdim"])
for _ in range(cfg.get("n_kernel_layers", 1)):
    model.add_kernel_layer()
model.load_state_dict(ckpt["model"])
model.eval()

tok = ByteTokenizer()

print(f"\nModel: d={cfg['d_model']} L={cfg.get('n_kernel_layers',1)} "
      f"wd={cfg.get('weight_decay',0)}")
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

# Test parity
print(f"\n{'='*60}")
print(f"PARITY — 30 examples")
print(f"{'='*60}")
exact_correct = 0
byte_correct_total = 0
byte_total = 0
for _ in range(30):
    ex = gen_parity(min_len=3, max_len=5)
    tokens, sep_pos = tok.encode_curriculum(ex)
    out_bytes = list(ex["output"].encode("utf-8"))

    t = torch.tensor([tokens], dtype=torch.long)
    with torch.no_grad():
        logits = model(t)

    predicted_bytes = []
    bc = 0
    for j in range(len(out_bytes)):
        p = sep_pos + j
        if p < logits.shape[1]:
            pred = logits[0, p].argmax().item()
            predicted_bytes.append(pred)
            if pred == out_bytes[j]:
                bc += 1

    pred_str = bytes(b for b in predicted_bytes if b < 256).decode("utf-8", errors="replace")
    exact = pred_str == ex["output"]
    if exact:
        exact_correct += 1
    byte_correct_total += bc
    byte_total += len(out_bytes)

    inp = ex["input"]
    exp_out = ex["output"]
    marker = "OK" if exact else "XX"
    print(f"  {marker}  {inp:20s} -> expected={exp_out:4s}  got={pred_str:6s}  "
          f"bytes={bc}/{len(out_bytes)}")

print(f"\n  Exact match: {exact_correct}/30 = {exact_correct/30:.0%}")
print(f"  Per-byte:    {byte_correct_total}/{byte_total} = {byte_correct_total/byte_total:.0%}")

# Test same_different
print(f"\n{'='*60}")
print(f"SAME_DIFFERENT — 30 examples")
print(f"{'='*60}")
exact_correct = 0
byte_correct_total = 0
byte_total = 0
for _ in range(30):
    ex = gen_same_different(max_val=3)
    tokens, sep_pos = tok.encode_curriculum(ex)
    out_bytes = list(ex["output"].encode("utf-8"))

    t = torch.tensor([tokens], dtype=torch.long)
    with torch.no_grad():
        logits = model(t)

    predicted_bytes = []
    bc = 0
    for j in range(len(out_bytes)):
        p = sep_pos + j
        if p < logits.shape[1]:
            pred = logits[0, p].argmax().item()
            predicted_bytes.append(pred)
            if pred == out_bytes[j]:
                bc += 1

    pred_str = bytes(b for b in predicted_bytes if b < 256).decode("utf-8", errors="replace")
    exact = pred_str == ex["output"]
    if exact:
        exact_correct += 1
    byte_correct_total += bc
    byte_total += len(out_bytes)

    inp = ex["input"]
    exp_out = ex["output"]
    marker = "OK" if exact else "XX"
    print(f"  {marker}  {inp:20s} -> expected={exp_out:4s}  got={pred_str:6s}  "
          f"bytes={bc}/{len(out_bytes)}")

print(f"\n  Exact match: {exact_correct}/30 = {exact_correct/30:.0%}")
print(f"  Per-byte:    {byte_correct_total}/{byte_total} = {byte_correct_total/byte_total:.0%}")

# What does the model output most often?
print(f"\n{'='*60}")
print(f"ANALYSIS: What does the model predict at the first output position?")
print(f"{'='*60}")
from collections import Counter
first_byte_preds = Counter()
for _ in range(100):
    ex = gen_parity(min_len=3, max_len=5)
    tokens, sep_pos = tok.encode_curriculum(ex)
    t = torch.tensor([tokens], dtype=torch.long)
    with torch.no_grad():
        logits = model(t)
    pred = logits[0, sep_pos].argmax().item()
    if pred < 128:
        first_byte_preds[chr(pred)] += 1
    else:
        first_byte_preds[f"<{pred}>"] += 1

print("  First output byte predictions:")
for char, count in first_byte_preds.most_common(10):
    print(f"    '{char}' (byte {ord(char) if len(char)==1 else char}): {count}%")
