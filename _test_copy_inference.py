"""Quick inference test of the partial CopyMamba3LM checkpoint."""
import torch
from train_tool_renderer_copy import CopyMamba3LM, BOA, EOS

ck = torch.load("checkpoints/tool_renderer_copy.pt", map_location="cpu", weights_only=False)
cfg = ck["config"]
model = CopyMamba3LM(**cfg)
model.load_state_dict(ck["state_dict"])
model.eval()
print(f"Loaded: best_val_loss = {ck['best_val_loss']:.4f}")

prompts = [
    "hanoi_solver|n=12|optimal=4095|params=45318|timing=2864",
    "gcd|a=462|b=252|gcd=42",
    "gcdhanoi|a=6|b=9|moves_a=63|moves_b=511|gcd=7",
]

for payload in prompts:
    prefix = list(payload.encode("utf-8")) + [BOA]
    gen, trace = model.generate(prefix, max_new=120, temperature=0.1, top_k=1, return_trace=True)
    if EOS in gen:
        gen = gen[:gen.index(EOS)]
    text = bytes([b for b in gen if 32 <= b < 256]).decode("utf-8", errors="ignore")
    print(f"\npayload : {payload}")
    print(f"output  : {text}")
    # Show low-gate (copy) decisions
    copies = [(i, t) for i, t in enumerate(trace[:len(gen)]) if t["gate"] < 0.4]
    if copies:
        print(f"copy events (gate < 0.4):")
        for i, t in copies[:10]:
            print(f"  pos {i:3d} byte='{t['char']}' gate={t['gate']:.3f}  ←  prefix[{t['attn_argmax']:3d}]='{t['attn_byte']}'")
