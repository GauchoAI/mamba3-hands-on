"""One-shot canary retention eval on a single checkpoint.

When the eval daemon backlog is too long to wait through, point this at
a specific .pt and get retention/drift/diversity in ~30 seconds.

Usage:
  python jepa/eval_one.py path/to/light_step_0002200.pt
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cortex_counting import CortexLM, CortexLMConfig

CANARY_PROMPTS = [
    b"Hello, how are you?\n",
    b"Hola, como estas?\n",
    b"The cat sat on the\n",
    b"En un lugar de la Mancha\n",
    b"It's getting cold today.\n",
    b"What did you do yesterday?\n",
    b"Tell me a short story.\n",
    b"Cuentame una historia corta.\n",
]


@torch.no_grad()
def eval_canary(model, device):
    model.eval()
    completions, h_p_list, h_r_list = [], [], []
    for prompt in CANARY_PROMPTS:
        ids = torch.tensor([list(prompt)], dtype=torch.long, device=device)
        plens = torch.tensor([len(prompt)], device=device)
        _, _, residual_p, _ = model(ids, return_jepa=True, prompt_lens=plens)
        h_p = residual_p[0, -1].float().cpu()
        out = model.generate_greedy(list(prompt), max_new=60)
        completions.append(bytes(out).decode("utf-8", errors="replace"))
        full_ids = torch.tensor([list(prompt) + list(out)],
                                 dtype=torch.long, device=device)
        _, _, residual_r, _ = model(full_ids, return_jepa=True,
                                     prompt_lens=plens)
        h_r = residual_r[0, -1].float().cpu()
        h_p_list.append(h_p)
        h_r_list.append(h_r)
    H_p = torch.stack(h_p_list)
    H_r = torch.stack(h_r_list)
    cos = F.cosine_similarity(H_p, H_r, dim=-1)
    drift = (H_r - H_p).norm(dim=-1) / H_p.norm(dim=-1).clamp_min(1e-6)
    sigs = []
    for c in completions:
        b = c.encode("utf-8")
        sigs.append(set(zip(b[:-1], b[1:])))
    n = len(sigs)
    sims = []
    for i in range(n):
        for j in range(i + 1, n):
            inter = len(sigs[i] & sigs[j])
            uni = len(sigs[i] | sigs[j])
            sims.append(inter / max(uni, 1))
    diversity = 1.0 - (sum(sims) / max(len(sims), 1))
    return {
        "retention": float(cos.mean()),
        "drift": float(drift.mean()),
        "diversity": diversity,
        "completions": completions,
    }


def main() -> None:
    ckpt_path = Path(sys.argv[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = payload["config"]
    cfg = CortexLMConfig(
        n_layers=cfg_dict["n_layers"], d_model=cfg_dict["d_model"],
        d_state=cfg_dict["d_state"], expand=cfg_dict["expand"],
        headdim=cfg_dict["headdim"], vocab_size=cfg_dict["vocab_size"],
        max_seq_len=cfg_dict["max_seq_len"],
        use_counter=cfg_dict["use_counter"],
    )
    model = CortexLM(cfg).to(device)
    model.load_state_dict(payload["model"], strict=False)
    metrics = eval_canary(model, device)
    print(f"step={payload.get('step', '?')}")
    print(f"retention={metrics['retention']:.4f}")
    print(f"drift={metrics['drift']:.4f}")
    print(f"diversity={metrics['diversity']:.4f}")
    print(f"sample_0={metrics['completions'][0][:80]!r}")
    print(f"sample_1={metrics['completions'][1][:80]!r}")
    print(f"sample_2={metrics['completions'][2][:80]!r}")


if __name__ == "__main__":
    main()
