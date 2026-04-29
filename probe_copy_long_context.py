"""probe_copy_long_context — does CopyMamba3LM hold up when we prepend
filler bytes before the real payload?

The model was trained at max_seq_len=256. The SSM scan itself has no
length limit (O(L) time, fixed-size hidden state). So inference can
technically run at any length. The question is whether the trained
hidden state encodes "ignore the noise, use the recent payload" well
enough to still produce the right copy.

We prepend N bytes of random ASCII filler before the real payload, run
inference, and check whether the answer matches the original (no
filler) output. Done at filler lengths 0, 64, 256, 1024, 4096, and
report copy accuracy + wall-clock + RSS.

This is the linear-memory probe: a transformer of the same size would
need O(L²) attention memory to process the same prefix and would OOM
or slow to a crawl. Mamba should keep the same per-step cost.
"""
import math, random, resource, time
import torch

from train_tool_renderer_copy import CopyMamba3LM, BOA, EOS


def load_model():
    ck = torch.load("checkpoints/tool_renderer_copy.pt", map_location="cpu", weights_only=False)
    cfg = ck["config"]
    m = CopyMamba3LM(**cfg)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m


PAYLOADS = [
    ("hanoi_solver|n=12|optimal=4095|params=45318|timing=2864",
     "The optimal solution to Tower of Hanoi with 12 disks requires 4,095 moves."),
    ("gcdhanoi|a=6|b=9|moves_a=63|moves_b=511|gcd=7",
     "Hanoi(6) needs 63 moves; Hanoi(9) needs 511 moves; their gcd is 7."),
]


def random_filler(n: int, seed: int) -> bytes:
    """N bytes of printable random ASCII. Excludes BOA/EOS/PAD bytes (0,1,2)."""
    rng = random.Random(seed)
    chars = bytes(rng.randint(33, 126) for _ in range(n))
    return chars


def run_with_filler(model, payload: str, expected: str, filler_len: int, seed: int = 0):
    filler = random_filler(filler_len, seed)
    prefix_str = filler + b" " + payload.encode("utf-8") if filler else payload.encode("utf-8")
    prefix = list(prefix_str) + [BOA]
    t0 = time.time()
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    gen = model.generate(prefix, max_new=200, temperature=0.1, top_k=1)
    dt = time.time() - t0
    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if EOS in gen:
        gen = gen[:gen.index(EOS)]
    text = bytes([b for b in gen if 32 <= b < 256]).decode("utf-8", errors="ignore")
    match = text.strip() == expected.strip()
    return {
        "match": match,
        "text": text,
        "expected": expected,
        "prefix_len": len(prefix),
        "wall_ms": dt * 1000,
        "peak_rss_mb": rss1 / 1024,  # macOS: maxrss is in bytes, but on Linux it's KB; this gives a rough number
        "rss_delta_mb": (rss1 - rss0) / 1024,
    }


def main():
    print("Loading CopyMamba3LM …")
    model = load_model()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params={n_params:,}\n")

    filler_lens = [0, 64, 256, 1024, 4096]
    print(f"{'task':>10} | {'filler':>6} | {'prefix':>6} | {'wall_ms':>8} | {'rss_mb':>8} | {'match':>5} | text/expected")
    print("-" * 110)
    for payload, expected in PAYLOADS:
        for fl in filler_lens:
            r = run_with_filler(model, payload, expected, fl)
            mark = "✓" if r["match"] else "✗"
            label = payload.split("|", 1)[0]
            preview = r["text"][:60] + ("…" if len(r["text"]) > 60 else "")
            print(f"{label:>10} | {fl:>6} | {r['prefix_len']:>6} | {r['wall_ms']:>8.0f} | {r['peak_rss_mb']:>8.0f} | {mark:>5} | {preview}")
        print()


if __name__ == "__main__":
    main()
