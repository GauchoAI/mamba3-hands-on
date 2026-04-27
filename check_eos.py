"""check_eos — sanity check: for each n in a range, does the model
under teacher forcing predict EOS at the *right* position?

This separates "model knows the count" from "model can autoregress."
If teacher-forced EOS placement is correct but autoregressive emits
'1' forever, the issue is exposure bias / weak EOS gradient. If
teacher-forced placement is also wrong, the model never learned the
count.
"""
import argparse, sys
import torch
sys.path.insert(0, ".")
from progressive_model import ProgressiveModel, ByteTokenizer

EOS = 257
ONE = ord("1")


def load(pt_path, device):
    ck = torch.load(pt_path, map_location=device, weights_only=False)
    cfg = ck["config"]
    model = ProgressiveModel(d_model=cfg["d_model"], d_state=cfg["d_state"],
                             expand=2, headdim=cfg["headdim"])
    for _ in range(cfg["n_kernel_layers"]):
        model.add_kernel_layer()
    model.load_state_dict(ck["model"])
    model.eval()
    return model.to(device), ck


def check_n(model, tok, n, bidir, device):
    inp = f"HANOIBIN {n}"
    if bidir:
        inp = inp + " " + inp[::-1]
    ex = {"input": inp, "output": "1" * n}
    toks, sep = tok.encode_curriculum(ex)
    x = torch.tensor([toks], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(x)
    # Position sep+n predicts the (n+1)-th answer-position output, which
    # should be EOS. Position sep+n-1 predicts the n-th '1'. So we want
    # argmax at position sep+n-1 (the LAST '1' in the answer) to be EOS,
    # because that prediction lands at sep+n in the sequence.
    # Wait — actually predictions are at position k for what comes at
    # position k+1. So at position sep+n-1 (the n-th '1' in the input
    # context), prediction should be the (n+1)-th token = EOS.
    target_pos = sep + n - 1
    target_pos_eos = sep + n  # would predict whatever comes after EOS
    if target_pos < logits.shape[1]:
        probs = torch.softmax(logits[0, target_pos], dim=-1)
        argmax = int(logits[0, target_pos].argmax().item())
        p_eos = float(probs[EOS].item())
        p_one = float(probs[ONE].item())
        return argmax, p_eos, p_one
    return None, 0.0, 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", default="checkpoints/specialists/tower_of_hanoi_binary.pt")
    ap.add_argument("--bidir", action="store_true")
    ap.add_argument("--ns", type=str, default="1,3,5,8,12,16,20,25,30,50,75,100")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()

    model, ck = load(args.pt, args.device)
    tok = ByteTokenizer()
    print(f"{'n':>4}  {'argmax':>8}  {'p(EOS)':>10}  {'p(1)':>10}  verdict")
    print("-" * 60)
    for s in args.ns.split(","):
        n = int(s)
        argmax, p_eos, p_one = check_n(model, tok, n, args.bidir, args.device)
        if argmax is None:
            print(f"{n:>4}  out of range")
            continue
        ok = "✓" if argmax == EOS else f"✗ (argmax={argmax}=ord('1') if 49)"
        print(f"{n:>4}  {argmax:>8}  {p_eos:>10.4f}  {p_one:>10.4f}  {ok}")


if __name__ == "__main__":
    main()
