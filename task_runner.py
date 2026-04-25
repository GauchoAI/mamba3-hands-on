"""Task-runner glue between Python generators and ptxd's batch protocol.

Owns the seam where `generators/<task>` (which produce {"input", "output"}
dicts) meet the binary batch format ptxd consumes. Keeping the curriculum
logic here — instead of in Rust scheduler.rs — means adding a new task is
a Python-only change: drop a generator, list it in problem_registry's
YAML, no Rust rebuild.

Phase 1 scope: just `parity` via the existing generator + ByteTokenizer.
Phase 2 will generalise to all tasks by routing through ProblemRegistry.
"""
import os, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

# Mirrors progressive_model.ByteTokenizer constants. Defined locally so
# this module is importable without torch (the registry lookup itself
# doesn't need torch even though progressive_model.py does).
BYTE_VOCAB = 256
BOS = 256
EOS = 257
SEP = 258
PAD = 259


def encode_curriculum(example: dict) -> tuple[list[int], int]:
    """Tokenise as [BOS input_bytes SEP output_bytes EOS]. Returns
    (token_ids, sep_position). Bit-exact match for the equivalent method
    in progressive_model.ByteTokenizer."""
    inp_bytes = list(example["input"].encode("utf-8"))
    out_bytes = list(example["output"].encode("utf-8"))
    tokens = [BOS] + inp_bytes + [SEP] + out_bytes + [EOS]
    sep_pos = len(inp_bytes) + 1
    return tokens, sep_pos


from batch_writer import write_examples, IGNORE


_REGISTRY_CACHE = None


def _get_registry():
    """Lazy registry init — discovery walks problems/ directory once."""
    global _REGISTRY_CACHE
    if _REGISTRY_CACHE is None:
        from registry.problem_registry import ProblemRegistry
        r = ProblemRegistry()
        r.discover([str(REPO_ROOT / "problems")])
        _REGISTRY_CACHE = r
    return _REGISTRY_CACHE


def make_examples_for_task(task, n_examples, stage=0, seed=None):
    """Generate `n_examples` (tokens, targets) tuples for the given task.

    `stage` is the curriculum stage int (0 = defaults). `seed` is for
    reproducibility. Phase 1: parity only — but the routing through
    ProblemRegistry means any task in `problems/` works once we drop the
    legacy parity codepath in scheduler.rs.
    """
    if seed is not None:
        import random
        random.seed(seed)

    registry = _get_registry()
    if task not in registry.problems:
        raise ValueError(f"unknown task {task!r}; known: {sorted(registry.problems)}")
    gen_fn = registry.get_generator(task, stage=stage)

    out = []
    for _ in range(n_examples):
        ex = gen_fn()
        tokens, sep_pos = encode_curriculum(ex)
        # Format from encode_curriculum:
        #   [BOS, input_bytes, SEP, output_bytes, EOS]
        # We supervise next-token prediction at every position from SEP up to
        # (but not including) EOS. So target[i] = tokens[i+1] for
        # sep_pos <= i < len(tokens) - 1. Everything else is IGNORE.
        # This handles single-byte answers (parity: "S"/"D") and multi-byte
        # answers (cumulative_sum: "160", arithmetic_next: "47") uniformly —
        # matches the convention specialist_trainer.py uses for its CE loss.
        if sep_pos >= len(tokens) - 1:
            continue  # malformed — no output region
        targets = [IGNORE] * len(tokens)
        for i in range(sep_pos, len(tokens) - 1):
            targets[i] = tokens[i + 1]
        out.append((tokens, targets))
    return out


def write_task_batches(path, task, n_examples, stage=0, seed=None):
    """Convenience wrapper: generate + write in one call. Returns path."""
    examples = make_examples_for_task(task, n_examples, stage=stage, seed=seed)
    write_examples(path, examples)
    return path


if __name__ == "__main__":
    # Smoke test: write a tiny parity batch and read it back.
    import tempfile
    from batch_writer import read_examples
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        path = tmp.name
    write_task_batches(path, "parity", n_examples=4, seed=7)
    rt = read_examples(path)
    os.unlink(path)
    print(f"parity: wrote 4 examples, read back {len(rt)}")
    for (toks, tgts) in rt:
        sup = [(i, t) for i, t in enumerate(tgts) if t != IGNORE]
        print(f"  tokens={toks}  supervised={sup}")
