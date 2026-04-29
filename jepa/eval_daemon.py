"""Live eval daemon: watches checkpoints, computes metrics, optionally serves HTTP.

W3.0 of the JEPA-Cortex plan, paired with checkpoint_policy.py. Watches one
or more `MANIFEST.jsonl` files for new entries, loads each new light
checkpoint, computes a small metric battery, appends to `metrics.jsonl`,
and (optionally) exposes everything on an HTTP endpoint for the SSH-tunneled
dashboard at port 8080.

Decoupled from training so a slow eval never blocks the GPU. Runs on CPU
by default; pass --device cuda:N to use a spare card.

Metrics computed per checkpoint:
  byte_ce_teacher  — held-out byte CE on a slice of teacher_thoughts records
  byte_ce_biling   — held-out byte CE on bilingual.txt
  count_acc_30     — counter accuracy at N=30   (in-distribution)
  count_acc_100    — counter accuracy at N=100  (4× OOD)
  intent_var       — across-batch variance of the intent embedding
                     (should grow with training under SIGreg pressure)
"""
from __future__ import annotations
import argparse
import json
import os
import threading
import time
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cortex_counting import CortexLM, CortexLMConfig
from data_loader import (
    TeacherThoughtsDataset, TeacherIterator, BilingualByteIterator,
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
@torch.no_grad()
def byte_ce(model: CortexLM, batch, device) -> float:
    """Mean byte-level cross-entropy under the standard next-byte target."""
    tokens = batch.tokens.to(device)
    mask = batch.byte_pad_mask.to(device)
    logits = model(tokens)
    # Predict next byte: shift by 1.
    pred = logits[:, :-1].reshape(-1, logits.size(-1))
    tgt = tokens[:, 1:].reshape(-1)
    m = mask[:, 1:].reshape(-1).float()
    raw = F.cross_entropy(pred, tgt, reduction="none")
    return float((raw * m).sum() / m.sum().clamp_min(1.0))


@torch.no_grad()
def count_accuracy(model: CortexLM, n: int, device,
                   max_new: int | None = None) -> float:
    """Generate from prompt `*** ... :` and check the model emits exactly N a's."""
    if max_new is None:
        max_new = n + 4
    prompt = b"*" * n + b":"
    out = model.generate_greedy(list(prompt), max_new=max_new)
    expected = b"a" * n + b"\n"
    actual = bytes(out[:len(expected)])
    return 1.0 if actual == expected else 0.0


@torch.no_grad()
def intent_variance(model: CortexLM, batch, device) -> float:
    tokens = batch.tokens.to(device)
    plens = batch.prompt_lens.to(device)
    _, _, _, intent = model(tokens, return_jepa=True, prompt_lens=plens)
    return float(intent.var(dim=0).mean())


# ---------------------------------------------------------------------------
# Worker: poll manifest, load each new ckpt, evaluate, log.
# ---------------------------------------------------------------------------
class RunWatcher:
    """Watches one run's MANIFEST and appends to its metrics.jsonl.

    State held in self._state for the HTTP endpoint (most recent N entries).
    """

    MAX_HISTORY = 1000

    def __init__(self, run_dir: Path, device: str,
                 teacher_path: str | None,
                 bilingual_path: str | None):
        self.run_dir = Path(run_dir)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self._teacher_iter = None
        self._biling_iter = None
        if teacher_path and Path(teacher_path + ".bin").exists():
            self._teacher_iter = TeacherIterator(
                TeacherThoughtsDataset(teacher_path), batch_size=16, seed=12345,
            )
        if bilingual_path and Path(bilingual_path).exists():
            self._biling_iter = BilingualByteIterator(
                bilingual_path, batch_size=16, seq_len=256, seed=54321,
            )
        self._seen_offset = 0
        self._state: list[dict] = []
        self._lock = threading.Lock()

    @property
    def manifest(self) -> Path:
        # Manifest lives next to checkpoints, not next to metrics. We rewrite
        # the first occurrence of a 'runs' path component to 'checkpoints'.
        # Operating on path *components* (not a substring) makes this work
        # for both relative ('runs/jepa_cortex/...') and absolute
        # ('/workspace/.../runs/...') run directories — the previous version
        # used `str(p).replace('/runs/', '/checkpoints/')` which silently
        # no-op'd on relative paths and watched the wrong directory.
        parts = list(self.run_dir.parts)
        for i, p in enumerate(parts):
            if p == "runs":
                parts[i] = "checkpoints"
                break
        return Path(*parts) / "MANIFEST.jsonl"

    def poll_once(self) -> int:
        """Read any new manifest entries and evaluate the light ones."""
        if not self.manifest.exists():
            return 0
        size = self.manifest.stat().st_size
        if size <= self._seen_offset:
            return 0
        with self.manifest.open("rb") as f:
            f.seek(self._seen_offset)
            new = f.read().decode("utf-8", errors="ignore")
        self._seen_offset = size
        n_evaluated = 0
        for line in new.splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("kind") != "light":
                continue
            ckpt_path = self.manifest.parent / entry["name"]
            if not ckpt_path.exists():
                continue
            metrics = self._evaluate(ckpt_path)
            row = {
                "step": entry["step"],
                "ts": time.time(),
                "ckpt": entry["name"],
                **metrics,
            }
            with self.metrics_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            with self._lock:
                self._state.append(row)
                if len(self._state) > self.MAX_HISTORY:
                    self._state = self._state[-self.MAX_HISTORY:]
            n_evaluated += 1
        return n_evaluated

    def _evaluate(self, ckpt_path: Path) -> dict:
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg_dict = payload["config"]
        # Pull the cortex sub-config out. The trainer stashes the full
        # TrainConfig; we only need the model fields.
        model_cfg_keys = {f for f in CortexLMConfig.__dataclass_fields__}
        model_cfg = CortexLMConfig(**{
            k: cfg_dict[k] for k in cfg_dict if k in model_cfg_keys
        })
        model = CortexLM(model_cfg).to(self.device)
        # State dict may include a ThoughtHead under a different prefix —
        # filter to keys the cortex model recognizes.
        own = set(model.state_dict().keys())
        sd = {k: v for k, v in payload["model"].items() if k in own}
        model.load_state_dict(sd, strict=False)
        model.eval()

        out: dict = {}
        if self._teacher_iter is not None:
            out["byte_ce_teacher"] = byte_ce(model, next(self._teacher_iter),
                                             self.device)
        if self._biling_iter is not None:
            out["byte_ce_biling"] = byte_ce(model, next(self._biling_iter),
                                            self.device)
        # Counter only if model has one.
        if any(p.name == "counter" for p in model.primitives):
            out["count_acc_30"] = count_accuracy(model, 30, self.device)
            out["count_acc_100"] = count_accuracy(model, 100, self.device)
        if self._teacher_iter is not None:
            out["intent_var"] = intent_variance(model, next(self._teacher_iter),
                                                self.device)
        return out

    def snapshot(self) -> list[dict]:
        with self._lock:
            return list(self._state)


# ---------------------------------------------------------------------------
# Tiny HTTP endpoint for the SSH-tunneled dashboard
# ---------------------------------------------------------------------------
class DashboardHandler(BaseHTTPRequestHandler):
    watchers: dict[str, RunWatcher] = {}

    def log_message(self, format, *args):  # silence default access log
        return

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self._send(200, "text/html", _INDEX_HTML.encode("utf-8"))
            return
        if self.path == "/api/runs":
            data = {name: w.snapshot() for name, w in self.watchers.items()}
            self._send(200, "application/json",
                       json.dumps(data).encode("utf-8"))
            return
        self._send(404, "text/plain", b"not found")

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


_INDEX_HTML = """<!doctype html>
<html><head><title>JEPA-Cortex live</title>
<style>
body{font-family:monospace;margin:20px;background:#0c0c0e;color:#cfd1d4}
h1{font-size:14pt}
table{border-collapse:collapse;margin-top:1em}
td,th{border:1px solid #2a2a30;padding:4px 8px;text-align:right}
th{background:#1a1a1e;text-align:left}
.run{margin-bottom:30px}
</style></head><body>
<h1>JEPA-Cortex — live runs</h1>
<div id=root>loading…</div>
<script>
async function refresh() {
  const r = await fetch('/api/runs'); const data = await r.json();
  const html = Object.entries(data).map(([name, rows]) => {
    if (!rows.length) return `<div class=run><h2>${name}</h2>(no data yet)</div>`;
    const last = rows.slice(-30);
    const cols = Object.keys(last[0]).filter(k => k !== 'ts' && k !== 'ckpt');
    const head = '<tr>' + cols.map(c=>`<th>${c}</th>`).join('') + '</tr>';
    const body = last.map(r => '<tr>' + cols.map(c => {
      const v = r[c]; return `<td>${typeof v==='number' ? v.toFixed(4) : v}</td>`;
    }).join('') + '</tr>').join('');
    return `<div class=run><h2>${name}</h2><table>${head}${body}</table></div>`;
  }).join('');
  document.getElementById('root').innerHTML = html;
}
refresh(); setInterval(refresh, 5000);
</script></body></html>"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True,
                    help="Comma-separated run dirs (runs/jepa_cortex/gpu0,...)")
    ap.add_argument("--teacher", default="data/teacher_thoughts",
                    help="Basename for held-out teacher data")
    ap.add_argument("--bilingual", default="data/bilingual.txt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--poll-interval", type=float, default=5.0)
    ap.add_argument("--serve", action="store_true",
                    help="Also start the HTTP dashboard")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    watchers = {
        Path(r).name: RunWatcher(Path(r), args.device, args.teacher,
                                 args.bilingual)
        for r in args.runs.split(",")
    }

    if args.serve:
        DashboardHandler.watchers = watchers
        server = HTTPServer(("0.0.0.0", args.port), DashboardHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        print(f"[dashboard] http://localhost:{args.port}", flush=True)

    print(f"[eval_daemon] watching {len(watchers)} run(s) on {args.device}",
          flush=True)
    while True:
        for name, w in watchers.items():
            n = w.poll_once()
            if n:
                print(f"  {name}: evaluated {n} new ckpt(s)", flush=True)
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
