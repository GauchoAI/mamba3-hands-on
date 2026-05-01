#!/usr/bin/env python3
"""
🐀 Ratatouille — Web UI for the Mamba-3 kernel.

Play with the trained model in your browser.

Usage:
    python play_server.py                    # http://localhost:8888
    python play_server.py --port 9999        # custom port
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import json
import time
import torch
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

from progressive_model import ProgressiveModel, ByteTokenizer, VOCAB_SIZE, PAD, BOS, EOS, SEP


# ── Model ───────────────────────────────────────────────────────────

MODEL = None
CFG = None
DEVICE = None
N_PARAMS = 0


def load_model(ckpt_path):
    global MODEL, CFG, DEVICE, N_PARAMS

    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        DEVICE = "mps"
    elif torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    CFG = ckpt.get("config", {})

    MODEL = ProgressiveModel(
        d_model=CFG.get("d_model", 64),
        d_state=CFG.get("d_state", 16),
        expand=2,
        headdim=CFG.get("headdim", 16),
    )
    for _ in range(CFG.get("n_kernel_layers", 1)):
        MODEL.add_kernel_layer()
    MODEL.load_state_dict(ckpt["model"])
    MODEL.eval()
    MODEL = MODEL.to(DEVICE)
    N_PARAMS = sum(p.numel() for p in MODEL.parameters())
    print(f"Model loaded: d={CFG.get('d_model')}, L={CFG.get('n_kernel_layers')}, "
          f"{N_PARAMS:,} params, device={DEVICE}")


def predict(input_text):
    tok = ByteTokenizer()
    input_bytes = list(input_text.encode("utf-8"))
    tokens = [BOS] + input_bytes + [SEP]
    t = torch.tensor([tokens], dtype=torch.long, device=DEVICE)

    if DEVICE == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = MODEL(t)
    if DEVICE == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - t0

    sep_pos = len(tokens) - 1
    predicted_bytes = []
    confidences = []
    top3s = []
    for i in range(10):
        pos = sep_pos + i
        if pos >= logits.shape[1]:
            break
        probs = torch.softmax(logits[0, pos], dim=-1)
        pred_byte = probs.argmax().item()
        confidence = probs[pred_byte].item()
        if pred_byte == EOS or pred_byte == PAD:
            break
        predicted_bytes.append(pred_byte)
        confidences.append(confidence)
        # Top 3 predictions
        topk = torch.topk(probs, 3)
        top3 = []
        for v, idx in zip(topk.values.tolist(), topk.indices.tolist()):
            ch = chr(idx) if 32 <= idx < 127 else f"<{idx}>"
            top3.append({"byte": idx, "char": ch, "prob": round(v * 100, 1)})
        top3s.append(top3)

    output_text = bytes(b for b in predicted_bytes if b < 256).decode("utf-8", errors="replace")
    n_tokens = logits.shape[1]
    tps = n_tokens / elapsed if elapsed > 0 else 0

    return {
        "output": output_text,
        "confidences": [round(c * 100, 1) for c in confidences],
        "top3": top3s,
        "elapsed_ms": round(elapsed * 1000, 1),
        "tokens_per_sec": round(tps),
        "n_tokens": n_tokens,
    }


# ── HTML ────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>🐀 Ratatouille — Mamba-3 Kernel</title>
<script src="https://cdn.tailwindcss.com"></script>
<style>
  body { font-family: 'Inter', system-ui, sans-serif; }
  .mono { font-family: 'JetBrains Mono', 'Fira Code', monospace; }
  input:focus { outline: none; }
  @keyframes pulse { 0%,100% { opacity:1 } 50% { opacity:0.5 } }
  .thinking { animation: pulse 0.8s infinite; }
</style>
</head>
<body class="bg-gray-950 text-gray-100 min-h-screen">

<div class="max-w-2xl mx-auto px-6 py-8">

<!-- Header -->
<div class="mb-8">
  <h1 class="text-3xl font-bold">🐀 Ratatouille</h1>
  <p class="text-gray-400 mt-1">Mamba-3 Kernel — interactive playground</p>
  <div class="flex gap-4 mt-3 text-sm text-gray-500">
    <span>d=<span id="d_model" class="text-gray-300">?</span></span>
    <span>layers=<span id="n_layers" class="text-gray-300">?</span></span>
    <span>params=<span id="n_params" class="text-gray-300">?</span></span>
    <span>device=<span id="device" class="text-gray-300">?</span></span>
  </div>
</div>

<!-- Task selector -->
<div class="flex gap-2 mb-6 flex-wrap">
  <button onclick="setTask('parity')" id="btn_parity"
    class="task-btn px-3 py-1.5 rounded-lg text-sm bg-blue-900 text-blue-200">Parity</button>
  <button onclick="setTask('same_different')" id="btn_same_different"
    class="task-btn px-3 py-1.5 rounded-lg text-sm bg-gray-800 text-gray-400">Same/Different</button>
  <button onclick="setTask('mirror')" id="btn_mirror"
    class="task-btn px-3 py-1.5 rounded-lg text-sm bg-gray-800 text-gray-400">Mirror</button>
  <button onclick="setTask('arithmetic')" id="btn_arithmetic"
    class="task-btn px-3 py-1.5 rounded-lg text-sm bg-gray-800 text-gray-400">Arithmetic</button>
  <button onclick="setTask('logic')" id="btn_logic"
    class="task-btn px-3 py-1.5 rounded-lg text-sm bg-gray-800 text-gray-400">Logic</button>
  <button onclick="setTask('free')" id="btn_free"
    class="task-btn px-3 py-1.5 rounded-lg text-sm bg-gray-800 text-gray-400">Free</button>
</div>

<!-- Task description -->
<div id="task_desc" class="text-sm text-gray-400 mb-2">Count 1s in binary. Even → S, Odd → D</div>
<div id="task_hint" class="text-xs text-gray-600 mb-4">Try: 0 1 1 0 · 1 0 1 · 0 0 0 1 1</div>

<!-- Input -->
<div class="flex gap-3 mb-6">
  <input id="input" type="text" placeholder="0 1 1 0"
    class="flex-1 bg-gray-900 border border-gray-700 rounded-lg px-4 py-3 mono text-lg
           focus:border-blue-500 transition-colors"
    onkeypress="if(event.key==='Enter')send()">
  <button onclick="send()"
    class="bg-blue-600 hover:bg-blue-500 px-6 py-3 rounded-lg font-bold transition-colors">
    →
  </button>
</div>

<!-- Result -->
<div id="result" class="hidden mb-6">
  <div class="bg-gray-900 rounded-lg p-5 border border-gray-800">
    <div class="flex justify-between items-baseline mb-3">
      <div>
        <span class="text-gray-500 text-sm">Output:</span>
        <span id="output" class="text-3xl font-bold mono ml-2 text-yellow-400"></span>
      </div>
      <div class="text-right text-xs text-gray-500">
        <span id="elapsed"></span>ms ·
        <span id="tps"></span> tok/s
      </div>
    </div>
    <!-- Confidence bars -->
    <div id="confidence" class="space-y-1"></div>
    <!-- Top 3 per position -->
    <div id="top3" class="mt-4 text-xs text-gray-500"></div>
  </div>
</div>

<!-- History -->
<div id="history" class="space-y-2"></div>

</div>

<script>
const TASKS = {
  parity: { desc: "Count 1s in binary. Even → S, Odd → D", hint: "Try: 0 1 1 0 · 1 0 1 · 0 0 0 1 1", placeholder: "0 1 1 0" },
  same_different: { desc: "Are two numbers the same? Same → S, Different → D", hint: "Try: 3 3 · 2 5 · 0 0 · 7 3", placeholder: "3 3" },
  mirror: { desc: "Is the sequence a palindrome? Yes → M, No → N", hint: "Try: 1 2 3 2 1 · 1 2 3 4", placeholder: "1 2 3 2 1" },
  arithmetic: { desc: "What comes next in the sequence?", hint: "Try: 2 5 8 11 ? · 10 8 6 4 ?", placeholder: "2 5 8 11 ?" },
  logic: { desc: "Evaluate a logic gate", hint: "Try: AND 1 0 · OR 0 1 · XOR 1 1 · NOT 0", placeholder: "AND 1 0" },
  free: { desc: "Type anything — see what the model thinks", hint: "The model will try to predict what comes after your input", placeholder: "hello" },
};

let currentTask = 'parity';

function setTask(name) {
  currentTask = name;
  const t = TASKS[name];
  document.getElementById('task_desc').textContent = t.desc;
  document.getElementById('task_hint').textContent = t.hint;
  document.getElementById('input').placeholder = t.placeholder;
  document.querySelectorAll('.task-btn').forEach(b => {
    b.className = b.className.replace(/bg-blue-900 text-blue-200/, 'bg-gray-800 text-gray-400');
  });
  const btn = document.getElementById('btn_' + name);
  if (btn) btn.className = btn.className.replace(/bg-gray-800 text-gray-400/, 'bg-blue-900 text-blue-200');
}

async function send() {
  const input = document.getElementById('input').value.trim();
  if (!input) return;

  document.getElementById('result').classList.remove('hidden');
  document.getElementById('output').textContent = '...';
  document.getElementById('output').classList.add('thinking');

  try {
    const resp = await fetch('/predict?input=' + encodeURIComponent(input));
    const data = await resp.json();

    document.getElementById('output').classList.remove('thinking');
    document.getElementById('output').textContent = data.output || '(empty)';
    document.getElementById('elapsed').textContent = data.elapsed_ms;
    document.getElementById('tps').textContent = data.tokens_per_sec;

    // Confidence bars
    const confDiv = document.getElementById('confidence');
    confDiv.innerHTML = '';
    if (data.confidences) {
      data.confidences.forEach((c, i) => {
        const byte = data.output.charCodeAt(i);
        const ch = data.output[i] || '?';
        confDiv.innerHTML += `
          <div class="flex items-center gap-2">
            <span class="w-6 text-right mono text-gray-500 text-xs">'${ch}'</span>
            <div class="flex-1 bg-gray-800 rounded-full h-2">
              <div class="h-2 rounded-full ${c > 80 ? 'bg-green-500' : c > 50 ? 'bg-yellow-500' : 'bg-red-500'}"
                   style="width: ${c}%"></div>
            </div>
            <span class="w-12 text-right text-xs text-gray-500">${c}%</span>
          </div>`;
      });
    }

    // Top 3
    const top3Div = document.getElementById('top3');
    if (data.top3 && data.top3.length > 0) {
      let html = '<div class="flex gap-4">';
      data.top3.forEach((pos, i) => {
        html += '<div>';
        html += `<div class="text-gray-600 mb-1">pos ${i}:</div>`;
        pos.forEach(p => {
          html += `<div class="mono">'${p.char}' ${p.prob}%</div>`;
        });
        html += '</div>';
      });
      html += '</div>';
      top3Div.innerHTML = html;
    }

    // Add to history
    const hist = document.getElementById('history');
    const isCorrect = checkAnswer(input, data.output);
    const mark = isCorrect === null ? '·' : isCorrect ? '✓' : '✗';
    const markColor = isCorrect === null ? 'text-gray-600' : isCorrect ? 'text-green-500' : 'text-red-500';
    hist.innerHTML = `
      <div class="flex items-center gap-3 text-sm py-1 border-b border-gray-900">
        <span class="${markColor}">${mark}</span>
        <span class="mono text-gray-400">${input}</span>
        <span class="text-gray-600">→</span>
        <span class="mono font-bold text-yellow-400">${data.output}</span>
        <span class="text-gray-700 text-xs">${data.elapsed_ms}ms</span>
      </div>` + hist.innerHTML;

  } catch (e) {
    document.getElementById('output').classList.remove('thinking');
    document.getElementById('output').textContent = 'Error: ' + e.message;
  }

  document.getElementById('input').value = '';
  document.getElementById('input').focus();
}

function checkAnswer(input, output) {
  // Auto-check parity
  if (currentTask === 'parity') {
    const ones = input.split(' ').filter(x => x === '1').length;
    const expected = ones % 2 === 0 ? 'S' : 'D';
    return output === expected;
  }
  if (currentTask === 'same_different') {
    const parts = input.split(' ');
    if (parts.length === 2) {
      const expected = parts[0] === parts[1] ? 'S' : 'D';
      return output === expected;
    }
  }
  if (currentTask === 'logic') {
    const parts = input.split(' ');
    if (parts[0] === 'NOT' && parts.length === 2) {
      return output === String(1 - parseInt(parts[1]));
    }
    if (parts.length === 3) {
      const a = parseInt(parts[1]), b = parseInt(parts[2]);
      let expected;
      if (parts[0] === 'AND') expected = a & b;
      else if (parts[0] === 'OR') expected = a | b;
      else if (parts[0] === 'XOR') expected = a ^ b;
      if (expected !== undefined) return output === String(expected);
    }
  }
  return null;
}

// Load model info
fetch('/info').then(r => r.json()).then(data => {
  document.getElementById('d_model').textContent = data.d_model;
  document.getElementById('n_layers').textContent = data.n_layers;
  document.getElementById('n_params').textContent = data.n_params.toLocaleString();
  document.getElementById('device').textContent = data.device;
});

document.getElementById('input').focus();
</script>

</body>
</html>"""


# ── Server ──────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/" or parsed.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(HTML.encode())

        elif parsed.path == "/predict":
            params = parse_qs(parsed.query)
            input_text = params.get("input", [""])[0]
            result = predict(input_text)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())

        elif parsed.path == "/info":
            info = {
                "d_model": CFG.get("d_model", "?"),
                "n_layers": CFG.get("n_kernel_layers", "?"),
                "n_params": N_PARAMS,
                "device": DEVICE,
                "weight_decay": CFG.get("weight_decay", "?"),
                "loss_fn": CFG.get("loss_fn", "?"),
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(info).encode())

        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # quiet


def main():
    import argparse
    parser = argparse.ArgumentParser(description="🐀 Ratatouille Web UI")
    parser.add_argument("--checkpoint", "-c", type=str, default=None)
    parser.add_argument("--port", "-p", type=int, default=8888)
    args = parser.parse_args()

    # Find checkpoint
    ckpt_path = args.checkpoint
    if not ckpt_path:
        candidates = list(Path("checkpoints").glob("best_parity*.pt"))
        candidates += list(Path("checkpoints").glob("progressive_best.pt"))
        if candidates:
            ckpt_path = str(candidates[0])
        else:
            print("No checkpoint found. Download one first:")
            print("  scp -P 32783 root@ssh2.vast.ai:/root/mamba3-hands-on/runs/exp_128/checkpoint.pt checkpoints/best_parity.pt")
            return

    load_model(ckpt_path)

    server = HTTPServer(("0.0.0.0", args.port), Handler)
    print(f"\n🐀 Ratatouille running at http://localhost:{args.port}")
    print(f"   Press Ctrl+C to stop\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Adiós!")
        server.server_close()


if __name__ == "__main__":
    main()
