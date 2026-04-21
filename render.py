"""
Renderers — read from SQLite, write HTML + markdown.

Completely decoupled from training code. Run standalone:
    python render.py                          # one-shot
    python render.py --watch --interval 30    # poll and re-render
    python render.py --db /path/to/metrics.db # custom db path
"""
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

from metrics_db import MetricsReader


COLORS = [
    "#1a5276", "#c0392b", "#27ae60", "#e67e22", "#8e44ad",
    "#2980b9", "#d35400", "#16a085", "#c0392b", "#7d3c98",
    "#1abc9c", "#e74c3c", "#3498db", "#f39c12", "#9b59b6",
]

TASK_DESC = {
    "parity": "Parity — count 1s mod 2",
    "binary_pattern_next": "Binary Patterns — detect cycles",
    "same_different": "Same/Different — compare values",
    "odd_one_out": "Odd One Out — find outlier",
    "sequence_completion": "Sequence Completion — predict next",
    "pattern_period": "Pattern Period — cycle length",
    "run_length_next": "Run Length — run-length patterns",
    "mirror_detection": "Mirror — palindrome detection",
    "repeat_count": "Repeat Count — accumulate",
    "arithmetic_next": "Arithmetic — detect step",
    "geometric_next": "Geometric — detect ratio",
    "alternating_next": "Alternating — interleaved",
    "logic_gate": "Logic Gates — AND/OR/XOR",
    "logic_chain": "Logic Chains — chained gates",
    "modus_ponens": "Modus Ponens — propositional logic",
}

STAGE_MAP = {
    "parity": 0, "binary_pattern_next": 0,
    "same_different": 1, "odd_one_out": 1,
    "sequence_completion": 2, "pattern_period": 2, "run_length_next": 2,
    "mirror_detection": 3, "repeat_count": 3,
    "arithmetic_next": 4, "geometric_next": 4, "alternating_next": 4,
    "logic_gate": 5, "logic_chain": 5, "modus_ponens": 5,
}


def render_html(reader: MetricsReader, output="index.html"):
    experiments = reader.get_experiments()
    active_tasks = reader.get_active_tasks()
    gpu_history = reader.get_gpu_history(limit=500)
    events = reader.get_events(limit=100)
    teacher = reader.get_latest_teacher()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    gpu_pct = gpu_history[-1]["gpu_pct"] if gpu_history else 0
    mem_pct = gpu_history[-1]["mem_pct"] if gpu_history else 0
    n_workers = gpu_history[-1]["n_workers"] if gpu_history else 0
    n_total = gpu_history[-1]["n_experiments"] if gpu_history else len(experiments)
    n_running = sum(1 for e in experiments if e["status"] == "running")

    # ── Chart data ──

    # Fresh accuracy timeseries per experiment (top 8)
    fresh_datasets = []
    for i, exp in enumerate(experiments[:8]):
        history = reader.get_cycle_history(exp["exp_id"])
        if not history:
            continue
        color = COLORS[i % len(COLORS)]
        alive = exp["status"] == "running"
        cfg = json.loads(exp["config_json"]) if exp.get("config_json") else {}
        wd = cfg.get("weight_decay", 0)
        backend = cfg.get("backend", "pt")
        method = f"wd={wd}" if wd > 0 else "perp"
        if backend == "tinygrad":
            method += "/tg"
        fresh_datasets.append({
            "label": f"{exp['exp_id']} d={exp.get('d_model','')} [{method}]",
            "data": [{"x": h["cycle"], "y": round((h["fresh_acc"] or 0) * 100, 1)} for h in history],
            "borderColor": color,
            "backgroundColor": "transparent",
            "borderWidth": 2 if alive else 1,
            "borderDash": [] if alive else [5, 5],
            "pointRadius": 0, "tension": 0.3,
        })

    # Loss timeseries (top 4)
    loss_datasets = []
    for i, exp in enumerate(experiments[:4]):
        history = reader.get_cycle_history(exp["exp_id"])
        if not history:
            continue
        color = COLORS[i % len(COLORS)]
        loss_datasets.append({
            "label": exp["exp_id"],
            "data": [{"x": h["cycle"], "y": round(h["loss"] or 0, 3)} for h in history],
            "borderColor": color,
            "backgroundColor": "transparent",
            "borderWidth": 1.5, "pointRadius": 0, "tension": 0.3,
        })

    # Per-task accuracy + difficulty (top 6 experiments)
    task_chart_data = {}
    for task in active_tasks:
        datasets = []
        for i, exp in enumerate(experiments[:6]):
            history = reader.get_task_history(exp["exp_id"], task)
            if not history:
                continue
            color = COLORS[i % len(COLORS)]
            alive = exp["status"] == "running"
            # Accuracy line (solid, left Y)
            datasets.append({
                "label": f"{exp['exp_id']} acc",
                "data": [{"x": h["cycle"], "y": round(h["accuracy"] * 100, 1)} for h in history],
                "borderColor": color, "backgroundColor": "transparent",
                "borderWidth": 2 if alive else 1,
                "borderDash": [] if alive else [5, 5],
                "pointRadius": 0, "tension": 0.3,
                "yAxisID": "y",
            })
            # Difficulty line (dashed, right Y, same color lighter)
            if any(h.get("difficulty", 0) > 0 for h in history):
                datasets.append({
                    "label": f"{exp['exp_id']} diff",
                    "data": [{"x": h["cycle"], "y": round(h.get("difficulty", 0) * 100, 1)} for h in history],
                    "borderColor": color + "66",  # 40% opacity
                    "backgroundColor": "transparent",
                    "borderWidth": 1,
                    "borderDash": [4, 4],
                    "pointRadius": 0, "tension": 0.3,
                    "yAxisID": "y2",
                })
        if datasets:
            task_chart_data[task] = datasets

    # GPU timeseries
    gpu_datasets = [{
        "label": "GPU %",
        "data": [{"x": i, "y": g["gpu_pct"]} for i, g in enumerate(gpu_history)],
        "borderColor": "#1a5276", "backgroundColor": "rgba(26,82,118,0.1)",
        "fill": True, "pointRadius": 0, "tension": 0.3,
    }, {
        "label": "VRAM %",
        "data": [{"x": i, "y": g["mem_pct"]} for i, g in enumerate(gpu_history)],
        "borderColor": "#c0392b", "backgroundColor": "transparent",
        "pointRadius": 0, "tension": 0.3,
    }, {
        "label": "Workers",
        "data": [{"x": i, "y": g["n_workers"] * 5} for i, g in enumerate(gpu_history)],
        "borderColor": "#27ae60", "backgroundColor": "transparent",
        "borderDash": [3, 3], "pointRadius": 0, "tension": 0.3,
        "yAxisID": "y",
    }]

    # ── Build HTML sections ──

    # Leaderboard
    leaderboard = ""
    for i, exp in enumerate(experiments[:15]):
        cfg = json.loads(exp["config_json"]) if exp.get("config_json") else {}
        wd = cfg.get("weight_decay", 0)
        backend = cfg.get("backend", "pytorch")
        method = f"wd={wd}" if wd > 0 else "PerpGrad"
        if backend == "tinygrad":
            method = f"tinygrad/{method}"
        status = exp["status"]
        sc = {"running": "text-green-700 font-semibold",
              "paused": "text-yellow-600",
              }.get(status, "text-gray-400")
        star = "★ " if i == 0 else ""
        peak = exp.get("peak_fresh", 0) or 0
        cycles = exp.get("max_cycle", 0) or 0
        leaderboard += f"""<tr class="border-b border-gray-100 hover:bg-blue-50 text-sm">
            <td class="py-1.5 pr-2 font-mono">{star}{exp['exp_id']}</td>
            <td class="py-1.5 pr-2 text-right tabular-nums">{exp.get('n_params',0):,}</td>
            <td class="py-1.5 pr-2 font-mono text-xs">d={exp.get('d_model','')} L={exp.get('n_kernel_layers','')}</td>
            <td class="py-1.5 pr-2 text-xs">{method}</td>
            <td class="py-1.5 pr-2 text-right font-bold tabular-nums">{peak:.1%}</td>
            <td class="py-1.5 pr-2 text-right tabular-nums">{cycles:,}</td>
            <td class="py-1.5 {sc}">{status}</td>
        </tr>\n"""

    # Task charts
    task_charts_html = ""
    task_charts_js = ""
    sorted_tasks = sorted(active_tasks, key=lambda t: STAGE_MAP.get(t, 99))
    for task in sorted_tasks:
        stage = STAGE_MAP.get(task, "?")
        desc = TASK_DESC.get(task, task)
        task_charts_html += f"""<div>
            <h3 class="text-xs font-bold text-gray-600 mb-1">Stage {stage}: {desc}</h3>
            <div style="height:180px"><canvas id="task_{task}"></canvas></div>
        </div>\n"""

    # Events
    events_html = ""
    for e in events[-30:]:
        ts = datetime.fromtimestamp(e["timestamp"]).strftime("%H:%M:%S")
        icons = {"mastery": "★", "unlock": "🔓", "evolve": "🧬",
                 "pause": "⏸", "spawn": "➕", "error": "❌"}
        icon = icons.get(e["event_type"], "·")
        color = "text-red-600" if e["event_type"] == "error" else "text-gray-700"
        events_html += (f'<div class="text-xs py-0.5 {color}">'
                       f'<span class="text-gray-400 tabular-nums">{ts}</span> '
                       f'{icon} <b>{e.get("exp_id","")}</b> '
                       f'{e.get("details","")}</div>\n')

    # Best per task (from actual data, not just teacher state)
    all_task_data = reader.get_all_task_latest()
    best_per_task = {}
    for t in all_task_data:
        tt = t["task_type"]
        if tt not in best_per_task or t["accuracy"] > best_per_task[tt][1]:
            best_per_task[tt] = (t["exp_id"], t["accuracy"])

    # Teacher + per-task summary
    teacher_html = ""
    # Show best scores per task first
    all_tasks_ordered = ["parity", "binary_pattern_next", "same_different", "odd_one_out",
                         "sequence_completion", "pattern_period", "run_length_next",
                         "mirror_detection", "repeat_count", "arithmetic_next",
                         "geometric_next", "alternating_next", "logic_gate",
                         "logic_chain", "modus_ponens"]
    for task in all_tasks_ordered:
        best_acc = best_per_task.get(task, (None, 0))[1]
        best_eid = best_per_task.get(task, ("—", 0))[0]
        stage = STAGE_MAP.get(task, "?")
        if best_acc >= 0.9:
            icon, cls = "✅", "text-green-700"
        elif best_acc > 0:
            icon, cls = "🔄", "text-yellow-700"
        else:
            icon, cls = "🔒", "text-gray-400"
        acc_str = f"{best_acc:.0%} ({best_eid})" if best_acc > 0 else "locked"
        teacher_html += f'<div class="text-xs py-0.5 {cls}">{icon} S{stage} <b>{task}</b>: {acc_str}</div>'

    # Learning to learn — from mastery events
    mastery_events = reader.get_events("mastery", limit=20)
    learning_html = ""
    if mastery_events:
        learning_html = '<table class="text-xs w-full">'
        learning_html += '<tr class="border-b font-bold"><td>Time</td><td>Experiment</td><td>Details</td></tr>'
        for e in mastery_events:
            ts = datetime.fromtimestamp(e["timestamp"]).strftime("%H:%M")
            learning_html += (f'<tr class="border-b border-gray-100">'
                             f'<td class="tabular-nums">{ts}</td>'
                             f'<td class="font-mono">{e.get("exp_id","")}</td>'
                             f'<td>{e.get("details","")}</td></tr>')
        learning_html += '</table>'

    if teacher:
        ml = json.loads(teacher.get("mastery_log", "[]"))
        if ml:
            learning_html += '<div class="mt-2 pt-2 border-t">'
            learning_html += '<table class="text-xs w-full">'
            learning_html += '<tr class="font-bold"><td>Task</td><td>Steps</td><td>Examples</td></tr>'
            for entry in ml:
                learning_html += (f'<tr class="border-b border-gray-100">'
                                 f'<td>{entry["task"]}</td>'
                                 f'<td class="tabular-nums">{entry["steps_to_master"]:,}</td>'
                                 f'<td class="tabular-nums">{entry["examples_to_master"]:,}</td></tr>')
            if len(ml) >= 2:
                ratio = ml[-1]["steps_to_master"] / max(ml[0]["steps_to_master"], 1)
                trend = "🚀 accelerating" if ratio < 0.7 else ("📈 improving" if ratio < 1 else "→ stable")
                learning_html += f'<tr><td colspan="3" class="pt-1 font-bold">Ratio: {ratio:.2f} {trend}</td></tr>'
            learning_html += '</table></div>'

    if not learning_html:
        learning_html = '<span class="text-xs text-gray-400">No mastery events yet</span>'

    # Curriculum progress bar
    all_tasks_ordered = ["parity", "binary_pattern_next", "same_different", "odd_one_out",
                         "sequence_completion", "pattern_period", "run_length_next",
                         "mirror_detection", "repeat_count", "arithmetic_next",
                         "geometric_next", "alternating_next", "logic_gate",
                         "logic_chain", "modus_ponens"]
    unlocked = json.loads(teacher.get("unlocked_tasks", "[]")) if teacher else []
    progress_html = ""
    for t in all_tasks_ordered:
        if t in active_tasks:
            cls = "bg-green-500 text-white"
        elif t in unlocked:
            cls = "bg-yellow-400 text-gray-800"
        else:
            cls = "bg-gray-200 text-gray-400"
        short = t.replace("_", " ")[:12]
        progress_html += f'<span class="inline-block px-1.5 py-0.5 rounded text-xs mr-1 mb-1 {cls}">{short}</span>'

    # ── Assemble all chart data ──
    all_data = json.dumps({
        "fresh": fresh_datasets,
        "loss": loss_datasets,
        "gpu": gpu_datasets,
        **{f"task_{k}": v for k, v in task_chart_data.items()},
    })

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta http-equiv="refresh" content="30">
<meta http-equiv="cache-control" content="no-cache, no-store, must-revalidate">
<meta http-equiv="pragma" content="no-cache">
<meta http-equiv="expires" content="0">
<title>Mamba-3 Evolution</title>
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  body {{ font-family: 'Inter', system-ui, sans-serif; font-size: 14px; }}
  .tabular-nums {{ font-variant-numeric: tabular-nums; }}
</style>
</head>
<body class="bg-gray-50 text-gray-900">

<div class="max-w-7xl mx-auto px-4 py-3">

<!-- Header -->
<div class="bg-white rounded-lg shadow-sm p-4 mb-3">
  <div class="flex justify-between items-center">
    <div>
      <h1 class="text-xl font-bold">Mamba-3 Curriculum Evolution</h1>
      <p class="text-gray-400 text-xs mt-0.5">{now} · Auto-refreshes 30s</p>
    </div>
    <div class="flex gap-4 text-sm">
      <div class="text-center">
        <div class="text-2xl font-bold text-blue-600">{gpu_pct:.0f}%</div>
        <div class="text-xs text-gray-400">GPU</div>
      </div>
      <div class="text-center">
        <div class="text-2xl font-bold text-orange-600">{mem_pct:.0f}%</div>
        <div class="text-xs text-gray-400">VRAM</div>
      </div>
      <div class="text-center">
        <div class="text-2xl font-bold text-green-600">{n_running}</div>
        <div class="text-xs text-gray-400">Workers</div>
      </div>
      <div class="text-center">
        <div class="text-2xl font-bold text-purple-600">{n_total}</div>
        <div class="text-xs text-gray-400">Total Exp</div>
      </div>
    </div>
  </div>
  <!-- Curriculum progress -->
  <div class="mt-2 pt-2 border-t border-gray-100">
    <span class="text-xs text-gray-400 mr-2">Curriculum:</span>{progress_html}
  </div>
</div>

<!-- Main charts row -->
<div class="grid grid-cols-3 gap-3 mb-3">
  <div class="col-span-2 bg-white rounded-lg shadow-sm p-3">
    <h2 class="text-xs font-bold text-gray-500 mb-1">Fresh Accuracy Over Time (%)</h2>
    <div style="height:280px"><canvas id="chart_fresh"></canvas></div>
  </div>
  <div class="bg-white rounded-lg shadow-sm p-3">
    <h2 class="text-xs font-bold text-gray-500 mb-1">GPU & VRAM</h2>
    <div style="height:120px"><canvas id="chart_gpu"></canvas></div>
    <h2 class="text-xs font-bold text-gray-500 mb-1 mt-2">Loss (top 4)</h2>
    <div style="height:120px"><canvas id="chart_loss"></canvas></div>
  </div>
</div>

<!-- Leaderboard + Events -->
<div class="grid grid-cols-3 gap-3 mb-3">
  <div class="col-span-2 bg-white rounded-lg shadow-sm p-3">
    <h2 class="text-xs font-bold text-gray-500 mb-2">Leaderboard</h2>
    <table class="w-full">
      <thead><tr class="border-b-2 border-gray-200 text-xs text-gray-400">
        <th class="py-1 text-left pr-2">ID</th>
        <th class="py-1 text-right pr-2">Params</th>
        <th class="py-1 text-left pr-2">Arch</th>
        <th class="py-1 text-left pr-2">Method</th>
        <th class="py-1 text-right pr-2">Best Fresh</th>
        <th class="py-1 text-right pr-2">Cycles</th>
        <th class="py-1 text-left">Status</th>
      </tr></thead>
      <tbody>{leaderboard}</tbody>
    </table>
  </div>
  <div class="bg-white rounded-lg shadow-sm p-3">
    <h2 class="text-xs font-bold text-gray-500 mb-2">Events</h2>
    <div class="max-h-64 overflow-y-auto">{events_html or '<span class="text-xs text-gray-400">No events yet</span>'}</div>
  </div>
</div>

<!-- Per-task charts -->
<div class="bg-white rounded-lg shadow-sm p-3 mb-3">
  <h2 class="text-xs font-bold text-gray-500 mb-2">Per-Task Accuracy</h2>
  <div class="grid grid-cols-3 gap-3">
    {task_charts_html}
  </div>
</div>

<!-- Teacher + Learning -->
<div class="grid grid-cols-2 gap-3 mb-3">
  <div class="bg-white rounded-lg shadow-sm p-3">
    <h2 class="text-xs font-bold text-gray-500 mb-2">Curriculum Progress</h2>
    <div class="font-mono text-xs leading-relaxed max-h-48 overflow-y-auto">{teacher_html}</div>
  </div>
  <div class="bg-white rounded-lg shadow-sm p-3">
    <h2 class="text-xs font-bold text-gray-500 mb-2">Mastery Events & Learning to Learn</h2>
    <div class="max-h-48 overflow-y-auto">{learning_html or '<span class="text-xs text-gray-400">No tasks mastered yet</span>'}</div>
  </div>
</div>

</div>

<script>
const D = {all_data};

const base = {{
  responsive: true, maintainAspectRatio: false, animation: false,
  plugins: {{ legend: {{ position: 'bottom', labels: {{ font: {{ size: 9 }}, boxWidth: 10, padding: 6 }} }} }},
}};

function mkChart(id, datasets, yMax, yLabel) {{
  const el = document.getElementById(id);
  if (!el || !datasets) return;
  new Chart(el, {{
    type: 'line',
    data: {{ datasets }},
    options: {{
      ...base,
      scales: {{
        x: {{ type: 'linear', title: {{ display: true, text: 'Cycle', font: {{ size: 9 }} }},
               ticks: {{ font: {{ size: 8 }} }} }},
        y: {{ min: 0, max: yMax, title: {{ display: true, text: yLabel || '%', font: {{ size: 9 }} }},
               ticks: {{ font: {{ size: 8 }} }} }},
      }},
    }},
  }});
}}

mkChart('chart_fresh', D.fresh, 100, 'Fresh %');
mkChart('chart_gpu', D.gpu, 100, '%');
mkChart('chart_loss', D.loss, null, 'Loss');

for (const [key, datasets] of Object.entries(D)) {{
  if (key.startsWith('task_')) {{
    // Dual axis: accuracy (left) + difficulty (right)
    const el = document.getElementById(key);
    if (!el) continue;
    new Chart(el, {{
      type: 'line',
      data: {{ datasets }},
      options: {{
        ...base,
        scales: {{
          x: {{ type: 'linear', title: {{ display: true, text: 'Cycle', font: {{ size: 9 }} }},
                 ticks: {{ font: {{ size: 8 }} }} }},
          y: {{ position: 'left', min: 0, max: 100,
                title: {{ display: true, text: 'Accuracy %', font: {{ size: 9 }} }},
                ticks: {{ font: {{ size: 8 }} }} }},
          y2: {{ position: 'right', min: 0, max: 100,
                 title: {{ display: true, text: 'Difficulty %', font: {{ size: 9 }}, color: '#999' }},
                 ticks: {{ font: {{ size: 8 }}, color: '#999' }},
                 grid: {{ drawOnChartArea: false }} }},
        }},
      }},
    }});
  }}
}}
</script>

</body></html>"""

    Path(output).write_text(html)


def render_md(reader: MetricsReader, output="dashboard.md"):
    experiments = reader.get_experiments()
    active_tasks = reader.get_active_tasks()
    gpu_history = reader.get_gpu_history(limit=1)
    events = reader.get_events(limit=20)
    teacher = reader.get_latest_teacher()

    gpu_pct = gpu_history[-1]["gpu_pct"] if gpu_history else 0
    mem_pct = gpu_history[-1]["mem_pct"] if gpu_history else 0
    n_running = sum(1 for e in experiments if e["status"] == "running")

    now = datetime.now().strftime('%H:%M:%S')
    n_total = len(experiments)
    n_paused = sum(1 for e in experiments if e["status"] == "paused")

    # Best per task
    all_task_data = reader.get_all_task_latest()
    best_per_task = {}
    for t in all_task_data:
        tt = t["task_type"]
        if tt not in best_per_task or t["accuracy"] > best_per_task[tt][1]:
            best_per_task[tt] = (t["exp_id"], t["accuracy"])

    # Winning config analysis
    top3_configs = []
    for e in experiments[:3]:
        cfg = json.loads(e["config_json"]) if e.get("config_json") else {}
        top3_configs.append(cfg)

    lines = [
        f"# Evolution Dashboard",
        f"GPU {gpu_pct:.0f}% · VRAM {mem_pct:.0f}% · {n_running} running · "
        f"{n_paused} paused · {n_total} total · {now}",
        "",
        "## Summary",
    ]

    # Best fresh
    if experiments:
        best = experiments[0]
        cfg = json.loads(best["config_json"]) if best.get("config_json") else {}
        peak = best.get("peak_fresh", 0) or 0
        lines.append(f"- **Best overall**: {best['exp_id']} at **{peak:.1%}** fresh "
                    f"(d={cfg.get('d_model','?')}, L={cfg.get('n_kernel_layers','?')}, "
                    f"wd={cfg.get('weight_decay','?')})")

    # Best per task summary
    for task in sorted(best_per_task, key=lambda t: STAGE_MAP.get(t, 99)):
        eid, acc = best_per_task[task]
        if acc > 0:
            stage = STAGE_MAP.get(task, "?")
            desc = TASK_DESC.get(task, task)
            marker = "✓" if acc >= 0.9 else ("→" if acc >= 0.5 else "·")
            lines.append(f"- {marker} **{task}**: {acc:.0%} ({eid}) — {desc}")

    # Winning method
    wd_wins = sum(1 for e in experiments[:5] if (json.loads(e.get("config_json","{}")) or {}).get("weight_decay",0) > 0)
    if wd_wins >= 3:
        lines.append(f"- **Grokking (weight decay) dominates top 5** ({wd_wins}/5)")
    perp_wins = 5 - wd_wins
    if perp_wins >= 3:
        lines.append(f"- **PerpGrad dominates top 5** ({perp_wins}/5)")

    lines.extend([
        "",
        "## Leaderboard",
        "| # | ID | Params | Arch | Method | Fresh | Cycles | Status |",
        "|---|-----|--------|------|--------|-------|--------|--------|",
    ])
    for i, exp in enumerate(experiments[:15]):
        cfg = json.loads(exp["config_json"]) if exp.get("config_json") else {}
        wd = cfg.get("weight_decay", 0)
        backend = cfg.get("backend", "pt")
        method = f"wd={wd}" if wd > 0 else "PerpGrad"
        if backend == "tinygrad":
            method = f"tg/{method}"
        star = "★" if i == 0 else ""
        peak = exp.get("peak_fresh", 0) or 0
        cycles = exp.get("max_cycle", 0) or 0
        lines.append(f"| {i+1} | {star}{exp['exp_id']} | {exp.get('n_params',0):,} | "
                    f"d={exp.get('d_model','')}L={exp.get('n_kernel_layers','')} | "
                    f"{method} | {peak:.1%} | {cycles:,} | {exp['status']} |")

    # Curriculum progress
    all_tasks_ordered = ["parity", "binary_pattern_next", "same_different", "odd_one_out",
                         "sequence_completion", "pattern_period", "run_length_next",
                         "mirror_detection", "repeat_count", "arithmetic_next",
                         "geometric_next", "alternating_next", "logic_gate",
                         "logic_chain", "modus_ponens"]
    unlocked = json.loads(teacher.get("unlocked_tasks", "[]")) if teacher else []

    lines.extend(["", "## Curriculum Progress"])
    for task in all_tasks_ordered:
        stage = STAGE_MAP.get(task, "?")
        desc = TASK_DESC.get(task, task)
        best_acc = best_per_task.get(task, (None, 0))[1]
        best_eid = best_per_task.get(task, ("none", 0))[0]
        if task in active_tasks and best_acc >= 0.9:
            icon = "✅"
        elif task in active_tasks:
            icon = "🔄"
        elif task in unlocked:
            icon = "🔓"
        else:
            icon = "🔒"
        acc_str = f" — best: {best_acc:.0%} ({best_eid})" if best_acc > 0 else ""
        lines.append(f"- {icon} Stage {stage}: **{task}**{acc_str}")

    if teacher:
        lines.extend(["", "## Teacher", "```", teacher.get("status_text", ""), "```"])
        ml = json.loads(teacher.get("mastery_log", "[]"))
        if ml:
            lines.extend(["", "## Learning-to-Learn",
                          "| # | Task | Steps | Examples |",
                          "|---|------|-------|----------|"])
            for entry in ml:
                lines.append(f"| {entry['task_index']+1} | **{entry['task']}** | "
                           f"{entry['steps_to_master']:,} | "
                           f"{entry['examples_to_master']:,} |")
            if len(ml) >= 2:
                ratio = ml[-1]["steps_to_master"] / max(ml[0]["steps_to_master"], 1)
                trend = "🚀 accelerating" if ratio < 0.7 else ("📈 improving" if ratio < 1 else "→ stable")
                lines.append(f"\nRatio (last/first): **{ratio:.2f}** {trend}")

    if events:
        lines.extend(["", "## Recent Events"])
        for e in events[-15:]:
            ts = datetime.fromtimestamp(e["timestamp"]).strftime("%H:%M:%S")
            icon = {"mastery": "★", "unlock": "🔓", "evolve": "🧬",
                    "pause": "⏸", "error": "❌"}.get(e["event_type"], "·")
            lines.append(f"- `{ts}` {icon} **{e['event_type']}** "
                        f"{e.get('exp_id','')} {e.get('details','')}")

    lines.append("")
    Path(output).write_text("\n".join(lines))


def render_once(db_path="metrics.db"):
    reader = MetricsReader(db_path)
    render_html(reader)
    render_md(reader)


def watch(db_path="metrics.db", interval=30):
    print(f"Watching {db_path}, rendering every {interval}s...", flush=True)
    fail_count = 0
    while True:
        try:
            render_once(db_path)
            fail_count = 0
            print(f"  rendered at {datetime.now().strftime('%H:%M:%S')}", flush=True)
        except Exception as e:
            fail_count += 1
            import traceback
            print(f"\n  ❌ RENDER ERROR #{fail_count}: {e}", flush=True)
            traceback.print_exc()
            if fail_count > 10:
                print(f"  ⚠ {fail_count} consecutive failures — still retrying", flush=True)
        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="metrics.db")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=30)
    args = parser.parse_args()
    if args.watch:
        watch(args.db, args.interval)
    else:
        render_once(args.db)
        print("Rendered dashboard.html + dashboard.md")
