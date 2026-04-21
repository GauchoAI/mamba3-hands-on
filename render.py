"""
Renderers — read from SQLite, write HTML + markdown.

Completely decoupled from training code. Can be run standalone:
    python render.py                      # one-shot render
    python render.py --watch --interval 30  # poll and re-render

Reads: metrics.db (written by workers + coordinator)
Writes: dashboard.html, dashboard.md
"""
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

from metrics_db import MetricsReader


# ── Color palette (WSJ-inspired) ───────────────────────────────────

COLORS = [
    "#0066CC", "#CC3300", "#339933", "#CC6600", "#6633CC",
    "#CC3399", "#336699", "#996633", "#009999", "#993366",
    "#666600", "#003366", "#660066", "#006633", "#CC6633",
]

TASK_DESC = {
    "parity": "Stage 0: Parity — count 1s mod 2",
    "binary_pattern_next": "Stage 0: Binary Patterns — detect cycles",
    "same_different": "Stage 1: Same/Different — compare values",
    "odd_one_out": "Stage 1: Odd One Out — find the outlier",
    "sequence_completion": "Stage 2: Sequence Completion",
    "pattern_period": "Stage 2: Pattern Period",
    "run_length_next": "Stage 2: Run Length",
    "mirror_detection": "Stage 3: Mirror Detection",
    "repeat_count": "Stage 3: Repeat Count",
    "arithmetic_next": "Stage 4: Arithmetic",
    "geometric_next": "Stage 4: Geometric",
    "alternating_next": "Stage 4: Alternating",
    "logic_gate": "Stage 5: Logic Gates",
    "logic_chain": "Stage 5: Logic Chains",
    "modus_ponens": "Stage 5: Modus Ponens",
}


# ── HTML Renderer ───────────────────────────────────────────────────

def render_html(reader: MetricsReader, output="dashboard.html"):
    experiments = reader.get_experiments()
    active_tasks = reader.get_active_tasks()
    gpu_history = reader.get_gpu_history(limit=200)
    events = reader.get_events(limit=50)
    teacher = reader.get_latest_teacher()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Latest GPU
    gpu_pct = gpu_history[-1]["gpu_pct"] if gpu_history else 0
    mem_pct = gpu_history[-1]["mem_pct"] if gpu_history else 0
    n_workers = gpu_history[-1]["n_workers"] if gpu_history else 0
    n_total = gpu_history[-1]["n_experiments"] if gpu_history else 0

    # ── Build chart datasets ──

    # 1. Fresh accuracy over cycles (line chart per experiment)
    fresh_datasets = []
    for i, exp in enumerate(experiments[:12]):
        history = reader.get_cycle_history(exp["exp_id"])
        if not history:
            continue
        color = COLORS[i % len(COLORS)]
        alive = exp["status"] == "running"
        cfg = json.loads(exp["config_json"]) if exp.get("config_json") else {}
        wd = cfg.get("weight_decay", 0)
        method = f"wd={wd}" if wd > 0 else "perp"
        fresh_datasets.append({
            "label": f"{exp['exp_id']} d={exp.get('d_model','')} [{method}]",
            "data": [{"x": h["cycle"], "y": round(h["fresh_acc"] * 100, 1)} for h in history],
            "borderColor": color,
            "backgroundColor": "transparent",
            "borderWidth": 2 if alive else 1,
            "borderDash": [] if alive else [5, 5],
            "pointRadius": 0,
            "tension": 0.3,
        })

    # 2. Per-task accuracy charts
    task_datasets = {}
    for task in active_tasks:
        datasets = []
        for i, exp in enumerate(experiments[:12]):
            history = reader.get_task_history(exp["exp_id"], task)
            if not history:
                continue
            color = COLORS[i % len(COLORS)]
            alive = exp["status"] == "running"
            cfg = json.loads(exp["config_json"]) if exp.get("config_json") else {}
            wd = cfg.get("weight_decay", 0)
            method = f"wd={wd}" if wd > 0 else "perp"
            datasets.append({
                "label": f"{exp['exp_id']} [{method}]",
                "data": [{"x": h["cycle"], "y": round(h["accuracy"] * 100, 1)} for h in history],
                "borderColor": color,
                "backgroundColor": "transparent",
                "borderWidth": 2 if alive else 1,
                "borderDash": [] if alive else [5, 5],
                "pointRadius": 0,
                "tension": 0.3,
            })
        if datasets:
            task_datasets[task] = datasets

    # 3. GPU utilization over time
    gpu_datasets = [{
        "label": "GPU %",
        "data": [{"x": i, "y": g["gpu_pct"]} for i, g in enumerate(gpu_history)],
        "borderColor": "#0066CC",
        "backgroundColor": "rgba(0,102,204,0.1)",
        "fill": True,
        "pointRadius": 0,
        "tension": 0.3,
    }, {
        "label": "VRAM %",
        "data": [{"x": i, "y": g["mem_pct"]} for i, g in enumerate(gpu_history)],
        "borderColor": "#CC3300",
        "backgroundColor": "transparent",
        "pointRadius": 0,
        "tension": 0.3,
    }]

    # ── Leaderboard HTML ──
    leaderboard = ""
    for i, exp in enumerate(experiments[:15]):
        cfg = json.loads(exp["config_json"]) if exp.get("config_json") else {}
        wd = cfg.get("weight_decay", 0)
        method = f"wd={wd}" if wd > 0 else "PerpGrad"
        status = exp["status"]
        sc = "text-green-600" if status == "running" else "text-yellow-600"
        star = "★" if i == 0 else ""
        peak = exp.get("peak_fresh", 0) or 0
        cycles = exp.get("max_cycle", 0) or 0
        leaderboard += f"""<tr class="border-b border-gray-100 hover:bg-gray-50">
            <td class="py-1.5 pr-3 font-mono text-sm">{star} {exp['exp_id']}</td>
            <td class="py-1.5 pr-3 text-right">{exp.get('n_params',0):,}</td>
            <td class="py-1.5 pr-3">d={exp.get('d_model','')} L={exp.get('n_kernel_layers','')}</td>
            <td class="py-1.5 pr-3">{method}</td>
            <td class="py-1.5 pr-3 text-right font-bold">{peak:.1%}</td>
            <td class="py-1.5 pr-3 text-right">{cycles}</td>
            <td class="py-1.5 pr-3 {sc}">{status}</td>
        </tr>"""

    # ── Task charts HTML ──
    charts_html = ""
    chart_keys = list(task_datasets.keys())
    for task in chart_keys:
        title = TASK_DESC.get(task, task.replace("_", " ").title())
        charts_html += f"""<div>
            <h3 class="text-sm font-bold mb-1 border-b border-gray-200 pb-1">{title}</h3>
            <div style="height:220px"><canvas id="chart_{task}"></canvas></div>
        </div>\n"""

    # ── Events timeline ──
    events_html = ""
    for e in events[-20:]:
        ts = datetime.fromtimestamp(e["timestamp"]).strftime("%H:%M:%S")
        icon = {"mastery": "★", "unlock": "🔓", "evolve": "🧬",
                "pause": "⏸", "spawn": "+", "gpu": "📊"}.get(e["event_type"], "·")
        events_html += f'<div class="text-xs font-mono py-0.5">'
        events_html += f'<span class="text-gray-400">{ts}</span> '
        events_html += f'{icon} <span class="font-semibold">{e.get("exp_id","")}</span> '
        events_html += f'{e.get("details","")}</div>\n'

    # ── Teacher status ──
    teacher_html = ""
    if teacher:
        teacher_html = teacher.get("status_text", "").replace("\n", "<br>")
        ml = json.loads(teacher.get("mastery_log", "[]"))
        if ml:
            teacher_html += "<br><br><b>Learning-to-Learn:</b><br>"
            for entry in ml:
                teacher_html += (f"Task {entry['task_index']+1}: {entry['task']} — "
                               f"{entry['steps_to_master']} steps<br>")

    # ── All chart data as JSON ──
    all_chart_data = {
        "fresh": fresh_datasets,
        "gpu": gpu_datasets,
    }
    all_chart_data.update(task_datasets)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta http-equiv="refresh" content="30">
<title>Mamba-3 Evolution</title>
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>body {{ font-family: 'Georgia', serif; }}</style>
</head>
<body class="bg-white text-gray-900 max-w-7xl mx-auto px-6 py-4">

<header class="border-b-2 border-black pb-2 mb-4">
  <h1 class="text-2xl font-bold tracking-tight">Mamba-3 Curriculum Evolution</h1>
  <p class="text-gray-500 text-sm">{now} · GPU {gpu_pct:.0f}% · VRAM {mem_pct:.0f}% · {n_workers} workers · {n_total} experiments · Auto-refreshes 30s</p>
</header>

<div class="grid grid-cols-3 gap-4 mb-4">
  <div class="col-span-2">
    <h2 class="text-sm font-bold mb-1 border-b border-gray-200 pb-1">Overall Fresh Accuracy (%)</h2>
    <div style="height:250px"><canvas id="chart_fresh"></canvas></div>
  </div>
  <div>
    <h2 class="text-sm font-bold mb-1 border-b border-gray-200 pb-1">GPU Utilization</h2>
    <div style="height:250px"><canvas id="chart_gpu"></canvas></div>
  </div>
</div>

<h2 class="text-sm font-bold mb-1 border-b border-gray-200 pb-1">Leaderboard</h2>
<table class="w-full mb-4 text-sm">
<thead><tr class="border-b-2 border-gray-300 text-left text-gray-500 text-xs">
  <th class="py-1 pr-3">ID</th><th class="py-1 pr-3 text-right">Params</th>
  <th class="py-1 pr-3">Arch</th><th class="py-1 pr-3">Method</th>
  <th class="py-1 pr-3 text-right">Best Fresh</th>
  <th class="py-1 pr-3 text-right">Cycles</th><th class="py-1 pr-3">Status</th>
</tr></thead>
<tbody>{leaderboard}</tbody>
</table>

<h2 class="text-sm font-bold mb-2 border-b border-gray-200 pb-1">Per-Task Progress</h2>
<div class="grid grid-cols-2 gap-4 mb-4">
{charts_html}
</div>

<div class="grid grid-cols-2 gap-4 mb-4">
  <div>
    <h2 class="text-sm font-bold mb-1 border-b border-gray-200 pb-1">Teacher</h2>
    <div class="bg-gray-50 p-3 rounded font-mono text-xs leading-relaxed max-h-48 overflow-y-auto">{teacher_html or 'No data yet'}</div>
  </div>
  <div>
    <h2 class="text-sm font-bold mb-1 border-b border-gray-200 pb-1">Events</h2>
    <div class="bg-gray-50 p-3 rounded max-h-48 overflow-y-auto">{events_html or 'No events yet'}</div>
  </div>
</div>

<script>
const allData = {json.dumps(all_chart_data)};

const lineOpts = {{
  responsive: true, maintainAspectRatio: false,
  plugins: {{ legend: {{ position: 'bottom', labels: {{ font: {{ size: 9 }}, boxWidth: 12 }} }} }},
  scales: {{ x: {{ type: 'linear', title: {{ display: true, text: 'Cycle', font: {{ size: 10 }} }} }},
             y: {{ min: 0, max: 100, title: {{ display: true, text: '%', font: {{ size: 10 }} }} }} }},
  animation: false,
}};

for (const [key, datasets] of Object.entries(allData)) {{
  const el = document.getElementById('chart_' + key);
  if (el) {{
    const opts = JSON.parse(JSON.stringify(lineOpts));
    if (key === 'gpu') {{ opts.scales.y.max = 100; }}
    new Chart(el, {{ type: 'line', data: {{ datasets }}, options: opts }});
  }}
}}
</script>

<footer class="text-gray-400 text-xs mt-4 border-t pt-2">
  Mamba-3 · GauchoAI · Evolutionary Model Shopping · SQLite-backed
</footer>
</body></html>"""

    Path(output).write_text(html)


# ── Markdown Renderer ───────────────────────────────────────────────

def render_md(reader: MetricsReader, output="dashboard.md"):
    experiments = reader.get_experiments()
    active_tasks = reader.get_active_tasks()
    gpu_history = reader.get_gpu_history(limit=1)
    events = reader.get_events(limit=20)
    teacher = reader.get_latest_teacher()
    now = datetime.now().strftime("%H:%M:%S")

    gpu_pct = gpu_history[-1]["gpu_pct"] if gpu_history else 0
    mem_pct = gpu_history[-1]["mem_pct"] if gpu_history else 0

    lines = [
        f"# Evolution Dashboard",
        f"GPU {gpu_pct:.0f}% · VRAM {mem_pct:.0f}% · {now}",
        "",
        "## Leaderboard",
        "| # | ID | Params | Arch | Method | Fresh | Cycles | Status |",
        "|---|-----|--------|------|--------|-------|--------|--------|",
    ]

    for i, exp in enumerate(experiments[:15]):
        cfg = json.loads(exp["config_json"]) if exp.get("config_json") else {}
        wd = cfg.get("weight_decay", 0)
        method = f"wd={wd}" if wd > 0 else "PerpGrad"
        star = "★" if i == 0 else ""
        peak = exp.get("peak_fresh", 0) or 0
        cycles = exp.get("max_cycle", 0) or 0
        lines.append(f"| {i+1} | {star}{exp['exp_id']} | {exp.get('n_params',0):,} | "
                    f"d={exp.get('d_model','')}L={exp.get('n_kernel_layers','')} | "
                    f"{method} | {peak:.1%} | {cycles} | {exp['status']} |")

    if active_tasks:
        lines.extend(["", "## Active Tasks"])
        for task in active_tasks:
            desc = TASK_DESC.get(task, task)
            lines.append(f"- {desc}")

    if teacher:
        lines.extend(["", "## Teacher", "```", teacher.get("status_text", ""), "```"])
        ml = json.loads(teacher.get("mastery_log", "[]"))
        if ml:
            lines.extend(["", "## Learning-to-Learn"])
            for entry in ml:
                lines.append(f"- Task {entry['task_index']+1}: **{entry['task']}** — "
                           f"{entry['steps_to_master']} steps, "
                           f"{entry['examples_to_master']:,} examples")

    if events:
        lines.extend(["", "## Recent Events"])
        for e in events[-10:]:
            ts = datetime.fromtimestamp(e["timestamp"]).strftime("%H:%M:%S")
            lines.append(f"- `{ts}` **{e['event_type']}** {e.get('exp_id','')} "
                        f"{e.get('details','')}")

    lines.append("")
    Path(output).write_text("\n".join(lines))


# ── Main ────────────────────────────────────────────────────────────

def render_once(db_path="metrics.db"):
    reader = MetricsReader(db_path)
    render_html(reader)
    render_md(reader)


def watch(db_path="metrics.db", interval=30):
    """Re-render every interval seconds."""
    print(f"Watching {db_path}, rendering every {interval}s...", flush=True)
    while True:
        try:
            render_once(db_path)
            print(f"  rendered at {datetime.now().strftime('%H:%M:%S')}", flush=True)
        except Exception as e:
            print(f"  render error: {e}", flush=True)
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
