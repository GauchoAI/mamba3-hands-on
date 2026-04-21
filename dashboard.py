"""
Dashboard generator — writes static HTML + markdown from tuner state.

Called by auto_tuner every N cycles. Produces:
  - dashboard.html: Chart.js + Tailwind, WSJ-clean style
  - dashboard.md: markdown summary for Claude to read

The H100 already has Jupyter on port 8080. We write to the workspace
so it's accessible via the Jupyter file browser, or serve separately.
"""
import json
from pathlib import Path
from datetime import datetime


def generate_dashboard(experiments, teacher, cycle, output_dir="."):
    """Generate HTML + MD dashboard from current tuner state."""
    output_dir = Path(output_dir)

    # Collect data
    exp_data = []
    for exp in experiments:
        if not hasattr(exp, 'history'):
            continue
        series = []
        for c, fresh, type_accs in exp.history:
            series.append({
                "cycle": c,
                "fresh": round(fresh * 100, 1),
                "parity": round(type_accs.get("parity", 0) * 100, 1),
                "same_different": round(type_accs.get("same_different", 0) * 100, 1),
            })
        exp_data.append({
            "name": exp.cfg.name(),
            "params": exp.n_params,
            "alive": exp.alive,
            "best": round(exp.best_acc * 100, 1),
            "wd": exp.cfg.weight_decay,
            "perp": exp.use_perp,
            "series": series,
        })

    # Sort by best accuracy
    exp_data.sort(key=lambda x: -x["best"])

    # Teacher status
    teacher_lines = teacher.get_status().split("\n")
    learning_report = teacher.get_learning_report() if teacher.mastery_log else ""

    _write_html(exp_data, teacher_lines, learning_report, cycle, output_dir)
    _write_md(exp_data, teacher_lines, learning_report, cycle, output_dir)


def _write_html(exp_data, teacher_lines, learning_report, cycle, output_dir):
    """Generate Chart.js + Tailwind dashboard."""
    # Color palette (WSJ-inspired)
    colors = [
        "#0066CC", "#CC3300", "#339933", "#CC6600",
        "#6633CC", "#CC3399", "#336699", "#996633",
    ]

    # Build chart datasets
    datasets_fresh = []
    datasets_parity = []
    for i, exp in enumerate(exp_data):
        color = colors[i % len(colors)]
        opacity = "1.0" if exp["alive"] else "0.3"
        tag = "★ " if i == 0 else ""
        method = "grok" if exp["wd"] > 0 else "perp"

        ds_fresh = {
            "label": f'{tag}{exp["name"]} [{method}]',
            "data": [{"x": s["cycle"], "y": s["fresh"]} for s in exp["series"]],
            "borderColor": color,
            "backgroundColor": "transparent",
            "borderWidth": 2 if exp["alive"] else 1,
            "pointRadius": 0,
            "tension": 0.3,
        }
        ds_parity = {
            "label": f'{tag}{exp["name"]} [{method}]',
            "data": [{"x": s["cycle"], "y": s["parity"]} for s in exp["series"]],
            "borderColor": color,
            "backgroundColor": "transparent",
            "borderWidth": 2 if exp["alive"] else 1,
            "borderDash": [] if exp["alive"] else [5, 5],
            "pointRadius": 0,
            "tension": 0.3,
        }
        datasets_fresh.append(ds_fresh)
        datasets_parity.append(ds_parity)

    # Leaderboard rows
    leaderboard_html = ""
    for i, exp in enumerate(exp_data):
        method = "weight_decay=0.1" if exp["wd"] > 0 else "PerpGrad"
        status = "Alive" if exp["alive"] else "Pruned"
        status_color = "text-green-600" if exp["alive"] else "text-red-400"
        star = "★" if i == 0 else ""
        leaderboard_html += f"""
        <tr class="border-b border-gray-100">
            <td class="py-2 pr-4 font-mono text-sm">{star} {exp['name']}</td>
            <td class="py-2 pr-4 text-right">{exp['params']:,}</td>
            <td class="py-2 pr-4">{method}</td>
            <td class="py-2 pr-4 text-right font-bold">{exp['best']}%</td>
            <td class="py-2 pr-4 {status_color}">{status}</td>
        </tr>"""

    # Teacher status HTML
    teacher_html = "<br>".join(
        f'<span class="font-mono text-sm">{line.strip()}</span>'
        for line in teacher_lines
    )

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta http-equiv="refresh" content="30">
<title>Mamba-3 Training Dashboard</title>
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  body {{ font-family: 'Georgia', serif; }}
  .chart-container {{ position: relative; height: 300px; }}
</style>
</head>
<body class="bg-white text-gray-900 max-w-6xl mx-auto px-8 py-6">

<header class="border-b-2 border-black pb-2 mb-6">
  <h1 class="text-3xl font-bold tracking-tight">Mamba-3 Curriculum Training</h1>
  <p class="text-gray-500 text-sm mt-1">Cycle {cycle} &middot; {now} &middot; Auto-refreshes every 30s</p>
</header>

<div class="grid grid-cols-2 gap-8 mb-8">
  <div>
    <h2 class="text-lg font-bold mb-2 border-b border-gray-200 pb-1">Fresh Accuracy (%)</h2>
    <div class="chart-container">
      <canvas id="freshChart"></canvas>
    </div>
  </div>
  <div>
    <h2 class="text-lg font-bold mb-2 border-b border-gray-200 pb-1">Parity Accuracy (%)</h2>
    <div class="chart-container">
      <canvas id="parityChart"></canvas>
    </div>
  </div>
</div>

<h2 class="text-lg font-bold mb-2 border-b border-gray-200 pb-1">Leaderboard</h2>
<table class="w-full mb-8">
  <thead>
    <tr class="border-b-2 border-gray-300 text-left text-sm text-gray-500">
      <th class="py-1 pr-4">Config</th>
      <th class="py-1 pr-4 text-right">Params</th>
      <th class="py-1 pr-4">Method</th>
      <th class="py-1 pr-4 text-right">Best Fresh</th>
      <th class="py-1 pr-4">Status</th>
    </tr>
  </thead>
  <tbody>{leaderboard_html}</tbody>
</table>

<h2 class="text-lg font-bold mb-2 border-b border-gray-200 pb-1">Curriculum Teacher</h2>
<div class="bg-gray-50 p-4 rounded mb-4 font-mono text-sm leading-relaxed">
  {teacher_html}
</div>

{f'<div class="bg-blue-50 p-4 rounded mb-4 font-mono text-sm"><pre>{learning_report}</pre></div>' if learning_report else ''}

<script>
const freshData = {json.dumps(datasets_fresh)};
const parityData = {json.dumps(datasets_parity)};

const chartOpts = {{
  responsive: true,
  maintainAspectRatio: false,
  plugins: {{ legend: {{ position: 'bottom', labels: {{ font: {{ size: 10 }} }} }} }},
  scales: {{
    x: {{ type: 'linear', title: {{ display: true, text: 'Cycle' }} }},
    y: {{ min: 0, max: 100, title: {{ display: true, text: '%' }} }}
  }}
}};

new Chart(document.getElementById('freshChart'), {{
  type: 'line', data: {{ datasets: freshData }}, options: chartOpts
}});
new Chart(document.getElementById('parityChart'), {{
  type: 'line', data: {{ datasets: parityData }}, options: chartOpts
}});
</script>

<footer class="text-gray-400 text-xs mt-8 border-t pt-2">
  Mamba-3 Hands-On &middot; GauchoAI &middot; Progressive Growing + Auto-Tuner
</footer>
</body>
</html>"""

    (output_dir / "dashboard.html").write_text(html)


def _write_md(exp_data, teacher_lines, learning_report, cycle, output_dir):
    """Generate markdown summary."""
    lines = [f"# Training Dashboard — Cycle {cycle}", ""]

    lines.append("## Leaderboard")
    lines.append("| Config | Params | Method | Best Fresh | Status |")
    lines.append("|--------|--------|--------|-----------|--------|")
    for i, exp in enumerate(exp_data):
        method = "wd=0.1" if exp["wd"] > 0 else "PerpGrad"
        status = "Alive" if exp["alive"] else "Pruned"
        star = "★" if i == 0 else ""
        lines.append(f"| {star}{exp['name']} | {exp['params']:,} | {method} | {exp['best']}% | {status} |")

    lines.append("")
    lines.append("## Teacher Status")
    lines.extend(teacher_lines)

    if learning_report:
        lines.append("")
        lines.append(learning_report)

    lines.append("")
    (output_dir / "dashboard.md").write_text("\n".join(lines))
