"""Render markdown docs to styled HTML."""
import re
from pathlib import Path

def md_to_html(md_text):
    """Simple markdown to HTML converter with Tailwind styling."""
    lines = md_text.split("\n")
    html_lines = []
    in_code = False
    in_table = False

    for line in lines:
        # Code blocks
        if line.strip().startswith("```"):
            if in_code:
                html_lines.append("</code></pre>")
                in_code = False
            else:
                lang = line.strip().replace("```", "")
                html_lines.append(f'<pre class="bg-gray-900 text-green-300 p-4 rounded-lg text-sm overflow-x-auto my-3"><code>')
                in_code = True
            continue
        if in_code:
            html_lines.append(line.replace("<", "&lt;").replace(">", "&gt;"))
            continue

        # Tables
        if "|" in line and line.strip().startswith("|"):
            cells = [c.strip() for c in line.split("|")[1:-1]]
            if all(c.replace("-", "").replace(":", "") == "" for c in cells):
                continue  # separator row
            if not in_table:
                html_lines.append('<table class="w-full text-sm my-4 border-collapse">')
                in_table = True
                html_lines.append("<thead><tr>")
                for cell in cells:
                    html_lines.append(f'<th class="border-b-2 border-gray-300 py-2 px-3 text-left">{_inline(cell)}</th>')
                html_lines.append("</tr></thead><tbody>")
                continue
            html_lines.append("<tr>")
            for cell in cells:
                cls = "font-bold text-green-700" if "★" in cell or "Leader" in cell else ""
                html_lines.append(f'<td class="border-b border-gray-100 py-1.5 px-3 {cls}">{_inline(cell)}</td>')
            html_lines.append("</tr>")
            continue
        elif in_table:
            html_lines.append("</tbody></table>")
            in_table = False

        # Headers
        if line.startswith("# "):
            html_lines.append(f'<h1 class="text-3xl font-bold mt-8 mb-4 border-b-2 border-black pb-2">{_inline(line[2:])}</h1>')
        elif line.startswith("## "):
            html_lines.append(f'<h2 class="text-2xl font-bold mt-8 mb-3 text-blue-900">{_inline(line[3:])}</h2>')
        elif line.startswith("### "):
            html_lines.append(f'<h3 class="text-xl font-semibold mt-6 mb-2 text-gray-700">{_inline(line[4:])}</h3>')
        elif line.startswith("> "):
            html_lines.append(f'<blockquote class="border-l-4 border-blue-400 bg-blue-50 p-3 my-3 text-sm italic">{_inline(line[2:])}</blockquote>')
        elif line.startswith("---"):
            html_lines.append('<hr class="my-8 border-gray-300">')
        elif line.strip() == "":
            html_lines.append("")
        else:
            html_lines.append(f'<p class="my-2 leading-relaxed">{_inline(line)}</p>')

    if in_table:
        html_lines.append("</tbody></table>")

    return "\n".join(html_lines)


def _inline(text):
    """Process inline markdown: bold, italic, code, links."""
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
    text = re.sub(r'`(.+?)`', r'<code class="bg-gray-100 px-1.5 py-0.5 rounded text-sm font-mono text-red-700">\1</code>', text)
    return text


def render_file(md_path, output_path=None):
    md_text = Path(md_path).read_text()
    body = md_to_html(md_text)

    if not output_path:
        output_path = Path(md_path).with_suffix(".html")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<script src="https://cdn.tailwindcss.com"></script>
<style>
  body {{ font-family: 'Georgia', 'Times New Roman', serif; }}
  h1, h2, h3 {{ font-family: 'Inter', system-ui, sans-serif; }}
</style>
<title>Training Strategies</title>
</head>
<body class="bg-white text-gray-800 max-w-4xl mx-auto px-8 py-6">
{body}
<footer class="text-gray-400 text-xs mt-12 border-t pt-4">
  Mamba-3 Hands-On · GauchoAI · Generated from training_strategies.md
</footer>
</body>
</html>"""

    Path(output_path).write_text(html)
    print(f"Rendered {md_path} → {output_path}")


if __name__ == "__main__":
    render_file("docs/training_strategies.md", "docs/training_strategies.html")
