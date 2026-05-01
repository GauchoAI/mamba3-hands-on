from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "lab_book" / "manifest.json"

LEGACY_FINDINGS = {
    "01_ga_tournament": "docs/findings/ga_tournament.md",
    "03_synapse_parity": "docs/findings/synapse.md",
    "04_hanoi": "docs/findings/hanoi.md",
    "05_lego_library": "docs/findings/lego.md",
    "06_cortex_existence": "docs/findings/cortex.md",
}


def parse_frontmatter(path: Path) -> tuple[dict[str, str], str]:
    text = path.read_text()
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---", 4)
    if end < 0:
        return {}, text
    attrs: dict[str, str] = {}
    for line in text[4:end].strip().splitlines():
        match = re.match(r"^([A-Za-z0-9_-]+):\s*(.*)$", line)
        if not match:
            continue
        value = match.group(2).strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        attrs[match.group(1)] = value
    body_start = text.find("\n", end + 4)
    return attrs, text[body_start + 1 :] if body_start >= 0 else ""


def title_from_body(body: str, fallback: str) -> str:
    for line in body.splitlines():
        match = re.match(r"^#\s+(.+?)\s*$", line)
        if match:
            return match.group(1).replace("—", "-").strip()
    return fallback.replace("_", " ").title()


def experiment_key(exp_dir: Path) -> str:
    return re.sub(r"^\d+_", "", exp_dir.name)


def source_for(exp_dir: Path) -> dict | None:
    readme = exp_dir / "README.md"
    if not readme.exists():
        return None
    attrs, body = parse_frontmatter(readme)
    if attrs.get("lab_book", "").lower() in {"false", "hidden", "subsection"}:
        return None
    chapter = attrs.get("chapter")
    if not chapter:
        match = re.match(r"^(\d+)_", exp_dir.name)
        chapter = str(int(match.group(1))) if match else ""
    key = experiment_key(exp_dir)
    title = attrs.get("title") or title_from_body(body, key)
    if chapter:
        title = f"{chapter.zfill(2)} - {title}"
    source = {
        "id": f"ch{chapter.zfill(2)}" if chapter else key,
        "title": title,
        "path": str(readme.relative_to(ROOT)),
        "group": "experiment",
        "experiment": key,
        "chapter": chapter,
    }
    finding = exp_dir / "findings.md"
    legacy = LEGACY_FINDINGS.get(exp_dir.name)
    if finding.exists():
        source["finding"] = str(finding.relative_to(ROOT))
    elif legacy and (ROOT / legacy).exists():
        source["finding"] = legacy
    return source


def sort_key(source: dict) -> tuple[int, str]:
    chapter = source.get("chapter")
    if chapter and str(chapter).isdigit():
        return int(chapter), source["id"]
    return 9999, source["id"]


def main() -> None:
    sources = [
        source
        for exp_dir in sorted((ROOT / "experiments").glob("[0-9][0-9]_*"))
        if (source := source_for(exp_dir))
    ]
    sources.sort(key=sort_key)
    OUT.write_text(json.dumps({"sources": sources}, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)} with {len(sources)} experiment sources")


if __name__ == "__main__":
    main()
