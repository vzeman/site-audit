"""On-disk evidence workspace for the SERP-gap AI agent.

For each analyzed URL we write a self-contained directory the coding agent
(Harnext) can explore with file tools: the full computed evidence, our page's
complete content, competitor outlines and paragraphs, SERP features, and the
task instructions including the recommendation contract.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from urllib.parse import urlparse


def _slug(url: str) -> str:
    parsed = urlparse(url if "://" in url else f"https://{url}")
    path = (parsed.path or "/").strip("/") or "home"
    if parsed.query:
        path = f"{path}-{parsed.query}"
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", path).strip("-").lower() or "home"
    return slug[:90].strip("-") or "home"


def _evidence_payload(page: dict) -> dict:
    payload = copy.deepcopy(page)
    payload.pop("ai_editor_brief", None)
    payload.pop("ai_recommendation", None)
    for analysis in payload.get("analyses") or []:
        analysis.pop("scatter", None)
        analysis.pop("visual_summary", None)
    return payload


def _heading_lines(headings: list[dict]) -> list[str]:
    lines: list[str] = []
    for heading in headings or []:
        text = str(heading.get("text") or "").strip()
        if not text:
            continue
        try:
            level = int(heading.get("level") or 2)
        except (TypeError, ValueError):
            level = 2
        level = min(max(level, 1), 6)
        lines.append(f"{'  ' * (level - 1)}- (H{level}) {text}")
    return lines


def _our_page_markdown(page: dict, own_ext) -> str:
    paragraphs = list(getattr(own_ext, "paragraphs", None) or [])
    headings = getattr(own_ext, "headers_rich", None) or []
    lines = [
        f"# Our page: {page.get('url', '')}",
        "",
        f"- Title: {getattr(own_ext, 'title', '') or page.get('title', '')}",
        f"- Meta description: {getattr(own_ext, 'description', '')}",
        f"- H1: {getattr(own_ext, 'h1', '') or page.get('h1', '')}",
        f"- Word count: {getattr(own_ext, 'word_count', 0)}",
        "",
        "## Heading outline",
        "",
        *(_heading_lines(headings) or ["- (none)"]),
        "",
        "## Paragraphs",
        "",
    ]
    for index, paragraph in enumerate(paragraphs):
        text = str(paragraph or "").strip()
        if not text:
            continue
        lines.append(f"### [P{index}] ({len(text.split())} words)")
        lines.append("")
        lines.append(text)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _competitor_markdown(url: str, content: dict) -> str:
    lines = [
        f"# Competitor: {url}",
        "",
        f"- Rank: {content.get('rank', '')}",
        f"- Title: {content.get('title', '')}",
        f"- H1: {content.get('h1', '')}",
        "",
        "## Heading outline",
        "",
        *(_heading_lines(content.get("headings") or []) or ["- (none)"]),
        "",
        "## Paragraphs",
        "",
    ]
    for index, paragraph in enumerate(content.get("paragraphs") or []):
        text = str(paragraph or "").strip()
        if not text:
            continue
        lines.append(f"### [C{index}]")
        lines.append("")
        lines.append(text)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _serp_payload(page: dict) -> dict:
    out: dict = {}
    for analysis in page.get("analyses") or []:
        keyword = str((analysis.get("keyword") or {}).get("keyword") or analysis.get("query") or "").strip()
        if not keyword:
            continue
        rankings = [
            {"url": row.get("url", ""), "rank": row.get("rank"), "title": row.get("title", "")}
            for row in analysis.get("competitor_pages") or []
            if not row.get("error")
        ]
        out[keyword] = {
            "features": analysis.get("serp_features") or {},
            "paa_coverage": analysis.get("paa_coverage") or [],
            "rankings": rankings,
        }
    return out


def _task_markdown(page: dict, schema_doc: str) -> str:
    return "\n".join([
        "# Task: page edit recommendation",
        "",
        f"Target URL: {page.get('url', '')}",
        "",
        "You are a senior SEO/GEO editor. Work only from the files in this directory:",
        "",
        "- `evidence.json` — every metric computed when comparing this page against ranking competitors:",
        "  topic coverage (missing/partial/covered with priorities), paragraph match heatmap with",
        "  per-competitor similarities, content comparison benchmark, structural/GEO patterns,",
        "  People Also Ask coverage, content order analysis, and prioritized action points.",
        "- `our_page.md` — our page's full content; paragraphs are numbered [P0], [P1], ...",
        "- `competitors/` — one file per ranking competitor page (outline + paragraphs).",
        "- `serp.json` — SERP rankings and features (People Also Ask, related searches) per keyword.",
        "",
        "Produce exactly two files in this directory:",
        "",
        "1. `recommendation.json` — machine-readable recommendation following the contract below, exactly.",
        "2. `brief.md` — a human-readable editorial brief summarizing the same plan.",
        "",
        "Rules:",
        "",
        "- Decide every paragraph: each [P<index>] from our_page.md must appear exactly once in",
        "  `paragraph_decisions` with decision keep, rewrite, move, merge, or remove.",
        "- Write original copy only. Never reuse competitor wording.",
        "- Cover every missing-status People Also Ask question in a section or FAQ block.",
        "- Respect `structural_patterns` advice (tables, question-form headings, statistics, schema).",
        "- Use the benchmark medians in evidence.json as the size target; do not pad with filler.",
        "- Do not duplicate topics whose coverage is already `covered`.",
        "- If demand metrics (impressions, clicks, traffic, volume) are absent, write",
        "  \"demand metrics absent\" instead of guessing numbers.",
        "- Do not modify any other file in this directory.",
        "",
        schema_doc.strip(),
        "",
    ])


def write_agent_workspace(
    report_dir: Path,
    page: dict,
    own_ext,
    competitor_content: dict[str, dict],
    schema_doc: str = "",
) -> Path:
    workspace = Path(report_dir) / "agent" / _slug(str(page.get("url") or ""))
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "evidence.json").write_text(
        json.dumps(_evidence_payload(page), ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    (workspace / "our_page.md").write_text(_our_page_markdown(page, own_ext), encoding="utf-8")
    competitors_dir = workspace / "competitors"
    competitors_dir.mkdir(exist_ok=True)
    for url, content in sorted(
        (competitor_content or {}).items(),
        key=lambda item: (int(item[1].get("rank") or 999), item[0]),
    ):
        rank = int(content.get("rank") or 0)
        domain = re.sub(r"[^a-zA-Z0-9.-]+", "-", urlparse(url).netloc or "competitor").strip("-")
        name = f"{rank:02d}-{domain}.md" if rank else f"00-{domain}.md"
        path = competitors_dir / name
        if path.exists():
            stem = path.stem
            suffix = 2
            while path.exists():
                path = competitors_dir / f"{stem}-{suffix}.md"
                suffix += 1
        path.write_text(_competitor_markdown(url, content), encoding="utf-8")
    (workspace / "serp.json").write_text(
        json.dumps(_serp_payload(page), ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    (workspace / "TASK.md").write_text(_task_markdown(page, schema_doc), encoding="utf-8")
    return workspace
