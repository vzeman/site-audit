"""Score each page on how citable it is to AI answer engines.

Generative engines (ChatGPT, Perplexity, Google AI Overviews) are biased
toward pages that:

* declare structured intent (FAQ schema, HowTo, Article, Product)
* answer questions atomically (Q-form H2/H3 + a short paragraph below)
* show their work (lists, comparison tables, statistics, dated facts)
* link out to authoritative sources

We turn those into a coarse 0–10 score per page. The score isn't a hard
truth — it's a checklist of what an LLM "looks for" when picking which
URLs to cite.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .extractor import ExtractedPage


_FAQ_SCHEMA_TYPES = {"FAQPage", "QAPage", "Question"}
_STRUCTURED_TYPES = {"HowTo", "Article", "NewsArticle", "BlogPosting", "Product", "Recipe", "VideoObject", "TechArticle"}


@dataclass
class AnswerabilityScore:
    url: str
    score: float                  # 0..10
    breakdown: dict[str, float]   # human-readable signal -> contribution
    flags: list[str]              # textual notes for the report


def _question_heading_count(page: ExtractedPage) -> int:
    n = 0
    for h in page.headings or []:
        h_l = h.strip().lower()
        if h_l.endswith("?"):
            n += 1
            continue
        first = h_l.split()[0] if h_l else ""
        if first in {"how", "what", "why", "when", "where", "which", "who", "can", "is", "are", "should", "do", "does", "will"}:
            n += 1
    return n


def score_page(page: ExtractedPage) -> AnswerabilityScore:
    breakdown: dict[str, float] = {}
    flags: list[str] = []

    schema_types = set(page.schema_types or [])

    if schema_types & _FAQ_SCHEMA_TYPES:
        breakdown["faq_schema"] = 3.0
        flags.append("FAQ/QA schema present")
    elif schema_types & _STRUCTURED_TYPES:
        breakdown["structured_schema"] = 1.5
        flags.append(f"structured schema: {sorted(schema_types & _STRUCTURED_TYPES)[0]}")
    elif schema_types:
        other = ", ".join(sorted(schema_types)[:3])
        flags.append(f"schema present ({other}) — consider adding Article/FAQ/HowTo")
    else:
        flags.append("no schema markup")

    q_h = _question_heading_count(page)
    if q_h >= 3:
        breakdown["q_headings"] = 2.0
        flags.append(f"{q_h} question-form headings")
    elif q_h >= 1:
        breakdown["q_headings"] = 1.0
        flags.append(f"{q_h} question-form heading")
    else:
        flags.append("no question-form headings")

    if page.list_count >= 1:
        breakdown["lists"] = min(1.0, 0.4 + 0.2 * page.list_count)
    if page.table_count >= 1:
        breakdown["tables"] = 1.0
        flags.append(f"{page.table_count} table(s)")

    if page.has_dates:
        breakdown["dates"] = 0.5

    if page.stat_count >= 5:
        breakdown["statistics"] = 1.5
        flags.append(f"{page.stat_count} stat-like phrases")
    elif page.stat_count >= 2:
        breakdown["statistics"] = 0.8
    else:
        flags.append("few/no statistics")

    if page.external_link_count >= 3:
        breakdown["external_citations"] = 1.0
        flags.append(f"{page.external_link_count} outbound citations")
    elif page.external_link_count >= 1:
        breakdown["external_citations"] = 0.5

    title = (page.title or "").strip()
    if title.endswith("?") or any(title.lower().startswith(w + " ") for w in ("how", "what", "why", "when", "which")):
        breakdown["question_title"] = 0.5
        flags.append("question-form title")

    if page.word_count >= 800:
        breakdown["long_form"] = 0.3

    score = float(min(10.0, sum(breakdown.values())))
    return AnswerabilityScore(url=page.url, score=score, breakdown=breakdown, flags=flags)


def score_all(pages: Iterable[ExtractedPage]) -> list[AnswerabilityScore]:
    return [score_page(p) for p in pages]


def to_payload(scores: list[AnswerabilityScore]) -> list[dict]:
    rows = [
        {
            "url": s.url,
            "score": round(s.score, 2),
            "breakdown": {k: round(v, 2) for k, v in s.breakdown.items()},
            "flags": s.flags,
        }
        for s in scores
    ]
    rows.sort(key=lambda r: r["score"])  # weakest first — those need attention
    return rows
