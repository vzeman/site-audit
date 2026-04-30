"""Map a list of target queries onto pages in the same vector space.

Three things drop out of one cosine-similarity pass:

* **Best page per query** (the canonical answer for that query)
* **Coverage gaps**     — queries with no page above a usefulness floor
* **Cannibalization**   — queries with N+ pages above a "too similar"
                          ceiling, meaning the site is competing with
                          itself for that intent

Queries can be:
  - explicit (``--queries-file`` with one query per line, optionally
    grouped by section: ``# section name`` then queries underneath)
  - auto-mined from the crawl: page titles + question-form H2/H3
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

LOG = logging.getLogger(__name__)


_QUESTION_PREFIXES = (
    "how ", "what ", "why ", "when ", "where ", "which ", "who ", "can ",
    "is ", "are ", "should ", "do ", "does ", "will ",
)


@dataclass
class QueryMatch:
    query: str
    source: str              # 'manual', 'title', 'h2'
    best_url: str
    best_title: str
    best_similarity: float
    runner_ups: list[dict]   # [{url, title, similarity}]
    status: str              # 'covered', 'gap', 'cannibalized'
    candidates_above_threshold: int


def load_queries_from_file(path: Path) -> list[str]:
    queries: list[str] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        queries.append(line)
    # de-dupe while preserving order
    seen: set[str] = set()
    out = []
    for q in queries:
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(q)
    return out


def _looks_like_question(text: str) -> bool:
    t = text.strip().lower()
    if t.endswith("?"):
        return True
    return any(t.startswith(prefix) for prefix in _QUESTION_PREFIXES)


def auto_mine_queries(pages, max_queries: int = 200) -> list[tuple[str, str]]:
    """Return list of (query, source). Source is 'title' or 'h2'."""
    seen: set[str] = set()
    out: list[tuple[str, str]] = []

    # 1) question-form headings first (highest signal)
    for p in pages:
        for h in getattr(p, "headings", []) or []:
            if not _looks_like_question(h):
                continue
            if len(h) < 8 or len(h) > 140:
                continue
            key = re.sub(r"\s+", " ", h.lower()).strip("? ")
            if key in seen:
                continue
            seen.add(key)
            out.append((h, "h2"))

    # 2) titles (cap each title to a reasonable phrase length)
    for p in pages:
        title = (p.title or "").strip()
        if not title or len(title) < 6 or len(title) > 140:
            continue
        # strip the typical " | Brand" suffix
        title_clean = re.split(r"[|·–—-]\s+\w[\w\s&]+$", title)[0].strip()
        key = title_clean.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append((title_clean, "title"))

    return out[:max_queries]


def match_queries(
    queries_with_source: list[tuple[str, str]],
    query_embeddings: np.ndarray,
    pages,
    page_embeddings: np.ndarray,
    coverage_threshold: float = 0.55,
    cannibalization_threshold: float = 0.72,
    max_runner_ups: int = 4,
) -> list[QueryMatch]:
    if len(queries_with_source) == 0 or len(pages) == 0:
        return []

    sims = query_embeddings @ page_embeddings.T  # [Q, P]
    sims = np.clip(sims, -1.0, 1.0)

    out: list[QueryMatch] = []
    for qi, (query, source) in enumerate(queries_with_source):
        row = sims[qi]
        order = np.argsort(-row)
        best = int(order[0])
        best_sim = float(row[best])

        runner_ups = []
        for pi in order[1:max_runner_ups + 1]:
            runner_ups.append({
                "url": pages[int(pi)].url,
                "title": pages[int(pi)].title,
                "similarity": float(round(row[int(pi)], 4)),
            })

        above = int(np.sum(row >= cannibalization_threshold))
        if best_sim < coverage_threshold:
            status = "gap"
        elif above >= 3:
            status = "cannibalized"
        else:
            status = "covered"

        out.append(QueryMatch(
            query=query,
            source=source,
            best_url=pages[best].url,
            best_title=pages[best].title,
            best_similarity=float(round(best_sim, 4)),
            runner_ups=runner_ups,
            status=status,
            candidates_above_threshold=above,
        ))

    return out


def to_payload(matches: Iterable[QueryMatch]) -> list[dict]:
    return [
        {
            "query": m.query,
            "source": m.source,
            "status": m.status,
            "best_similarity": m.best_similarity,
            "best_url": m.best_url,
            "best_title": m.best_title,
            "candidates_above_threshold": m.candidates_above_threshold,
            "runner_ups": m.runner_ups,
        }
        for m in matches
    ]
