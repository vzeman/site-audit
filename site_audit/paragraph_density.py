"""Per-paragraph link density.

For every paragraph we already extract, count the internal and external
``<a href>`` elements it contains and divide by the paragraph's word count
(× 100) to get a "links per 100 words" density. Two practical uses:

* **Editorial signal** — paragraphs at one extreme look like SEO link
  spam; at the other, they're large blocks of unlinked text that pass no
  authority. Both deserve attention.
* **Filter for paragraph-link recommendations** — when the
  ``paragraph_links.recommend`` step suggests adding an inline link to a
  paragraph, skip the suggestion if the paragraph is already saturated.

The output payload mirrors the shape of other paragraph-level analyses
(summary + per-page rollup + flagged-paragraphs list).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


_WORD_RE = re.compile(r"[A-Za-zÀ-ÿ0-9][A-Za-zÀ-ÿ0-9'-]*")


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


@dataclass
class _Row:
    page_index: int
    paragraph_index: int
    url: str
    title: str
    text: str
    words: int
    internal: int
    external: int
    density_per_100: float  # (internal+external) / max(words, 1) * 100


def _rows(
    pages,
    paragraph_records: list,
    paragraph_link_counts_by_page: list[list[tuple[int, int]]],
) -> list[_Row]:
    rows: list[_Row] = []
    for page_i, para_i, text, _ in paragraph_records:
        counts = paragraph_link_counts_by_page[page_i] if page_i < len(paragraph_link_counts_by_page) else []
        internal, external = (counts[para_i] if para_i < len(counts) else (0, 0))
        words = _word_count(text)
        density = ((internal + external) / words * 100.0) if words else 0.0
        rows.append(_Row(
            page_index=page_i,
            paragraph_index=para_i,
            url=pages[page_i].url,
            title=pages[page_i].title,
            text=text,
            words=words,
            internal=int(internal),
            external=int(external),
            density_per_100=round(density, 2),
        ))
    return rows


def density_lookup(rows: Iterable[_Row]) -> dict[tuple[int, int], float]:
    """(page_index, paragraph_index) → links/100w. Used by paragraph_links.recommend."""
    return {(r.page_index, r.paragraph_index): r.density_per_100 for r in rows}


def total_links_lookup(rows: Iterable[_Row]) -> dict[tuple[int, int], int]:
    return {(r.page_index, r.paragraph_index): r.internal + r.external for r in rows}


def compute_rows(
    pages,
    paragraph_records: list,
    extracted_pages,
) -> list[_Row]:
    counts_by_page = [
        list(getattr(ext, "paragraph_link_counts", []) or []) for ext in extracted_pages
    ]
    return _rows(pages, paragraph_records, counts_by_page)


def to_payload(
    rows: list[_Row],
    high_density_threshold: float = 5.0,  # links per 100 words above this = "spammy"
    long_paragraph_words: int = 80,        # only flag zero-link paragraphs at least this long
    top_n_per_section: int = 25,
) -> dict:
    if not rows:
        return {"summary": {}, "per_page": [], "spammy": [], "unlinked_long": []}

    densities = sorted(r.density_per_100 for r in rows)
    n = len(densities)

    def _percentile(p):
        if not n:
            return 0.0
        idx = max(0, min(n - 1, int(round(p * (n - 1)))))
        return float(densities[idx])

    total_internal = sum(r.internal for r in rows)
    total_external = sum(r.external for r in rows)
    total_words = sum(r.words for r in rows)

    # Per-page rollup (sorted by avg density desc — link-stuffed pages)
    by_page: dict[int, dict] = {}
    for r in rows:
        agg = by_page.setdefault(r.page_index, {
            "page_index": r.page_index,
            "url": r.url,
            "title": r.title,
            "paragraphs": 0,
            "words": 0,
            "internal": 0,
            "external": 0,
        })
        agg["paragraphs"] += 1
        agg["words"] += r.words
        agg["internal"] += r.internal
        agg["external"] += r.external
    per_page = []
    for agg in by_page.values():
        words = max(agg["words"], 1)
        per_page.append({
            "url": agg["url"],
            "title": agg["title"],
            "paragraphs": agg["paragraphs"],
            "words": agg["words"],
            "internal_links": agg["internal"],
            "external_links": agg["external"],
            "links_per_100w": round((agg["internal"] + agg["external"]) / words * 100.0, 2),
        })
    per_page.sort(key=lambda x: x["links_per_100w"], reverse=True)
    page_densities = sorted(float(p["links_per_100w"]) for p in per_page)

    def _page_percentile(p):
        if not page_densities:
            return 0.0
        idx = max(0, min(len(page_densities) - 1, int(round(p * (len(page_densities) - 1)))))
        return float(page_densities[idx])

    summary = {
        "total_paragraphs": n,
        "total_internal_links": total_internal,
        "total_external_links": total_external,
        "total_words": total_words,
        "median_density_per_100w": _percentile(0.5),
        "p90_density_per_100w": _percentile(0.9),
        "p99_density_per_100w": _percentile(0.99),
        "median_page_density_per_100w": _page_percentile(0.5),
        "p90_page_density_per_100w": _page_percentile(0.9),
        "p99_page_density_per_100w": _page_percentile(0.99),
        "zero_link_share": sum(1 for r in rows if r.internal + r.external == 0) / n,
        "spammy_threshold_per_100w": high_density_threshold,
        "spammy_count": sum(1 for r in rows if r.density_per_100 >= high_density_threshold),
    }

    # Flag spammy paragraphs (high density AND have at least a couple of links)
    spammy_rows = [
        r for r in rows
        if r.density_per_100 >= high_density_threshold and (r.internal + r.external) >= 2
    ]
    spammy_rows.sort(key=lambda r: r.density_per_100, reverse=True)
    spammy = [
        {
            "url": r.url,
            "title": r.title,
            "paragraph_index": r.paragraph_index,
            "excerpt": r.text[:240],
            "words": r.words,
            "internal_links": r.internal,
            "external_links": r.external,
            "density_per_100w": r.density_per_100,
        }
        for r in spammy_rows[:top_n_per_section]
    ]

    # Long unlinked paragraphs — could host an inline internal link
    unlinked_rows = [
        r for r in rows
        if r.internal + r.external == 0 and r.words >= long_paragraph_words
    ]
    unlinked_rows.sort(key=lambda r: r.words, reverse=True)
    unlinked_long = [
        {
            "url": r.url,
            "title": r.title,
            "paragraph_index": r.paragraph_index,
            "excerpt": r.text[:240],
            "words": r.words,
        }
        for r in unlinked_rows[:top_n_per_section]
    ]

    return {
        "summary": summary,
        "per_page": per_page,
        "spammy": spammy,
        "unlinked_long": unlinked_long,
    }
