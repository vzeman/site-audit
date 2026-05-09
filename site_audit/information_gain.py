"""Information-gain and originality scoring for pages and paragraphs."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Iterable

from .analyzer import PageInfo
from .entities import extract_entities_from_text

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9'_-]*", re.I)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_STAT_RE = re.compile(
    r"\b\d+(?:[,.]\d+)*\s*(?:%|percent|million|billion|thousand|users?|customers?|clients?|"
    r"seconds?|minutes?|hours?|days?|weeks?|months?|years?|\$|€|£|x|times)\b",
    re.I,
)
_EXAMPLE_RE = re.compile(r"\b(for example|case study|customer story|benchmark|we tested|our data|in practice|implementation|workflow|step-by-step)\b", re.I)
_GENERIC_RE = re.compile(r"\b(learn more|read more|contact us|in today's digital world|it is important to|this article will|whether you are|businesses need)\b", re.I)
_BOILERPLATE_RE = re.compile(r"\b(cookie|privacy policy|terms|all rights reserved|subscribe|newsletter|copyright)\b", re.I)


def _tokens(text: str) -> list[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(text or "")]


def _fingerprint(text: str) -> str:
    return " ".join(_tokens(text))


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text or "") if len(s.strip()) >= 30]


def _clip(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def _cluster_labels(cluster_summaries) -> dict[int, str]:
    out: dict[int, str] = {}
    for summary in cluster_summaries or []:
        cid = getattr(summary, "cluster_id", None)
        keywords = getattr(summary, "keywords", []) or []
        if cid is not None:
            out[int(cid)] = ", ".join(k.get("keyword", "") for k in keywords[:3] if k.get("keyword")) or f"cluster {cid}"
    return out


def _entity_cluster_counts(page_entities: list[set[str]], clusters: list[int]) -> dict[int, Counter[str]]:
    out: dict[int, Counter[str]] = defaultdict(Counter)
    for ents, cid in zip(page_entities, clusters):
        out[int(cid)].update(ents)
    return out


def _unique_fact_snippets(text: str, cluster_common: set[str], limit: int = 6) -> list[str]:
    snippets: list[str] = []
    for sentence in _sentences(text):
        ents = set(extract_entities_from_text(sentence))
        has_stat = bool(_STAT_RE.search(sentence))
        has_example = bool(_EXAMPLE_RE.search(sentence))
        rare_entities = ents - cluster_common
        if has_stat or has_example or rare_entities:
            snippets.append(sentence[:260])
        if len(snippets) >= limit:
            break
    return snippets


def _score_text(
    text: str,
    *,
    duplicate_count: int,
    external_links: int = 0,
    media_count: int = 0,
    schema_count: int = 0,
    cluster_common: set[str] | None = None,
) -> tuple[float, list[str], list[str], list[str]]:
    cluster_common = cluster_common or set()
    tokens = _tokens(text)
    words = len(tokens)
    entities = set(extract_entities_from_text(text))
    rare_entities = entities - cluster_common
    stat_count = len(_STAT_RE.findall(text or ""))
    example_count = len(_EXAMPLE_RE.findall(text or ""))
    generic_hits = len(_GENERIC_RE.findall(text or ""))
    boilerplate = bool(_BOILERPLATE_RE.search(text or ""))
    unique_ratio = len(set(tokens)) / max(words, 1)

    positives: list[str] = []
    negatives: list[str] = []
    if stat_count:
        positives.append(f"{stat_count} data/statistic signals")
    if example_count:
        positives.append(f"{example_count} example or implementation signals")
    if external_links:
        positives.append(f"{external_links} outbound citation links")
    if media_count:
        positives.append(f"{media_count} media/screenshot signals")
    if schema_count:
        positives.append(f"{schema_count} structured-data types")
    if rare_entities:
        positives.append(f"{len(rare_entities)} less-common named entities")

    if words < 80:
        negatives.append("thin content block")
    if duplicate_count >= 2:
        negatives.append(f"duplicated text appears {duplicate_count} times")
    if generic_hits:
        negatives.append("generic SEO phrasing detected")
    if boilerplate:
        negatives.append("boilerplate/template language")
    if not stat_count and not example_count and not external_links:
        negatives.append("no data, examples, or citations")
    if unique_ratio < 0.42 and words >= 40:
        negatives.append("low unique-word ratio")

    score = 45.0
    score += min(18.0, stat_count * 5.0)
    score += min(16.0, example_count * 6.0)
    score += min(12.0, external_links * 3.0)
    score += min(10.0, media_count * 2.0)
    score += min(8.0, schema_count * 2.0)
    score += min(16.0, len(rare_entities) * 1.8)
    score += min(8.0, max(0.0, unique_ratio - 0.45) * 20.0)
    score -= min(22.0, duplicate_count * 7.0) if duplicate_count >= 2 else 0.0
    score -= min(16.0, generic_hits * 5.0)
    score -= 16.0 if boilerplate else 0.0
    score -= 12.0 if words < 80 else 0.0
    score -= 10.0 if "no data, examples, or citations" in negatives else 0.0
    snippets = _unique_fact_snippets(text, cluster_common)
    return round(_clip(score), 2), positives[:8], negatives[:8], snippets


def _recommendations(negatives: list[str]) -> list[str]:
    recs: list[str] = []
    joined = " ".join(negatives)
    if "no data" in joined:
        recs.append("Add original data, measured benchmarks, examples, or citations.")
    if "thin" in joined:
        recs.append("Expand the section with implementation detail and concrete examples.")
    if "duplicated" in joined or "boilerplate" in joined:
        recs.append("Replace repeated boilerplate with page-specific insight.")
    if "generic" in joined:
        recs.append("Replace generic summaries with product-specific claims and named use cases.")
    return recs[:5]


def build_information_gain(
    pages: list[PageInfo],
    extracted_pages: list,
    paragraph_records: list[tuple[int, int, str, object]] | None = None,
    *,
    cluster_labels: Iterable[int] | None = None,
    cluster_summaries=None,
    top_n: int = 700,
) -> dict:
    if not pages:
        return {"summary": {"status": "no_pages", "pages": 0}, "pages": [], "paragraphs": []}
    clusters = list(cluster_labels) if cluster_labels is not None else [0] * len(pages)
    label_lookup = _cluster_labels(cluster_summaries)
    page_entity_sets: list[set[str]] = []
    page_texts: list[str] = []
    for i, page in enumerate(pages):
        ext = extracted_pages[i] if i < len(extracted_pages) else None
        text = " ".join([
            page.title or "",
            page.description or "",
            getattr(ext, "h1", "") if ext is not None else "",
            " ".join(str(h.get("text", "")) for h in (getattr(ext, "headers_rich", []) or [])) if ext is not None else "",
            getattr(ext, "body", "") if ext is not None else "",
        ])
        page_texts.append(text)
        page_entity_sets.append(set(extract_entities_from_text(text)))
    cluster_entity_counts = _entity_cluster_counts(page_entity_sets, clusters)
    cluster_common = {
        cid: {entity for entity, count in counter.items() if count >= 2}
        for cid, counter in cluster_entity_counts.items()
    }
    page_fingerprints = Counter(_fingerprint(text) for text in page_texts)
    paragraph_texts = [text for _, _, text, _ in (paragraph_records or [])]
    paragraph_fingerprints = Counter(_fingerprint(text) for text in paragraph_texts)

    page_rows: list[dict] = []
    paragraph_rows: list[dict] = []
    for i, page in enumerate(pages):
        ext = extracted_pages[i] if i < len(extracted_pages) else None
        cid = int(clusters[i]) if i < len(clusters) else 0
        common = cluster_common.get(cid, set())
        media_count = len(getattr(ext, "media_items", []) or []) if ext is not None else 0
        schema_count = len(getattr(ext, "schema_types", []) or []) if ext is not None else 0
        external_links = int(getattr(ext, "external_link_count", 0) or 0) if ext is not None else 0
        score, positives, negatives, snippets = _score_text(
            page_texts[i],
            duplicate_count=page_fingerprints.get(_fingerprint(page_texts[i]), 0),
            external_links=external_links,
            media_count=media_count,
            schema_count=schema_count,
            cluster_common=common,
        )
        page_rows.append({
            "url": page.url,
            "title": page.title,
            "section": page.section,
            "cluster": cid,
            "cluster_label": label_lookup.get(cid, f"cluster {cid}"),
            "information_gain_score": score,
            "positive_evidence": positives,
            "negative_reasons": negatives,
            "unique_facts": snippets,
            "recommendations": _recommendations(negatives),
            "entity_count": len(page_entity_sets[i]),
            "external_links": external_links,
            "media_count": media_count,
            "schema_count": schema_count,
        })

    for page_i, para_i, text, _ in paragraph_records or []:
        if page_i >= len(pages):
            continue
        page = pages[page_i]
        cid = int(clusters[page_i]) if page_i < len(clusters) else 0
        score, positives, negatives, snippets = _score_text(
            text,
            duplicate_count=paragraph_fingerprints.get(_fingerprint(text), 0),
            cluster_common=cluster_common.get(cid, set()),
        )
        if score >= 75 and positives or score < 55 or negatives:
            paragraph_rows.append({
                "url": page.url,
                "title": page.title,
                "section": page.section,
                "cluster": cid,
                "cluster_label": label_lookup.get(cid, f"cluster {cid}"),
                "paragraph_index": int(para_i),
                "paragraph_excerpt": text[:360],
                "information_gain_score": score,
                "positive_evidence": positives,
                "negative_reasons": negatives,
                "unique_facts": snippets[:3],
                "recommendations": _recommendations(negatives),
            })

    page_rows.sort(key=lambda r: (float(r.get("information_gain_score", 0.0)), r.get("url", "")))
    paragraph_rows.sort(key=lambda r: (float(r.get("information_gain_score", 0.0)), r.get("url", "")))
    by_cluster: dict[int, list[dict]] = defaultdict(list)
    for row in page_rows:
        by_cluster[int(row["cluster"])].append(row)
    clusters_payload = []
    for cid, rows in by_cluster.items():
        clusters_payload.append({
            "cluster": cid,
            "label": label_lookup.get(cid, f"cluster {cid}"),
            "pages": len(rows),
            "avg_score": round(sum(float(r["information_gain_score"]) for r in rows) / max(len(rows), 1), 2),
            "low_score_pages": sum(1 for r in rows if float(r["information_gain_score"]) < 55),
        })
    clusters_payload.sort(key=lambda r: r["avg_score"])
    summary = {
        "status": "ok",
        "model": "information_gain_v1",
        "pages": len(page_rows),
        "paragraphs": len(paragraph_rows),
        "avg_page_score": round(sum(float(r["information_gain_score"]) for r in page_rows) / max(len(page_rows), 1), 2),
        "low_score_pages": sum(1 for r in page_rows if float(r["information_gain_score"]) < 55),
        "high_score_pages": sum(1 for r in page_rows if float(r["information_gain_score"]) >= 75),
    }
    return {
        "summary": summary,
        "pages": page_rows[:top_n],
        "paragraphs": paragraph_rows[:top_n],
        "clusters": clusters_payload,
        "interpretation": {
            "information_gain_score": "Heuristic score rewarding original data, examples, citations, media, schema, and rare named entities while penalizing generic, duplicated, boilerplate, and evidence-free content.",
        },
    }
