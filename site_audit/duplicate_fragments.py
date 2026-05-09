"""Detect repeated strong fragments and harmful boilerplate."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any
from urllib.parse import urlparse

import numpy as np

from .analyzer import PageInfo
from .entities import extract_entities_from_text
from .paragraph_impact import _heading_for_paragraph, _match_page, _page_lookup, _to_int

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9'_-]*", re.I)
_LEGAL_RE = re.compile(r"\b(cookie|privacy policy|terms|copyright|all rights reserved|gdpr|legal)\b", re.I)
_CTA_RE = re.compile(r"\b(get started|start free trial|try for free|contact us|book a demo|request demo|learn more|subscribe|sign up)\b", re.I)
_NAV_RE = re.compile(r"\b(home|pricing|features|solutions|resources|login|menu|navigation|footer)\b", re.I)
_GENERIC_RE = re.compile(r"\b(learn more|read more|in today's digital world|it is important to|businesses need|whether you are)\b", re.I)
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "for", "from", "has", "have",
    "in", "is", "it", "of", "on", "or", "our", "that", "the", "this", "to", "with", "you", "your",
}


def _tokens(text: str) -> list[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(text or "")]


def _fingerprint(text: str) -> str:
    toks = [tok for tok in _tokens(text) if tok not in _STOPWORDS]
    return " ".join(toks)


def _signature(text: str) -> str:
    toks = [tok for tok in _tokens(text) if tok not in _STOPWORDS and len(tok) > 2]
    counts = Counter(toks)
    top = sorted(counts, key=lambda tok: (-counts[tok], tok))[:18]
    return " ".join(sorted(top))


def _url_template(url: str) -> str:
    path = urlparse(url).path or "/"
    parts = [p for p in path.split("/") if p]
    if not parts:
        return "/"
    normalized = []
    for part in parts[:4]:
        if re.search(r"\d", part):
            normalized.append(":num")
        elif len(part) > 24:
            normalized.append(":slug")
        else:
            normalized.append(part)
    return "/" + "/".join(normalized)


def _traffic_lookup(pages: list[PageInfo], search_payload: dict | None) -> dict[int, dict]:
    lookup = _page_lookup(pages)
    out: dict[int, dict] = defaultdict(lambda: {"traffic": 0, "keywords": 0, "top_keyword": ""})
    for row in (search_payload or {}).get("top_pages") or []:
        page_i = _match_page(row.get("matched_url") or row.get("url") or "", lookup)
        if page_i is None:
            continue
        traffic = _to_int(row.get("traffic"))
        if traffic >= out[page_i]["traffic"]:
            out[page_i]["traffic"] = traffic
            out[page_i]["top_keyword"] = row.get("top_keyword") or out[page_i]["top_keyword"]
        out[page_i]["keywords"] = max(out[page_i]["keywords"], _to_int(row.get("keywords")))
    return out


def _keyword_paragraph_lookup(keyword_attribution: dict | None) -> dict[tuple[str, int], dict]:
    out: dict[tuple[str, int], dict] = defaultdict(lambda: {"traffic": 0, "keywords": []})
    for row in (keyword_attribution or {}).get("keywords") or []:
        para_i = row.get("best_paragraph_index")
        if para_i is None:
            continue
        try:
            idx = int(para_i)
        except (TypeError, ValueError):
            continue
        key = (str(row.get("url") or ""), idx)
        out[key]["traffic"] += _to_int(row.get("traffic"))
        if len(out[key]["keywords"]) < 8 and row.get("keyword"):
            out[key]["keywords"].append({
                "keyword": row.get("keyword"),
                "traffic": _to_int(row.get("traffic")),
                "position": _to_int(row.get("position")),
            })
    return out


def _context_kind(text: str, heading: str, count: int) -> str:
    haystack = f"{heading} {text}"
    if _LEGAL_RE.search(haystack):
        return "legal_template"
    if _CTA_RE.search(haystack):
        return "cta_template"
    if count >= 12 and _NAV_RE.search(haystack):
        return "nav_footer_template"
    return "main_content"


def _specificity(text: str) -> float:
    toks = _tokens(text)
    if not toks:
        return 0.0
    unique_ratio = len(set(toks)) / len(toks)
    entities = len(set(extract_entities_from_text(text)))
    length_component = min(1.0, len(toks) / 70)
    generic_penalty = 0.25 if _GENERIC_RE.search(text) else 0.0
    return max(0.0, min(1.0, unique_ratio * 0.45 + min(0.35, entities * 0.08) + length_component * 0.25 - generic_penalty))


def _cohesion(records: list[dict]) -> float:
    embs = [np.asarray(r["embedding"], dtype=np.float32) for r in records if r.get("embedding") is not None]
    if len(embs) < 2:
        return 1.0
    arr = np.stack(embs[:30])
    sims = arr @ arr.T
    tri = sims[np.triu_indices(len(arr), k=1)]
    return float(np.mean(tri)) if len(tri) else 1.0


def _classification(kind: str, specificity: float, count: int, traffic_urls: int, attributed_traffic: int) -> tuple[str, str]:
    if kind in {"legal_template", "nav_footer_template"}:
        if attributed_traffic:
            return kind, "keep_template_but_exclude_from_main_content_scoring"
        return "harmful_boilerplate", "prune_or_suppress_template"
    if kind == "cta_template":
        if traffic_urls >= 2:
            return "reusable_template", "keep_as_template_pattern"
        return "low_value_boilerplate", "simplify_or_remove"
    if attributed_traffic > 0 and specificity >= 0.55:
        return "strong_reusable_pattern", "reuse_pattern_with_page_specific_context"
    if count >= 5 and specificity < 0.45:
        return "harmful_boilerplate", "rewrite_or_remove_repeated_block"
    if specificity >= 0.5:
        return "main_content_duplicate", "differentiate_examples_or_merge_repeated_sections"
    return "low_value_boilerplate", "rewrite_or_remove_repeated_block"


def build_duplicate_fragments(
    pages: list[PageInfo],
    extracted_pages: list,
    paragraph_records: list[tuple[int, int, str, Any]] | None,
    *,
    search_payload: dict | None = None,
    keyword_attribution: dict | None = None,
    min_repetitions: int = 2,
    top_n: int = 700,
) -> dict:
    if not paragraph_records:
        return {"summary": {"status": "no_paragraphs", "groups": 0}, "groups": [], "pattern_library": [], "noise": []}
    page_traffic = _traffic_lookup(pages, search_payload)
    keyword_lookup = _keyword_paragraph_lookup(keyword_attribution)
    records: list[dict] = []
    for page_i, para_i, text, emb in paragraph_records:
        if page_i >= len(pages) or page_i >= len(extracted_pages):
            continue
        clean = " ".join(str(text or "").split())
        if len(_tokens(clean)) < 8:
            continue
        page = pages[int(page_i)]
        heading = _heading_for_paragraph(extracted_pages[int(page_i)], int(para_i))
        kw = keyword_lookup.get((page.url, int(para_i)), {"traffic": 0, "keywords": []})
        records.append({
            "record_id": len(records),
            "page_index": int(page_i),
            "paragraph_index": int(para_i),
            "url": page.url,
            "title": page.title,
            "section": page.section,
            "url_template": _url_template(page.url),
            "heading": heading,
            "text": clean,
            "embedding": np.asarray(emb, dtype=np.float32),
            "fingerprint": _fingerprint(clean),
            "signature": _signature(clean),
            "page_traffic": _to_int(page_traffic.get(int(page_i), {}).get("traffic")),
            "top_keyword": page_traffic.get(int(page_i), {}).get("top_keyword", ""),
            "attributed_traffic": _to_int(kw.get("traffic")),
            "keywords": kw.get("keywords") or [],
        })
    if not records:
        return {"summary": {"status": "no_paragraphs", "groups": 0}, "groups": [], "pattern_library": [], "noise": []}

    candidate_sets: set[frozenset[int]] = set()
    for key_name in ("fingerprint", "signature"):
        grouped: dict[str, list[int]] = defaultdict(list)
        for rec in records:
            key = rec.get(key_name) or ""
            if len(key) >= 12:
                grouped[key].append(int(rec["record_id"]))
        for ids in grouped.values():
            if len(ids) >= min_repetitions:
                candidate_sets.add(frozenset(ids))

    groups: list[dict] = []
    for group_i, ids in enumerate(candidate_sets):
        recs = [records[i] for i in sorted(ids)]
        if len({r["url"] for r in recs}) < 2:
            continue
        count = len(recs)
        sample = max(recs, key=lambda r: (r["attributed_traffic"], r["page_traffic"], len(r["text"])))
        kind = _context_kind(sample["text"], sample["heading"], count)
        spec = _specificity(sample["text"])
        traffic_urls = len({r["url"] for r in recs if _to_int(r.get("page_traffic")) > 0 or _to_int(r.get("attributed_traffic")) > 0})
        attributed_traffic = sum(_to_int(r.get("attributed_traffic")) for r in recs)
        page_traffic_sum = sum(_to_int(r.get("page_traffic")) for r in recs)
        classification, handling = _classification(kind, spec, count, traffic_urls, attributed_traffic)
        examples = []
        for r in sorted(recs, key=lambda row: (_to_int(row.get("attributed_traffic")), _to_int(row.get("page_traffic"))), reverse=True)[:12]:
            examples.append({
                "url": r["url"],
                "title": r["title"],
                "section": r["section"],
                "url_template": r["url_template"],
                "heading": r["heading"],
                "paragraph_index": r["paragraph_index"],
                "page_traffic": r["page_traffic"],
                "attributed_traffic": r["attributed_traffic"],
                "top_keyword": r["top_keyword"],
                "keywords": r["keywords"],
            })
        groups.append({
            "group_id": f"frag_{group_i}",
            "classification": classification,
            "context_kind": kind,
            "recommended_handling": handling,
            "count": count,
            "affected_urls": len({r["url"] for r in recs}),
            "traffic_bearing_urls": traffic_urls,
            "page_traffic_sum": page_traffic_sum,
            "attributed_traffic": attributed_traffic,
            "specificity_score": round(spec, 3),
            "semantic_cohesion": round(_cohesion(recs), 4),
            "url_templates": [{"template": t, "count": c} for t, c in Counter(r["url_template"] for r in recs).most_common(8)],
            "headings": [{"heading": h, "count": c} for h, c in Counter(r["heading"] or "(no heading)" for r in recs).most_common(8)],
            "sample_text": sample["text"][:520],
            "examples": examples,
        })

    groups.sort(key=lambda r: (
        r["classification"] not in {"strong_reusable_pattern", "main_content_duplicate"},
        -(r["attributed_traffic"] + r["page_traffic_sum"]),
        -r["count"],
    ))
    pattern_library = [
        row for row in groups
        if row["classification"] in {"strong_reusable_pattern", "reusable_template"}
    ]
    noise = [
        row for row in groups
        if row["classification"] in {"harmful_boilerplate", "low_value_boilerplate", "legal_template", "nav_footer_template"}
    ]
    summary = {
        "status": "ok",
        "model": "duplicate_fragments_v1",
        "paragraphs": len(records),
        "groups": len(groups),
        "strong_patterns": len(pattern_library),
        "harmful_boilerplate": sum(1 for row in groups if row["classification"] == "harmful_boilerplate"),
        "main_content_duplicates": sum(1 for row in groups if row["classification"] == "main_content_duplicate"),
        "affected_urls": len({ex["url"] for row in groups for ex in row["examples"]}),
        "traffic_in_repeated_fragments": sum(_to_int(row.get("attributed_traffic")) for row in groups),
    }
    return {
        "summary": summary,
        "groups": groups[:top_n],
        "pattern_library": pattern_library[:top_n],
        "noise": noise[:top_n],
        "interpretation": {
            "strong_reusable_pattern": "Repeated main-content fragment with traffic or keyword attribution and enough semantic specificity to reuse as a pattern.",
            "harmful_boilerplate": "Repeated low-specificity or template-like text that can dilute page uniqueness.",
        },
    }
