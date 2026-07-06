"""Per-link internal anchor relevance scoring."""

from __future__ import annotations

import math
import re
import logging
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import numpy as np

from .extractor import ExtractedPage

LOG = logging.getLogger(__name__)
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9'_-]*", re.I)
_GENERIC = {
    "click here", "here", "read more", "learn more", "more", "details", "view",
    "view more", "this", "this page", "link", "website", "page", "go",
}


def _tokens(text: str) -> set[str]:
    return {m.group(0).lower() for m in _TOKEN_RE.finditer(text or "") if len(m.group(0)) > 1}


def _canonical(url: str) -> str:
    try:
        parsed = urlparse(url)
    except Exception:
        return str(url or "").rstrip("/")
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return f"{parsed.scheme.lower() or 'https'}://{host}{path}".rstrip("/")


def _directory(url: str) -> str:
    try:
        parts = [p for p in (urlparse(url).path or "/").split("/") if p]
    except Exception:
        parts = []
    return f"/{parts[0]}/" if parts else "/"


def _safe_int(value) -> int:
    try:
        if value is None or value == "":
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _phrase_candidates(target: ExtractedPage, keywords: list[str], entities: list[str]) -> list[str]:
    candidates = []
    candidates.extend(k for k in keywords if 3 <= len(k) <= 70)
    candidates.extend(e for e in entities if 3 <= len(e) <= 70)
    candidates.extend([target.h1, target.title])
    candidates.extend(h for h in (target.headings or []) if 3 <= len(h) <= 70)
    out = []
    seen = set()
    for value in candidates:
        clean = re.sub(r"\s+", " ", value or "").strip(" -:|")
        key = clean.lower()
        if clean and key not in seen and key not in _GENERIC:
            seen.add(key)
            out.append(clean[:80])
    return out


def _search_keywords(search_payload: dict | None) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for row in (search_payload or {}).get("top_pages") or []:
        url = row.get("matched_url") or row.get("url") or ""
        kw = row.get("top_keyword") or ""
        if url and kw:
            out[_canonical(url)].append(str(kw))
    for row in (search_payload or {}).get("organic_keywords") or []:
        url = row.get("matched_url") or row.get("url") or ""
        kw = row.get("keyword") or ""
        if url and kw and len(out[_canonical(url)]) < 12:
            out[_canonical(url)].append(str(kw))
    return out


def _entity_lookup(entities_payload: dict | None) -> dict[str, list[str]]:
    out = {}
    for row in (entities_payload or {}).get("per_page") or []:
        url = row.get("url") or ""
        if not url:
            continue
        out[_canonical(url)] = [str(e.get("entity") or "") for e in row.get("top_entities") or [] if e.get("entity")]
    return out


def _label(row: dict, target_anchor_counts: Counter[tuple[str, str]]) -> str:
    anchor = (row.get("anchor") or "").strip().lower()
    if row.get("is_empty"):
        return "empty"
    if row.get("is_image_only"):
        return "image_only"
    if anchor in _GENERIC or len(_tokens(anchor)) <= 1:
        return "vague"
    if row.get("score", 0) < 35:
        return "mismatched"
    if target_anchor_counts[(row.get("target_url"), anchor)] >= 5 and row.get("keyword_overlap", 0) >= 0.8:
        return "over_optimized"
    if target_anchor_counts[(row.get("target_url"), anchor)] >= 8:
        return "duplicated"
    if row.get("score", 0) >= 65:
        return "descriptive"
    return "acceptable"


def _chunked(items: list[dict], chunk_size: int) -> list[list[dict]]:
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def _score_row_chunk(
    rows: list[dict],
    *,
    target_features: dict[str, dict],
    index_by_url: dict[str, int],
    anchor_vectors: dict[str, np.ndarray],
    page_embeddings: np.ndarray | None,
    target_anchor_counts: Counter[tuple[str, str]],
) -> list[dict]:
    scored: list[dict] = []
    for row in rows:
        target_key = row["target_key"]
        features = target_features[target_key]
        anchor_tokens = _tokens(row["anchor"])
        context_tokens = _tokens(row["context"])
        target_tokens = features["target_tokens"]
        keyword_tokens = features["keyword_tokens"]
        entity_tokens = features["entity_tokens"]
        overlap = len(anchor_tokens & target_tokens) / max(1, len(anchor_tokens)) if anchor_tokens else 0.0
        context_overlap = len(context_tokens & target_tokens) / max(1, min(len(context_tokens), 24)) if context_tokens else 0.0
        keyword_overlap = len(anchor_tokens & keyword_tokens) / max(1, len(anchor_tokens)) if anchor_tokens and keyword_tokens else 0.0
        entity_overlap = len(anchor_tokens & entity_tokens) / max(1, len(anchor_tokens)) if anchor_tokens and entity_tokens else 0.0
        semantic = 0.0
        anchor_vec = anchor_vectors.get(row["anchor"])
        target_idx = index_by_url.get(row["target_url"])
        if anchor_vec is not None and page_embeddings is not None and target_idx is not None:
            semantic = max(0.0, min(1.0, (float(np.clip(anchor_vec @ page_embeddings[target_idx], -1.0, 1.0)) + 1.0) / 2.0))
        lexical_score = overlap * 100.0
        semantic_score = semantic * 100.0 if semantic else lexical_score
        score = (
            semantic_score * 0.35
            + lexical_score * 0.25
            + max(keyword_overlap, entity_overlap) * 100.0 * 0.20
            + context_overlap * 100.0 * 0.10
            + (0.0 if row["is_empty"] else 10.0)
        )
        if row["is_image_only"]:
            score -= 18.0
        out = dict(row)
        out.pop("target_key", None)
        out.update({
            "score": round(max(0.0, min(100.0, score)), 2),
            "semantic_score": round(semantic_score, 2),
            "lexical_overlap": round(overlap, 4),
            "keyword_overlap": round(keyword_overlap, 4),
            "entity_overlap": round(entity_overlap, 4),
            "context_overlap": round(context_overlap, 4),
            "suggested_anchor": features["suggested_anchor"],
        })
        out["label"] = _label(out, target_anchor_counts)
        out["recommended_action"] = (
            "Add visible descriptive anchor text or useful image alt text."
            if out["label"] in {"empty", "image_only"}
            else "Replace with a target-specific phrase from the suggested anchor."
            if out["label"] in {"vague", "mismatched"}
            else "Vary repeated anchors with natural alternatives from headings or entities."
            if out["label"] in {"over_optimized", "duplicated"}
            else "Keep this anchor."
        )
        scored.append(out)
    return scored


def build_anchor_relevance(
    extracted_pages: list[ExtractedPage],
    page_embeddings: np.ndarray | None = None,
    *,
    search_payload: dict | None = None,
    entities_payload: dict | None = None,
    embedder=None,
    semantic_anchor_cap: int = 3000,
    max_workers: int = 1,
    chunk_size: int = 50000,
) -> dict:
    if not extracted_pages:
        return {"summary": {"status": "no_pages", "total_internal_links": 0}, "links": []}

    target_by_canonical = {_canonical(p.url): p for p in extracted_pages}
    index_by_url = {p.url: i for i, p in enumerate(extracted_pages)}
    keywords_by_url = _search_keywords(search_payload)
    entities_by_url = _entity_lookup(entities_payload)
    raw_rows: list[dict] = []
    for source in extracted_pages:
        for link in source.link_audit_rows or []:
            if not link.get("is_internal"):
                continue
            target = target_by_canonical.get(_canonical(link.get("target_url") or ""))
            if target is None or target.url == source.url:
                continue
            target_key = _canonical(target.url)
            raw_rows.append({
                "source_url": source.url,
                "source_title": source.title,
                "source_directory": _directory(source.url),
                "target_url": target.url,
                "target_key": target_key,
                "target_title": target.title,
                "target_directory": _directory(target.url),
                "anchor": (link.get("anchor") or "").strip(),
                "context": link.get("context") or "",
                "is_empty": bool(link.get("is_empty")),
                "is_image_only": bool(link.get("is_image_only")),
                "has_text": bool(link.get("has_text")),
            })
    if not raw_rows:
        return {"summary": {"status": "no_links", "total_internal_links": 0}, "links": []}

    anchor_vectors: dict[str, np.ndarray] = {}
    if embedder is not None and page_embeddings is not None:
        unique = sorted({r["anchor"] for r in raw_rows if r["anchor"]})[:semantic_anchor_cap]
        if unique:
            embs = embedder.encode(unique, batch_size=256, show_progress=False)
            anchor_vectors = {anchor: vec for anchor, vec in zip(unique, embs)}

    target_anchor_counts = Counter((r["target_url"], r["anchor"].lower()) for r in raw_rows if r["anchor"])
    target_features: dict[str, dict] = {}
    for target_key, target in target_by_canonical.items():
        target_keywords = keywords_by_url.get(target_key, [])
        target_entities = entities_by_url.get(target_key, [])
        replacement_candidates = _phrase_candidates(target, target_keywords, target_entities)
        target_text = " ".join([target.title, target.h1, " ".join(target.headings or []), " ".join(target_keywords), " ".join(target_entities)])
        target_features[target_key] = {
            "target_tokens": _tokens(target_text),
            "keyword_tokens": _tokens(" ".join(target_keywords)),
            "entity_tokens": _tokens(" ".join(target_entities)),
            "suggested_anchor": replacement_candidates[0] if replacement_candidates else target.title,
        }

    workers = max(1, int(max_workers or 1))
    if workers > 1 and len(raw_rows) > chunk_size:
        chunks = _chunked(raw_rows, max(1, int(chunk_size)))
        LOG.info("  anchor relevance: scoring %d links in %d chunks with %d workers", len(raw_rows), len(chunks), workers)
        links = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for chunk_links in pool.map(
                lambda chunk: _score_row_chunk(
                    chunk,
                    target_features=target_features,
                    index_by_url=index_by_url,
                    anchor_vectors=anchor_vectors,
                    page_embeddings=page_embeddings,
                    target_anchor_counts=target_anchor_counts,
                ),
                chunks,
            ):
                links.extend(chunk_links)
    else:
        links = _score_row_chunk(
            raw_rows,
            target_features=target_features,
            index_by_url=index_by_url,
            anchor_vectors=anchor_vectors,
            page_embeddings=page_embeddings,
            target_anchor_counts=target_anchor_counts,
        )

    labels = Counter(row["label"] for row in links)

    def aggregate(key: str) -> list[dict]:
        buckets: dict[str, list[dict]] = defaultdict(list)
        for row in links:
            buckets[row.get(key) or "unknown"].append(row)
        out = []
        for name, rows in buckets.items():
            total = len(rows)
            descriptive = sum(1 for r in rows if r["label"] in {"descriptive", "acceptable"})
            out.append({
                key: name,
                "label": name,
                "links": total,
                "avg_score": round(sum(r["score"] for r in rows) / max(1, total), 2),
                "descriptive_rate": round(descriptive / max(1, total), 4),
                "weak_links": sum(1 for r in rows if r["label"] in {"empty", "image_only", "vague", "mismatched"}),
            })
        out.sort(key=lambda r: (r["weak_links"], r["links"]), reverse=True)
        return out

    weak_labels = {"empty", "image_only", "vague", "mismatched", "over_optimized", "duplicated"}
    weak_links = [r for r in links if r["label"] in weak_labels]
    weak_links.sort(key=lambda r: (r["score"], -target_anchor_counts[(r["target_url"], r["anchor"].lower())]))
    descriptive_count = sum(1 for r in links if r["label"] in {"descriptive", "acceptable"})
    return {
        "summary": {
            "status": "ok",
            "model": "anchor_relevance_v1",
            "total_internal_links": len(links),
            "avg_score": round(sum(r["score"] for r in links) / max(1, len(links)), 2),
            "descriptive_rate": round(descriptive_count / max(1, len(links)), 4),
            "weak_links": len(weak_links),
            "empty_links": labels.get("empty", 0),
            "image_only_links": labels.get("image_only", 0),
            "vague_links": labels.get("vague", 0),
            "mismatched_links": labels.get("mismatched", 0),
            "over_optimized_links": labels.get("over_optimized", 0),
            "duplicated_links": labels.get("duplicated", 0),
        },
        "links": links,
        "weak_links": weak_links[:500],
        "by_source_directory": aggregate("source_directory"),
        "by_target_directory": aggregate("target_directory"),
        "labels": [{"label": label, "count": count} for label, count in labels.most_common()],
        "interpretation": {
            "score": "Blend of anchor-target semantic similarity, exact target/title/heading overlap, keyword/entity overlap, and nearby source context overlap.",
            "labels": "Weak labels prioritize empty/image-only anchors, generic text, semantic mismatches, and repeated exact anchors.",
        },
    }
