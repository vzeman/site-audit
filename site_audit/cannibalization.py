"""Content cannibalization by keyword intent and paragraph overlap."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from itertools import combinations
from typing import Any

import numpy as np

from .analyzer import PageInfo
from .paragraph_impact import _match_page, _normalize_url, _page_lookup, _to_int

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9'_-]*", re.I)
_LANG_OR_GEO_RE = re.compile(
    r"(^|/)(en|de|fr|es|it|pt|nl|pl|sk|cz|cs|uk|us|ca|au|in|br|mx)(/|$)|"
    r"\b(united states|usa|uk|canada|australia|india|germany|france|spain|italy|brazil|mexico)\b",
    re.I,
)
_PRODUCT_VARIANT_RE = re.compile(r"/(integrations?|features?|templates?|solutions?|industries?|use-cases?)/|\b(for|integration|template|feature)\b", re.I)
_STOPWORDS = {
    "a", "about", "and", "are", "as", "best", "by", "can", "for", "from", "guide", "how",
    "in", "is", "of", "on", "or", "software", "the", "to", "tool", "tools", "top", "vs",
    "what", "with", "your", "germany", "usa", "us", "uk", "canada", "india", "australia",
    "france", "spain", "italy", "brazil", "mexico",
}


def _tokens(text: str) -> list[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(text or "")]


def _intent_key(keyword: str) -> str:
    toks = [tok for tok in _tokens(keyword) if tok not in _STOPWORDS and len(tok) > 1]
    if not toks:
        toks = [tok for tok in _tokens(keyword) if len(tok) > 1]
    return " ".join(toks[:4]) or keyword.lower().strip()


def _jaccard(a: str, b: str) -> float:
    ta = set(_tokens(a))
    tb = set(_tokens(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _page_search_context(pages: list[PageInfo], search_payload: dict | None) -> dict[str, dict]:
    lookup = _page_lookup(pages)
    out: dict[str, dict] = defaultdict(lambda: {"traffic": 0, "keywords": 0, "top_keyword": ""})
    for row in (search_payload or {}).get("top_pages") or []:
        page_i = _match_page(row.get("matched_url") or row.get("url") or "", lookup)
        if page_i is None:
            continue
        url = pages[page_i].url
        traffic = _to_int(row.get("traffic"))
        if traffic >= out[url]["traffic"]:
            out[url]["traffic"] = traffic
            out[url]["top_keyword"] = row.get("top_keyword") or out[url]["top_keyword"]
        out[url]["keywords"] = max(out[url]["keywords"], _to_int(row.get("keywords")))
    return out


def _keyword_rows(pages: list[PageInfo], keyword_attribution: dict | None, search_payload: dict | None) -> list[dict]:
    lookup = _page_lookup(pages)
    rows: list[dict] = []
    if (keyword_attribution or {}).get("keywords"):
        for row in (keyword_attribution or {}).get("keywords") or []:
            url = row.get("url") or ""
            page_i = _match_page(url, lookup)
            if page_i is None:
                continue
            keyword = str(row.get("keyword") or "").strip()
            if not keyword:
                continue
            rows.append({
                "keyword": keyword,
                "url": pages[page_i].url,
                "page_index": page_i,
                "title": pages[page_i].title,
                "traffic": _to_int(row.get("traffic")),
                "volume": _to_int(row.get("volume")),
                "position": _to_int(row.get("position")),
                "paragraph_index": row.get("best_paragraph_index"),
                "paragraph_excerpt": row.get("best_paragraph_excerpt") or "",
                "heading": row.get("best_paragraph_heading") or row.get("best_heading") or "",
                "source": "keyword_attribution",
            })
    else:
        for row in (search_payload or {}).get("organic_keywords") or []:
            page_i = _match_page(row.get("matched_url") or row.get("url") or "", lookup)
            if page_i is None:
                continue
            keyword = str(row.get("keyword") or "").strip()
            if not keyword:
                continue
            rows.append({
                "keyword": keyword,
                "url": pages[page_i].url,
                "page_index": page_i,
                "title": pages[page_i].title,
                "traffic": _to_int(row.get("traffic")),
                "volume": _to_int(row.get("volume")),
                "position": _to_int(row.get("position")),
                "paragraph_index": None,
                "paragraph_excerpt": "",
                "heading": "",
                "source": "organic_keyword",
            })
    rows.sort(key=lambda r: (_to_int(r.get("traffic")), _to_int(r.get("volume"))), reverse=True)
    return rows


def _link_context(linkgraph: dict | None) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in (linkgraph or {}).get("page_link_counts") or []:
        url = row.get("url") or ""
        if url:
            out[_normalize_url(url)] = row
    return out


def _indexability_context(indexability: dict | None) -> set[str]:
    return {
        _normalize_url(row.get("url") or "")
        for row in (indexability or {}).get("noindex_pages") or []
        if row.get("url")
    }


def _outlink_set(outlinks_map: dict[str, list[tuple[str, str]]] | None) -> set[tuple[str, str]]:
    edges: set[tuple[str, str]] = set()
    for src, outs in (outlinks_map or {}).items():
        src_norm = _normalize_url(src)
        for target, _anchor in outs or []:
            edges.add((src_norm, _normalize_url(target)))
    return edges


def _winner_score(url: str, page_rows: list[dict], page_ctx: dict, link_ctx: dict, noindex_urls: set[str]) -> float:
    traffic = sum(_to_int(row.get("traffic")) for row in page_rows)
    best_pos = min([_to_int(row.get("position")) for row in page_rows if _to_int(row.get("position")) > 0] or [100])
    link = link_ctx.get(_normalize_url(url), {})
    score = traffic * 2.0
    score += max(0, 100 - best_pos) * 4.0
    score += _to_int(link.get("in_degree")) * 2.0
    score += float(link.get("pagerank", 0.0) or 0.0) * 1000
    score += _to_int(page_ctx.get("traffic")) * 0.6
    if _normalize_url(url) in noindex_urls:
        score -= 10000
    return score


def _classify(
    *,
    winner_url: str,
    competitor_url: str,
    winner_title: str,
    competitor_title: str,
    similarity: float,
    linked: bool,
    link_ctx: dict,
) -> tuple[str, str, str]:
    pair_text = f"{winner_url} {competitor_url} {winner_title} {competitor_title}"
    if _LANG_OR_GEO_RE.search(pair_text):
        return (
            "localized_variant",
            "filter_localized",
            "URLs or titles look locale/geography-specific; verify hreflang/canonical intent before merging.",
        )
    if _PRODUCT_VARIANT_RE.search(pair_text):
        return (
            "product_variant",
            "differentiate",
            "URLs or titles look product/feature-specific; keep only if each page has distinct intent and internal links.",
        )
    if linked and similarity < 0.9:
        return (
            "healthy_hub_spoke",
            "interlink",
            "Pages are similar but already linked as a hub/spoke or supporting path; strengthen anchor specificity.",
        )
    if similarity >= 0.88 or _jaccard(winner_title, competitor_title) >= 0.62:
        return (
            "duplicate_competing_page",
            "merge_or_canonical",
            "Pages are highly similar for the same intent; consolidate content or canonicalize the weaker URL.",
        )
    winner_links = link_ctx.get(_normalize_url(winner_url), {})
    competitor_links = link_ctx.get(_normalize_url(competitor_url), {})
    if _to_int(winner_links.get("in_degree")) >= _to_int(competitor_links.get("in_degree")) + 3:
        return (
            "consolidation_candidate",
            "retarget_or_merge",
            "Winner has stronger internal authority; move overlapping intent blocks there or retarget the weaker page.",
        )
    return (
        "consolidation_candidate",
        "differentiate_or_merge",
        "Multiple URLs target the same intent without a clear variant reason; choose a primary URL and differentiate the rest.",
    )


def _paragraph_map(paragraph_records: list[tuple[int, int, str, Any]] | None) -> dict[tuple[int, int], tuple[str, np.ndarray]]:
    out: dict[tuple[int, int], tuple[str, np.ndarray]] = {}
    for page_i, para_i, text, emb in paragraph_records or []:
        out[(int(page_i), int(para_i))] = (str(text or ""), np.asarray(emb, dtype=np.float32))
    return out


def _cluster_intents(keyword_rows: list[dict]) -> dict[str, dict]:
    groups: dict[str, dict] = {}
    for row in keyword_rows:
        key = _intent_key(row["keyword"])
        group = groups.setdefault(key, {
            "intent_key": key,
            "label": row["keyword"],
            "keywords": Counter(),
            "rows": [],
            "traffic": 0,
        })
        weight = max(1, _to_int(row.get("traffic")), _to_int(row.get("volume")) // 20)
        group["keywords"][row["keyword"]] += weight
        group["rows"].append(row)
        group["traffic"] += _to_int(row.get("traffic"))
        group["label"] = group["keywords"].most_common(1)[0][0]
    return groups


def build_cannibalization(
    pages: list[PageInfo],
    page_embeddings: np.ndarray,
    extracted_pages: list | None = None,
    paragraph_records: list[tuple[int, int, str, Any]] | None = None,
    *,
    keyword_attribution: dict | None = None,
    search_payload: dict | None = None,
    linkgraph: dict | None = None,
    indexability: dict | None = None,
    outlinks_map: dict[str, list[tuple[str, str]]] | None = None,
    top_n: int = 500,
) -> dict:
    if not pages:
        return {"summary": {"status": "no_pages", "pages": 0}, "intents": [], "matrix": [], "page_conflicts": [], "paragraph_overlaps": []}
    rows = _keyword_rows(pages, keyword_attribution, search_payload)
    if not rows:
        return {"summary": {"status": "no_keyword_data", "pages": len(pages)}, "intents": [], "matrix": [], "page_conflicts": [], "paragraph_overlaps": []}
    link_ctx = _link_context(linkgraph)
    noindex_urls = _indexability_context(indexability)
    edges = _outlink_set(outlinks_map)
    search_ctx = _page_search_context(pages, search_payload)
    pmap = _paragraph_map(paragraph_records)
    intents = _cluster_intents(rows)
    page_conflicts: list[dict] = []
    matrix: list[dict] = []
    paragraph_overlaps: list[dict] = []
    intent_payload: list[dict] = []

    for intent_key, group in intents.items():
        by_url: dict[str, list[dict]] = defaultdict(list)
        for row in group["rows"]:
            by_url[row["url"]].append(row)
        if len(by_url) < 2:
            continue
        winner_url = max(
            by_url,
            key=lambda url: _winner_score(url, by_url[url], search_ctx.get(url, {}), link_ctx, noindex_urls),
        )
        winner_page = next((p for p in pages if p.url == winner_url), None)
        winner_i = pages.index(winner_page) if winner_page in pages else None
        winner_rows = by_url[winner_url]
        competitors = []
        intent_classes: Counter[str] = Counter()
        intent_action = ""
        traffic_at_risk = 0

        for url, page_rows in by_url.items():
            page = next((p for p in pages if p.url == url), None)
            if page is None:
                continue
            page_i = pages.index(page)
            similarity = 0.0
            if winner_i is not None and page_i != winner_i and page_i < len(page_embeddings) and winner_i < len(page_embeddings):
                similarity = float(np.clip(page_embeddings[winner_i] @ page_embeddings[page_i], -1.0, 1.0))
            linked = (
                (_normalize_url(winner_url), _normalize_url(url)) in edges
                or (_normalize_url(url), _normalize_url(winner_url)) in edges
            )
            if url == winner_url:
                classification = "winner"
                action = "keep_primary"
                reason = "Preferred URL for this intent based on traffic, position, and internal authority."
            else:
                classification, action, reason = _classify(
                    winner_url=winner_url,
                    competitor_url=url,
                    winner_title=winner_page.title if winner_page else winner_url,
                    competitor_title=page.title,
                    similarity=similarity,
                    linked=linked,
                    link_ctx=link_ctx,
                )
                intent_classes[classification] += 1
                if classification not in {"localized_variant", "product_variant", "healthy_hub_spoke"}:
                    traffic_at_risk += sum(_to_int(r.get("traffic")) for r in page_rows)
                if not intent_action and action != "interlink":
                    intent_action = action
                competitors.append({
                    "url": url,
                    "title": page.title,
                    "similarity_to_winner": round(similarity, 4),
                    "traffic": sum(_to_int(r.get("traffic")) for r in page_rows),
                    "keywords": len({r["keyword"] for r in page_rows}),
                    "best_position": min([_to_int(r.get("position")) for r in page_rows if _to_int(r.get("position")) > 0] or [0]),
                    "classification": classification,
                    "recommended_action": action,
                    "reason": reason,
                })
            matrix.append({
                "intent_key": intent_key,
                "intent_label": group["label"],
                "url": url,
                "title": page.title,
                "role": "winner" if url == winner_url else "competitor",
                "preferred_winner_url": winner_url,
                "classification": classification,
                "recommended_action": action,
                "traffic": sum(_to_int(r.get("traffic")) for r in page_rows),
                "keywords": len({r["keyword"] for r in page_rows}),
                "best_position": min([_to_int(r.get("position")) for r in page_rows if _to_int(r.get("position")) > 0] or [0]),
                "similarity_to_winner": round(similarity, 4),
                "in_degree": _to_int((link_ctx.get(_normalize_url(url)) or {}).get("in_degree")),
                "click_depth": (link_ctx.get(_normalize_url(url)) or {}).get("click_depth"),
                "top_keywords": [
                    {"keyword": kw, "weight": int(weight)}
                    for kw, weight in Counter(r["keyword"] for r in page_rows).most_common(5)
                ],
            })

        competitors.sort(key=lambda r: (r["classification"] in {"localized_variant", "product_variant", "healthy_hub_spoke"}, -r["traffic"]))
        classification = intent_classes.most_common(1)[0][0] if intent_classes else "competing_pages"
        recommendation = intent_action or ("interlink" if classification == "healthy_hub_spoke" else "differentiate_or_merge")
        intent = {
            "intent_key": intent_key,
            "label": group["label"],
            "keywords": [{"keyword": kw, "weight": int(weight)} for kw, weight in group["keywords"].most_common(8)],
            "url_count": len(by_url),
            "preferred_winner_url": winner_url,
            "preferred_winner_title": winner_page.title if winner_page else winner_url,
            "classification": classification,
            "recommended_action": recommendation,
            "traffic": int(group["traffic"]),
            "traffic_at_risk": int(traffic_at_risk),
            "competitors": competitors[:12],
            "reason": competitors[0]["reason"] if competitors else "Multiple URLs rank for the same intent.",
        }
        intent_payload.append(intent)
        if competitors:
            page_conflicts.append(intent)

        para_candidates: list[dict] = []
        for url, page_rows in by_url.items():
            page = next((p for p in pages if p.url == url), None)
            if page is None:
                continue
            page_i = pages.index(page)
            by_para: dict[int, dict] = {}
            for row in page_rows:
                para_i = row.get("paragraph_index")
                if para_i is None:
                    continue
                try:
                    para_i_int = int(para_i)
                except (TypeError, ValueError):
                    continue
                text_emb = pmap.get((page_i, para_i_int))
                if text_emb is None:
                    continue
                text, emb = text_emb
                payload = by_para.setdefault(para_i_int, {
                    "url": url,
                    "title": page.title,
                    "page_index": page_i,
                    "paragraph_index": para_i_int,
                    "text": text,
                    "embedding": emb,
                    "keywords": Counter(),
                    "traffic": 0,
                    "heading": row.get("heading") or "",
                })
                payload["keywords"][row["keyword"]] += max(1, _to_int(row.get("traffic")) or 1)
                payload["traffic"] += _to_int(row.get("traffic"))
            para_candidates.extend(by_para.values())
        for a, b in combinations(para_candidates, 2):
            if a["url"] == b["url"]:
                continue
            sim = float(np.clip(a["embedding"] @ b["embedding"], -1.0, 1.0))
            lexical = _jaccard(a["text"], b["text"])
            if sim < 0.8 and lexical < 0.45:
                continue
            winner = winner_url if winner_url in {a["url"], b["url"]} else (a["url"] if a["traffic"] >= b["traffic"] else b["url"])
            paragraph_overlaps.append({
                "intent_key": intent_key,
                "intent_label": group["label"],
                "preferred_winner_url": winner,
                "url_a": a["url"],
                "title_a": a["title"],
                "paragraph_index_a": a["paragraph_index"],
                "heading_a": a.get("heading", ""),
                "excerpt_a": a["text"][:260],
                "traffic_a": int(a["traffic"]),
                "url_b": b["url"],
                "title_b": b["title"],
                "paragraph_index_b": b["paragraph_index"],
                "heading_b": b.get("heading", ""),
                "excerpt_b": b["text"][:260],
                "traffic_b": int(b["traffic"]),
                "paragraph_similarity": round(sim, 4),
                "lexical_overlap": round(lexical, 4),
                "shared_keywords": [
                    kw for kw in (set(a["keywords"]) & set(b["keywords"]))
                ][:8],
                "recommended_action": "move_to_winner_or_rewrite",
                "reason": "Paragraphs on different URLs support the same intent with high semantic or lexical overlap.",
            })

    page_conflicts.sort(key=lambda r: (int(r.get("traffic_at_risk", 0)), int(r.get("traffic", 0))), reverse=True)
    paragraph_overlaps.sort(key=lambda r: (float(r.get("paragraph_similarity", 0.0)), int(r.get("traffic_a", 0)) + int(r.get("traffic_b", 0))), reverse=True)
    matrix.sort(key=lambda r: (r["intent_label"], r["role"] != "winner", -_to_int(r.get("traffic"))))
    intent_payload.sort(key=lambda r: (int(r.get("traffic_at_risk", 0)), int(r.get("traffic", 0))), reverse=True)

    summary = {
        "status": "ok",
        "model": "cannibalization_v1",
        "keyword_rows": len(rows),
        "intent_groups": len(intent_payload),
        "page_conflicts": len(page_conflicts),
        "paragraph_conflicts": len(paragraph_overlaps),
        "localized_or_variant_groups": sum(1 for r in page_conflicts if r.get("classification") in {"localized_variant", "product_variant"}),
        "traffic_at_risk": sum(int(r.get("traffic_at_risk", 0)) for r in page_conflicts),
    }
    return {
        "summary": summary,
        "intents": intent_payload[:top_n],
        "matrix": matrix[:top_n * 3],
        "page_conflicts": page_conflicts[:top_n],
        "paragraph_overlaps": paragraph_overlaps[:top_n],
        "interpretation": {
            "classification": "localized_variant and product_variant are filter buckets; duplicate_competing_page and consolidation_candidate are actionable conflicts.",
            "preferred_winner_url": "Winner chosen from keyword traffic, ranking position, internal authority, indexability, and page traffic.",
        },
    }
