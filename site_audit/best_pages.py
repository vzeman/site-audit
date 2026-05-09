"""Best-page reverse engineering payloads.

This module turns already-collected audit signals into concise page explainers.
It deliberately reports likely supporting factors as observed correlations, not
confirmed ranking causes.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

from .analyzer import PageInfo


CAUSATION_NOTE = (
    "These findings combine observed search, content, freshness, schema, and internal-link signals. "
    "They are correlations in this audit data, not proof that any single element directly caused rankings."
)


def _safe_int(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(float(str(value).replace(",", "")))
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return default


def _url_keys(url: Any) -> set[str]:
    raw = str(url or "").strip()
    if not raw:
        return set()
    keys = {raw, raw.rstrip("/")}
    try:
        parts = urlsplit(raw)
    except ValueError:
        return {k for k in keys if k}
    if not parts.netloc:
        return {k for k in keys if k}
    scheme = (parts.scheme or "https").lower()
    netloc = parts.netloc.lower()
    path = parts.path or "/"
    path_trimmed = path.rstrip("/") or "/"
    netlocs = {netloc, netloc[4:] if netloc.startswith("www.") else f"www.{netloc}"}
    schemes = {scheme}
    if scheme in {"http", "https"}:
        schemes.add("https" if scheme == "http" else "http")
    for candidate_scheme in schemes:
        for candidate_netloc in netlocs:
            for candidate_path in {path, path_trimmed}:
                normalized = urlunsplit((candidate_scheme, candidate_netloc, candidate_path, "", ""))
                keys.add(normalized)
                keys.add(normalized.rstrip("/"))
    return {k for k in keys if k}


def _store_lookup(out: dict[str, dict], url: Any, row: dict, score_key: str = "") -> None:
    for key in _url_keys(url):
        current = out.get(key)
        if current is None:
            out[key] = row
        elif score_key and _safe_float(row.get(score_key)) > _safe_float(current.get(score_key)):
            out[key] = row


def _lookup_from_rows(rows: Iterable[dict] | None, fields: tuple[str, ...] = ("url",), score_key: str = "") -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        for field in fields:
            if row.get(field):
                _store_lookup(out, row.get(field), row, score_key=score_key)
    return out


def _lookup(lookup: Mapping[str, dict], *urls: Any) -> dict:
    for url in urls:
        for key in _url_keys(url):
            row = lookup.get(key)
            if row is not None:
                return row
    return {}


def _page_to_dict(page: PageInfo | Mapping[str, Any]) -> dict:
    if isinstance(page, Mapping):
        return dict(page)
    return {
        "url": page.url,
        "title": page.title,
        "section": page.section,
        "word_count": page.word_count,
        "description": page.description,
    }


def _top_pages(search_payload: Mapping[str, Any] | None, pages: Sequence[PageInfo | Mapping[str, Any]], limit: int) -> list[dict]:
    rows = [dict(r) for r in ((search_payload or {}).get("top_pages") or []) if isinstance(r, dict)]
    if rows:
        rows.sort(key=lambda r: (_safe_int(r.get("traffic")), _safe_int(r.get("keywords"))), reverse=True)
        return rows[:limit]

    fallback = []
    for page in pages:
        p = _page_to_dict(page)
        fallback.append({
            "url": p.get("url") or "",
            "matched_url": p.get("url") or "",
            "matched": True,
            "title": p.get("title") or "",
            "section": p.get("section") or "",
            "traffic": 0,
            "keywords": 0,
            "top_keyword": "",
        })
    return fallback[:limit]


def _keyword_lookup(search_payload: Mapping[str, Any] | None) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for row in ((search_payload or {}).get("organic_keywords") or []):
        if not isinstance(row, dict):
            continue
        url = row.get("matched_url") or row.get("url") or ""
        if not url:
            continue
        for key in _url_keys(url):
            out[key].append(row)
    for rows in out.values():
        rows.sort(key=lambda r: (_safe_int(r.get("traffic")), -_safe_int(r.get("position"))), reverse=True)
    return out


def _keywords_for_url(keyword_lookup: Mapping[str, list[dict]], *urls: Any) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    rows: list[dict] = []
    for url in urls:
        for key in _url_keys(url):
            for row in keyword_lookup.get(key, []):
                ident = (str(row.get("keyword") or ""), str(row.get("matched_url") or row.get("url") or ""))
                if ident in seen:
                    continue
                seen.add(ident)
                rows.append(row)
    rows.sort(key=lambda r: (_safe_int(r.get("traffic")), -_safe_int(r.get("position"))), reverse=True)
    return rows


def _group_by_url(rows: Iterable[dict] | None, fields: tuple[str, ...] = ("url",)) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        for field in fields:
            if row.get(field):
                for key in _url_keys(row.get(field)):
                    out[key].append(row)
    return out


def _rows_for_url(grouped: Mapping[str, list[dict]], *urls: Any) -> list[dict]:
    seen: set[int] = set()
    rows: list[dict] = []
    for url in urls:
        for key in _url_keys(url):
            for row in grouped.get(key, []):
                ident = id(row)
                if ident in seen:
                    continue
                seen.add(ident)
                rows.append(row)
    return rows


def _cluster_key(row: Mapping[str, Any]) -> str:
    value = row.get("cluster_label") or row.get("cluster") or row.get("section") or row.get("directory") or "unclustered"
    return str(value) if value not in (None, "") else "unclustered"


def _add_signal(
    target: list[dict],
    category: str,
    label: str,
    evidence: str,
    *,
    value: Any = None,
    source: str = "",
    confidence: float = 0.6,
) -> None:
    target.append({
        "category": category,
        "label": label,
        "evidence": evidence,
        "value": value,
        "source": source,
        "confidence": round(max(0.0, min(1.0, confidence)), 2),
    })


def _page_patterns(template_patterns: Mapping[str, Any] | None, *urls: Any) -> tuple[list[dict], list[dict]]:
    matching_patterns: list[dict] = []
    matching_recommendations: list[dict] = []
    for pattern in ((template_patterns or {}).get("patterns") or []):
        if not isinstance(pattern, dict):
            continue
        samples = pattern.get("sample_urls") or []
        if any(_lookup_from_rows(samples, ("url",)).get(key) for url in urls for key in _url_keys(url)):
            matching_patterns.append(pattern)
    for rec in ((template_patterns or {}).get("recommendations") or []):
        if not isinstance(rec, dict):
            continue
        if any(key in _url_keys(rec.get("url")) for url in urls for key in _url_keys(url)):
            matching_recommendations.append(rec)
    return matching_patterns[:8], matching_recommendations[:8]


def _performance_score(
    *,
    traffic: int,
    keywords: int,
    keyword_rows: list[dict],
    link: Mapping[str, Any],
    authority: Mapping[str, Any],
    structured: Mapping[str, Any],
    freshness: Mapping[str, Any],
    entity: Mapping[str, Any],
    information_gain: Mapping[str, Any],
    paragraphs: list[dict],
) -> float:
    top3 = sum(1 for r in keyword_rows if 0 < _safe_int(r.get("position")) <= 3)
    top10 = sum(1 for r in keyword_rows if 0 < _safe_int(r.get("position")) <= 10)
    demand = min(32.0, math.log1p(max(traffic, 0)) * 4.2 + min(keywords, 80) * 0.08 + top3 * 1.8 + top10 * 0.5)
    link_score = min(20.0, _safe_int(link.get("in_degree")) * 2.0 + max(0.0, _safe_float(authority.get("weighted_pagerank_percentile"))) * 9.0)
    schema_score = 8.0 if _safe_int(structured.get("valid_blocks")) else (3.0 if structured.get("types") else 0.0)
    freshness_score = {"fresh": 8.0, "aging": 6.0, "stale": 3.0, "very_stale": 0.0, "future": 4.0}.get(str(freshness.get("bucket") or "unknown"), 2.0)
    content_score = min(20.0, _safe_float(entity.get("coverage")) * 9.0 + _safe_float(information_gain.get("information_gain_score")) * 0.08 + min(len(paragraphs), 5) * 1.2)
    return round(min(100.0, demand + link_score + schema_score + freshness_score + content_score + 12.0), 1)


def _explanation(strengths: list[dict], weak_spots: list[dict]) -> str:
    top_strengths = [s.get("label", "") for s in strengths[:3] if s.get("label")]
    if not top_strengths:
        return "This page is selected because it is one of the highest-demand pages in the available search dataset."
    sentence = "Likely support comes from " + ", ".join(top_strengths).rstrip(".") + "."
    if weak_spots:
        sentence += " The main visible constraint is " + str(weak_spots[0].get("label", "a remaining audit gap")).lower().rstrip(".") + "."
    return sentence


def build_best_page_explainers(
    pages: Sequence[PageInfo | Mapping[str, Any]],
    *,
    search_payload: Mapping[str, Any] | None = None,
    linkgraph: Mapping[str, Any] | None = None,
    structured_data: Mapping[str, Any] | None = None,
    freshness: Mapping[str, Any] | None = None,
    entity_coverage: Mapping[str, Any] | None = None,
    information_gain: Mapping[str, Any] | None = None,
    paragraph_impact: Mapping[str, Any] | None = None,
    winning_paragraphs: Mapping[str, Any] | None = None,
    template_patterns: Mapping[str, Any] | None = None,
    header_analysis: Mapping[str, Any] | None = None,
    answerability: Sequence[Mapping[str, Any]] | None = None,
    conversion_balance: Mapping[str, Any] | None = None,
    metadata_quality: Mapping[str, Any] | None = None,
    page_types: Mapping[str, Any] | None = None,
    limit: int = 40,
) -> dict:
    """Build domain-level best-page explainers from existing audit payloads."""

    page_lookup = _lookup_from_rows([_page_to_dict(p) for p in pages], ("url",))
    keyword_by_url = _keyword_lookup(search_payload)
    link_lookup = _lookup_from_rows((linkgraph or {}).get("page_link_counts") or [], ("url",))
    authority_lookup = _lookup_from_rows(((linkgraph or {}).get("traffic_weighted_pagerank") or {}).get("pages") or [], ("url",), score_key="traffic")
    structured_lookup = _lookup_from_rows((structured_data or {}).get("per_page") or [], ("url",))
    freshness_lookup = _lookup_from_rows((freshness or {}).get("per_page") or [], ("url",))
    entity_lookup = _lookup_from_rows((entity_coverage or {}).get("pages") or [], ("url",), score_key="traffic")
    information_lookup = _lookup_from_rows((information_gain or {}).get("pages") or [], ("url",))
    header_lookup = _lookup_from_rows((header_analysis or {}).get("per_page") or [], ("url",))
    answer_lookup = _lookup_from_rows(list(answerability or []), ("url",))
    conversion_lookup = _lookup_from_rows((conversion_balance or {}).get("pages") or [], ("url",), score_key="traffic")
    metadata_lookup = _lookup_from_rows((metadata_quality or {}).get("per_page") or [], ("url",))
    page_type_lookup = _lookup_from_rows((page_types or {}).get("per_page") or [], ("url",))
    paragraph_rows = (winning_paragraphs or {}).get("rows") or (paragraph_impact or {}).get("top_paragraphs") or []
    paragraphs_by_url = _group_by_url(paragraph_rows, ("url",))

    explainers: list[dict] = []
    for rank, top in enumerate(_top_pages(search_payload, pages, limit), 1):
        if not isinstance(top, dict):
            continue
        url = top.get("matched_url") or top.get("url") or ""
        source_url = top.get("url") or url
        if not url and not source_url:
            continue
        page = _lookup(page_lookup, url, source_url)
        keywords = _keywords_for_url(keyword_by_url, url, source_url)
        link = _lookup(link_lookup, url, source_url)
        authority = _lookup(authority_lookup, url, source_url)
        structured = _lookup(structured_lookup, url, source_url)
        fresh = _lookup(freshness_lookup, url, source_url)
        entity = _lookup(entity_lookup, url, source_url)
        info = _lookup(information_lookup, url, source_url)
        header = _lookup(header_lookup, url, source_url)
        answer = _lookup(answer_lookup, url, source_url)
        conversion = _lookup(conversion_lookup, url, source_url)
        metadata = _lookup(metadata_lookup, url, source_url)
        page_type = _lookup(page_type_lookup, url, source_url)
        paragraphs = _rows_for_url(paragraphs_by_url, url, source_url)
        paragraphs.sort(key=lambda r: (_safe_float(r.get("impact_score")), _safe_float(r.get("attributed_traffic"))), reverse=True)
        patterns, pattern_recs = _page_patterns(template_patterns, url, source_url)

        traffic = _safe_int(top.get("traffic"))
        keyword_count = _safe_int(top.get("keywords")) or len(keywords)
        title = top.get("title") or page.get("title") or top.get("top_keyword_title") or source_url or url
        cluster_label = top.get("cluster_label") or entity.get("cluster_label") or info.get("cluster_label") or top.get("section") or page.get("section") or ""
        strengths: list[dict] = []
        weak_spots: list[dict] = []

        if traffic or keyword_count:
            _add_signal(
                strengths,
                "Demand",
                "Search demand is concentrated on this URL",
                f"{traffic:,} estimated visits and {keyword_count:,} ranking keywords in the provider snapshot.",
                value=traffic,
                source="search",
                confidence=0.82,
            )
        if keywords:
            best = keywords[0]
            pos = _safe_int(best.get("position"))
            _add_signal(
                strengths if pos and pos <= 10 else weak_spots,
                "Keyword Fit",
                "Top keywords map to the page",
                f"{best.get('keyword') or top.get('top_keyword') or 'top keyword'} ranks at position {pos or 'unknown'} with {_safe_int(best.get('traffic')):,} estimated visits.",
                value=pos,
                source="search_keywords",
                confidence=0.74,
            )
        if _safe_int(structured.get("valid_blocks")) or structured.get("types"):
            _add_signal(
                strengths,
                "Schema",
                "Structured data is present",
                f"{_safe_int(structured.get('valid_blocks')):,} valid blocks · {', '.join(str(t) for t in (structured.get('types') or [])[:4]) or 'schema types detected'}.",
                value=_safe_int(structured.get("valid_blocks")),
                source="structured_data",
                confidence=0.64,
            )
        elif structured or traffic > 0:
            _add_signal(weak_spots, "Schema", "No valid schema found", "The page has no valid structured-data block in the crawl extraction.", source="structured_data", confidence=0.62)
        if _safe_int(structured.get("invalid_blocks")):
            _add_signal(weak_spots, "Schema", "Invalid schema blocks", f"{_safe_int(structured.get('invalid_blocks'))} invalid structured-data blocks were found.", source="structured_data", confidence=0.78)

        bucket = str(fresh.get("bucket") or "unknown")
        age = fresh.get("age_days")
        if bucket in {"fresh", "aging"}:
            _add_signal(strengths, "Freshness", "Freshness signal is available", f"Freshness bucket is {bucket}{' · ' + str(age) + ' days old' if age is not None else ''}.", value=age, source="freshness", confidence=0.62)
        elif bucket in {"stale", "very_stale", "unknown"}:
            _add_signal(weak_spots, "Freshness", "Freshness is weak or missing", f"Freshness bucket is {bucket}{' · ' + str(age) + ' days old' if age is not None else ''}.", value=age, source="freshness", confidence=0.62)

        in_degree = _safe_int(link.get("in_degree"))
        click_depth = link.get("click_depth")
        authority_pct = _safe_float(authority.get("weighted_pagerank_percentile"))
        if in_degree >= 3 or authority_pct >= 0.65:
            _add_signal(strengths, "Internal Links", "Internal link support is visible", f"{in_degree:,} inbound internal links · PageRank percentile {authority_pct:.2f}.", value=in_degree, source="linkgraph", confidence=0.76)
        elif traffic > 0:
            _add_signal(weak_spots, "Internal Links", "Demand may be under-supported by internal links", f"{in_degree:,} inbound internal links · click depth {click_depth if click_depth is not None else 'unknown'}.", value=in_degree, source="linkgraph", confidence=0.72)

        coverage = _safe_float(entity.get("coverage"))
        if coverage >= 0.68:
            _add_signal(strengths, "Entities", "Expected cluster entities are covered", f"{_safe_float(entity.get('coverage_pct'), coverage * 100):.1f}% weighted entity coverage.", value=coverage, source="entity_coverage", confidence=0.7)
        elif entity:
            missing = entity.get("missing_core_entities") or []
            _add_signal(weak_spots, "Entities", "Core entity gaps remain", f"{_safe_float(entity.get('coverage_pct'), coverage * 100):.1f}% coverage · {len(missing)} core gaps.", value=coverage, source="entity_coverage", confidence=0.7)

        info_score = _safe_float(info.get("information_gain_score"))
        if info_score >= 75:
            _add_signal(strengths, "Information Gain", "Originality and evidence signals are strong", f"Information-gain score {info_score:.1f}; positives: {', '.join((info.get('positive_evidence') or [])[:3])}.", value=info_score, source="information_gain", confidence=0.66)
        elif info and info_score < 55:
            _add_signal(weak_spots, "Information Gain", "Content may be generic or thin", f"Information-gain score {info_score:.1f}; issues: {', '.join((info.get('negative_reasons') or [])[:3])}.", value=info_score, source="information_gain", confidence=0.66)

        if paragraphs:
            top_para = paragraphs[0]
            _add_signal(
                strengths,
                "Paragraphs",
                "High-impact paragraph section detected",
                f"Top paragraph impact {float(top_para.get('impact_score') or 0):.1f} with {float(top_para.get('attributed_traffic') or 0):,.0f} attributed visits.",
                value=top_para.get("impact_score"),
                source="paragraph_impact",
                confidence=0.72,
            )

        if _safe_float(answer.get("score")) >= 7:
            _add_signal(strengths, "Answerability", "Page is answer-ready", f"GEO answerability score {_safe_float(answer.get('score')):.1f}.", value=answer.get("score"), source="answerability", confidence=0.56)
        if _safe_float(conversion.get("conversion_support")) >= 70:
            _add_signal(strengths, "Conversion", "Commercial page has conversion support", f"Conversion support score {_safe_float(conversion.get('conversion_support')):.1f}.", value=conversion.get("conversion_support"), source="conversion_balance", confidence=0.54)
        if metadata.get("issues"):
            _add_signal(weak_spots, "Metadata", "Metadata issues remain", f"{', '.join(str(v) for v in (metadata.get('issues') or [])[:4])}.", source="metadata_quality", confidence=0.64)
        if header and (header.get("level_skips") or _safe_int(header.get("h1_count")) != 1):
            _add_signal(weak_spots, "Headers", "Header structure needs review", f"H1 count {_safe_int(header.get('h1_count'))} · level skips {_safe_int(header.get('level_skips'))}.", source="header_analysis", confidence=0.66)

        for pattern in patterns[:3]:
            _add_signal(
                strengths,
                "Template",
                f"Winning template pattern: {pattern.get('label') or pattern.get('feature_key')}",
                f"Observed lift {float(pattern.get('observed_lift') or 0):.2f} · confidence {float(pattern.get('confidence') or 0):.2f}.",
                value=pattern.get("observed_lift"),
                source="template_patterns",
                confidence=_safe_float(pattern.get("confidence"), 0.55),
            )
        for rec in pattern_recs[:3]:
            _add_signal(weak_spots, "Template", f"Missing template pattern: {rec.get('missing_pattern') or rec.get('feature_key')}", rec.get("recommendation") or "Add the missing repeated winner feature.", value=rec.get("observed_lift"), source="template_patterns", confidence=_safe_float(rec.get("confidence"), 0.55))

        strengths.sort(key=lambda s: (_safe_float(s.get("confidence")), _safe_float(s.get("value"))), reverse=True)
        weak_spots.sort(key=lambda s: (_safe_float(s.get("confidence")), _safe_float(s.get("value"))), reverse=True)

        transferable_patterns: list[dict] = []
        for pattern in patterns[:5]:
            transferable_patterns.append({
                "type": "template",
                "label": pattern.get("label") or pattern.get("feature_key"),
                "action": pattern.get("recommendation") or "Reuse this pattern on similar pages.",
                "evidence": f"Observed lift {float(pattern.get('observed_lift') or 0):.2f} · confidence {float(pattern.get('confidence') or 0):.2f}.",
                "source": "template_patterns",
            })
        if paragraphs:
            transferable_patterns.append({
                "type": "content",
                "label": "Protect and expand high-impact paragraphs",
                "action": "Use the same paragraph role, keyword vocabulary, and proof depth on weaker same-cluster pages.",
                "evidence": f"{len(paragraphs[:5])} high-impact paragraph examples on this page.",
                "source": "paragraph_impact",
            })
        if in_degree >= 3:
            transferable_patterns.append({
                "type": "links",
                "label": "Replicate internal promotion pattern",
                "action": "Link similar pages from relevant hubs and supporting articles until they reach comparable in-degree and click depth.",
                "evidence": f"{in_degree:,} inbound links and click depth {click_depth if click_depth is not None else 'unknown'}.",
                "source": "linkgraph",
            })
        if structured.get("types"):
            transferable_patterns.append({
                "type": "schema",
                "label": "Reuse schema coverage",
                "action": f"Add equivalent {', '.join(str(t) for t in (structured.get('types') or [])[:3])} schema where the page type matches.",
                "evidence": f"{_safe_int(structured.get('valid_blocks')):,} valid blocks on the winner.",
                "source": "structured_data",
            })

        copy_recommendations = [p["action"] for p in transferable_patterns[:5]]
        if not copy_recommendations and keywords:
            copy_recommendations.append("Align the page title, H1, intro, and internal anchors with the same high-demand keyword vocabulary.")
        avoid_recommendations = [
            f"Do not copy the weakness: {spot.get('label')}. {spot.get('evidence')}"
            for spot in weak_spots[:4]
        ]

        perf = _performance_score(
            traffic=traffic,
            keywords=keyword_count,
            keyword_rows=keywords,
            link=link,
            authority=authority,
            structured=structured,
            freshness=fresh,
            entity=entity,
            information_gain=info,
            paragraphs=paragraphs,
        )

        explainers.append({
            "rank": rank,
            "url": url or source_url,
            "source_url": source_url,
            "title": title,
            "section": top.get("section") or page.get("section") or "",
            "directory": authority.get("directory") or top.get("directory") or "",
            "page_type": page_type.get("page_type") or authority.get("page_type") or top.get("page_type") or "",
            "cluster": top.get("cluster") if top.get("cluster") is not None else entity.get("cluster", info.get("cluster")),
            "cluster_label": cluster_label,
            "traffic": traffic,
            "keywords": keyword_count,
            "top_keyword": top.get("top_keyword") or ((keywords[0] or {}).get("keyword") if keywords else ""),
            "top_keyword_position": _safe_int(top.get("top_keyword_position")) or (_safe_int((keywords[0] or {}).get("position")) if keywords else 0),
            "performance_score": perf,
            "explanation": _explanation(strengths, weak_spots),
            "causation_note": CAUSATION_NOTE,
            "strengths": strengths[:10],
            "weak_spots": weak_spots[:8],
            "transferable_patterns": transferable_patterns[:8],
            "copy_recommendations": copy_recommendations[:6],
            "avoid_recommendations": avoid_recommendations[:5],
            "evidence_links": [
                {"label": "Open analyzed page", "url": url or source_url},
            ],
            "top_keywords": [
                {
                    "keyword": kw.get("keyword") or "",
                    "position": _safe_int(kw.get("position")),
                    "traffic": _safe_int(kw.get("traffic")),
                    "volume": _safe_int(kw.get("volume")),
                    "intents": list(kw.get("intents") or [])[:4],
                    "serp_features": list(kw.get("serp_features") or [])[:6],
                }
                for kw in keywords[:10]
            ],
            "top_paragraphs": [
                {
                    "paragraph_index": row.get("paragraph_index"),
                    "excerpt": row.get("excerpt") or row.get("paragraph_excerpt") or row.get("text") or "",
                    "impact_score": _safe_float(row.get("impact_score")),
                    "impact_tier": row.get("impact_tier") or "",
                    "attributed_traffic": _safe_float(row.get("attributed_traffic")),
                    "recommended_action": row.get("recommended_action_label") or row.get("recommended_action") or "",
                }
                for row in paragraphs[:5]
            ],
            "signals": {
                "valid_schema_blocks": _safe_int(structured.get("valid_blocks")),
                "schema_types": list(structured.get("types") or [])[:8],
                "freshness_bucket": fresh.get("bucket") or "",
                "freshness_age_days": fresh.get("age_days"),
                "in_degree": in_degree,
                "out_degree": _safe_int(link.get("out_degree")),
                "click_depth": click_depth,
                "pagerank": _safe_float(authority.get("pagerank")),
                "weighted_pagerank_percentile": authority_pct,
                "authority_traffic_gap": _safe_float(authority.get("authority_traffic_gap")),
                "entity_coverage": coverage,
                "information_gain_score": info_score,
                "answerability_score": _safe_float(answer.get("score")),
                "conversion_support": _safe_float(conversion.get("conversion_support")),
            },
        })

    explainers.sort(key=lambda r: (_safe_int(r.get("traffic")), _safe_float(r.get("performance_score"))), reverse=True)
    for i, row in enumerate(explainers, 1):
        row["rank"] = i

    cluster_groups: dict[str, list[dict]] = defaultdict(list)
    for row in explainers:
        cluster_groups[_cluster_key(row)].append(row)
    cluster_summaries = []
    for key, rows in cluster_groups.items():
        rows.sort(key=lambda r: (_safe_int(r.get("traffic")), _safe_float(r.get("performance_score"))), reverse=True)
        cluster_summaries.append({
            "cluster": key,
            "label": rows[0].get("cluster_label") or key,
            "pages": len(rows),
            "traffic": sum(_safe_int(r.get("traffic")) for r in rows),
            "keywords": sum(_safe_int(r.get("keywords")) for r in rows),
            "leader_url": rows[0].get("url"),
            "leader_title": rows[0].get("title"),
            "leader_score": rows[0].get("performance_score"),
            "leader_traffic": rows[0].get("traffic"),
            "peer_pages": [
                {
                    "url": peer.get("url"),
                    "title": peer.get("title"),
                    "traffic": peer.get("traffic"),
                    "keywords": peer.get("keywords"),
                    "performance_score": peer.get("performance_score"),
                }
                for peer in rows[1:8]
            ],
        })
    cluster_summaries.sort(key=lambda r: (_safe_int(r.get("traffic")), _safe_float(r.get("leader_score"))), reverse=True)

    return {
        "summary": {
            "status": "ok" if explainers else "no_top_pages",
            "model": "best_page_explainer_v1",
            "pages": len(explainers),
            "clusters": len(cluster_summaries),
            "top_pages_source": "search_provider" if (search_payload or {}).get("top_pages") else "crawl_pages",
            "causation_note": CAUSATION_NOTE,
        },
        "pages": explainers,
        "clusters": cluster_summaries,
        "interpretation": {
            "performance_score": "0-100 diagnostic score combining demand, keyword positions, internal links, schema, freshness, entity coverage, information gain, and paragraph impact.",
            "strengths": "Observed signals that plausibly support performance. They are ranked by confidence and measured value.",
            "weak_spots": "Observed constraints that may limit the page or should not be copied blindly.",
            "transferable_patterns": "Patterns to test on weaker pages in the same semantic or commercial cluster.",
            "causation_note": CAUSATION_NOTE,
        },
    }


def build_best_page_comparison(domain_payloads: Sequence[Mapping[str, Any]], limit_per_domain: int = 30) -> dict:
    """Aggregate generated best-page payloads for cross-domain reports."""

    pages: list[dict] = []
    for domain_payload in domain_payloads:
        domain = str(domain_payload.get("domain") or "")
        best_pages = domain_payload.get("best_pages") or {}
        for page in (best_pages.get("pages") or [])[:limit_per_domain]:
            if not isinstance(page, dict):
                continue
            pages.append({"domain": domain, **page})
    if not pages:
        return {"summary": {"status": "no_best_pages", "pages": 0}, "pages": [], "clusters": [], "recommendations": [], "side_by_side": []}

    pages.sort(key=lambda r: (_safe_int(r.get("traffic")), _safe_float(r.get("performance_score"))), reverse=True)
    domains = [str(p.get("domain")) for p in domain_payloads if p.get("domain")]
    by_cluster: dict[str, list[dict]] = defaultdict(list)
    for page in pages:
        by_cluster[_cluster_key(page)].append(page)

    clusters: list[dict] = []
    recommendations: list[dict] = []
    side_by_side: list[dict] = []
    for cluster, rows in by_cluster.items():
        rows.sort(key=lambda r: (_safe_int(r.get("traffic")), _safe_float(r.get("performance_score"))), reverse=True)
        leader = rows[0]
        domain_cells = []
        for domain in domains:
            domain_rows = [r for r in rows if r.get("domain") == domain]
            best = domain_rows[0] if domain_rows else {}
            domain_cells.append({
                "domain": domain,
                "url": best.get("url", ""),
                "title": best.get("title", ""),
                "traffic": _safe_int(best.get("traffic")),
                "keywords": _safe_int(best.get("keywords")),
                "performance_score": _safe_float(best.get("performance_score")),
                "strengths": (best.get("strengths") or [])[:4],
                "weak_spots": (best.get("weak_spots") or [])[:4],
                "missing": not bool(best),
            })
        clusters.append({
            "cluster": cluster,
            "label": leader.get("cluster_label") or cluster,
            "domains": domain_cells,
            "leader_domain": leader.get("domain"),
            "leader_url": leader.get("url"),
            "leader_title": leader.get("title"),
            "leader_traffic": _safe_int(leader.get("traffic")),
            "leader_score": _safe_float(leader.get("performance_score")),
            "total_traffic": sum(_safe_int(r.get("traffic")) for r in rows),
            "pages": len(rows),
        })

        for cell in domain_cells:
            if cell["domain"] == leader.get("domain"):
                continue
            traffic_gap = _safe_int(leader.get("traffic")) - _safe_int(cell.get("traffic"))
            score_gap = _safe_float(leader.get("performance_score")) - _safe_float(cell.get("performance_score"))
            if traffic_gap <= 0 and score_gap <= 8 and not cell.get("missing"):
                continue
            rec = {
                "cluster": cluster,
                "cluster_label": leader.get("cluster_label") or cluster,
                "source_domain": leader.get("domain"),
                "source_url": leader.get("url"),
                "source_title": leader.get("title"),
                "target_domain": cell["domain"],
                "target_url": cell.get("url", ""),
                "target_title": cell.get("title", ""),
                "priority_score": round(max(0.0, math.log1p(max(traffic_gap, 0)) * 8.0 + max(score_gap, 0) * 0.8), 1),
                "traffic_gap": traffic_gap,
                "score_gap": round(score_gap, 1),
                "copy": list(leader.get("copy_recommendations") or [])[:5],
                "avoid": list(leader.get("avoid_recommendations") or [])[:4],
                "source_strengths": list(leader.get("strengths") or [])[:5],
                "target_weak_spots": list(cell.get("weak_spots") or [])[:5],
                "causation_note": CAUSATION_NOTE,
            }
            recommendations.append(rec)
            side_by_side.append({
                "cluster": cluster,
                "cluster_label": leader.get("cluster_label") or cluster,
                "leader": leader,
                "target": next((r for r in rows if r.get("domain") == cell["domain"]), {"domain": cell["domain"], "missing": True}),
                "recommendation": rec,
            })

    clusters.sort(key=lambda r: (_safe_int(r.get("total_traffic")), _safe_float(r.get("leader_score"))), reverse=True)
    recommendations.sort(key=lambda r: (_safe_float(r.get("priority_score")), _safe_int(r.get("traffic_gap"))), reverse=True)
    side_by_side.sort(key=lambda r: _safe_float((r.get("recommendation") or {}).get("priority_score")), reverse=True)

    return {
        "summary": {
            "status": "ok",
            "model": "best_page_comparison_v1",
            "pages": len(pages),
            "clusters": len(clusters),
            "recommendations": len(recommendations),
            "causation_note": CAUSATION_NOTE,
        },
        "pages": pages[:300],
        "clusters": clusters[:160],
        "recommendations": recommendations[:160],
        "side_by_side": side_by_side[:80],
        "interpretation": {
            "cluster": "Pages are grouped by their search/audit cluster label when available, otherwise by section or directory.",
            "priority_score": "Higher when the leader has materially more traffic or stronger diagnostic support than a compared domain.",
            "causation_note": CAUSATION_NOTE,
        },
    }
