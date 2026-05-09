"""Detect answer blocks and snippet candidates for query clusters."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Iterable

from .analyzer import PageInfo
from .entities import extract_entities_from_text
from .paragraph_impact import _heading_for_paragraph, _match_page, _page_lookup, _to_int

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9'_-]*", re.I)
_STAT_RE = re.compile(
    r"\b\d+(?:[,.]\d+)*\s*(?:%|percent|million|billion|thousand|users?|customers?|"
    r"seconds?|minutes?|hours?|days?|weeks?|months?|years?|\$|€|£|x|times)\b",
    re.I,
)
_QUESTION_PREFIXES = ("how", "what", "why", "when", "where", "which", "who", "can", "is", "are", "do", "does", "should")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in", "is", "it",
    "of", "on", "or", "the", "to", "vs", "what", "with", "your",
}


def _tokens(text: str) -> list[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(text or "")]


def _token_set(text: str) -> set[str]:
    return {tok for tok in _tokens(text) if tok not in _STOPWORDS and len(tok) > 1}


def _clip(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def _cluster_label_lookup(cluster_summaries) -> dict[int, str]:
    out: dict[int, str] = {}
    for summary in cluster_summaries or []:
        cid = getattr(summary, "cluster_id", None)
        if cid is None:
            continue
        keywords = getattr(summary, "keywords", []) or []
        label = ", ".join(k.get("keyword", "") for k in keywords[:4] if k.get("keyword"))
        out[int(cid)] = label or f"cluster {cid}"
    return out


def _query_format(query: str) -> str:
    q = f" {query.lower().strip()} "
    if re.search(r"\b(vs|versus|compare|comparison|difference|alternative|alternatives)\b", q):
        return "comparison"
    if re.search(r"\b(pros|cons|advantages|disadvantages|benefits|drawbacks)\b", q):
        return "pros_cons"
    if re.search(r"\b(price|pricing|cost|costs|plans?|subscription|license)\b", q):
        return "pricing"
    if re.search(r"\b(integrat|api|webhook|zapier|slack|salesforce|hubspot)\b", q):
        return "integration"
    if re.search(r"\b(error|fix|issue|problem|troubleshoot|not working|failed|failure)\b", q):
        return "troubleshooting"
    if re.search(r"\b(statistic|statistics|benchmark|data|average|rate|percent|percentage)\b", q):
        return "statistic"
    if re.search(r"\b(how to|steps?|setup|set up|install|configure|create|build|use)\b", q):
        return "steps"
    if re.search(r"\b(best|top|types?|examples?|ideas?|ways?|tools?|features?|checklist)\b", q):
        return "list"
    if re.search(r"\b(what is|what are|definition|meaning|means|does .* mean)\b", q):
        return "definition"
    if query.strip().endswith("?") or any(query.lower().startswith(prefix + " ") for prefix in _QUESTION_PREFIXES):
        return "faq"
    return "definition"


def _schema_for_format(answer_type: str) -> str:
    return {
        "faq": "FAQPage",
        "steps": "HowTo",
        "list": "ItemList",
        "definition": "Article",
        "comparison": "Article",
        "pros_cons": "Article",
        "pricing": "Product",
        "integration": "TechArticle",
        "troubleshooting": "HowTo",
        "statistic": "Article",
    }.get(answer_type, "Article")


def _detect_answer_type(text: str, heading: str, query: str = "") -> tuple[str, float]:
    haystack = f"{heading} {text}".lower()
    expected = _query_format(query) if query else ""
    scores: Counter[str] = Counter()
    if _STAT_RE.search(text):
        scores["statistic"] += 2.0
    if re.search(r"\b(vs|versus|compare|comparison|difference between|whereas)\b", haystack):
        scores["comparison"] += 2.3
    if re.search(r"\b(pros and cons|advantages|disadvantages|benefits|drawbacks)\b", haystack):
        scores["pros_cons"] += 2.2
    if re.search(r"\b(error|fix|troubleshoot|problem|issue|not working|failed)\b", haystack):
        scores["troubleshooting"] += 2.0
    if re.search(r"\b(price|pricing|cost|plans?|subscription)\b", haystack):
        scores["pricing"] += 1.8
    if re.search(r"\b(integrat|api|webhook|zapier|salesforce|hubspot|slack)\b", haystack):
        scores["integration"] += 1.8
    if re.search(r"\b(step|first|second|third|then|finally|setup|set up|install|configure)\b", haystack):
        scores["steps"] += 2.0
    if re.search(r"\b(include|includes|types of|examples of|ways to|best|top|checklist|features)\b", haystack):
        scores["list"] += 1.6
    if re.search(r"\b(is a|is an|are a|are an|means|refers to|defined as|lets you|helps)\b", text.lower()[:220]):
        scores["definition"] += 1.8
    if heading.strip().endswith("?") or any(heading.lower().startswith(prefix + " ") for prefix in _QUESTION_PREFIXES):
        scores["faq"] += 1.5
    if expected:
        scores[expected] += 1.4
    if not scores:
        return expected or "definition", 0.35
    answer_type, raw = scores.most_common(1)[0]
    confidence = min(1.0, 0.35 + raw / 3.8)
    return answer_type, confidence


def _heading_match(heading: str, query: str) -> float:
    heading_tokens = _token_set(heading)
    query_tokens = _token_set(query)
    if not query_tokens:
        return 0.7 if heading else 0.2
    if not heading_tokens:
        return 0.15
    overlap = len(heading_tokens & query_tokens) / max(len(query_tokens), 1)
    if heading.strip().endswith("?"):
        overlap += 0.25
    return min(1.0, overlap)


def _directness(text: str, query: str, answer_type: str) -> float:
    first = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)[0].lower()
    q_tokens = _token_set(query)
    overlap = len(_token_set(first) & q_tokens) / max(len(q_tokens), 1) if q_tokens else 0.35
    markers = {
        "definition": r"\b(is a|is an|are a|are an|means|refers to|defined as|helps|lets you)\b",
        "steps": r"\b(step|first|start|then|next|finally)\b",
        "list": r"\b(include|includes|these are|the main|best|top)\b",
        "comparison": r"\b(compared with|whereas|while|difference|versus|vs)\b",
        "pros_cons": r"\b(advantage|disadvantage|benefit|drawback|pros|cons)\b",
        "pricing": r"\b(costs?|pricing|plan|starts at|per month|per user)\b",
        "integration": r"\b(integrates?|connects?|api|webhook)\b",
        "troubleshooting": r"\b(fix|solve|check|restart|verify|troubleshoot)\b",
        "statistic": r"\b\d",
        "faq": r"\b(is|are|can|does|do|should|will|helps|lets you)\b",
    }
    marker_bonus = 0.45 if re.search(markers.get(answer_type, r"\b(is|are)\b"), first) else 0.0
    return min(1.0, 0.25 + overlap * 0.55 + marker_bonus)


def _concision(words: int) -> float:
    if 35 <= words <= 95:
        return 1.0
    if 18 <= words <= 140:
        return 0.75
    if 10 <= words <= 220:
        return 0.45
    return 0.2


def _schema_score(schema_types: Iterable[str], answer_type: str) -> tuple[float, list[str]]:
    types = set(schema_types or [])
    wanted = _schema_for_format(answer_type)
    if wanted in types:
        return 1.0, []
    if answer_type == "list" and {"ItemList", "BreadcrumbList"} & types:
        return 0.9, []
    if answer_type in {"definition", "comparison", "pros_cons", "statistic"} and {"Article", "BlogPosting", "NewsArticle", "TechArticle"} & types:
        return 0.8, []
    if answer_type in {"troubleshooting", "integration"} and {"HowTo", "TechArticle", "Article", "BlogPosting"} & types:
        return 0.75, []
    return 0.3, [wanted]


def _score_block(text: str, heading: str, query: str, ext) -> dict:
    tokens = _tokens(text)
    words = len(tokens)
    answer_type, type_confidence = _detect_answer_type(text, heading, query)
    directness = _directness(text, query, answer_type)
    concision = _concision(words)
    heading_score = _heading_match(heading, query)
    entities = extract_entities_from_text(text)
    entity_clarity = min(1.0, 0.3 + len(set(entities)) * 0.18)
    schema_compatibility, missing_schema = _schema_score(getattr(ext, "schema_types", []) or [], answer_type)
    stat_count = len(_STAT_RE.findall(text or ""))
    list_table_bonus = 0.0
    if answer_type == "list" and int(getattr(ext, "list_count", 0) or 0):
        list_table_bonus += 0.45
    if answer_type in {"comparison", "pricing", "statistic"} and int(getattr(ext, "table_count", 0) or 0):
        list_table_bonus += 0.45
    if stat_count:
        list_table_bonus += min(0.4, stat_count * 0.15)

    score = (
        directness * 24
        + concision * 18
        + heading_score * 18
        + entity_clarity * 14
        + schema_compatibility * 12
        + type_confidence * 10
        + min(1.0, list_table_bonus) * 8
    )
    evidence: list[str] = []
    risks: list[str] = []
    if directness >= 0.75:
        evidence.append("direct answer opening")
    elif directness < 0.45:
        risks.append("answer opening is indirect")
    if concision >= 0.75:
        evidence.append(f"concise {words}-word block")
    elif words < 18:
        risks.append("too thin for a standalone answer")
    elif words > 140:
        risks.append("too long for a clean snippet")
    if heading_score >= 0.65:
        evidence.append("heading matches target query")
    else:
        risks.append("heading does not clearly match target query")
    if entities:
        evidence.append(f"{len(set(entities))} named entity signals")
    else:
        risks.append("few named entities")
    if stat_count:
        evidence.append(f"{stat_count} statistic signals")
    if missing_schema:
        risks.append(f"missing {missing_schema[0]} schema opportunity")
    else:
        evidence.append("compatible structured data present")

    return {
        "answer_type": answer_type,
        "recommended_format": _query_format(query) if query else answer_type,
        "score": round(_clip(score), 2),
        "directness": round(directness, 3),
        "concision": round(concision, 3),
        "entity_clarity": round(entity_clarity, 3),
        "heading_match": round(heading_score, 3),
        "schema_compatibility": round(schema_compatibility, 3),
        "type_confidence": round(type_confidence, 3),
        "schema_opportunities": missing_schema,
        "evidence": evidence[:7],
        "risks": risks[:7],
    }


def _query_rows(
    pages: list[PageInfo],
    coverage: list[dict] | None,
    paragraph_fanout: list[dict] | None,
    search_payload: dict | None,
    *,
    max_queries: int,
) -> list[dict]:
    lookup = _page_lookup(pages)
    rows: dict[tuple[int, str], dict] = {}

    def put(page_i: int | None, query: str, source: str, **extra) -> None:
        if page_i is None or page_i >= len(pages):
            return
        query = " ".join(str(query or "").split())
        if len(query) < 3:
            return
        key = (page_i, query.lower())
        traffic = _to_int(extra.get("traffic"))
        current = rows.get(key)
        if current is None or traffic > _to_int(current.get("traffic")):
            rows[key] = {
                "page_index": page_i,
                "query": query,
                "source": source,
                "traffic": traffic,
                "volume": _to_int(extra.get("volume")),
                "position": _to_int(extra.get("position")),
                "serp_features": list(extra.get("serp_features") or []),
                "intent": extra.get("intent") or "",
                "coverage_status": extra.get("coverage_status") or "",
                "best_similarity": extra.get("best_similarity", 0.0),
            }

    search_payload = search_payload or {}
    for row in search_payload.get("organic_keywords") or []:
        intents = row.get("intents") or []
        put(
            _match_page(row.get("matched_url") or row.get("url") or "", lookup),
            row.get("keyword") or "",
            "organic_keyword",
            traffic=row.get("traffic"),
            volume=row.get("volume"),
            position=row.get("position"),
            serp_features=row.get("serp_features") or [],
            intent=intents[0] if intents else "",
        )
    for row in search_payload.get("top_pages") or []:
        put(
            _match_page(row.get("matched_url") or row.get("url") or "", lookup),
            row.get("top_keyword") or "",
            "top_page",
            traffic=row.get("traffic"),
            volume=row.get("top_keyword_volume"),
            position=row.get("top_keyword_position"),
        )
    for row in coverage or []:
        put(
            _match_page(row.get("best_url") or "", lookup),
            row.get("query") or "",
            f"coverage:{row.get('source') or 'query'}",
            coverage_status=row.get("status") or "",
            best_similarity=row.get("best_similarity", 0.0),
        )
    for row in paragraph_fanout or []:
        put(
            _match_page(row.get("best_url") or "", lookup),
            row.get("query") or "",
            f"paragraph:{row.get('source') or 'query'}",
            coverage_status=row.get("status") or "",
            best_similarity=row.get("best_similarity", 0.0),
        )

    if not rows:
        for page_i, page in enumerate(pages):
            put(page_i, page.title or page.url, "title")

    out = list(rows.values())
    out.sort(key=lambda r: (_to_int(r.get("traffic")), _to_int(r.get("volume"))), reverse=True)
    return out[:max_queries]


def _paragraphs_by_page(paragraph_records: list[tuple[int, int, str, Any]] | None) -> dict[int, list[tuple[int, str]]]:
    out: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for page_i, para_i, text, _ in paragraph_records or []:
        out[int(page_i)].append((int(para_i), str(text or "")))
    return out


def _cluster_for(page_i: int, cluster_labels: Iterable[int] | None) -> int:
    if cluster_labels is None:
        return 0
    labels = list(cluster_labels)
    if page_i < len(labels):
        return int(labels[page_i])
    return 0


def _build_block_row(page: PageInfo, ext, page_i: int, para_i: int, text: str, heading: str, query: dict, cluster: int, cluster_label: str) -> dict:
    scored = _score_block(text, heading, str(query.get("query") or ""), ext)
    return {
        "url": page.url,
        "title": page.title,
        "section": page.section,
        "cluster": cluster,
        "cluster_label": cluster_label,
        "query": query.get("query") or "",
        "query_source": query.get("source") or "",
        "keyword_traffic": _to_int(query.get("traffic")),
        "keyword_volume": _to_int(query.get("volume")),
        "keyword_position": _to_int(query.get("position")),
        "serp_features": query.get("serp_features") or [],
        "paragraph_index": int(para_i),
        "heading": heading,
        "excerpt": text[:420],
        **scored,
    }


def build_answer_blocks(
    pages: list[PageInfo],
    extracted_pages: list,
    paragraph_records: list[tuple[int, int, str, Any]] | None = None,
    *,
    coverage: list[dict] | None = None,
    paragraph_fanout: list[dict] | None = None,
    search_payload: dict | None = None,
    cluster_labels: Iterable[int] | None = None,
    cluster_summaries=None,
    max_queries: int = 1000,
    top_n: int = 800,
) -> dict:
    if not pages:
        return {"summary": {"status": "no_pages", "pages": 0}, "blocks": [], "opportunities": [], "clusters": [], "pages": []}
    by_page = _paragraphs_by_page(paragraph_records)
    if not by_page:
        return {"summary": {"status": "no_paragraphs", "pages": len(pages)}, "blocks": [], "opportunities": [], "clusters": [], "pages": []}

    cluster_labels_list = list(cluster_labels) if cluster_labels is not None else [0] * len(pages)
    cluster_names = _cluster_label_lookup(cluster_summaries)
    queries = _query_rows(pages, coverage, paragraph_fanout, search_payload, max_queries=max_queries)
    block_lookup: dict[tuple[str, int, str], dict] = {}
    opportunities: list[dict] = []
    cluster_rollup: dict[int, dict] = {}
    page_rollup: dict[str, dict] = {}

    def cluster_payload(cid: int) -> dict:
        return cluster_rollup.setdefault(cid, {
            "cluster": cid,
            "label": cluster_names.get(cid, f"cluster {cid}"),
            "queries": 0,
            "traffic": 0,
            "strong_answer_blocks": 0,
            "opportunity_queries": 0,
            "recommended_formats": Counter(),
            "top_queries": [],
            "best_scores": [],
        })

    for query in queries:
        page_i = int(query["page_index"])
        if page_i >= len(pages) or page_i >= len(extracted_pages):
            continue
        page = pages[page_i]
        ext = extracted_pages[page_i]
        cid = _cluster_for(page_i, cluster_labels_list)
        label = cluster_names.get(cid, f"cluster {cid}")
        candidates = by_page.get(page_i) or []
        if not candidates:
            continue
        best: dict | None = None
        for para_i, text in candidates:
            heading = _heading_for_paragraph(ext, para_i)
            row = _build_block_row(page, ext, page_i, para_i, text, heading, query, cid, label)
            if best is None or float(row["score"]) > float(best["score"]):
                best = row
        if best is None:
            continue
        key = (best["url"], int(best["paragraph_index"]), str(best["query"]).lower())
        block_lookup[key] = best

        cluster = cluster_payload(cid)
        cluster["queries"] += 1
        cluster["traffic"] += _to_int(query.get("traffic"))
        cluster["best_scores"].append(float(best["score"]))
        if len(cluster["top_queries"]) < 8:
            cluster["top_queries"].append({
                "query": query.get("query"),
                "traffic": _to_int(query.get("traffic")),
                "position": _to_int(query.get("position")),
                "best_score": best["score"],
                "recommended_format": best["recommended_format"],
            })
        if float(best["score"]) >= 70:
            cluster["strong_answer_blocks"] += 1
        else:
            cluster["opportunity_queries"] += 1
            cluster["recommended_formats"][best["recommended_format"]] += max(1, _to_int(query.get("traffic")) or 1)
            opportunities.append({
                "query": query.get("query") or "",
                "source": query.get("source") or "",
                "url": page.url,
                "title": page.title,
                "section": page.section,
                "cluster": cid,
                "cluster_label": label,
                "recommended_format": best["recommended_format"],
                "current_answer_type": best["answer_type"],
                "best_score": best["score"],
                "keyword_traffic": _to_int(query.get("traffic")),
                "keyword_volume": _to_int(query.get("volume")),
                "keyword_position": _to_int(query.get("position")),
                "schema_recommendation": (best.get("schema_opportunities") or [""])[0],
                "reason": "; ".join(best.get("risks") or ["no strong answer block detected"]),
                "candidate_heading": best.get("heading", ""),
                "candidate_excerpt": best.get("excerpt", ""),
            })

        p = page_rollup.setdefault(page.url, {
            "url": page.url,
            "title": page.title,
            "section": page.section,
            "cluster": cid,
            "cluster_label": label,
            "query_count": 0,
            "strong_blocks": 0,
            "best_score": 0.0,
            "best_answer_type": "",
            "formats": Counter(),
            "opportunities": 0,
        })
        p["query_count"] += 1
        p["best_score"] = max(float(p["best_score"]), float(best["score"]))
        if float(best["score"]) >= 70:
            p["strong_blocks"] += 1
        else:
            p["opportunities"] += 1
        p["formats"][best["answer_type"]] += 1
        if float(best["score"]) >= float(p["best_score"]):
            p["best_answer_type"] = best["answer_type"]

    # Add standalone high-quality answer candidates from pages without ranking-query rows.
    queried_pages = {int(q["page_index"]) for q in queries}
    for page_i, page in enumerate(pages):
        if page_i in queried_pages or page_i >= len(extracted_pages):
            continue
        ext = extracted_pages[page_i]
        cid = _cluster_for(page_i, cluster_labels_list)
        label = cluster_names.get(cid, f"cluster {cid}")
        pseudo_query = {"query": page.title or page.url, "source": "page_title", "traffic": 0, "volume": 0, "position": 0}
        page_best: list[dict] = []
        for para_i, text in by_page.get(page_i, []):
            heading = _heading_for_paragraph(ext, para_i)
            row = _build_block_row(page, ext, page_i, para_i, text, heading, pseudo_query, cid, label)
            if float(row["score"]) >= 70:
                page_best.append(row)
        page_best.sort(key=lambda r: float(r["score"]), reverse=True)
        for row in page_best[:2]:
            block_lookup[(row["url"], int(row["paragraph_index"]), "")] = row

    blocks = sorted(block_lookup.values(), key=lambda r: (float(r["score"]), _to_int(r.get("keyword_traffic"))), reverse=True)
    opportunities.sort(key=lambda r: (_to_int(r.get("keyword_traffic")), -float(r.get("best_score", 0.0))), reverse=True)

    clusters: list[dict] = []
    for raw in cluster_rollup.values():
        queries_count = int(raw["queries"])
        strong = int(raw["strong_answer_blocks"])
        scores = raw["best_scores"] or []
        formats = raw["recommended_formats"]
        recommended = formats.most_common(1)[0][0] if formats else ""
        share = strong / queries_count if queries_count else 0.0
        clusters.append({
            "cluster": raw["cluster"],
            "label": raw["label"],
            "queries": queries_count,
            "traffic": raw["traffic"],
            "strong_answer_blocks": strong,
            "opportunity_queries": raw["opportunity_queries"],
            "strong_query_share": round(share, 4),
            "avg_best_score": round(sum(scores) / max(len(scores), 1), 2),
            "status": "strong" if share >= 0.7 else ("partial" if strong else "gap"),
            "recommended_format": recommended,
            "top_queries": raw["top_queries"],
        })
    clusters.sort(key=lambda r: (r["strong_query_share"], -r["traffic"], r["label"]))

    page_rows = []
    for row in page_rollup.values():
        formats = row.pop("formats")
        row["formats"] = [{"format": fmt, "count": count} for fmt, count in formats.most_common()]
        row["best_score"] = round(float(row["best_score"]), 2)
        page_rows.append(row)
    page_rows.sort(key=lambda r: (float(r.get("best_score", 0.0)), -int(r.get("query_count", 0))))

    search_meta = (search_payload or {}).get("meta", {}) or {}
    search_summary = (search_payload or {}).get("summary", {}) or {}
    summary = {
        "status": "ok",
        "model": "answer_blocks_v1",
        "pages": len(pages),
        "queries": len(queries),
        "blocks": len(blocks),
        "strong_blocks": sum(1 for r in blocks if float(r.get("score", 0.0)) >= 70),
        "opportunity_queries": len(opportunities),
        "top_query_clusters": len(clusters),
        "strong_query_clusters": sum(1 for r in clusters if r["status"] == "strong"),
        "avg_best_score": round(sum(float(r.get("score", 0.0)) for r in blocks) / max(len(blocks), 1), 2),
        "provider": search_meta.get("provider_label") or search_summary.get("provider_label") or "site queries",
    }
    return {
        "summary": summary,
        "blocks": blocks[:top_n],
        "opportunities": opportunities[:top_n],
        "clusters": clusters[:top_n],
        "pages": page_rows[:top_n],
        "interpretation": {
            "score": "0-100 heuristic for snippet-ready answer blocks. It rewards directness, concision, entity clarity, heading/query alignment, schema compatibility, and answer-format signals.",
            "strong_threshold": 70,
        },
    }
