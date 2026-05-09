"""Mine repeatable internal-link patterns from high-performing pages."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from urllib.parse import urlparse


_GENERIC_ANCHORS = {
    "click here", "click", "here", "more", "read more", "learn more",
    "details", "see more", "view more", "view all", "this", "this page",
    "link", "open", "go", "next", "previous",
}


def _safe_int(value, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _canonical(url: str) -> str:
    try:
        p = urlparse(url)
        path = p.path or "/"
        if path != "/" and path.endswith("/"):
            path = path[:-1]
        return f"{p.scheme}://{p.netloc.lower()}{path}".rstrip("/")
    except Exception:
        return str(url or "").rstrip("/")


def _directory(url: str) -> str:
    try:
        path = urlparse(url).path or "/"
    except Exception:
        path = "/"
    if path == "/":
        return "/"
    parts = [p for p in path.split("/") if p]
    return f"/{parts[0]}/" if parts else "/"


def _depth(url: str) -> int:
    try:
        return len([p for p in (urlparse(url).path or "/").split("/") if p])
    except Exception:
        return 0


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9][a-z0-9-]{2,}", str(text or "").lower()))


def _anchor_form(anchor: str, target_title: str = "", target_keyword: str = "") -> str:
    clean = re.sub(r"\s+", " ", str(anchor or "").strip())
    lower = clean.lower()
    if not clean:
        return "empty"
    if lower in _GENERIC_ANCHORS or len(lower) <= 2:
        return "generic"
    words = re.findall(r"[a-z0-9]+", lower)
    title = str(target_title or "").lower()
    keyword = str(target_keyword or "").lower()
    if title and (lower in title or title in lower):
        return "title_match"
    if keyword and (_tokens(lower) & _tokens(keyword)):
        return "keyword_phrase"
    if len(words) <= 2:
        return "short_phrase"
    if len(words) >= 6:
        return "long_descriptive"
    return "descriptive"


def _depth_relation(source: str, target: str) -> str:
    diff = _depth(target) - _depth(source)
    if diff >= 2:
        return "much_deeper"
    if diff == 1:
        return "deeper"
    if diff == 0:
        return "same_depth"
    if diff == -1:
        return "shallower"
    return "much_shallower"


def _semantic_bucket(similarity: float) -> str:
    sim = _safe_float(similarity)
    if sim >= 0.78:
        return "very_close"
    if sim >= 0.62:
        return "close"
    if sim >= 0.45:
        return "loose"
    return "distant"


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    i = max(0, min(len(s) - 1, int(round(p * (len(s) - 1)))))
    return float(s[i])


def _page_type_lookup(page_types: dict | None) -> dict[str, str]:
    out = {}
    for row in (page_types or {}).get("per_page") or []:
        if row.get("url"):
            out[_canonical(row["url"])] = row.get("page_type") or "other"
    return out


def _authority_lookup(linkgraph: dict | None) -> dict[str, dict]:
    rows = (((linkgraph or {}).get("traffic_weighted_pagerank") or {}).get("pages") or [])
    if not rows:
        rows = (linkgraph or {}).get("top_authority_pages") or []
    return {row.get("url"): row for row in rows if row.get("url")}


def _contextual_lookup(linkgraph: dict | None) -> dict[tuple[str, str, str], dict]:
    out = {}
    for row in (((linkgraph or {}).get("contextual_link_impact") or {}).get("links") or []):
        key = (row.get("source_url") or "", row.get("target_url") or "", str(row.get("anchor") or "").strip().lower())
        current = out.get(key)
        if current is None or _safe_float(row.get("contextual_link_impact")) > _safe_float(current.get("contextual_link_impact")):
            out[key] = row
    return out


def _performance_score(row: dict) -> float:
    traffic = _safe_int(row.get("traffic"))
    keywords = _safe_int(row.get("keywords"))
    return (
        math.log1p(traffic) * 14.0
        + math.sqrt(max(0, keywords)) * 5.0
        + _safe_float(row.get("weighted_pagerank_percentile")) * 34.0
        + _safe_float(row.get("pagerank_percentile")) * 18.0
        + _safe_float(row.get("link_flow_percentile")) * 12.0
        + _safe_float(row.get("pagerank")) * 100.0
    )


def _rule_label(rule: dict) -> str:
    return (
        f"{rule['source_page_type']} pages link to {rule['target_page_type']} pages "
        f"with {rule['anchor_form'].replace('_', ' ')} anchors in {rule['context_type'].replace('_', ' ')} "
        f"({rule['cluster_relation'].replace('_', ' ')}, {rule['depth_relation'].replace('_', ' ')})"
    )


def build_internal_link_patterns(
    pages,
    extracted_pages,
    *,
    page_types: dict | None = None,
    linkgraph: dict | None = None,
    min_segment_size: int = 3,
    top_n: int = 120,
) -> dict:
    if not pages or not extracted_pages:
        return {"summary": {"status": "no_pages", "patterns": 0, "recommendations": 0}, "patterns": [], "recommendations": []}

    by_canonical = {_canonical(p.url): p for p in pages}
    extracted_by_canonical = {_canonical(p.url): p for p in extracted_pages}
    type_by_url = _page_type_lookup(page_types)
    authority_by_url = _authority_lookup(linkgraph)
    contextual_by_edge = _contextual_lookup(linkgraph)

    page_rows: dict[str, dict] = {}
    for page in pages:
        auth = authority_by_url.get(page.url) or {}
        page_type = type_by_url.get(_canonical(page.url), "other")
        row = {
            "url": page.url,
            "title": page.title,
            "page_type": page_type,
            "cluster": auth.get("cluster") or page.section or _directory(page.url),
            "directory": auth.get("directory") or _directory(page.url),
            "traffic": _safe_int(auth.get("traffic")),
            "keywords": _safe_int(auth.get("keywords")),
            "top_keyword": auth.get("top_keyword") or "",
            "pagerank": _safe_float(auth.get("pagerank")),
            "pagerank_percentile": _safe_float(auth.get("pagerank_percentile")),
            "weighted_pagerank_percentile": _safe_float(auth.get("weighted_pagerank_percentile")),
            "link_flow_percentile": _safe_float(auth.get("link_flow_percentile")),
            "in_degree": _safe_int(auth.get("in_degree")),
            "out_degree": _safe_int(auth.get("out_degree")),
        }
        row["performance_score"] = round(_performance_score(row), 4)
        page_rows[page.url] = row

    raw_links: list[dict] = []
    for source in extracted_pages:
        source_row = page_rows.get(source.url)
        if not source_row:
            continue
        for link in source.link_audit_rows or []:
            if not link.get("is_internal"):
                continue
            target_page = by_canonical.get(_canonical(link.get("target_url") or ""))
            if target_page is None or target_page.url == source.url:
                continue
            target_row = page_rows.get(target_page.url)
            if not target_row:
                continue
            anchor = str(link.get("anchor") or "").strip()
            contextual = (
                contextual_by_edge.get((source.url, target_page.url, anchor.lower()))
                or contextual_by_edge.get((source.url, target_page.url, ""))
                or {}
            )
            context_type = contextual.get("context_type") or ("main_content" if link.get("context") else "template")
            similarity = _safe_float(contextual.get("contextual_similarity") or contextual.get("paragraph_match"))
            cluster_relation = "same_cluster" if source_row["cluster"] == target_row["cluster"] else "cross_cluster"
            directory_relation = "same_directory" if source_row["directory"] == target_row["directory"] else "cross_directory"
            rule = {
                "source_page_type": source_row["page_type"],
                "source_cluster": source_row["cluster"],
                "target_page_type": target_row["page_type"],
                "anchor_form": _anchor_form(anchor, target_row["title"], target_row.get("top_keyword", "")),
                "context_type": context_type,
                "cluster_relation": cluster_relation,
                "directory_relation": directory_relation,
                "depth_relation": _depth_relation(source.url, target_page.url),
                "semantic_distance_bucket": _semantic_bucket(similarity),
            }
            rule_key = "|".join(str(rule[k]) for k in (
                "source_page_type", "target_page_type", "anchor_form", "context_type",
                "cluster_relation", "directory_relation", "depth_relation", "semantic_distance_bucket",
            ))
            raw_links.append({
                **rule,
                "rule_key": rule_key,
                "source_url": source.url,
                "source_title": source_row["title"],
                "target_url": target_page.url,
                "target_title": target_row["title"],
                "anchor": anchor,
                "context": link.get("context") or "",
                "contextual_similarity": round(similarity, 4),
                "source_performance_score": source_row["performance_score"],
                "target_traffic": target_row["traffic"],
                "target_keywords": target_row["keywords"],
            })

    if not raw_links:
        return {
            "summary": {"status": "no_links", "total_pages": len(page_rows), "patterns": 0, "recommendations": 0},
            "patterns": [],
            "recommendations": [],
            "links": [],
        }

    page_pattern_keys: dict[str, set[str]] = defaultdict(set)
    page_links_by_pattern: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    linked_targets_by_source: dict[str, set[str]] = defaultdict(set)
    for link in raw_links:
        page_pattern_keys[link["source_url"]].add(link["rule_key"])
        page_links_by_pattern[link["rule_key"]][link["source_url"]].append(link)
        linked_targets_by_source[link["source_url"]].add(link["target_url"])

    by_type: dict[str, list[dict]] = defaultdict(list)
    by_type_cluster: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in page_rows.values():
        by_type[row["page_type"]].append(row)
        by_type_cluster[(row["page_type"], row["cluster"])].append(row)

    segments: list[dict] = []
    clustered_page_types: set[str] = set()
    for (page_type, cluster), rows in by_type_cluster.items():
        if len(rows) >= min_segment_size:
            segments.append({"mode": "page_type_cluster", "page_type": page_type, "cluster": cluster, "rows": rows})
            clustered_page_types.add(page_type)
    for page_type, rows in by_type.items():
        if len(rows) >= min_segment_size and page_type not in clustered_page_types:
            segments.append({"mode": "page_type", "page_type": page_type, "cluster": "all", "rows": rows})
    if not segments:
        segments = [{"mode": "site", "page_type": "all", "cluster": "all", "rows": list(page_rows.values())}]

    target_candidates_by_type: dict[str, list[dict]] = defaultdict(list)
    for row in sorted(page_rows.values(), key=lambda r: (r["performance_score"], r["traffic"]), reverse=True):
        target_candidates_by_type[row["page_type"]].append(row)

    patterns: list[dict] = []
    recommendations: list[dict] = []
    seen_patterns: set[tuple[str, str, str]] = set()
    for segment in segments:
        rows = sorted(segment["rows"], key=lambda r: r["performance_score"], reverse=True)
        if len(rows) < 2:
            continue
        scores = [float(r["performance_score"]) for r in rows]
        top_cut = _percentile(scores, 0.66)
        weak_cut = _percentile(scores, 0.5)
        top_rows = [r for r in rows if float(r["performance_score"]) >= top_cut] or rows[: max(1, len(rows) // 3)]
        weak_rows = [r for r in rows if float(r["performance_score"]) <= weak_cut] or rows[-max(1, len(rows) // 2):]
        top_urls = {r["url"] for r in top_rows}
        weak_urls = {r["url"] for r in weak_rows}
        pattern_counts = Counter()
        for row in top_rows:
            pattern_counts.update(page_pattern_keys.get(row["url"], set()))
        min_support = max(1, min(2, len(top_rows)))
        for rule_key, top_support in pattern_counts.most_common():
            if top_support < min_support:
                continue
            sample_links_all = [link for links in page_links_by_pattern.get(rule_key, {}).values() for link in links]
            if not sample_links_all:
                continue
            rule = {k: sample_links_all[0].get(k) for k in (
                "source_page_type", "target_page_type", "anchor_form", "context_type",
                "cluster_relation", "directory_relation", "depth_relation", "semantic_distance_bucket",
            )}
            if segment["page_type"] != "all" and rule["source_page_type"] != segment["page_type"]:
                continue
            present_urls = set(page_links_by_pattern.get(rule_key, {}).keys())
            top_present = top_urls & present_urls
            weak_present = weak_urls & present_urls
            weak_missing_rows = [r for r in weak_rows if r["url"] not in present_urls]
            if not weak_missing_rows:
                continue
            top_rate = len(top_present) / max(1, len(top_rows))
            weak_rate = len(weak_present) / max(1, len(weak_rows))
            if top_rate < 0.5 or (top_rate - weak_rate) < 0.2:
                continue
            present_scores = [page_rows[url]["performance_score"] for url in present_urls if url in page_rows]
            absent_scores = [r["performance_score"] for r in rows if r["url"] not in present_urls]
            avg_present = sum(present_scores) / max(1, len(present_scores))
            avg_absent = sum(absent_scores) / max(1, len(absent_scores))
            lift = avg_present - avg_absent
            support_count = len(present_urls)
            confidence = min(
                0.98,
                0.25
                + min(0.25, support_count / 12.0)
                + min(0.25, top_rate * 0.25)
                + min(0.18, max(0.0, top_rate - weak_rate) * 0.35)
                + min(0.10, max(0.0, lift) / 100.0),
            )
            pattern_key = (segment["mode"], segment["page_type"], rule_key)
            if pattern_key in seen_patterns:
                continue
            seen_patterns.add(pattern_key)
            anchors = Counter(link["anchor"] for link in sample_links_all if link.get("anchor"))
            sample_links = sorted(sample_links_all, key=lambda r: r["source_performance_score"], reverse=True)[:8]
            pattern_id = f"link_pattern_{len(patterns) + 1}"
            pattern = {
                "pattern_id": pattern_id,
                "rule_key": rule_key,
                **rule,
                "source_segment_mode": segment["mode"],
                "source_cluster": segment["cluster"],
                "inferred_rule": _rule_label(rule),
                "support_count": support_count,
                "top_support_count": len(top_present),
                "weak_support_count": len(weak_present),
                "top_rate": round(top_rate, 3),
                "weak_rate": round(weak_rate, 3),
                "confidence": round(confidence, 3),
                "lift_score_difference": round(lift, 2),
                "avg_score_with_pattern": round(avg_present, 2),
                "avg_score_without_pattern": round(avg_absent, 2),
                "sample_links": [
                    {
                        "source_url": link["source_url"],
                        "source_title": link["source_title"],
                        "target_url": link["target_url"],
                        "target_title": link["target_title"],
                        "anchor": link["anchor"],
                        "context_type": link["context_type"],
                        "contextual_similarity": link["contextual_similarity"],
                    }
                    for link in sample_links
                ],
                "sample_anchors": [{"anchor": anchor, "count": count} for anchor, count in anchors.most_common(6)],
                "affected_weak_pages": [
                    {"url": r["url"], "title": r["title"], "traffic": r["traffic"], "keywords": r["keywords"], "performance_score": r["performance_score"]}
                    for r in weak_missing_rows[:12]
                ],
            }
            patterns.append(pattern)

            sample_targets = []
            seen_targets = set()
            for link in sample_links:
                if link["target_url"] not in seen_targets:
                    seen_targets.add(link["target_url"])
                    sample_targets.append(page_rows[link["target_url"]])
            fallback_targets = target_candidates_by_type.get(str(rule["target_page_type"]), [])
            for weak in weak_missing_rows[:12]:
                candidates = []
                for target in sample_targets + fallback_targets:
                    if target["url"] == weak["url"] or target["url"] in linked_targets_by_source.get(weak["url"], set()):
                        continue
                    if rule["cluster_relation"] == "same_cluster" and target["cluster"] != weak["cluster"]:
                        continue
                    candidates.append(target)
                    if len(candidates) >= 5:
                        break
                if not candidates:
                    continue
                anchor = (patterns[-1]["sample_anchors"][0]["anchor"] if patterns[-1]["sample_anchors"] else "") or candidates[0].get("top_keyword") or candidates[0].get("title") or ""
                recommendations.append({
                    "pattern_id": pattern_id,
                    "source_url": weak["url"],
                    "source_title": weak["title"],
                    "source_page_type": weak["page_type"],
                    "source_cluster": weak["cluster"],
                    "missing_pattern": pattern["inferred_rule"],
                    "target_page_type": rule["target_page_type"],
                    "suggested_anchor": anchor,
                    "suggested_targets": [
                        {"url": c["url"], "title": c["title"], "traffic": c["traffic"], "keywords": c["keywords"], "cluster": c["cluster"]}
                        for c in candidates[:5]
                    ],
                    "confidence": pattern["confidence"],
                    "lift_score_difference": pattern["lift_score_difference"],
                    "traffic": weak["traffic"],
                    "keywords": weak["keywords"],
                    "recommended_action": "Add a contextual internal link that follows this high-performing source-page pattern.",
                })

    patterns.sort(key=lambda r: (r["confidence"], r["lift_score_difference"], r["support_count"]), reverse=True)
    recommendations.sort(key=lambda r: (r["confidence"], r["lift_score_difference"], r["traffic"]), reverse=True)
    pattern_types = Counter(p["source_page_type"] for p in patterns)
    target_types = Counter(p["target_page_type"] for p in patterns)
    return {
        "summary": {
            "status": "ok" if patterns else "no_patterns",
            "model": "internal_link_patterns_v1",
            "total_pages": len(page_rows),
            "total_links": len(raw_links),
            "patterns": len(patterns),
            "recommendations": len(recommendations),
            "page_types_with_patterns": len(pattern_types),
            "target_types_with_patterns": len(target_types),
            "avg_confidence": round(sum(_safe_float(p.get("confidence")) for p in patterns) / max(1, len(patterns)), 4),
            "min_segment_size": min_segment_size,
        },
        "patterns": patterns[:top_n],
        "recommendations": recommendations[:300],
        "links": raw_links[:1000],
        "top_source_page_types": [{"page_type": k, "patterns": v} for k, v in pattern_types.most_common(30)],
        "top_target_page_types": [{"page_type": k, "patterns": v} for k, v in target_types.most_common(30)],
        "interpretation": {
            "confidence": "Support-weighted estimate that high-performing pages in the same source segment use this link pattern more often than weak pages.",
            "recommendations": "Generated only for weak source pages in the same segment that do not already use the mined target-link pattern.",
        },
    }
