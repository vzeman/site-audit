"""Synthesize the analyses into a prioritized action plan.

Every other module in this package answers "what is true about this site".
This module answers "what should the editor *do* next, in what order".

Inputs are the already-built payloads (no embeddings or re-analysis here),
so this stage is cheap. Output is a flat list of :class:`Recommendation`
records grouped by ``category`` and ranked by ``priority`` + per-category
score so the UI can render an action list and a per-category breakdown.

Categories
----------

* ``content_debt`` — duplicates to merge, outliers to refocus, paragraphs
  that belong on a different page.
* ``coverage`` — missing topic pages (gaps) and cannibalization.
* ``geo`` — AI answer-ability fixes (FAQ schema, question headings,
  citations) targeted at high-PR pages first.
* ``linking`` — top page-level + paragraph-level internal-link
  recommendations, orphans to surface, buried pages to lift.
* ``onpage`` — title mismatches, CTR-anomaly title rewrites, generic anchors.

Priority is one of ``high`` / ``medium`` / ``low``. The pickers favour
high-PageRank pages — fixing the load-bearing pages compounds.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Iterable, Optional
from urllib.parse import urlsplit, urlunsplit

from .ai_access import format_blocked_agent_recommendation
from .ctr_curve import estimate_clicks_gain, expected_ctr


@dataclass
class Recommendation:
    id: str
    category: str        # content_debt | coverage | geo | linking | onpage
    priority: str        # high | medium | low
    title: str
    instruction: str
    targets: list[str] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)
    effort: str = "medium"  # quick | medium | deep
    score: float = 0.0
    impact: float = 0.0
    confidence: float = 0.0
    effort_score: float = 0.0
    risk: float = 0.0
    priority_score: float = 0.0
    owner: str = ""
    type: str = ""
    cluster: str = ""
    traffic_opportunity: float = 0.0
    estimated_clicks_gain: float | None = None
    estimate_basis: str | None = None
    suppressed: bool = False
    suppressed_by: str | None = None
    suppressed_reason: str = ""


_PRI_RANK = {"high": 0, "medium": 1, "low": 2}
_EFFORT_SCORE = {"quick": 25.0, "medium": 55.0, "deep": 85.0}
_CATEGORY_CONFIDENCE = {
    "content_debt": 72.0,
    "coverage": 68.0,
    "geo": 62.0,
    "linking": 70.0,
    "onpage": 66.0,
}
_CATEGORY_RISK = {
    "content_debt": 44.0,
    "coverage": 38.0,
    "geo": 22.0,
    "linking": 16.0,
    "onpage": 20.0,
}
_CATEGORY_OWNER = {
    "content_debt": "Content",
    "coverage": "Content strategy",
    "geo": "Content",
    "linking": "SEO",
    "onpage": "SEO",
}
_TYPE_BY_PREFIX = {
    "dup": "merge_duplicate",
    "out": "refocus_outlier",
    "wh": "move_paragraph",
    "gap": "coverage_gap",
    "cann": "cannibalization",
    "geo": "answerability",
    "geo-cite": "citation_gap",
    "geo-access": "ai_crawler_access",
    "link": "internal_link",
    "plink": "paragraph_link",
    "orphan": "orphan_page",
    "deep": "click_depth",
    "title": "title_rewrite",
    "ctr": "title_rewrite",
    "anchor": "anchor_rewrite",
}

SCORE_MODEL = {
    "model": "fix_priority_score_v1",
    "components": {
        "impact": "0-100 blended estimate from issue severity and percentile-normalized modeled clicks gain, traffic, and PageRank.",
        "confidence": "0-100 estimate from recommendation class and evidence completeness.",
        "effort_score": "0-100 estimated implementation effort. Quick edits are lower, deep content/template work is higher.",
        "risk": "0-100 estimated downside risk. Redirects and consolidation are higher risk than links or metadata edits.",
        "priority_score": "0.45*impact + 0.25*confidence - 0.18*effort_score - 0.12*risk, clamped to 0-100.",
        "traffic_opportunity": "Sum of modeled estimated_clicks_gain when available, falling back to the legacy traffic index for actions without a clicks model.",
    },
    "priority_thresholds": {"high": 60, "medium": 35, "low": 0},
}


def _clip(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def _safe_float(value, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return default


def _safe_int(value, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(float(str(value).replace(",", "")))
    except (TypeError, ValueError):
        return default


def _url_keys(url: object) -> set[str]:
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


def _merge_query_pages(left: list[dict] | None, right: list[dict] | None) -> list[dict]:
    rows = list(left or []) + list(right or [])
    rows.sort(key=lambda r: _safe_float(r.get("impressions")), reverse=True)
    return rows[:3]


def _store_context(out: dict[str, dict], url: object, context: dict) -> None:
    for key in _url_keys(url):
        current = out.get(key)
        if current is None:
            out[key] = dict(context)
            continue
        merged = dict(current)
        incoming_traffic = _safe_float(context.get("traffic"))
        current_traffic = _safe_float(current.get("traffic"))
        prefer_incoming = incoming_traffic > current_traffic
        for name, value in context.items():
            if name == "query_pages":
                merged[name] = _merge_query_pages(current.get(name), value)
                continue
            if value in (None, "", [], {}):
                continue
            if name in {"traffic", "keywords", "pagerank", "authority_gap", "top_keyword_volume"}:
                merged[name] = max(_safe_float(current.get(name)), _safe_float(value))
                continue
            if name == "top_keyword_position":
                if prefer_incoming or not _safe_float(current.get(name)):
                    merged[name] = _safe_float(value)
                continue
            if prefer_incoming or not merged.get(name):
                merged[name] = value
        out[key] = merged


def _target_context_lookup(linkgraph_payload: dict | None, search_payload: dict | None) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in ((search_payload or {}).get("top_pages") or []):
        url = row.get("matched_url") or row.get("url")
        if not url:
            continue
        _store_context(out, url, {
            "traffic": _safe_float(row.get("traffic")),
            "keywords": _safe_int(row.get("keywords")),
            "cluster": row.get("cluster_label") or row.get("cluster") or row.get("section") or "",
            "top_keyword": row.get("top_keyword") or "",
            "top_keyword_position": _safe_float(row.get("top_keyword_position") or row.get("position")),
            "top_keyword_volume": _safe_float(
                row.get("top_keyword_volume")
                or row.get("top_keyword_impressions")
                or row.get("impressions")
            ),
            "top_keyword_source": str(row.get("provider") or row.get("source") or ""),
        })
    for row in (((linkgraph_payload or {}).get("traffic_weighted_pagerank") or {}).get("pages") or []):
        url = row.get("url")
        if not url:
            continue
        _store_context(out, url, {
            "traffic": _safe_float(row.get("traffic")),
            "keywords": _safe_int(row.get("keywords")),
            "cluster": row.get("cluster") or row.get("section") or row.get("directory") or "",
            "pagerank": _safe_float(row.get("pagerank")),
            "authority_gap": _safe_float(row.get("authority_traffic_gap")),
        })
    for row in ((search_payload or {}).get("query_pages") or []):
        url = row.get("matched_url") or row.get("url")
        if not url:
            continue
        query = row.get("query") or row.get("keyword") or ""
        if not query:
            continue
        _store_context(out, url, {
            "query_pages": [{
                "query": query,
                "position": _safe_float(row.get("position")),
                "impressions": _safe_float(row.get("impressions") or row.get("volume")),
            }],
        })
    return out


def _lookup_context(context: dict[str, dict], targets: Iterable[str]) -> dict:
    best: dict = {}
    for target in targets:
        for key in _url_keys(target):
            row = context.get(key)
            if row and _safe_float(row.get("traffic")) >= _safe_float(best.get("traffic")):
                best = row
    return best


def _rec_type(rec: Recommendation) -> str:
    if rec.id.startswith("geo-aio"):
        return "ai_overview_citation"
    if rec.id.startswith("geo-access"):
        return "ai_crawler_access"
    prefix = rec.id.split("-", 1)[0]
    if rec.id.startswith("geo-cite"):
        return "citation_gap"
    return _TYPE_BY_PREFIX.get(prefix, rec.category)


def _component_scores(rec: Recommendation, target_context: dict) -> dict:
    evidence = rec.evidence or {}
    traffic_terms = [
        _safe_float(target_context.get("traffic")),
        _safe_float(evidence.get("traffic")),
        _safe_float(evidence.get("target_traffic")),
    ]
    if rec.estimated_clicks_gain is None:
        # A first-class gain is blended separately; counting the evidence copy
        # here as well would double-weight the same clicks.
        traffic_terms.append(_safe_float(evidence.get("traffic_opportunity")))
        traffic_terms.append(_safe_float(evidence.get("estimated_clicks_gain")))
    traffic = max(traffic_terms)
    pagerank = max(_safe_float(evidence.get("pagerank")), _safe_float(target_context.get("pagerank")))
    severity = max(
        _safe_float(rec.score),
        _safe_float(evidence.get("similarity")) * 100.0,
        _safe_float(evidence.get("lift")) * 200.0,
        _safe_float(evidence.get("lift_to_suggested")) * 180.0,
        max(0.0, 4.0 - _safe_float(evidence.get("answerability_score"), 4.0)) * 18.0,
        max(0.0, 0.55 - _safe_float(evidence.get("title_to_content"), 0.55)) * 140.0,
        _safe_float(evidence.get("candidates_above_threshold")) * 14.0,
        _safe_float(evidence.get("generic_anchor_share")) * 80.0,
        _safe_float(evidence.get("click_depth")) * 6.0,
    )
    legacy_impact = _clip(
        min(62.0, severity)
        + min(25.0, math.log1p(max(traffic, 0.0)) * 3.2)
        + min(18.0, pagerank * 2500.0)
        + min(14.0, _safe_float(target_context.get("authority_gap")) * 14.0)
    )
    evidence_count = sum(1 for value in evidence.values() if value not in (None, "", [], {}))
    confidence = _clip(_CATEGORY_CONFIDENCE.get(rec.category, 62.0) + min(16.0, evidence_count * 2.5))
    effort_score = _clip(_EFFORT_SCORE.get(rec.effort, 55.0) + max(0, len([t for t in rec.targets if t]) - 1) * 5.0)
    risk = _CATEGORY_RISK.get(rec.category, 25.0)
    if rec.id.startswith("dup") or rec.id.startswith("cann"):
        risk += 18.0
    if rec.effort == "deep":
        risk += 8.0
    if len([t for t in rec.targets if t]) > 1:
        risk += 6.0
    risk = _clip(risk)
    return {
        "severity_norm": _clip(severity),
        "legacy_impact": round(legacy_impact, 1),
        "confidence": round(confidence, 1),
        "effort_score": round(effort_score, 1),
        "risk": round(risk, 1),
        "traffic_opportunity": round(traffic, 1),
        "traffic": traffic,
        "pagerank": pagerank,
        "cluster": str(target_context.get("cluster") or evidence.get("query") or ""),
    }


def _estimated_clicks_gain(context: dict, category: str) -> tuple[float | None, str | None]:
    """Return modeled incremental clicks and the data source used."""
    if category == "coverage_gap":
        # Only the gap query's own volume counts here — the target context
        # belongs to the nearest *existing* page, not to the missing one.
        volume = _safe_float(context.get("volume"))
        if volume > 0:
            return volume * expected_ctr(5.0), "keyword_volume"
        return None, None

    query_rows = [
        row for row in (context.get("query_pages") or [])
        if _safe_float(row.get("position")) > 3.0 and _safe_float(row.get("impressions")) > 0.0
    ]
    if query_rows:
        row = sorted(query_rows, key=lambda r: _safe_float(r.get("impressions")), reverse=True)[0]
        position = _safe_float(row.get("position"))
        gain = estimate_clicks_gain(
            _safe_float(row.get("impressions")),
            position,
            max(3.0, position * 0.5),
        )
        return gain, "gsc_impressions"

    position = _safe_float(context.get("top_keyword_position"))
    volume = _safe_float(context.get("top_keyword_volume"))
    if position > 3.0 and volume > 0.0:
        gain = estimate_clicks_gain(volume, position, max(3.0, position * 0.5))
        source = str(context.get("top_keyword_source") or "").lower()
        return gain, ("gsc_impressions" if source == "gsc" else "keyword_volume")
    return None, None


def _estimate_context(rec: Recommendation, target_context: dict) -> dict:
    evidence = rec.evidence or {}
    if rec.id.startswith("gap-"):
        return {
            **target_context,
            "volume": max(
                _safe_float(evidence.get("volume")),
                _safe_float(evidence.get("impressions")),
                _safe_float(evidence.get("top_keyword_volume")),
                _safe_float(evidence.get("search_volume")),
            ),
        }
    return target_context


def _apply_estimated_gain(rec: Recommendation, target_context: dict) -> None:
    evidence_gain = _safe_float((rec.evidence or {}).get("estimated_clicks_gain"), default=0.0)
    if evidence_gain > 0:
        rec.estimated_clicks_gain = round(evidence_gain, 2)
        rec.estimate_basis = (
            rec.estimate_basis
            or (rec.evidence or {}).get("estimate_basis")
            or "gsc_impressions"
        )
        return
    gain, basis = _estimated_clicks_gain(_estimate_context(rec, target_context), _rec_type(rec))
    if gain is None or basis is None:
        return
    rec.estimated_clicks_gain = round(gain, 2)
    rec.estimate_basis = basis


def _append_top_query_instruction(rec: Recommendation, target_context: dict) -> None:
    if rec.id.startswith(("gap-", "cann-")):
        # These recs target a query, not an existing page — the neighbor
        # page's top keyword would be misleading here.
        return
    keyword = str(target_context.get("top_keyword") or "").strip()
    position = _safe_float(target_context.get("top_keyword_position"))
    volume = _safe_float(target_context.get("top_keyword_volume"))
    if not keyword or position <= 0.0 or volume <= 0.0:
        return
    if " Top query: " in rec.instruction:
        return
    source = str(target_context.get("top_keyword_source") or "").lower()
    demand = (
        f"{int(round(volume))} impressions/period"
        if source == "gsc"
        else f"{int(round(volume))}/mo"
    )
    rec.instruction = (
        f'{rec.instruction} Top query: "{keyword}" '
        f"(position {position:.0f}, {demand})."
    )


def _percentile_norm(values: dict[str, float]) -> dict[str, float]:
    positives = [(rec_id, value) for rec_id, value in values.items() if value > 0.0]
    if not positives:
        return {}
    ordered = sorted(positives, key=lambda item: (item[1], item[0]))
    n = len(ordered)
    return {rec_id: (rank / n) * 100.0 for rank, (rec_id, _value) in enumerate(ordered, start=1)}


def _gain_label(gain: float | None) -> str:
    if gain is None:
        return ""
    return f"≈ +{gain:.0f} clicks/period"


def _item_row(r: Recommendation) -> dict:
    return {
        "id": r.id,
        "category": r.category,
        "type": r.type,
        "priority": r.priority,
        "priority_score": r.priority_score,
        "impact": r.impact,
        "confidence": r.confidence,
        "effort_score": r.effort_score,
        "risk": r.risk,
        "owner": r.owner,
        "cluster": r.cluster,
        "traffic_opportunity": r.traffic_opportunity,
        "estimated_clicks_gain": r.estimated_clicks_gain,
        "estimate_basis": r.estimate_basis,
        "gain_label": _gain_label(r.estimated_clicks_gain),
        "title": r.title,
        "instruction": r.instruction,
        "targets": r.targets,
        "evidence": r.evidence,
        "effort": r.effort,
        "score": r.score,
    }


def _suppressed_row(r: Recommendation) -> dict:
    return {
        **_item_row(r),
        "suppressed": r.suppressed,
        "suppressed_by": r.suppressed_by,
        "suppressed_reason": r.suppressed_reason,
    }


def _priority_bucket(priority_score: float) -> str:
    if priority_score >= 60.0:
        return "high"
    if priority_score >= 35.0:
        return "medium"
    return "low"


def _finalize(recs: list[Recommendation], *, linkgraph_payload: dict | None = None, search_payload: dict | None = None) -> list[Recommendation]:
    context = _target_context_lookup(linkgraph_payload, search_payload)
    base_scores: dict[str, dict] = {}
    gain_values: dict[str, float] = {}
    traffic_values: dict[str, float] = {}
    pagerank_values: dict[str, float] = {}
    for rec in recs:
        target_context = _lookup_context(context, rec.targets)
        _apply_estimated_gain(rec, target_context)
        _append_top_query_instruction(rec, target_context)
        scores = _component_scores(rec, target_context)
        base_scores[rec.id] = scores
        gain_values[rec.id] = _safe_float(rec.estimated_clicks_gain)
        traffic_values[rec.id] = _safe_float(scores.get("traffic"))
        pagerank_values[rec.id] = _safe_float(scores.get("pagerank"))

    has_search_payload = bool((search_payload or {}).get("top_pages") or (search_payload or {}).get("query_pages"))
    has_business_data = has_search_payload or any(value > 0.0 for value in gain_values.values())
    gain_norms = _percentile_norm(gain_values) if has_business_data else {}
    traffic_norms = _percentile_norm(traffic_values) if has_business_data else {}
    pagerank_norms = _percentile_norm(pagerank_values) if has_business_data else {}

    for rec in recs:
        scores = base_scores[rec.id]
        if has_business_data:
            business_norm = (
                0.6 * gain_norms.get(rec.id, 0.0)
                + 0.25 * traffic_norms.get(rec.id, 0.0)
                + 0.15 * pagerank_norms.get(rec.id, 0.0)
            )
            rec.impact = round(_clip(0.5 * _safe_float(scores.get("severity_norm")) + 0.5 * business_norm), 1)
        else:
            # Without any business data the percentile blend is all zeros;
            # fall back to the legacy severity+linkgraph impact so ordering
            # and priority buckets match pre-model behaviour.
            rec.impact = _safe_float(scores.get("legacy_impact"))
        rec.confidence = scores["confidence"]
        rec.effort_score = scores["effort_score"]
        rec.risk = scores["risk"]
        rec.priority_score = round(_clip(0.45 * rec.impact + 0.25 * rec.confidence - 0.18 * rec.effort_score - 0.12 * rec.risk), 1)
        rec.priority = _priority_bucket(rec.priority_score)
        rec.owner = rec.owner or _CATEGORY_OWNER.get(rec.category, "SEO")
        rec.type = rec.type or _rec_type(rec)
        rec.cluster = rec.cluster or scores["cluster"]
        rec.traffic_opportunity = (
            round(float(rec.estimated_clicks_gain), 1)
            if rec.estimated_clicks_gain is not None
            else scores["traffic_opportunity"]
        )
        rec.evidence = {
            **(rec.evidence or {}),
            "score_components": {
                "impact": rec.impact,
                "confidence": rec.confidence,
                "effort_score": rec.effort_score,
                "risk": rec.risk,
                "priority_score": rec.priority_score,
            },
        }
    recs.sort(key=lambda r: (-r.priority_score, _PRI_RANK.get(r.priority, 9), -r.impact, r.effort_score, r.id))
    return recs


def _removal_targets(rec: Recommendation) -> list[str]:
    if rec.id.startswith("dup-") and len(rec.targets) > 1:
        return [rec.targets[1]]
    if rec.id.startswith("cann-") and len(rec.targets) > 1:
        return [target for target in rec.targets[1:] if target]
    return []


def _primary_target_keys(rec: Recommendation) -> set[str]:
    if not rec.targets:
        return set()
    return _url_keys(rec.targets[0])


def _improves_page(rec: Recommendation) -> bool:
    return bool(rec.targets) and not rec.id.startswith("gap-") and rec.type != "coverage_gap"


def _resolve_conflicts(recommendations: Iterable[Recommendation]) -> tuple[list[Recommendation], list[Recommendation]]:
    recs = list(recommendations)
    for rec in recs:
        rec.suppressed = False
        rec.suppressed_by = None
        rec.suppressed_reason = ""

    removal_by_url: dict[str, Recommendation] = {}
    removal_recs = sorted(
        (rec for rec in recs if _removal_targets(rec)),
        key=lambda r: (-_safe_float(r.priority_score), r.id),
    )
    for rec in removal_recs:
        for target in _removal_targets(rec):
            for key in _url_keys(target):
                removal_by_url.setdefault(key, rec)

    kept: list[Recommendation] = []
    suppressed: list[Recommendation] = []
    for rec in recs:
        if _removal_targets(rec):
            kept.append(rec)
            continue
        if not _improves_page(rec):
            kept.append(rec)
            continue
        suppressor: Recommendation | None = None
        reason = ""
        for key in sorted(_primary_target_keys(rec)):
            if key in removal_by_url:
                suppressor = removal_by_url[key]
                reason = f"page is slated for redirect/merge by {suppressor.id}"
                break
        if suppressor is None and rec.id.startswith(("link-", "plink-")):
            # Link recs edit targets[0] but point at targets[1]; a link into
            # a page slated for removal is wasted work too.
            for destination in rec.targets[1:]:
                for key in sorted(_url_keys(destination)):
                    if key in removal_by_url:
                        suppressor = removal_by_url[key]
                        reason = f"link destination is slated for redirect/merge by {suppressor.id}"
                        break
                if suppressor is not None:
                    break
        if suppressor is None:
            kept.append(rec)
            continue
        rec.suppressed = True
        rec.suppressed_by = suppressor.id
        rec.suppressed_reason = reason
        suppressed.append(rec)
    return kept, suppressed


def _card_title(members: list[Recommendation], url: str, query: str) -> str:
    for rec in members:
        evidence = rec.evidence or {}
        for key in (
            "page_title",
            "current_title",
            "title",
            "target_title",
            "source_title",
            "best_title",
            "canonical_title",
        ):
            value = str(evidence.get(key) or "").strip()
            if value:
                return value
    if query:
        return query
    return url


def _card_group(rec: Recommendation) -> tuple[str, str, str, list[str]]:
    evidence = rec.evidence or {}
    if rec.id.startswith("gap-") or rec.type == "coverage_gap":
        query = str(evidence.get("query") or rec.title or "").strip()
        return f"new-content:{query}", "", query, [target for target in rec.targets if target]
    url = rec.targets[0] if rec.targets else ""
    key = next(iter(sorted(_url_keys(url))), url) if url else f"site-wide:{rec.id}"
    return key, url, "", [target for target in rec.targets[1:] if target]


_EFFORT_RANK = {"quick": 0, "medium": 1, "deep": 2}


def _effort_total_label(members: list[Recommendation]) -> str:
    # The card label is the heaviest member's effort — summing scores would
    # make three quick edits read as "deep". The summed score is exposed
    # separately as effort_total_score.
    ranked = [rec.effort for rec in members if rec.effort in _EFFORT_RANK]
    if not ranked:
        return "medium"
    return max(ranked, key=lambda label: _EFFORT_RANK[label])


def to_cards(recommendations: Iterable[Recommendation]) -> list[dict]:
    groups: dict[str, dict] = {}
    for rec in recommendations:
        key, url, query, related = _card_group(rec)
        group = groups.get(key)
        if group is None:
            group = {
                "url": url,
                "query": query,
                "members": [],
                "related_urls": [],
            }
            groups[key] = group
        for related_url in related:
            if related_url and related_url != group["url"] and related_url not in group["related_urls"]:
                group["related_urls"].append(related_url)
        group["members"].append(rec)

    cards: list[dict] = []
    for group in groups.values():
        members = group["members"]
        gains = [float(rec.estimated_clicks_gain) for rec in members if rec.estimated_clicks_gain is not None]
        total_gain = round(sum(gains), 2) if gains else None
        top_priority_score = max(float(rec.priority_score or 0.0) for rec in members)
        top_priority = sorted({rec.priority for rec in members}, key=lambda p: _PRI_RANK.get(p, 9))[0]
        effort_score_total = round(sum(float(rec.effort_score or 0.0) for rec in members), 1)
        cards.append({
            "url": group["url"],
            "query": group["query"],
            "title": _card_title(members, group["url"], group["query"]),
            "related_urls": sorted(group["related_urls"]),
            "total_estimated_clicks_gain": total_gain,
            "top_priority": top_priority,
            "top_priority_score": top_priority_score,
            "categories": sorted({rec.category for rec in members}),
            "recommendation_ids": [rec.id for rec in members],
            "recommendations": [_item_row(rec) for rec in members],
            "effort_total": _effort_total_label(members),
            "effort_total_score": effort_score_total,
        })
    cards.sort(key=lambda card: (
        _PRI_RANK.get(card["top_priority"], 9),
        -(card["total_estimated_clicks_gain"] if card["total_estimated_clicks_gain"] is not None else -1.0),
        -float(card["top_priority_score"] or 0.0),
        card["url"] or f"new-content:{card.get('query', '')}",
    ))
    return cards


def _pr_lookup(linkgraph_payload: dict | None) -> dict[str, float]:
    if not linkgraph_payload:
        return {}
    out: dict[str, float] = {}
    for row in linkgraph_payload.get("top_authority_pages", []) or []:
        out[row["url"]] = float(row.get("pagerank", 0.0))
    return out


def _orphan_set(linkgraph_payload: dict | None) -> set[str]:
    if not linkgraph_payload:
        return set()
    return {r["url"] for r in linkgraph_payload.get("orphans", []) or []}


def _content_debt(
    duplicates_rows: list[dict],
    outliers_rows: list[dict],
    wrong_home_payload: list[dict],
    pr: dict[str, float],
) -> list[Recommendation]:
    out: list[Recommendation] = []

    # Duplicates: pick canonical = the side with higher PR; 301 the other.
    for d in duplicates_rows[:30]:
        sim = float(d.get("similarity", 0.0))
        url_a = d.get("url_a", "")
        url_b = d.get("url_b", "")
        if not url_a or not url_b:
            continue
        pr_a = pr.get(url_a, 0.0)
        pr_b = pr.get(url_b, 0.0)
        canonical, drop = (url_a, url_b) if pr_a >= pr_b else (url_b, url_a)
        priority = "high" if sim >= 0.95 else "medium"
        pair_key = sorted([url_a, url_b])
        out.append(Recommendation(
            id=_stable_rec_id("dup", "content_debt", "merge_duplicate", *pair_key),
            category="content_debt",
            priority=priority,
            title=f"Merge near-duplicate (sim {sim:.2f})",
            instruction=(
                f"Pick {canonical} as canonical (higher PageRank), "
                f"301 redirect {drop} → canonical, archive the duplicate's content."
            ),
            targets=[canonical, drop],
            evidence={"similarity": round(sim, 4),
                      "pagerank_canonical": round(pr.get(canonical, 0.0), 6),
                      "pagerank_dropped": round(pr.get(drop, 0.0), 6)},
            effort="quick",
            score=sim * 100,
        ))

    # Off-topic outliers: refocus or move to a different section.
    for o in outliers_rows[:15]:
        drift = float(o.get("distance_to_section_centroid", 0.0))
        p95 = float(o.get("section_p95_distance", drift))
        # keep only true outliers
        if drift < p95:
            continue
        url = o.get("url", "")
        if not url:
            continue
        # PR-top outliers compound — high priority
        page_pr = pr.get(url, 0.0)
        priority = "high" if page_pr >= 0.005 else "medium"
        out.append(Recommendation(
            id=_stable_rec_id("out", "content_debt", "refocus_outlier", url, o.get("section") or ""),
            category="content_debt",
            priority=priority,
            title=f"Off-topic page in {o.get('section') or '(root)'}",
            instruction=(
                f"{o.get('recommendation') or 'Refocus or move to a more relevant section.'} "
                f"Pick a section whose centroid this page is closer to, or rewrite to match its current section."
            ),
            targets=[url],
            evidence={
                "drift_to_section": round(drift, 4),
                "section_p95": round(p95, 4),
                "word_count": o.get("word_count"),
            },
            effort="medium",
            score=(drift - p95) * 50 + page_pr * 1000,
        ))

    # Paragraphs on the wrong page: move to suggested home.
    wh_sorted = sorted(wrong_home_payload or [], key=lambda r: float(r.get("lift", 0.0)), reverse=True)
    for w in wh_sorted[:15]:
        lift = float(w.get("lift", 0.0))
        if lift < 0.10:
            break
        out.append(Recommendation(
            id=_stable_rec_id(
                "wh",
                "content_debt",
                "move_paragraph",
                w.get("source_url", ""),
                w.get("paragraph_index", ""),
                w.get("suggested_home_url", ""),
                (w.get("paragraph_excerpt") or "")[:120],
            ),
            category="content_debt",
            priority="medium" if lift < 0.20 else "high",
            title="Paragraph belongs on a different page",
            instruction=(
                f"Move this paragraph from {w.get('source_url')} to "
                f"{w.get('suggested_home_url')} ({w.get('suggested_home_title') or 'better-fit page'})."
            ),
            targets=[w.get("source_url", ""), w.get("suggested_home_url", "")],
            evidence={
                "lift_to_suggested": lift,
                "sim_to_suggested": float(w.get("sim_to_suggested", 0.0)),
                "paragraph_excerpt": (w.get("paragraph_excerpt") or "")[:240],
            },
            effort="quick",
            score=lift * 100,
        ))

    return out


def _coverage(coverage_payload: list[dict]) -> list[Recommendation]:
    out: list[Recommendation] = []
    if not coverage_payload:
        return out

    gaps = [c for c in coverage_payload if c.get("status") == "gap"]
    gaps.sort(key=lambda c: float(c.get("best_similarity", 0.0)))
    for c in gaps[:15]:
        out.append(Recommendation(
            id=_stable_rec_id("gap", "coverage", "coverage_gap", c.get("best_url", ""), c.get("query", "")),
            category="coverage",
            priority="high",
            title=f'No page answers "{c.get("query", "")}"',
            instruction=(
                f"Write a dedicated page for this query. Closest existing page "
                f"({c.get('best_url')}) only scores {c.get('best_similarity', 0):.2f} — "
                f"insufficient. Title the new page with the query, target an FAQ + question H2 layout."
            ),
            targets=[c.get("best_url", "")] if c.get("best_url") else [],
            evidence={
                "query": c.get("query", ""),
                "source": c.get("source", ""),
                "best_similarity": float(c.get("best_similarity", 0.0)),
                "volume": _safe_float(c.get("volume") or c.get("impressions") or c.get("search_volume")),
            },
            effort="deep",
            score=(0.55 - float(c.get("best_similarity", 0.0))) * 100,
        ))

    cann = [c for c in coverage_payload if c.get("status") == "cannibalized"]
    cann.sort(key=lambda c: c.get("candidates_above_threshold", 0), reverse=True)
    for c in cann[:15]:
        n = int(c.get("candidates_above_threshold", 0))
        runners = c.get("runner_ups") or []
        urls = [c.get("best_url", "")] + [r.get("url") for r in runners[:3] if r.get("url")]
        urls = [u for u in urls if u]
        out.append(Recommendation(
            id=_stable_rec_id("cann", "coverage", "cannibalization", c.get("query", ""), *urls),
            category="coverage",
            priority="high",
            title=f'{n} pages compete for "{c.get("query", "")}"',
            instruction=(
                f"Pick the canonical page (best fit + highest PageRank), 301 redirect the rest "
                f"or rewrite them to target distinct sub-queries. Best current: {c.get('best_url')}"
            ),
            targets=urls,
            evidence={
                "query": c.get("query", ""),
                "candidates_above_threshold": n,
                "best_similarity": float(c.get("best_similarity", 0.0)),
            },
            effort="medium",
            score=n * 10,
        ))

    return out


def _geo(
    answerability_payload: list[dict],
    pr: dict[str, float],
    external_per_page: list[dict] | None,
    ai_access_payload: dict | None = None,
    ai_citations_payload: dict | None = None,
    chunk_retrievability_payload: dict | None = None,
) -> list[Recommendation]:
    out: list[Recommendation] = []
    out.extend(_ai_access_recommendations(ai_access_payload))
    out.extend(_ai_citation_recommendations(ai_citations_payload))
    out.extend(_chunk_retrievability_recommendations(chunk_retrievability_payload))

    # Bias to high-PR pages — fixing the load-bearing ones moves the needle.
    if answerability_payload:
        ranked = sorted(
            answerability_payload,
            key=lambda r: (float(r.get("score", 10.0)), -pr.get(r.get("url", ""), 0.0)),
        )
        for p in ranked:
            score = float(p.get("score", 10.0))
            if score >= 4.0:
                break  # rest are fine
            page_pr = pr.get(p.get("url", ""), 0.0)
            priority = "high" if page_pr >= 0.005 else "medium"
            flags = p.get("flags") or []
            flag_text = "; ".join(flags[:4]) if flags else "no specific flags"
            page_label = p.get("title") or p.get("url") or "page"
            authority_label = "high-PR page" if page_pr >= 0.005 else "page"
            out.append(Recommendation(
                id=_stable_rec_id("geo", "geo", "answerability", p.get("url", "")),
                category="geo",
                priority=priority,
                title=f"Low answer-ability ({score:.1f}/10) on {authority_label}: {page_label}",
                instruction=(
                    f"Add the missing GEO signals: {flag_text}. "
                    f"Concretely: add an FAQ block with question H2s, include 1–2 stats with "
                    f"numbers + dates, link to 2–3 authoritative sources."
                ),
                targets=[p.get("url", "")],
                evidence={
                    "answerability_score": score,
                    "pagerank": round(page_pr, 6),
                    "flags": flags,
                },
                effort="medium",
                score=(4.0 - score) * 10 + page_pr * 1000,
            ))
            if len(out) >= 20:
                break

    # Pages on PR-top with zero external citations look unsourced to LLMs.
    if external_per_page:
        ext_lookup = {p.get("url"): p for p in external_per_page}
        top_pr = sorted(pr.items(), key=lambda kv: kv[1], reverse=True)[:50]
        for url, page_pr in top_pr:
            row = ext_lookup.get(url)
            if not row:
                continue
            if int(row.get("external_count", 0)) == 0 and page_pr >= 0.003:
                out.append(Recommendation(
                    id=_stable_rec_id("geo-cite", "geo", "citation_gap", url),
                    category="geo",
                    priority="medium",
                    title="Authority page has no outbound citations",
                    instruction=(
                        "Add 2–3 links to authoritative sources (.gov, .edu, Wikipedia, "
                        "industry research) so LLM answer engines treat the page as sourced."
                    ),
                    targets=[url],
                    evidence={"pagerank": round(page_pr, 6), "external_count": 0},
                    effort="quick",
                    score=page_pr * 1000,
                ))
                if len(out) >= 30:
                    break

    return out


def _agent_slug(agent: str, index: int) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", agent.lower()).strip("-")
    return slug or f"agent-{index}"


def _stable_slug(value: object, fallback: str = "item") -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(value or "").lower()).strip("-")
    if len(slug) > 80:
        # Truncation can collide for long values differing only past the
        # cut — pin uniqueness with a short digest of the full slug.
        digest = hashlib.sha1(slug.encode("utf-8")).hexdigest()[:8]
        slug = f"{slug[:80].strip('-')}-{digest}"
    return slug or fallback


def _stable_rec_id(prefix: str, category: str, action_type: str, *parts: object) -> str:
    raw_parts = [str(part) for part in (category, action_type, *parts) if part not in (None, "")]
    raw = "\u241f".join(raw_parts)
    slug_parts = [str(part) for part in parts if part not in (None, "")]
    slug = _stable_slug("\u241f".join(slug_parts), prefix)
    digest = hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:10]
    return f"{prefix}-{slug}-{digest}"


def _ai_citation_recommendations(payload: dict | None) -> list[Recommendation]:
    if not isinstance(payload, dict) or payload.get("available") is not True:
        return []
    out: list[Recommendation] = []
    for row in (payload.get("at_risk") or [])[:15]:
        url = str(row.get("url") or "")
        keyword = str(row.get("top_keyword") or "")
        volume = _safe_int(row.get("top_keyword_volume"))
        if not url or not keyword:
            continue
        out.append(Recommendation(
            id=f"geo-aio-{_stable_slug(url, 'url')}",
            category="geo",
            priority="medium",
            title="Refresh page cited in Google AI Overviews",
            instruction=(
                f'{url} is cited in Google AI Overviews for "{keyword}" ({volume}/mo) '
                "but its content is stale — refresh it to keep the citation."
            ),
            targets=[url],
            evidence={
                "keyword": keyword,
                "search_volume": volume,
                "bucket": row.get("bucket", ""),
                "age_days": row.get("age_days"),
                "query_count": _safe_int(row.get("query_count")),
            },
            effort="medium",
            score=72.0 + min(18.0, math.log1p(max(volume, 0)) * 2.0),
        ))

    for row in (payload.get("opportunities") or [])[:20]:
        keyword = str(row.get("keyword") or "")
        url = str(row.get("url") or "")
        position = _safe_float(row.get("position"))
        if not keyword or not url or position <= 0:
            continue
        out.append(Recommendation(
            id=f"geo-aio-opp-{_stable_slug(url, 'url')}-{_stable_slug(keyword, 'kw')}",
            category="geo",
            priority="medium",
            title="Win missing Google AI Overview citation",
            instruction=(
                f'AI Overview exists for "{keyword}" but does not cite {url} '
                f"(ranks #{position:.0f}). Strengthen the matching answer block: "
                "direct answer under a question H2, cited evidence."
            ),
            targets=[url],
            evidence={
                "keyword": keyword,
                "position": position,
                "search_volume": _safe_int(row.get("search_volume")),
                "traffic": _safe_int(row.get("traffic")),
            },
            effort="medium",
            score=64.0 + min(16.0, math.log1p(max(_safe_int(row.get("search_volume")), 0)) * 1.8),
        ))
    return out


def _ai_access_recommendations(payload: dict | None) -> list[Recommendation]:
    if not isinstance(payload, dict):
        return []
    recommendations = list(payload.get("recommendations") or [])
    if not recommendations:
        return []
    summary = payload.get("summary") or {}
    agents = list(payload.get("agents") or [])
    if summary.get("blanket_block"):
        return [Recommendation(
            id="geo-access-blanket",
            category="geo",
            priority="high",
            title="AI search and fetch crawlers blocked by robots.txt",
            instruction=str(recommendations[0]),
            targets=[str(payload.get("base_url") or "")],
            evidence={
                "blanket_block": True,
                "search_blocked": _safe_int(summary.get("search_blocked")),
                "user_fetch_blocked": _safe_int(summary.get("user_fetch_blocked")),
            },
            effort="quick",
            score=95.0,
        )]

    out: list[Recommendation] = []
    blocked = [
        row for row in agents
        if row.get("allowed_root") is False and row.get("purpose") in {"search", "user_fetch"}
    ]
    for i, row in enumerate(blocked[:10]):
        # The payload's recommendation strings are built from the same
        # blocked rows in the same order; fall back to the shared formatter
        # only when a hand-crafted payload disagrees.
        instruction = (
            str(recommendations[i])
            if i < len(recommendations)
            else format_blocked_agent_recommendation(row)
        )
        agent_slug = _agent_slug(str(row.get("agent") or ""), i)
        out.append(Recommendation(
            id=f"geo-access-{agent_slug}",
            category="geo",
            priority="high",
            title=f"robots.txt blocks {row.get('agent')}",
            instruction=instruction,
            targets=[str(payload.get("base_url") or "")],
            evidence={
                "agent": row.get("agent", ""),
                "operator": row.get("operator", ""),
                "purpose": row.get("purpose", ""),
                "matched_group": row.get("matched_group", ""),
                "explicitly_named": bool(row.get("explicitly_named")),
            },
            effort="quick",
            score=90.0 - i,
        ))
    return out


def _chunk_retrievability_recommendations(payload: dict | None) -> list[Recommendation]:
    if not isinstance(payload, dict) or payload.get("available") is not True:
        return []
    out: list[Recommendation] = []
    for i, row in enumerate(payload.get("recommendations") or []):
        rec_id = str(row.get("id") or "")
        text = str(row.get("text") or "")
        url = str(row.get("url") or "")
        query = str(row.get("query") or "")
        if not rec_id or not text or not url:
            continue
        status = str(row.get("status") or "")
        title = (
            "Consolidate split answer into one retrievable chunk"
            if status == "split_answer"
            else "Add missing retrievable answer chunk"
        )
        out.append(Recommendation(
            id=rec_id,
            category="geo",
            priority="medium",
            title=title,
            instruction=text,
            targets=[url],
            evidence={
                "query": query,
                "status": status,
                "best_similarity": _safe_float(row.get("best_similarity")),
                "heading_a": row.get("heading_a", ""),
                "heading_b": row.get("heading_b", ""),
            },
            effort=str(row.get("effort") or "medium"),
            score=68.0 - i + max(0.0, 0.65 - _safe_float(row.get("best_similarity"))) * 20.0,
        ))
    return out


def _linking(
    linkgraph_payload: dict | None,
    paragraph_links: list[dict] | None,
    pr: dict[str, float],
) -> list[Recommendation]:
    out: list[Recommendation] = []
    lg = linkgraph_payload or {}

    # Page-level link recs
    for r in (lg.get("recommendations") or [])[:15]:
        sim = float(r.get("similarity", 0.0))
        source_url = r.get("source_url") or r.get("url_a") or ""
        target_url = r.get("target_url") or r.get("url_b") or ""
        source_label = r.get("source_title") or r.get("title_a") or source_url or "source page"
        target_label = r.get("target_title") or r.get("title_b") or target_url or "target page"
        out.append(Recommendation(
            id=_stable_rec_id("link", "linking", "internal_link", source_url, target_url, r.get("suggested_anchor") or ""),
            category="linking",
            priority="medium",
            title="Add internal link",
            instruction=(
                f"Add a contextual link from {source_label} to "
                f"{target_label} (the topics overlap at sim {sim:.2f} but no link exists today)."
            ),
            targets=[source_url, target_url],
            evidence={"similarity": sim, "anchor_hint": r.get("suggested_anchor")},
            effort="quick",
            score=sim * 100,
        ))

    # Paragraph-level link recs — these are the highest-signal because
    # they tell the editor *which paragraph* to edit.
    if paragraph_links:
        pl_sorted = sorted(paragraph_links, key=lambda r: float(r.get("lift", 0.0)), reverse=True)
        for r in pl_sorted[:15]:
            lift = float(r.get("lift", 0.0))
            out.append(Recommendation(
                id=_stable_rec_id(
                    "plink",
                    "linking",
                    "paragraph_link",
                    r.get("source_url", ""),
                    r.get("paragraph_index", ""),
                    r.get("target_url", ""),
                    r.get("suggested_anchor", ""),
                ),
                category="linking",
                priority="medium" if lift < 0.15 else "high",
                title="Add in-paragraph internal link",
                instruction=(
                    f"In paragraph {r.get('paragraph_index')} of {r.get('source_url')}, "
                    f"link the phrase \"{r.get('suggested_anchor')}\" to "
                    f"{r.get('target_url')} ({r.get('target_title') or 'target page'})."
                ),
                targets=[r.get("source_url", ""), r.get("target_url", "")],
                evidence={
                    "fit": float(r.get("fit", 0.0)),
                    "lift": lift,
                    "paragraph_excerpt": (r.get("paragraph_excerpt") or "")[:240],
                },
                effort="quick",
                score=lift * 200,
            ))

    # Orphans on the cluster-authority list are the most painful — fix first.
    cluster_auth_urls = {ca["authority"]["url"] for ca in (lg.get("cluster_authorities") or [])
                        if ca.get("authority")}
    orphans = lg.get("orphans") or []
    orphans_sorted = sorted(orphans, key=lambda r: float(r.get("pagerank", 0.0)), reverse=True)
    for o in orphans_sorted[:10]:
        url = o.get("url", "")
        is_auth = url in cluster_auth_urls
        out.append(Recommendation(
            id=_stable_rec_id("orphan", "linking", "orphan_page", url),
            category="linking",
            priority="high" if is_auth else "medium",
            title="Orphan page" + (" (cluster authority)" if is_auth else ""),
            instruction=(
                f"Add 2–3 inbound internal links from related pages — start with the "
                f"section landing page and the topic-cluster authority. Currently 0 inbound links."
            ),
            targets=[url],
            evidence={"pagerank": float(o.get("pagerank", 0.0)),
                      "out_degree": int(o.get("out_degree", 0)),
                      "is_cluster_authority": is_auth},
            effort="quick",
            score=10 + float(o.get("pagerank", 0.0)) * 1000,
        ))

    # Buried pages: top PR pages that take 4+ clicks
    deep = lg.get("deep_pages") or []
    pr_lookup_full = pr  # only top-N from linkgraph payload, but enough
    deep_with_pr = []
    for d in deep:
        url = d.get("url", "")
        deep_with_pr.append((url, pr_lookup_full.get(url, 0.0), d))
    deep_with_pr.sort(key=lambda x: x[1], reverse=True)
    for url, page_pr, d in deep_with_pr[:8]:
        if int(d.get("click_depth", 0)) < 4:
            continue
        out.append(Recommendation(
            id=_stable_rec_id("deep", "linking", "click_depth", url),
            category="linking",
            priority="medium",
            title=f"Buried page (depth {d.get('click_depth')})",
            instruction=(
                "Bring this page within 3 clicks of the homepage by linking it from a "
                "section landing page or sidebar."
            ),
            targets=[url],
            evidence={"click_depth": int(d.get("click_depth", 0)),
                      "pagerank": round(page_pr, 6)},
            effort="quick",
            score=int(d.get("click_depth", 0)) + page_pr * 100,
        ))

    return out


def _onpage(
    title_mismatch: list[dict] | None,
    anchor_analysis: list[dict] | None,
    ctr_anomalies: list[dict] | None,
    pr: dict[str, float],
) -> list[Recommendation]:
    out: list[Recommendation] = []

    if ctr_anomalies:
        rows = sorted(ctr_anomalies, key=lambda r: _safe_float(r.get("missed_clicks")), reverse=True)
        for r in rows[:20]:
            url = r.get("url", "")
            missed = _safe_float(r.get("missed_clicks") or r.get("estimated_clicks_gain"))
            if not url or missed <= 0:
                continue
            out.append(Recommendation(
                id=_stable_rec_id("ctr", "onpage", "title_rewrite", url, r.get("query", "")),
                category="onpage",
                priority="medium",
                title=str(r.get("title") or ""),
                instruction=str(r.get("action") or ""),
                targets=[url],
                evidence={
                    "query": r.get("query", ""),
                    "position": _safe_float(r.get("position")),
                    "actual_ctr": _safe_float(r.get("actual_ctr")),
                    "expected_ctr": _safe_float(r.get("expected_ctr")),
                    "estimated_clicks_gain": round(missed, 2),
                    "traffic_opportunity": round(missed, 2),
                    "probable_cause": r.get("probable_cause", ""),
                    "current_title": r.get("current_title", ""),
                    "period": r.get("period", ""),
                },
                effort="quick",
                score=min(100.0, missed),
            ))

    if title_mismatch:
        tm_sorted = sorted(title_mismatch, key=lambda r: float(r.get("title_to_content", 1.0)))
        for r in tm_sorted[:15]:
            ratio = float(r.get("title_to_content", 1.0))
            if ratio >= 0.55:
                break
            url = r.get("url", "")
            page_pr = pr.get(url, 0.0)
            priority = "high" if page_pr >= 0.005 else "medium"
            kws = r.get("suggested_keywords") or []
            kw_text = ", ".join(kws[:4]) if kws else "the page's actual topic"
            out.append(Recommendation(
                id=_stable_rec_id("title", "onpage", "title_rewrite", url),
                category="onpage",
                priority=priority,
                title=f"Misleading title (cosine {ratio:.2f})",
                instruction=(
                    f"Rewrite the title to reflect content. Suggested keywords: {kw_text}. "
                    f"Current title doesn't match what the page actually says."
                ),
                targets=[url],
                evidence={"title_to_content": ratio,
                          "current_title": r.get("title"),
                          "suggested_keywords": kws[:6],
                          "pagerank": round(page_pr, 6)},
                effort="quick",
                score=(0.55 - ratio) * 100 + page_pr * 1000,
            ))

    if anchor_analysis:
        bad = [a for a in anchor_analysis
               if float(a.get("generic_anchor_share", 0.0)) >= 0.5
               and int(a.get("inbound_link_count", 0)) >= 3]
        bad.sort(key=lambda a: float(a.get("generic_anchor_share", 0.0)), reverse=True)
        for a in bad[:8]:
            share = float(a.get("generic_anchor_share", 0.0))
            out.append(Recommendation(
                id=_stable_rec_id("anchor", "onpage", "anchor_rewrite", a.get("target_url", "")),
                category="onpage",
                priority="medium",
                title=f"Generic anchors dominate ({int(share*100)}%)",
                instruction=(
                    f'Replace "click here" / numeric / one-word anchors with descriptive '
                    f'phrases that include the target page\'s topic.'
                ),
                targets=[a.get("target_url", "")],
                evidence={"generic_anchor_share": share,
                          "inbound_link_count": int(a.get("inbound_link_count", 0)),
                          "top_anchors": (a.get("top_anchors") or [])[:5]},
                effort="quick",
                score=share * 50,
            ))

    return out


def synthesize(
    *,
    duplicates_rows: list[dict] | None = None,
    outliers_rows: list[dict] | None = None,
    coverage_payload: list[dict] | None = None,
    answerability_payload: list[dict] | None = None,
    linkgraph_payload: dict | None = None,
    search_payload: dict | None = None,
    paragraph_links: list[dict] | None = None,
    wrong_home_payload: list[dict] | None = None,
    title_mismatch: list[dict] | None = None,
    ctr_anomalies_payload: dict | list | None = None,
    ai_access_payload: dict | None = None,
    ai_citations_payload: dict | None = None,
    chunk_retrievability_payload: dict | None = None,
    external_links_payload: dict | None = None,
    max_total: int = 100,
) -> list[Recommendation]:
    pr = _pr_lookup(linkgraph_payload)
    external_per_page = (external_links_payload or {}).get("per_page") or []
    anchor_analysis = (linkgraph_payload or {}).get("anchor_analysis") or []
    ctr_anomalies = _ctr_anomaly_recommendations(ctr_anomalies_payload)

    recs: list[Recommendation] = []
    recs += _content_debt(duplicates_rows or [], outliers_rows or [], wrong_home_payload or [], pr)
    recs += _coverage(coverage_payload or [])
    recs += _geo(
        answerability_payload or [],
        pr,
        external_per_page,
        ai_access_payload,
        ai_citations_payload,
        chunk_retrievability_payload,
    )
    recs += _linking(linkgraph_payload, paragraph_links, pr)
    recs += _onpage(title_mismatch, anchor_analysis, ctr_anomalies, pr)

    finalized = _finalize(recs, linkgraph_payload=linkgraph_payload, search_payload=search_payload)
    kept, suppressed = _resolve_conflicts(finalized)
    # Truncate AFTER resolving conflicts, and only report suppressions whose
    # suppressor survived the cut — re-resolving on a truncated list would
    # resurrect improve-recs for pages still slated for removal.
    kept = kept[:max_total]
    kept_ids = {rec.id for rec in kept}
    suppressed = [rec for rec in suppressed if rec.suppressed_by in kept_ids]
    return kept + suppressed


def _ctr_anomaly_recommendations(payload: dict | list | None) -> list[dict]:
    if isinstance(payload, dict):
        return list(payload.get("recommendations") or [])
    return list(payload or [])


def to_payload(recs: Iterable[Recommendation]) -> dict:
    """Convert recommendations to the report payload.

    Per item, ``traffic_opportunity`` is the modeled ``estimated_clicks_gain``
    when a CTR/position basis exists, else the legacy traffic index. The
    summary total sums those per-item values.
    """
    recs_list = list(recs)
    items = [r for r in recs_list if not r.suppressed]
    suppressed = [r for r in recs_list if r.suppressed]
    by_category: dict[str, int] = {}
    by_priority: dict[str, int] = {"high": 0, "medium": 0, "low": 0}
    for r in items:
        by_category[r.category] = by_category.get(r.category, 0) + 1
        by_priority[r.priority] = by_priority.get(r.priority, 0) + 1
    owners = sorted({r.owner for r in items if r.owner})
    types = sorted({r.type for r in items if r.type})
    clusters = sorted({r.cluster for r in items if r.cluster})
    avg = lambda attr: round(sum(float(getattr(r, attr, 0.0) or 0.0) for r in items) / max(len(items), 1), 1)
    cards = to_cards(items)
    return {
        "total": len(items),
        "by_category": by_category,
        "by_priority": by_priority,
        "score_model": SCORE_MODEL,
        "filters": {
            "owners": owners,
            "types": types,
            "clusters": clusters[:200],
            "categories": sorted(by_category),
            "priorities": ["high", "medium", "low"],
        },
        "summary": {
            "avg_impact": avg("impact"),
            "avg_confidence": avg("confidence"),
            "avg_effort_score": avg("effort_score"),
            "avg_risk": avg("risk"),
            "avg_priority_score": avg("priority_score"),
            "traffic_opportunity": round(sum(float(r.traffic_opportunity or 0.0) for r in items), 1),
            "cards": len(cards),
            "suppressed": len(suppressed),
        },
        "cards": cards,
        "items": [_item_row(r) for r in items],
        "suppressed": [_suppressed_row(r) for r in suppressed],
    }
