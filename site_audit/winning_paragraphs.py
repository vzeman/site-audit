"""Curated winning-paragraph table built from paragraph impact signals."""

from __future__ import annotations

from collections import defaultdict


def _key(url: str, paragraph_index) -> str:
    try:
        idx = int(paragraph_index)
    except (TypeError, ValueError):
        idx = 0
    return f"{url or ''}#{idx}"


def _action(row: dict) -> str:
    comps = row.get("components") or {}
    tier = row.get("impact_tier") or ""
    ablation = row.get("ablation_classification") or ""
    if ablation == "topic_carrier" and tier == "high":
        return "protect_and_expand"
    if float(comps.get("heading_match", 0.0) or 0.0) < 0.25:
        return "align_heading"
    if float(comps.get("link_context", 0.0) or 0.0) < 0.25:
        return "add_contextual_links"
    if float(comps.get("freshness", 0.0) or 0.0) < 0.6:
        return "refresh_evidence"
    return "maintain"


def _action_label(action: str) -> str:
    return {
        "protect_and_expand": "Protect and expand",
        "align_heading": "Align heading",
        "add_contextual_links": "Add contextual links",
        "refresh_evidence": "Refresh evidence",
        "maintain": "Maintain",
    }.get(action, action)


def build_winning_paragraphs(
    paragraph_impact: dict,
    semantic_ablation: dict | None = None,
    keyword_attribution: dict | None = None,
    *,
    top_n: int = 500,
) -> dict:
    impact_rows = list((paragraph_impact or {}).get("top_paragraphs") or [])
    if not impact_rows:
        return {"summary": {"status": "no_impact_rows", "rows": 0}, "rows": []}

    ablation_lookup: dict[str, dict] = {}
    for row in (semantic_ablation or {}).get("rows") or []:
        ablation_lookup[_key(row.get("url", ""), row.get("paragraph_index"))] = row

    keyword_lookup: dict[str, list[dict]] = defaultdict(list)
    for row in (keyword_attribution or {}).get("keywords") or []:
        if row.get("best_paragraph_index") is None:
            continue
        keyword_lookup[_key(row.get("url", ""), row.get("best_paragraph_index"))].append({
            "keyword": row.get("keyword") or "",
            "traffic": int(row.get("traffic") or 0),
            "position": int(row.get("position") or 0),
            "status": row.get("status") or "",
        })

    rows: list[dict] = []
    for source in impact_rows:
        key = _key(source.get("url", ""), source.get("paragraph_index"))
        ablation = ablation_lookup.get(key) or {}
        keywords = sorted(keyword_lookup.get(key, []), key=lambda r: int(r.get("traffic", 0)), reverse=True)[:8]
        row = {
            **source,
            "ablation_classification": ablation.get("classification", ""),
            "ablation_label": ablation.get("classification_label", ""),
            "alignment_delta": ablation.get("alignment_delta"),
            "self_alignment": ablation.get("self_alignment"),
            "attributed_keywords": keywords,
            "attributed_keyword_count": len(keywords),
            "attributed_keyword_traffic": sum(int(k.get("traffic", 0)) for k in keywords),
        }
        action = _action(row)
        row["recommended_action"] = action
        row["recommended_action_label"] = _action_label(action)
        rows.append(row)

    rows.sort(key=lambda r: (float(r.get("impact_score", 0.0)), float(r.get("attributed_traffic", 0.0))), reverse=True)
    rows = rows[:top_n]
    actions: dict[str, int] = defaultdict(int)
    for i, row in enumerate(rows, 1):
        row["rank"] = i
        actions[row["recommended_action"]] += 1

    summary = {
        "status": "ok",
        "rows": len(rows),
        "source_impact_rows": len(impact_rows),
        "high_impact": sum(1 for r in rows if r.get("impact_tier") == "high"),
        "topic_carriers": sum(1 for r in rows if r.get("ablation_classification") == "topic_carrier"),
        "noise_candidates": sum(1 for r in rows if r.get("ablation_classification") == "noise_candidate"),
        "attributed_traffic": round(sum(float(r.get("attributed_traffic", 0.0)) for r in rows), 2),
        "actions": dict(sorted(actions.items())),
    }
    return {
        "summary": summary,
        "rows": rows,
        "interpretation": {
            "protect_and_expand": "High-impact topic carriers should be preserved during edits and expanded with stronger evidence where appropriate.",
            "align_heading": "The paragraph supports demand, but visible headings do not describe that demand clearly enough.",
            "add_contextual_links": "The paragraph carries search value but has weak inline-link support.",
            "refresh_evidence": "The paragraph carries demand but lacks freshness signals.",
        },
    }
