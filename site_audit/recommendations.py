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
* ``onpage`` — title mismatches, generic anchors.

Priority is one of ``high`` / ``medium`` / ``low``. The pickers favour
high-PageRank pages — fixing the load-bearing pages compounds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable, Optional
from urllib.parse import urlsplit, urlunsplit


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
    "link": "internal_link",
    "plink": "paragraph_link",
    "orphan": "orphan_page",
    "deep": "click_depth",
    "title": "title_rewrite",
    "anchor": "anchor_rewrite",
}

SCORE_MODEL = {
    "model": "fix_priority_score_v1",
    "components": {
        "impact": "0-100 estimate from traffic opportunity, existing recommendation score, PageRank, ranking/search gap, and issue severity.",
        "confidence": "0-100 estimate from recommendation class and evidence completeness.",
        "effort_score": "0-100 estimated implementation effort. Quick edits are lower, deep content/template work is higher.",
        "risk": "0-100 estimated downside risk. Redirects and consolidation are higher risk than links or metadata edits.",
        "priority_score": "0.45*impact + 0.25*confidence - 0.18*effort_score - 0.12*risk, clamped to 0-100.",
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


def _store_context(out: dict[str, dict], url: object, context: dict) -> None:
    for key in _url_keys(url):
        current = out.get(key)
        if current is None or _safe_float(context.get("traffic")) > _safe_float(current.get("traffic")):
            out[key] = context


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
    prefix = rec.id.split("-", 1)[0]
    if rec.id.startswith("geo-cite"):
        return "citation_gap"
    return _TYPE_BY_PREFIX.get(prefix, rec.category)


def _component_scores(rec: Recommendation, context: dict[str, dict]) -> dict:
    evidence = rec.evidence or {}
    target_context = _lookup_context(context, rec.targets)
    traffic = max(
        _safe_float(target_context.get("traffic")),
        _safe_float(evidence.get("traffic")),
        _safe_float(evidence.get("traffic_opportunity")),
        _safe_float(evidence.get("target_traffic")),
    )
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
    impact = _clip(
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
    priority_score = _clip(0.45 * impact + 0.25 * confidence - 0.18 * effort_score - 0.12 * risk)
    return {
        "impact": round(impact, 1),
        "confidence": round(confidence, 1),
        "effort_score": round(effort_score, 1),
        "risk": round(risk, 1),
        "priority_score": round(priority_score, 1),
        "traffic_opportunity": round(traffic, 1),
        "cluster": str(target_context.get("cluster") or evidence.get("query") or ""),
    }


def _priority_bucket(priority_score: float) -> str:
    if priority_score >= 60.0:
        return "high"
    if priority_score >= 35.0:
        return "medium"
    return "low"


def _finalize(recs: list[Recommendation], *, linkgraph_payload: dict | None = None, search_payload: dict | None = None) -> list[Recommendation]:
    context = _target_context_lookup(linkgraph_payload, search_payload)
    for rec in recs:
        scores = _component_scores(rec, context)
        rec.impact = scores["impact"]
        rec.confidence = scores["confidence"]
        rec.effort_score = scores["effort_score"]
        rec.risk = scores["risk"]
        rec.priority_score = scores["priority_score"]
        rec.priority = _priority_bucket(rec.priority_score)
        rec.owner = rec.owner or _CATEGORY_OWNER.get(rec.category, "SEO")
        rec.type = rec.type or _rec_type(rec)
        rec.cluster = rec.cluster or scores["cluster"]
        rec.traffic_opportunity = scores["traffic_opportunity"]
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
    for i, d in enumerate(duplicates_rows[:30]):
        sim = float(d.get("similarity", 0.0))
        url_a = d.get("url_a", "")
        url_b = d.get("url_b", "")
        if not url_a or not url_b:
            continue
        pr_a = pr.get(url_a, 0.0)
        pr_b = pr.get(url_b, 0.0)
        canonical, drop = (url_a, url_b) if pr_a >= pr_b else (url_b, url_a)
        priority = "high" if sim >= 0.95 else "medium"
        out.append(Recommendation(
            id=f"dup-{i}",
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
    for i, o in enumerate(outliers_rows[:15]):
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
            id=f"out-{i}",
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
    for i, w in enumerate(wh_sorted[:15]):
        lift = float(w.get("lift", 0.0))
        if lift < 0.10:
            break
        out.append(Recommendation(
            id=f"wh-{i}",
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
    for i, c in enumerate(gaps[:15]):
        out.append(Recommendation(
            id=f"gap-{i}",
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
            },
            effort="deep",
            score=(0.55 - float(c.get("best_similarity", 0.0))) * 100,
        ))

    cann = [c for c in coverage_payload if c.get("status") == "cannibalized"]
    cann.sort(key=lambda c: c.get("candidates_above_threshold", 0), reverse=True)
    for i, c in enumerate(cann[:15]):
        n = int(c.get("candidates_above_threshold", 0))
        runners = c.get("runner_ups") or []
        urls = [c.get("best_url", "")] + [r.get("url") for r in runners[:3] if r.get("url")]
        urls = [u for u in urls if u]
        out.append(Recommendation(
            id=f"cann-{i}",
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
) -> list[Recommendation]:
    out: list[Recommendation] = []

    # Bias to high-PR pages — fixing the load-bearing ones moves the needle.
    if answerability_payload:
        ranked = sorted(
            answerability_payload,
            key=lambda r: (float(r.get("score", 10.0)), -pr.get(r.get("url", ""), 0.0)),
        )
        for i, p in enumerate(ranked):
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
                id=f"geo-{i}",
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
                    id=f"geo-cite-{len(out)}",
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


def _linking(
    linkgraph_payload: dict | None,
    paragraph_links: list[dict] | None,
    pr: dict[str, float],
) -> list[Recommendation]:
    out: list[Recommendation] = []
    lg = linkgraph_payload or {}

    # Page-level link recs
    for i, r in enumerate((lg.get("recommendations") or [])[:15]):
        sim = float(r.get("similarity", 0.0))
        source_url = r.get("source_url") or r.get("url_a") or ""
        target_url = r.get("target_url") or r.get("url_b") or ""
        source_label = r.get("source_title") or r.get("title_a") or source_url or "source page"
        target_label = r.get("target_title") or r.get("title_b") or target_url or "target page"
        out.append(Recommendation(
            id=f"link-{i}",
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
        for i, r in enumerate(pl_sorted[:15]):
            lift = float(r.get("lift", 0.0))
            out.append(Recommendation(
                id=f"plink-{i}",
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
    for i, o in enumerate(orphans_sorted[:10]):
        url = o.get("url", "")
        is_auth = url in cluster_auth_urls
        out.append(Recommendation(
            id=f"orphan-{i}",
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
    for i, (url, page_pr, d) in enumerate(deep_with_pr[:8]):
        if int(d.get("click_depth", 0)) < 4:
            continue
        out.append(Recommendation(
            id=f"deep-{i}",
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
    pr: dict[str, float],
) -> list[Recommendation]:
    out: list[Recommendation] = []

    if title_mismatch:
        tm_sorted = sorted(title_mismatch, key=lambda r: float(r.get("title_to_content", 1.0)))
        for i, r in enumerate(tm_sorted[:15]):
            ratio = float(r.get("title_to_content", 1.0))
            if ratio >= 0.55:
                break
            url = r.get("url", "")
            page_pr = pr.get(url, 0.0)
            priority = "high" if page_pr >= 0.005 else "medium"
            kws = r.get("suggested_keywords") or []
            kw_text = ", ".join(kws[:4]) if kws else "the page's actual topic"
            out.append(Recommendation(
                id=f"title-{i}",
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
        for i, a in enumerate(bad[:8]):
            share = float(a.get("generic_anchor_share", 0.0))
            out.append(Recommendation(
                id=f"anchor-{i}",
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
    external_links_payload: dict | None = None,
    max_total: int = 100,
) -> list[Recommendation]:
    pr = _pr_lookup(linkgraph_payload)
    external_per_page = (external_links_payload or {}).get("per_page") or []
    anchor_analysis = (linkgraph_payload or {}).get("anchor_analysis") or []

    recs: list[Recommendation] = []
    recs += _content_debt(duplicates_rows or [], outliers_rows or [], wrong_home_payload or [], pr)
    recs += _coverage(coverage_payload or [])
    recs += _geo(answerability_payload or [], pr, external_per_page)
    recs += _linking(linkgraph_payload, paragraph_links, pr)
    recs += _onpage(title_mismatch, anchor_analysis, pr)

    return _finalize(recs, linkgraph_payload=linkgraph_payload, search_payload=search_payload)[:max_total]


def to_payload(recs: Iterable[Recommendation]) -> dict:
    items = list(recs)
    by_category: dict[str, int] = {}
    by_priority: dict[str, int] = {"high": 0, "medium": 0, "low": 0}
    for r in items:
        by_category[r.category] = by_category.get(r.category, 0) + 1
        by_priority[r.priority] = by_priority.get(r.priority, 0) + 1
    owners = sorted({r.owner for r in items if r.owner})
    types = sorted({r.type for r in items if r.type})
    clusters = sorted({r.cluster for r in items if r.cluster})
    avg = lambda attr: round(sum(float(getattr(r, attr, 0.0) or 0.0) for r in items) / max(len(items), 1), 1)
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
        },
        "items": [
            {
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
                "title": r.title,
                "instruction": r.instruction,
                "targets": r.targets,
                "evidence": r.evidence,
                "effort": r.effort,
                "score": r.score,
            }
            for r in items
        ],
    }
