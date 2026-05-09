"""Balance organic-search support against conversion support."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any
from urllib.parse import urlparse

from .analyzer import PageInfo
from .extractor import ExtractedPage
from .page_types import classify_page
from .paragraph_impact import _match_page, _page_lookup, _to_int

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9'_-]*", re.I)
_MONEY_RE = re.compile(r"\b(pricing|price|demo|trial|signup|sign up|quote|contact|book|buy|product|software|platform|solution)\b", re.I)
_GENERIC_CTA = {"learn more", "read more", "more", "details", "click here"}


def _tokens(text: str) -> set[str]:
    return {m.group(0).lower() for m in _WORD_RE.finditer(text or "") if len(m.group(0)) > 1}


def _search_lookup(pages: list[PageInfo], search_payload: dict | None) -> dict[int, dict]:
    lookup = _page_lookup(pages)
    out: dict[int, dict] = defaultdict(lambda: {"traffic": 0, "keywords": 0, "top_keyword": "", "top_position": 0, "intents": set(), "cluster": ""})
    for row in (search_payload or {}).get("top_pages") or []:
        idx = _match_page(row.get("matched_url") or row.get("url") or "", lookup)
        if idx is None:
            continue
        out[idx]["traffic"] = max(out[idx]["traffic"], _to_int(row.get("traffic")))
        out[idx]["keywords"] = max(out[idx]["keywords"], _to_int(row.get("keywords")))
        if row.get("top_keyword"):
            out[idx]["top_keyword"] = row.get("top_keyword")
        out[idx]["top_position"] = _to_int(row.get("top_keyword_position") or row.get("position"))
        if row.get("cluster_label"):
            out[idx]["cluster"] = row.get("cluster_label")
    for row in (search_payload or {}).get("organic_keywords") or []:
        idx = _match_page(row.get("matched_url") or row.get("url") or "", lookup)
        if idx is None:
            continue
        out[idx]["traffic"] += _to_int(row.get("traffic"))
        out[idx]["keywords"] += 1
        if not out[idx]["top_keyword"] or _to_int(row.get("traffic")) > out[idx].get("_kw_traffic", 0):
            out[idx]["top_keyword"] = row.get("keyword") or out[idx]["top_keyword"]
            out[idx]["top_position"] = _to_int(row.get("position"))
            out[idx]["_kw_traffic"] = _to_int(row.get("traffic"))
        for intent in row.get("intents") or []:
            out[idx]["intents"].add(str(intent))
        if row.get("cluster_label"):
            out[idx]["cluster"] = row.get("cluster_label")
    for ctx in out.values():
        ctx.pop("_kw_traffic", None)
        ctx["intents"] = sorted(ctx["intents"])
    return out


def _seo_score(page: PageInfo, ext: ExtractedPage, ctx: dict) -> tuple[float, dict]:
    traffic = _to_int(ctx.get("traffic"))
    keywords = _to_int(ctx.get("keywords"))
    top_keyword = str(ctx.get("top_keyword") or "")
    keyword_tokens = _tokens(top_keyword)
    visible_tokens = _tokens(f"{page.title} {ext.h1} {' '.join(ext.headings or [])} {ext.description}")
    overlap = len(keyword_tokens & visible_tokens) / max(1, len(keyword_tokens)) if keyword_tokens else 0.0
    position = _to_int(ctx.get("top_position"))
    score = min(45.0, math.log1p(max(0, traffic)) * 9.0)
    score += min(25.0, math.sqrt(max(0, keywords)) * 5.0)
    score += overlap * 20.0
    if 0 < position <= 3:
        score += 10.0
    elif 0 < position <= 10:
        score += 5.0
    return round(min(100.0, score), 2), {"traffic": traffic, "keywords": keywords, "top_keyword": top_keyword, "keyword_overlap": round(overlap, 3), "top_position": position}


def _conversion_score(ext: ExtractedPage) -> tuple[float, dict]:
    signals = ext.conversion_signals or {}
    ctas = list(signals.get("ctas") or [])
    cta_count = _to_int(signals.get("cta_count") or len(ctas))
    primary = _to_int(signals.get("primary_cta_count") or sum(1 for c in ctas if c.get("primary")))
    forms = _to_int(signals.get("form_count") or len(signals.get("forms") or []))
    contact = _to_int(signals.get("contact_link_count"))
    specific_ctas = [
        c for c in ctas
        if (c.get("text") or "").strip().lower() not in _GENERIC_CTA and len((c.get("text") or "").strip()) >= 6
    ]
    journey = sum(1 for c in ctas if _MONEY_RE.search(f"{c.get('text', '')} {c.get('href', '')}"))
    score = 0.0
    if primary:
        score += 35
    elif cta_count:
        score += 18
    if forms or contact:
        score += 25
    if specific_ctas:
        score += min(18, len(specific_ctas) * 6)
    if journey:
        score += min(17, journey * 8)
    if cta_count > 8:
        score -= 10
    return round(max(0.0, min(100.0, score)), 2), {"cta_count": cta_count, "primary_cta_count": primary, "form_count": forms, "contact_link_count": contact, "specific_ctas": len(specific_ctas), "journey_links": journey}


def _is_money_page(page: PageInfo, ext: ExtractedPage, page_type: str, ctx: dict) -> bool:
    haystack = f"{page.url} {page.title} {ext.h1} {' '.join(ext.headings or [])}"
    intents = set(ctx.get("intents") or [])
    return page_type in {"product", "service", "home", "contact"} or bool(_MONEY_RE.search(haystack)) or bool(intents & {"commercial", "transactional", "navigational"})


def _label(seo: float, conv: float, money_page: bool, traffic: int) -> tuple[str, str]:
    if seo >= 55 and conv >= 55:
        return "balanced", "Maintain the page; test CTA wording and keep search intent coverage intact."
    if seo >= 55 and conv < 45:
        if money_page or traffic >= 20:
            return "high_risk_money_page" if money_page else "seo_heavy", "Add a clear primary CTA, lead capture, or contextual journey link without weakening the ranking content."
        return "seo_heavy_informational", "Add a soft next step such as related product, demo, pricing, or newsletter link."
    if seo < 45 and conv >= 55:
        return "conversion_heavy", "Strengthen search-intent coverage in title, headings, and body copy while preserving the conversion path."
    if seo < 45 and conv < 45:
        return "weak_both", "Clarify the page purpose, add intent-focused copy, and include a relevant next step."
    return "mixed", "Review whether the page should serve search demand, conversion demand, or both."


def build_conversion_balance(
    pages: list[PageInfo],
    extracted_pages: list[ExtractedPage],
    *,
    search_payload: dict | None = None,
) -> dict:
    if not pages or not extracted_pages:
        return {"summary": {"status": "no_pages", "total_pages": 0}, "pages": [], "scatter": [], "high_traffic_weak_conversion": [], "cta_warnings": []}
    search = _search_lookup(pages, search_payload)
    rows = []
    warnings = []
    for i, page in enumerate(pages):
        if i >= len(extracted_pages):
            continue
        ext = extracted_pages[i]
        ctx = search.get(i, {})
        page_type = classify_page(ext).page_type
        seo, seo_parts = _seo_score(page, ext, ctx)
        conv, conv_parts = _conversion_score(ext)
        money = _is_money_page(page, ext, page_type, ctx)
        label, action = _label(seo, conv, money, _to_int(ctx.get("traffic")))
        row = {
            "url": page.url,
            "title": page.title,
            "page_type": page_type,
            "section": page.section,
            "cluster": ctx.get("cluster") or page.section or page_type,
            "traffic": _to_int(ctx.get("traffic")),
            "keywords": _to_int(ctx.get("keywords")),
            "money_page": money,
            "seo_support": seo,
            "conversion_support": conv,
            "balance_label": label,
            "recommended_action": action,
            "seo_components": seo_parts,
            "conversion_components": conv_parts,
        }
        rows.append(row)
        if row["traffic"] >= 20 and row["conversion_support"] < 45:
            warnings.append({
                "url": page.url,
                "title": page.title,
                "section": "Primary content",
                "traffic": row["traffic"],
                "page_type": page_type,
                "money_page": money,
                "warning": "High-traffic page has weak CTA/form/contact support near the main content.",
                "recommended_action": action,
            })
    rows.sort(key=lambda r: (r["balance_label"] != "high_risk_money_page", -r["traffic"], r["conversion_support"]))
    label_counts = Counter(r["balance_label"] for r in rows)
    total = len(rows)
    avg_seo = sum(r["seo_support"] for r in rows) / total if total else 0.0
    avg_conv = sum(r["conversion_support"] for r in rows) / total if total else 0.0
    conversion_efficiency = sum(r["traffic"] for r in rows if r["conversion_support"] >= 55) / max(1, sum(r["traffic"] for r in rows))
    return {
        "summary": {
            "status": "ok",
            "model": "conversion_balance_v1",
            "total_pages": total,
            "avg_seo_support": round(avg_seo, 2),
            "avg_conversion_support": round(avg_conv, 2),
            "conversion_efficiency": round(conversion_efficiency, 4),
            "high_risk_money_pages": label_counts.get("high_risk_money_page", 0),
            "seo_heavy_pages": label_counts.get("seo_heavy", 0) + label_counts.get("seo_heavy_informational", 0),
            "conversion_heavy_pages": label_counts.get("conversion_heavy", 0),
            "weak_both_pages": label_counts.get("weak_both", 0),
            "balanced_pages": label_counts.get("balanced", 0),
            "cta_warnings": len(warnings),
        },
        "pages": rows[:800],
        "scatter": rows[:1200],
        "high_traffic_weak_conversion": [r for r in rows if r["traffic"] >= 20 and r["conversion_support"] < 45][:200],
        "cta_warnings": warnings[:300],
        "interpretation": {
            "money_pages": "Product, service, home, contact, pricing/demo/signup, or commercial/transactional intent pages are held to a higher conversion-support standard.",
            "scores": "SEO support uses search traffic, keyword count, ranking position, and title/header keyword overlap. Conversion support uses CTA, primary CTA, form/contact, CTA specificity, and journey links.",
        },
    }
