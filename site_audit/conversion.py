"""Conversion and CTA signal analysis."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urlparse

from .extractor import ExtractedPage


_LEAD_PATH_TERMS = (
    "contact",
    "demo",
    "quote",
    "estimate",
    "pricing",
    "signup",
    "register",
    "book",
    "consult",
    "apply",
)
_GENERIC_CTA_TEXT = {
    "learn more",
    "read more",
    "click here",
    "more",
    "details",
}


@dataclass
class ConversionReport:
    summary: dict
    issues_by_type: dict[str, int]
    top_ctas: list[dict]
    per_page: list[dict]


def _is_lead_page(url: str, title: str) -> bool:
    parsed = urlparse(url)
    haystack = f"{parsed.path} {title}".lower().replace("-", " ")
    return any(term in haystack for term in _LEAD_PATH_TERMS)


def _generic_cta_count(ctas: list[dict]) -> int:
    return sum(1 for cta in ctas if (cta.get("text") or "").strip().lower() in _GENERIC_CTA_TEXT)


def analyze(pages: Iterable[ExtractedPage]) -> ConversionReport:
    page_list = list(pages)
    issues_by_type: Counter[str] = Counter()
    cta_text_counts: Counter[str] = Counter()
    rows: list[dict] = []

    total_ctas = 0
    total_primary_ctas = 0
    total_forms = 0
    total_form_fields = 0
    pages_with_cta = 0
    pages_with_primary_cta = 0
    pages_with_forms = 0
    pages_with_contact_link = 0

    for page in page_list:
        signals = page.conversion_signals or {}
        ctas = list(signals.get("ctas") or [])
        forms = list(signals.get("forms") or [])
        cta_count = int(signals.get("cta_count") or len(ctas))
        primary_cta_count = int(signals.get("primary_cta_count") or sum(1 for cta in ctas if cta.get("primary")))
        form_count = int(signals.get("form_count") or len(forms))
        form_field_count = int(signals.get("form_field_count") or sum(int(form.get("field_count") or 0) for form in forms))
        contact_link_count = int(signals.get("contact_link_count") or 0)
        generic_ctas = _generic_cta_count(ctas)
        lead_page = _is_lead_page(page.url, page.title)
        forms_without_submit = sum(1 for form in forms if not form.get("has_submit"))

        total_ctas += cta_count
        total_primary_ctas += primary_cta_count
        total_forms += form_count
        total_form_fields += form_field_count
        if cta_count:
            pages_with_cta += 1
        if primary_cta_count:
            pages_with_primary_cta += 1
        if form_count:
            pages_with_forms += 1
        if contact_link_count:
            pages_with_contact_link += 1
        cta_text_counts.update((cta.get("text") or "").strip() for cta in ctas if (cta.get("text") or "").strip())

        issues: list[str] = []
        if not cta_count:
            issues.append("no_cta")
        if not primary_cta_count:
            issues.append("no_primary_cta")
        if lead_page and not form_count and not contact_link_count:
            issues.append("lead_page_without_capture")
        if cta_count > 8:
            issues.append("cta_overload")
        if generic_ctas and generic_ctas == cta_count:
            issues.append("only_generic_ctas")
        if forms_without_submit:
            issues.append("form_without_submit")

        issues_by_type.update(issues)
        rows.append({
            "url": page.url,
            "title": page.title,
            "cta_count": cta_count,
            "primary_cta_count": primary_cta_count,
            "form_count": form_count,
            "form_field_count": form_field_count,
            "contact_link_count": contact_link_count,
            "lead_page": lead_page,
            "generic_cta_count": generic_ctas,
            "forms_without_submit": forms_without_submit,
            "top_ctas": ctas[:8],
            "issues": issues,
        })

    total_pages = len(page_list)
    summary = {
        "total_pages": total_pages,
        "total_ctas": total_ctas,
        "total_primary_ctas": total_primary_ctas,
        "total_forms": total_forms,
        "total_form_fields": total_form_fields,
        "pages_with_cta": pages_with_cta,
        "cta_coverage": pages_with_cta / total_pages if total_pages else 0.0,
        "pages_with_primary_cta": pages_with_primary_cta,
        "primary_cta_coverage": pages_with_primary_cta / total_pages if total_pages else 0.0,
        "pages_with_forms": pages_with_forms,
        "form_coverage": pages_with_forms / total_pages if total_pages else 0.0,
        "pages_with_contact_link": pages_with_contact_link,
        "contact_link_coverage": pages_with_contact_link / total_pages if total_pages else 0.0,
        "avg_ctas_per_page": total_ctas / total_pages if total_pages else 0.0,
        "pages_without_cta": issues_by_type.get("no_cta", 0),
        "pages_without_primary_cta": issues_by_type.get("no_primary_cta", 0),
        "lead_pages_without_capture": issues_by_type.get("lead_page_without_capture", 0),
        "cta_overload_pages": issues_by_type.get("cta_overload", 0),
        "forms_without_submit_pages": issues_by_type.get("form_without_submit", 0),
    }
    top_ctas = [
        {"text": text, "count": count}
        for text, count in cta_text_counts.most_common(50)
    ]
    rows.sort(key=lambda row: (-len(row["issues"]), -int(row["lead_page"]), row["url"]))
    return ConversionReport(
        summary=summary,
        issues_by_type=dict(issues_by_type),
        top_ctas=top_ctas,
        per_page=rows,
    )


def to_payload(report: ConversionReport) -> dict:
    return {
        "summary": report.summary,
        "issues_by_type": report.issues_by_type,
        "top_ctas": report.top_ctas,
        "per_page": report.per_page,
    }
