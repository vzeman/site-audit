"""Generate concrete fix text for top action-plan recommendations."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable
from urllib.parse import urlparse

from .ai_agent import (
    DEFAULT_OPENROUTER_MODEL,
    AgentClient,
    OpenRouterClient,
    openrouter_api_key,
    openrouter_model,
)
from .draft_verification import verify_numeric_claims


_DRAFTABLE_PREFIXES = ("title-", "ctr-", "geo-", "gap-")
_TITLE_PREFIXES = ("title-", "ctr-")
_SEPARATORS = (" | ", " — ", " - ", " – ", " :: ", " : ")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class _DraftItem:
    rec: dict
    context: dict
    kind: str
    current_title: str = ""
    top_query: str = ""
    url: str = ""


@dataclass
class _ValidationResult:
    ok: bool
    draft: dict | None = None
    errors: list[str] = field(default_factory=list)


def build_fix_drafts(
    recommendations_payload: dict,
    pages: Iterable[Any] | None,
    search_payload: dict | None,
    client: AgentClient | None,
    *,
    model: str | None = None,
    max_items: int = 20,
    batch_size: int = 8,
) -> dict:
    """Build exact draft text for draftable action-plan recommendations.

    Summary semantics: ``failed`` counts items whose LLM draft failed
    validation (after the repair turn) and fell back; items degraded by a
    client exception are counted under ``errors`` instead of ``failed``.
    ``model_used`` is null when no draft came from the LLM.
    """
    selected = _select_items(recommendations_payload, pages, search_payload, max_items=max_items)
    model_name = model or openrouter_model(DEFAULT_OPENROUTER_MODEL)
    summary: dict[str, Any] = {
        "requested": len(selected),
        "drafted": 0,
        "llm": 0,
        "fallback": 0,
        "repaired": 0,
        "failed": 0,
    }
    drafts: dict[str, dict] = {}
    if not selected:
        return {"available": False, "drafts": drafts, "summary": summary, "model_used": None}

    active_client = client
    if active_client is None and openrouter_api_key():
        active_client = OpenRouterClient()

    if active_client is None:
        for item in selected:
            drafts[item.rec["id"]] = _fallback_draft(item)
        summary["drafted"] = len(drafts)
        summary["fallback"] = len(drafts)
        return {"available": bool(drafts), "drafts": drafts, "summary": summary, "model_used": None}

    for batch in _chunks(selected, max(1, batch_size)):
        _build_batch(batch, active_client, model_name, drafts, summary)

    summary["drafted"] = len(drafts)
    model_used = model_name if summary["llm"] else None
    return {"available": bool(drafts), "drafts": drafts, "summary": summary, "model_used": model_used}


def attach_fix_drafts(recommendations_payload: dict, fix_drafts_payload: dict | None) -> dict:
    """Attach generated drafts to item rows and card rows in a recommendations payload."""
    if not isinstance(recommendations_payload, dict) or not isinstance(fix_drafts_payload, dict):
        return recommendations_payload
    drafts = fix_drafts_payload.get("drafts") or {}
    if not isinstance(drafts, dict):
        return recommendations_payload
    for row in recommendations_payload.get("items") or []:
        draft = drafts.get(row.get("id"))
        if draft:
            row["fix_draft"] = draft
    for card in recommendations_payload.get("cards") or []:
        for row in card.get("recommendations") or []:
            draft = drafts.get(row.get("id"))
            if draft:
                row["fix_draft"] = draft
    recommendations_payload["fix_drafts"] = fix_drafts_payload
    return recommendations_payload


def _build_batch(
    batch: list[_DraftItem],
    client: AgentClient,
    model: str,
    drafts: dict[str, dict],
    summary: dict,
) -> None:
    try:
        completion = client.complete(_messages(batch), model=model, temperature=0.1, timeout=120)
        payload = _extract_json(completion.text)
    except Exception as exc:
        _fallback_batch(batch, drafts, summary, error_type=exc.__class__.__name__)
        return

    parsed = _payload_drafts(payload)
    valid, failed = _validate_batch(batch, parsed)
    for item, draft in valid:
        drafts[item.rec["id"]] = draft
        summary["llm"] += 1
    if not failed:
        return

    try:
        repair_completion = client.complete(
            _repair_messages(failed, parsed),
            model=model,
            temperature=0.0,
            timeout=120,
        )
        repair_payload = _extract_json(repair_completion.text)
        repaired = _payload_drafts(repair_payload)
    except Exception as exc:
        _fallback_batch(failed, drafts, summary, error_type=exc.__class__.__name__)
        return

    repair_valid, still_failed = _validate_batch(failed, repaired)
    for item, draft in repair_valid:
        drafts[item.rec["id"]] = draft
        summary["llm"] += 1
        summary["repaired"] += 1
    if still_failed:
        _fallback_batch(still_failed, drafts, summary)


def _fallback_batch(
    items: list[_DraftItem],
    drafts: dict[str, dict],
    summary: dict,
    *,
    error_type: str = "",
) -> None:
    for item in items:
        drafts[item.rec["id"]] = _fallback_draft(item)
    summary["fallback"] += len(items)
    if error_type:
        summary.setdefault("errors", []).append({"type": error_type, "items": len(items)})
    else:
        summary["failed"] += len(items)


def _select_items(
    recommendations_payload: dict,
    pages,
    search_payload: dict | None,
    *,
    max_items: int,
) -> list[_DraftItem]:
    page_lookup = _page_lookup(pages)
    query_lookup = _query_lookup(search_payload)
    selected: list[_DraftItem] = []
    for rec in recommendations_payload.get("items") or []:
        rec_id = str(rec.get("id") or "")
        if not rec_id.startswith(_DRAFTABLE_PREFIXES):
            continue
        item = _draft_item(rec, page_lookup, query_lookup)
        if item:
            selected.append(item)
        if len(selected) >= max_items:
            break
    return selected


def _draft_item(rec: dict, page_lookup: dict[str, dict], query_lookup: dict[str, list[dict]]) -> _DraftItem | None:
    rec_id = str(rec.get("id") or "")
    targets = [str(target) for target in (rec.get("targets") or []) if target]
    url = targets[0] if targets else ""
    page = _lookup_page(page_lookup, url) if url else {}
    evidence = rec.get("evidence") or {}
    top_queries = _top_queries(rec, page, query_lookup.get(_url_key(url), []))
    current_title = str(evidence.get("current_title") or page.get("title") or "")
    context = {
        "id": rec_id,
        "recommendation_title": rec.get("title", ""),
        "instruction": rec.get("instruction", ""),
        "url": url,
        "current_title": current_title,
        "current_description": page.get("description", ""),
        "top_queries": top_queries,
        "word_count": page.get("word_count", 0),
        "headings": (page.get("headings") or [])[:12],
        "excerpt": _page_excerpt(page),
        "evidence": _compact_evidence(evidence),
    }
    if rec_id.startswith(_TITLE_PREFIXES):
        kind = "metadata"
    elif rec_id.startswith("gap-"):
        kind = "gap"
    else:
        kind = "faq"
    return _DraftItem(
        rec=rec,
        context=context,
        kind=kind,
        current_title=current_title,
        top_query=top_queries[0] if top_queries else "",
        url=url,
    )


def _messages(items: list[_DraftItem]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You write exact SEO implementation draft text from provided audit context. "
                "Use only the Evidence JSON supplied. Do not invent statistics, capabilities, "
                "case studies, certifications, product features, pricing, rankings, dates, or "
                "claims not present in the context. If a required fact is unknown, write [NEEDS DATA]. "
                "Return strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                "For each numbered item, return one object in this shape:\n"
                '{"drafts":[{"id":"...","proposed_title":"...","proposed_meta":"...",'
                '"questions":["..."],"outline":["..."]}]}\n'
                "Rules: title and CTR items need proposed_title <=65 chars and proposed_meta <=165 chars. "
                "GEO items need exactly 3 FAQ question H2s grounded in headings and top queries. "
                "Gap items need proposed_title <=65 chars and an H2 outline of 4-7 items. "
                "Questions must end with ?. Do not add numbers unless the same number appears in the item context.\n\n"
                f"Items JSON:\n{json.dumps([item.context for item in items], ensure_ascii=False, indent=2)}"
            ),
        },
    ]


def _repair_messages(items: list[_DraftItem], previous: dict[str, dict]) -> list[dict[str, str]]:
    repair_context = []
    for item in items:
        repair_context.append({
            "context": item.context,
            "previous_draft": previous.get(item.rec["id"], {}),
        })
    return [
        {
            "role": "system",
            "content": (
                "Repair invalid SEO draft JSON. Use only the supplied context. "
                "No invented numbers, claims, capabilities, or statistics. Return strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                "Return repaired drafts only for these failed items using the same JSON shape. "
                "Respect all length limits and make questions end with ?. "
                "Do not add any two-or-more-digit number unless it appears in the context.\n\n"
                f"Failed items JSON:\n{json.dumps(repair_context, ensure_ascii=False, indent=2)}"
            ),
        },
    ]


def _validate_batch(items: list[_DraftItem], drafts_by_id: dict[str, dict]) -> tuple[list[tuple[_DraftItem, dict]], list[_DraftItem]]:
    valid: list[tuple[_DraftItem, dict]] = []
    failed: list[_DraftItem] = []
    for item in items:
        result = _validate_draft(item, drafts_by_id.get(item.rec["id"]))
        if result.ok and result.draft:
            valid.append((item, result.draft))
        else:
            failed.append(item)
    return valid, failed


def _validate_draft(item: _DraftItem, raw: dict | None) -> _ValidationResult:
    if not isinstance(raw, dict):
        return _ValidationResult(False, errors=["missing draft"])
    draft: dict[str, Any] = {}
    context_text = json.dumps(item.context, ensure_ascii=False)
    errors: list[str] = []

    if item.kind in {"metadata", "gap"}:
        title = _clean_text(raw.get("proposed_title"))
        if not title or len(title) > 65:
            errors.append("title length")
        if item.current_title and title.casefold() == item.current_title.casefold():
            errors.append("title unchanged")
        if _has_unseen_number(title, context_text):
            errors.append("title invented number")
        if title:
            draft["proposed_title"] = title

    if item.kind == "metadata":
        meta = _clean_text(raw.get("proposed_meta"))
        if not meta or len(meta) > 165:
            errors.append("meta length")
        if _has_unseen_number(meta, context_text):
            errors.append("meta invented number")
        if meta:
            draft["proposed_meta"] = meta

    if item.kind == "faq":
        questions = [_clean_text(q) for q in _as_list(raw.get("questions"))]
        questions = [q for q in questions if q]
        if len(questions) != 3:
            errors.append("question count")
        for q in questions:
            if len(q) > 90 or not q.endswith("?"):
                errors.append("question shape")
            if _has_unseen_number(q, context_text):
                errors.append("question invented number")
        if questions:
            draft["questions"] = questions[:3]

    if item.kind == "gap":
        outline = [_clean_text(line) for line in _as_list(raw.get("outline"))]
        outline = [line for line in outline if line]
        if len(outline) < 4 or len(outline) > 7:
            errors.append("outline count")
        for line in outline:
            if len(line) > 80:
                errors.append("outline length")
            if _has_unseen_number(line, context_text):
                errors.append("outline invented number")
        if outline:
            draft["outline"] = outline[:7]

    if errors:
        return _ValidationResult(False, errors=errors)
    draft["generated_by"] = "llm"
    return _ValidationResult(True, draft=draft)


def _fallback_draft(item: _DraftItem) -> dict:
    query = item.top_query or _clean_text((item.context.get("evidence") or {}).get("query")) or _clean_text(item.rec.get("title"))
    site_name = _site_or_page_name(item.context, item.url)
    draft: dict[str, Any] = {"generated_by": "fallback"}
    if item.kind in {"metadata", "gap"}:
        title = _truncate(f"{query} — {site_name}", 65)
        if item.current_title and title.casefold() == item.current_title.casefold():
            title = _truncate(query or item.current_title, 65)
        draft["proposed_title"] = title
    if item.kind == "metadata":
        meta_source = (
            _clean_text(item.context.get("current_description"))
            or _clean_text(item.context.get("excerpt"))
        )
        if meta_source:
            draft["proposed_meta"] = _truncate(_first_sentence(meta_source), 165)
    if item.kind == "faq":
        draft["questions"] = _fallback_questions(item)
    return draft


def _fallback_questions(item: _DraftItem) -> list[str]:
    queries = list(item.context.get("top_queries") or [])
    if not queries:
        queries = [_clean_text((item.context.get("evidence") or {}).get("query")) or _clean_text(item.rec.get("title"))]
    out: list[str] = []
    for query in queries:
        q = _clean_text(query)
        if not q:
            continue
        if q.endswith("?"):
            text = q
        else:
            text = f"What is {q}?"
        if len(text) > 90:
            text = _truncate(text.rstrip("?"), 89) + "?"
        if text not in out:
            out.append(text)
        if len(out) >= 3:
            break
    filler_index = 0
    while len(out) < 3:
        seed = ["How can this help?", "Why does this matter?", "What should readers know?"][filler_index]
        filler_index += 1
        out.append(seed)
    return out[:3]


def _page_lookup(pages) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for page in pages or []:
        row = _object_dict(page)
        url = str(row.get("url") or "")
        if not url:
            continue
        normalized = {
            "url": url,
            "title": row.get("title", ""),
            "description": row.get("description", ""),
            "word_count": row.get("word_count", 0),
            "headings": row.get("headings") or row.get("headers") or [],
            "paragraphs": row.get("paragraphs") or [],
            "body": row.get("body", ""),
            "section": row.get("section", ""),
        }
        for key in _url_keys(url):
            out[key] = normalized
    return out


def _query_lookup(search_payload: dict | None) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for row in (search_payload or {}).get("query_pages") or []:
        url = str(row.get("matched_url") or row.get("url") or "")
        query = str(row.get("keyword") or row.get("query") or "")
        if not url or not query:
            continue
        for key in _url_keys(url):
            out.setdefault(key, []).append({"query": query, "impressions": _safe_float(row.get("impressions"))})
    for row in (search_payload or {}).get("top_pages") or []:
        url = str(row.get("matched_url") or row.get("url") or "")
        query = str(row.get("top_keyword") or row.get("keyword") or "")
        if not url or not query:
            continue
        for key in _url_keys(url):
            out.setdefault(key, []).append({"query": query, "impressions": _safe_float(row.get("impressions") or row.get("traffic"))})
    for rows in out.values():
        rows.sort(key=lambda r: (-_safe_float(r.get("impressions")), str(r.get("query") or "")))
    return out


def _top_queries(rec: dict, page: dict, query_rows: list[dict]) -> list[str]:
    evidence = rec.get("evidence") or {}
    candidates = [
        evidence.get("query"),
        evidence.get("keyword"),
        *((evidence.get("suggested_keywords") or [])[:4] if isinstance(evidence.get("suggested_keywords"), list) else []),
        *[row.get("query") for row in query_rows[:4]],
    ]
    out: list[str] = []
    for candidate in candidates:
        text = _clean_text(candidate)
        if text and text not in out:
            out.append(text)
        if len(out) >= 5:
            break
    if not out:
        title = _clean_text(page.get("title"))
        if title:
            out.append(title)
    return out


def _payload_drafts(payload: Any) -> dict[str, dict]:
    rows = payload.get("drafts") if isinstance(payload, dict) else []
    out: dict[str, dict] = {}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        rec_id = str(row.get("id") or "")
        if rec_id:
            out[rec_id] = row
    return out


def _extract_json(text: str) -> Any:
    raw = str(text or "").strip()
    if not raw:
        return None
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", raw, flags=re.IGNORECASE | re.DOTALL)
    candidates = [raw, *fenced, _slice_between(raw, "{", "}"), _slice_between(raw, "[", "]")]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _slice_between(text: str, start: str, end: str) -> str:
    try:
        lo = text.index(start)
        hi = text.rindex(end)
    except ValueError:
        return ""
    if hi <= lo:
        return ""
    return text[lo : hi + 1]


def _compact_evidence(evidence: dict) -> dict:
    allowed = {
        "query",
        "keyword",
        "current_title",
        "suggested_keywords",
        "probable_cause",
        "flags",
        "answerability_score",
        "best_similarity",
        "volume",
        "position",
        "period",
    }
    return {key: value for key, value in (evidence or {}).items() if key in allowed}


def _page_excerpt(page: dict) -> str:
    paragraphs = [_clean_text(p) for p in (page.get("paragraphs") or []) if _clean_text(p)]
    if paragraphs:
        return _truncate(" ".join(paragraphs[:3]), 500)
    return _truncate(_clean_text(page.get("body")), 500)


def _site_or_page_name(context: dict, url: str) -> str:
    title = _clean_text(context.get("current_title"))
    for sep in _SEPARATORS:
        if sep in title:
            tail = title.rsplit(sep, 1)[-1].strip()
            if tail:
                return tail
    parsed = urlparse(url or "")
    domain = parsed.netloc or parsed.path.split("/", 1)[0]
    return domain.removeprefix("www.") or "site"


def _first_sentence(text: str) -> str:
    cleaned = _clean_text(text)
    if not cleaned:
        return ""
    parts = _SENTENCE_RE.split(cleaned, maxsplit=1)
    return parts[0].strip() or cleaned


def _has_unseen_number(text: str, context_text: str) -> bool:
    """Flag numeric claims in ``text`` that are not grounded in the context.

    Delegates to :func:`site_audit.draft_verification.verify_numeric_claims` so the
    main report and the SERP-gap verification loop share one digit-grounding
    checker (typed claims, fuzzy formatting variants, year/list/date exclusions,
    [NEEDS DATA] exemption). Known, accepted limitation: any number present
    anywhere in the context JSON (e.g. word_count, evidence volume/position)
    legitimizes the same number in the draft regardless of meaning.
    """
    return bool(verify_numeric_claims(text or "", [context_text or ""])["unverified"])


def _object_dict(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    return dict(getattr(value, "__dict__", {}) or {})


def _lookup_page(lookup: dict[str, dict], url: str) -> dict:
    for key in _url_keys(url):
        if key in lookup:
            return lookup[key]
    return {}


def _url_key(url: str) -> str:
    keys = sorted(_url_keys(url))
    return keys[0] if keys else ""


def _url_keys(url: object) -> set[str]:
    raw = str(url or "").strip()
    if not raw:
        return set()
    keys = {raw, raw.rstrip("/")}
    parsed = urlparse(raw)
    if not parsed.netloc:
        return {key for key in keys if key}
    scheme = parsed.scheme or "https"
    netloc = parsed.netloc.lower()
    path = parsed.path or "/"
    path_trimmed = path.rstrip("/") or "/"
    netlocs = {netloc, netloc.removeprefix("www.")}
    if not netloc.startswith("www."):
        netlocs.add(f"www.{netloc}")
    for candidate_netloc in netlocs:
        for candidate_path in {path, path_trimmed}:
            full = f"{scheme}://{candidate_netloc}{candidate_path}"
            keys.add(full)
            keys.add(full.rstrip("/"))
    return {key for key in keys if key}


def _as_list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _chunks(items: list[_DraftItem], size: int) -> Iterable[list[_DraftItem]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _truncate(text: str, limit: int) -> str:
    cleaned = _clean_text(text)
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 1)].rstrip() + "…"
