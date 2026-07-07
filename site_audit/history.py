"""Historical report snapshots and before/after impact diffs."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from .cache import domain_slug


HISTORY_CORRELATION_CAVEATS = [
    "Impact rows are before/after associations, not causal proof.",
    "The configured comparison window is used as the observation window; confirm the same window in GSC or analytics when available.",
    "Validate against seasonality, provider sampling noise, ranking volatility, and unrelated site changes.",
    "Use wider windows for low-traffic pages before treating a delta as meaningful.",
]

RECOMMENDATION_OUTCOME_CAVEAT = HISTORY_CORRELATION_CAVEATS[0]

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in", "into", "is",
    "it", "of", "on", "or", "the", "this", "to", "vs", "with", "without", "your",
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(float(str(value).replace(",", "")))
    except (TypeError, ValueError):
        return default


def _hash(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


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
    path = parts.path or "/"
    path_trimmed = path.rstrip("/") or "/"
    netloc = parts.netloc.lower()
    netlocs = {netloc, netloc[4:] if netloc.startswith("www.") else f"www.{netloc}"}
    schemes = {(parts.scheme or "https").lower(), "https", "http"}
    for scheme in schemes:
        for host in netlocs:
            for candidate_path in {path, path_trimmed}:
                normalized = urlunsplit((scheme, host, candidate_path, "", ""))
                keys.add(normalized)
                keys.add(normalized.rstrip("/"))
    return {key for key in keys if key}


def _store_url(out: dict[str, dict], url: object, row: dict, score_key: str = "") -> None:
    if not url:
        return
    for key in _url_keys(url):
        current = out.get(key)
        if current is None or not score_key or _safe_float(row.get(score_key)) >= _safe_float(current.get(score_key)):
            out[key] = row


def _lookup_url(lookup: dict[str, dict], *urls: object) -> dict:
    best: dict = {}
    for url in urls:
        for key in _url_keys(url):
            row = lookup.get(key)
            if row and len(row) >= len(best):
                best = row
    return best


def _load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _search_lookup(search_payload: dict | None) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in (search_payload or {}).get("top_pages") or []:
        url = row.get("matched_url") or row.get("url") or ""
        if not url:
            continue
        _store_url(out, url, {
            "traffic": _safe_float(row.get("traffic")),
            "keywords": _safe_int(row.get("keywords") or row.get("keywords_total")),
            "position": _safe_float(row.get("top_keyword_position") or row.get("position")),
            "top_keyword": row.get("top_keyword") or "",
            "serp_title": row.get("top_keyword_title") or row.get("serp_title") or "",
            "clicks": _safe_float(row.get("clicks")),
            "impressions": _safe_float(row.get("impressions")),
        }, score_key="traffic")
    for row in (search_payload or {}).get("organic_keywords") or []:
        url = row.get("matched_url") or row.get("url") or ""
        if not url:
            continue
        current = _lookup_url(out, url)
        traffic = max(_safe_float(current.get("traffic")), _safe_float(row.get("traffic")))
        position = min(_safe_float(current.get("position"), 999.0) or 999.0, _safe_float(row.get("position"), 999.0))
        _store_url(out, url, {
            **current,
            "traffic": traffic,
            "keywords": max(_safe_int(current.get("keywords")), 1),
            "position": position if position < 999.0 else 0,
            "top_keyword": current.get("top_keyword") or row.get("keyword") or "",
            "serp_title": current.get("serp_title") or row.get("page_title") or row.get("serp_title") or "",
        }, score_key="traffic")
    return out


def _row_lookup(rows: list[dict], keys: tuple[str, ...] = ("url",), score_key: str = "") -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in rows or []:
        for key in keys:
            if row.get(key):
                _store_url(out, row.get(key), row, score_key=score_key)
    return out


def _snapshot_recommendations(recommendations_payload: dict | list | None, limit: int = 100) -> list[dict]:
    if isinstance(recommendations_payload, dict):
        rows = list(recommendations_payload.get("items") or [])
    elif isinstance(recommendations_payload, list):
        rows = list(recommendations_payload)
    else:
        rows = []
    compact: list[dict] = []
    for row in rows:
        if not isinstance(row, dict) or row.get("suppressed"):
            continue
        rec_id = str(row.get("id") or "").strip()
        if not rec_id:
            continue
        targets = [str(target) for target in (row.get("targets") or []) if target]
        compact.append({
            "id": rec_id,
            "category": str(row.get("category") or ""),
            "type": str(row.get("type") or ""),
            "priority": str(row.get("priority") or ""),
            "primary_url": str(row.get("primary_url") or (targets[0] if targets else "")),
            "targets": targets,
            "title": str(row.get("title") or ""),
            "estimated_clicks_gain": row.get("estimated_clicks_gain"),
        })
        if len(compact) >= limit:
            break
    return compact


def build_history_snapshot(
    domain: str,
    pages: list,
    extracted_pages: list | None = None,
    *,
    outlinks_map: dict[str, list[tuple[str, str]]] | None = None,
    structured_data: dict | None = None,
    freshness: dict | None = None,
    metadata_quality: dict | None = None,
    indexability: dict | None = None,
    search_payload: dict | None = None,
    recommendations_payload: dict | list | None = None,
    snapshot_id: str | None = None,
) -> dict:
    raw_max_pages = os.getenv("SITE_AUDIT_HISTORY_MAX_PAGES", "20000")
    try:
        max_pages = max(0, int(raw_max_pages))
    except ValueError:
        max_pages = 20000
    if max_pages and len(pages) > max_pages:
        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        return {
            "summary": {
                "status": "skipped_large_site",
                "model": "history_snapshot_v1",
                "domain": domain,
                "snapshot_id": snapshot_id or "",
                "created_at": now,
                "pages": len(pages),
                "max_pages": max_pages,
                "reason": "History snapshots include per-page headings, paragraphs, and links; skipped to keep large audits memory-safe.",
            },
            "pages": [],
        }
    extracted_pages = extracted_pages or []
    outlinks_map = outlinks_map or {}
    structured = _row_lookup((structured_data or {}).get("per_page") or [], ("url",))
    fresh = _row_lookup((freshness or {}).get("per_page") or [], ("url",))
    metadata = _row_lookup((metadata_quality or {}).get("per_page") or [], ("url",))
    index_rows = _row_lookup((indexability or {}).get("per_page") or [], ("url",))
    search = _search_lookup(search_payload)
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    page_rows: list[dict] = []
    for i, page in enumerate(pages):
        ext = extracted_pages[i] if i < len(extracted_pages) else None
        url = getattr(page, "url", "") or ""
        headings = []
        for header in getattr(ext, "headers_rich", []) or []:
            text = str(header.get("text") or "").strip()
            if text:
                headings.append({"level": _safe_int(header.get("level")), "text": text, "hash": _hash(text)})
        if not headings:
            headings = [{"level": 0, "text": str(text), "hash": _hash(text)} for text in (getattr(ext, "headings", []) or []) if str(text).strip()]
        paragraphs = [
            {
                "index": idx,
                "hash": _hash(text),
                "words": len(str(text).split()),
                "excerpt": str(text).strip()[:220],
            }
            for idx, text in enumerate(getattr(ext, "paragraphs", []) or [])
            if str(text).strip()
        ]
        links = sorted({
            str(target or "").rstrip("/")
            for target, _anchor in outlinks_map.get(url, []) or []
            if target
        })
        structured_row = _lookup_url(structured, url)
        fresh_row = _lookup_url(fresh, url)
        metadata_row = _lookup_url(metadata, url)
        index_row = _lookup_url(index_rows, url)
        search_row = _lookup_url(search, url)
        schema_types = sorted(set(structured_row.get("types") or getattr(ext, "schema_types", []) or []))
        title = getattr(page, "title", "") or getattr(ext, "title", "") or ""
        description = getattr(page, "description", "") or getattr(ext, "description", "") or ""
        canonical_url = metadata_row.get("canonical_url") or getattr(ext, "canonical_url", "") or ""
        h1 = getattr(ext, "h1", "") or ""
        redirect_target_url = index_row.get("redirect_target_url") or ""
        status_code = _safe_int(index_row.get("http_status") or index_row.get("status_code"))
        serp_title = search_row.get("serp_title") or ""
        page_rows.append({
            "url": url,
            "title": title,
            "description": description,
            "section": getattr(page, "section", "") or "",
            "word_count": _safe_int(getattr(page, "word_count", 0)),
            "word_count_hash": _hash(str(_safe_int(getattr(page, "word_count", 0)))),
            "title_hash": _hash(title),
            "description_hash": _hash(description),
            "canonical_url": canonical_url,
            "canonical_hash": _hash(canonical_url),
            "h1": h1,
            "h1_hash": _hash(h1),
            "serp_title": serp_title,
            "serp_title_hash": _hash(serp_title),
            "redirect_target_url": redirect_target_url,
            "status_code": status_code,
            "redirect_target_hash": _hash(redirect_target_url),
            "heading_hash": _hash("|".join(h["hash"] for h in headings)),
            "paragraph_hash": _hash("|".join(p["hash"] for p in paragraphs)),
            "link_hash": _hash("|".join(links)),
            "schema_hash": _hash("|".join(schema_types)),
            "metadata_hash": _hash(json.dumps(metadata_row, sort_keys=True, default=str)),
            "freshness_hash": _hash(json.dumps(fresh_row, sort_keys=True, default=str)),
            "headings": headings,
            "paragraphs": paragraphs,
            "links": links,
            "schema_types": schema_types,
            "metadata_issues": metadata_row.get("issues") or [],
            "freshness": {
                "bucket": fresh_row.get("bucket") or "",
                "age_days": fresh_row.get("age_days"),
                "date": fresh_row.get("date") or "",
                "issues": fresh_row.get("issues") or [],
            },
            "metrics": {
                "traffic": _safe_float(search_row.get("traffic")),
                "keywords": _safe_int(search_row.get("keywords")),
                "position": _safe_float(search_row.get("position")),
                "top_keyword": search_row.get("top_keyword") or "",
                "serp_title": serp_title,
                "clicks": _safe_float(search_row.get("clicks")),
                "impressions": _safe_float(search_row.get("impressions")),
            },
        })

    total_traffic = sum(_safe_float(row["metrics"].get("traffic")) for row in page_rows)
    total_clicks = sum(_safe_float(row["metrics"].get("clicks")) for row in page_rows)
    total_impressions = sum(_safe_float(row["metrics"].get("impressions")) for row in page_rows)
    positions = [
        _safe_float(row["metrics"].get("position"))
        for row in page_rows
        if _safe_float(row["metrics"].get("position")) > 0
    ]
    recommendations = _snapshot_recommendations(recommendations_payload)
    return {
        "summary": {
            "status": "ok",
            "model": "history_snapshot_v1",
            "domain": domain,
            "snapshot_id": snapshot_id or "",
            "created_at": now,
            "pages": len(page_rows),
            "total_traffic": round(total_traffic, 1),
            "total_clicks": round(total_clicks, 1),
            "total_impressions": round(total_impressions, 1),
            "avg_position": round(sum(positions) / len(positions), 2) if positions else 0,
            "total_keywords": sum(_safe_int(row["metrics"].get("keywords")) for row in page_rows),
            "paragraphs": sum(len(row.get("paragraphs") or []) for row in page_rows),
            "links": sum(len(row.get("links") or []) for row in page_rows),
            "recommendations": len(recommendations),
        },
        "pages": page_rows,
        "recommendations": recommendations,
    }


def snapshot_root(projects_root: Path, domain: str) -> Path:
    return Path(projects_root) / domain_slug(domain) / "snapshots"


def list_snapshots(domain: str, projects_root: Path) -> list[dict]:
    root = snapshot_root(projects_root, domain)
    rows = []
    for path in sorted(root.iterdir()) if root.exists() else []:
        if not path.is_dir():
            continue
        metadata = _load_json(path / "snapshot_metadata.json", {})
        history = _load_json(path / "report" / "history_snapshot.json", {})
        rows.append({
            "snapshot_id": path.name,
            "path": str(path),
            "created_at": metadata.get("created_at") or (history.get("summary") or {}).get("created_at") or "",
            "pages": (history.get("summary") or {}).get("pages", 0),
            "traffic": (history.get("summary") or {}).get("total_traffic", 0),
        })
    return rows


def save_report_snapshot(
    domain: str,
    projects_root: Path,
    report_dir: Path,
    *,
    snapshot_id: str | None = None,
    overwrite: bool = False,
) -> Path:
    snapshot_id = snapshot_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = snapshot_root(projects_root, domain) / snapshot_id
    if target.exists() and not overwrite:
        suffix = hashlib.sha1(str(report_dir).encode("utf-8")).hexdigest()[:6]
        target = snapshot_root(projects_root, domain) / f"{snapshot_id}-{suffix}"
    target.mkdir(parents=True, exist_ok=True)
    report_target = target / "report"
    if report_target.exists():
        shutil.rmtree(report_target)
    shutil.copytree(report_dir, report_target)
    metadata = {
        "snapshot_id": target.name,
        "domain": domain,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source_report_dir": str(report_dir),
        "report_dir": str(report_target),
    }
    (target / "snapshot_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return target


def _snapshot_payload(domain: str, projects_root: Path, snapshot_id: str) -> dict:
    path = snapshot_root(projects_root, domain) / snapshot_id
    payload = _load_json(path / "report" / "history_snapshot.json", {})
    if isinstance(payload, dict) and payload:
        summary = payload.setdefault("summary", {})
        if isinstance(summary, dict) and not summary.get("snapshot_id"):
            summary["snapshot_id"] = snapshot_id
        return payload
    report = path / "report"
    pages = _load_json(report / "pages.json", [])
    search = _load_json(report / "ahrefs.json", {}) or _load_json(report / "search.json", {})
    structured = _load_json(report / "structured_data.json", {})
    freshness = _load_json(report / "freshness.json", {})
    metadata = _load_json(report / "metadata_quality.json", {})
    recommendations = _load_json(report / "recommendations.json", {})
    page_objs = [type("Page", (), row)() for row in pages]
    return build_history_snapshot(
        domain,
        page_objs,
        [],
        structured_data=structured,
        freshness=freshness,
        metadata_quality=metadata,
        search_payload=search,
        recommendations_payload=recommendations,
        snapshot_id=snapshot_id,
    )


def load_snapshot_payload(domain: str, projects_root: Path, snapshot_id: str) -> dict:
    return _snapshot_payload(domain, projects_root, snapshot_id)


def _page_map(snapshot: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in snapshot.get("pages") or []:
        url = str(row.get("url") or "").rstrip("/")
        if url:
            out[url] = row
    return out


def _set_delta(before: list, after: list, key: str = "") -> tuple[list, list]:
    if key:
        b = {str(item.get(key) or "") for item in before or [] if item.get(key)}
        a = {str(item.get(key) or "") for item in after or [] if item.get(key)}
    else:
        b = {str(item) for item in before or [] if item}
        a = {str(item) for item in after or [] if item}
    return sorted(a - b), sorted(b - a)


def _snapshot_total(snapshot: dict, summary_key: str, metric_key: str) -> float:
    summary = snapshot.get("summary") or {}
    if summary.get(summary_key) not in (None, ""):
        return _safe_float(summary.get(summary_key))
    return sum(_safe_float((row.get("metrics") or {}).get(metric_key)) for row in snapshot.get("pages") or [])


def _snapshot_avg_position(snapshot: dict) -> float:
    summary = snapshot.get("summary") or {}
    if summary.get("avg_position") not in (None, ""):
        return _safe_float(summary.get("avg_position"))
    positions = [
        _safe_float((row.get("metrics") or {}).get("position"))
        for row in snapshot.get("pages") or []
        if _safe_float((row.get("metrics") or {}).get("position")) > 0
    ]
    return round(sum(positions) / len(positions), 2) if positions else 0.0


def _snapshot_lookup(snapshot: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in snapshot.get("pages") or []:
        _store_url(out, row.get("url"), row)
    return out


def _inbound_counts(snapshot: dict) -> Counter:
    counts: Counter = Counter()
    for row in snapshot.get("pages") or []:
        for target in row.get("links") or []:
            for key in _url_keys(target):
                counts[key] += 1
    return counts


def _rec_prefix(rec: dict) -> str:
    rec_id = str(rec.get("id") or "")
    if rec_id.startswith("geo-access-"):
        return "geo-access"
    if rec_id.startswith("geo-aio-"):
        return "geo-aio"
    if rec_id.startswith("geo-cite-"):
        return "geo-cite"
    if rec_id.startswith("geo-chunk-"):
        return "geo-chunk"
    return rec_id.split("-", 1)[0]


def _rec_query(rec: dict) -> str:
    evidence = rec.get("evidence") or {}
    for key in ("query", "keyword"):
        value = str(evidence.get(key) or "").strip()
        if value:
            return value
    title = str(rec.get("title") or "")
    match = re.search(r'"([^"]+)"', title)
    return match.group(1) if match else title


def _tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", str(text or "").lower())
        if len(token) > 1 and token not in _STOPWORDS
    ]


def _gap_page_match(rec: dict, before_lookup: dict[str, dict], after_snapshot: dict) -> dict:
    query_tokens = _tokens(_rec_query(rec))
    if not query_tokens:
        return {}
    required = max(1, (len(query_tokens) + 1) // 2)
    candidates = [
        row for row in after_snapshot.get("pages") or []
        if not _lookup_url(before_lookup, row.get("url"))
    ]
    for row in candidates:
        haystack = f"{row.get('url') or ''} {row.get('title') or ''}".lower()
        if sum(1 for token in query_tokens if token in haystack) >= required:
            return row
    return {}


def _is_redirect_page(page: dict) -> bool:
    status_code = _safe_int(page.get("status_code"))
    return bool(300 <= status_code < 400 or page.get("redirect_target_url"))


def _page_hash_changed(before: dict, after: dict, keys: tuple[str, ...]) -> bool:
    if not before or not after:
        return False
    return any(
        before.get(key) != after.get(key)
        for key in keys
        if key in before and key in after
    )


def _target_change_status(
    rec: dict,
    after_snapshot: dict,
    before_lookup: dict[str, dict],
    after_lookup: dict[str, dict],
    before_inbound: Counter,
    after_inbound: Counter,
) -> tuple[str, dict]:
    prefix = _rec_prefix(rec)
    targets = [str(target) for target in (rec.get("targets") or []) if target]
    primary_url = str(rec.get("primary_url") or (targets[0] if targets else "") or "")

    if prefix == "gap":
        match = _gap_page_match(rec, before_lookup, after_snapshot)
        return ("implemented" if match else "not_implemented"), match

    if prefix in {"dup", "cann"}:
        removal_targets = targets[1:] if len(targets) > 1 else targets
        if not removal_targets:
            return "unknown", {}
        outcomes = []
        metric_page: dict = {}
        for target in removal_targets:
            before_page = _lookup_url(before_lookup, target)
            after_page = _lookup_url(after_lookup, target)
            if not before_page:
                outcomes.append("unknown")
            elif not after_page or _is_redirect_page(after_page):
                outcomes.append("implemented")
            else:
                outcomes.append("not_implemented")
                metric_page = metric_page or after_page
        if all(status == "implemented" for status in outcomes):
            return "implemented", metric_page
        if any(status == "implemented" for status in outcomes):
            return "partially", metric_page
        if any(status == "unknown" for status in outcomes):
            return "unknown", metric_page
        return "not_implemented", metric_page

    if not primary_url:
        return "unknown", {}
    before_primary = _lookup_url(before_lookup, primary_url)
    after_primary = _lookup_url(after_lookup, primary_url)
    if not before_primary:
        return "unknown", {}
    if not after_primary:
        return "page_removed", before_primary

    if prefix in {"title", "ctr"}:
        changed = [_page_hash_changed(before_primary, after_primary, ("title_hash", "description_hash"))]
    elif prefix in {"geo", "geo-aio", "geo-cite", "geo-chunk", "out", "wh", "move"}:
        check_targets = targets or [primary_url]
        changed = []
        for target in check_targets:
            before_page = _lookup_url(before_lookup, target)
            after_page = _lookup_url(after_lookup, target)
            if before_page and after_page:
                changed.append(_page_hash_changed(before_page, after_page, ("paragraph_hash", "heading_hash", "h1_hash")))
        if not changed:
            return "unknown", after_primary
    elif prefix in {"link", "plink", "anchor"}:
        changed = [_page_hash_changed(before_primary, after_primary, ("link_hash",))]
    elif prefix in {"orphan", "deep"}:
        inbound_before = max((before_inbound.get(key, 0) for key in _url_keys(primary_url)), default=0)
        inbound_after = max((after_inbound.get(key, 0) for key in _url_keys(primary_url)), default=0)
        changed = [
            _page_hash_changed(before_primary, after_primary, ("link_hash",))
            or inbound_after > inbound_before
        ]
    else:
        return "unknown", after_primary

    if all(changed):
        return "implemented", after_primary
    if any(changed):
        return "partially", after_primary
    return "not_implemented", after_primary


def _metric_delta(before_page: dict, after_page: dict) -> dict:
    before_metrics = before_page.get("metrics") or {}
    after_metrics = after_page.get("metrics") or {}
    traffic_before = _safe_float(before_metrics.get("traffic"))
    traffic_after = _safe_float(after_metrics.get("traffic"))
    clicks_before = _safe_float(before_metrics.get("clicks"))
    clicks_after = _safe_float(after_metrics.get("clicks"))
    impressions_before = _safe_float(before_metrics.get("impressions"))
    impressions_after = _safe_float(after_metrics.get("impressions"))
    before_position = _safe_float(before_metrics.get("position"))
    after_position = _safe_float(after_metrics.get("position"))
    position_delta = round(after_position - before_position, 2) if before_position and after_position else None
    change_count = sum(
        1
        for key in ("title_hash", "description_hash", "heading_hash", "paragraph_hash", "link_hash", "schema_hash")
        if before_page.get(key) != after_page.get(key)
    )
    confidence, caveat = _confidence(
        change_count,
        traffic_before,
        traffic_after - traffic_before,
        clicks_after - clicks_before,
        impressions_after - impressions_before,
    )
    return {
        "traffic_delta": round(traffic_after - traffic_before, 1),
        "clicks_delta": round(clicks_after - clicks_before, 1),
        "position_delta": position_delta,
        "confidence": confidence,
        "confidence_caveat": caveat,
    }


def _avg(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 2) if values else None


def _outcome_aggregates(rows: list[dict]) -> dict:
    by_category: dict[str, dict] = {}
    by_status: dict[str, dict] = {}
    position_by_status: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        category = row.get("category") or "uncategorized"
        status = row.get("change_status") or "unknown"
        by_category.setdefault(category, {"count": 0, "statuses": {}})
        by_category[category]["count"] += 1
        by_category[category]["statuses"][status] = by_category[category]["statuses"].get(status, 0) + 1
        by_status.setdefault(status, {"count": 0})
        by_status[status]["count"] += 1
        if row.get("position_delta") is not None:
            position_by_status[status].append(_safe_float(row.get("position_delta")))
    for status, values in position_by_status.items():
        by_status.setdefault(status, {"count": 0})["avg_position_delta"] = _avg(values)
    implemented = position_by_status.get("implemented", [])
    not_implemented = position_by_status.get("not_implemented", [])
    return {
        "by_category": by_category,
        "by_status": by_status,
        "implemented_count": by_status.get("implemented", {}).get("count", 0),
        "not_implemented_count": by_status.get("not_implemented", {}).get("count", 0),
        "avg_position_delta_implemented": _avg(implemented),
        "avg_position_delta_not_implemented": _avg(not_implemented),
    }


def detect_recommendation_outcomes(prev_snapshot: dict, curr_snapshot: dict) -> dict:
    """Classify whether recommendations from a prior snapshot appear implemented."""
    prev_recommendations = list(prev_snapshot.get("recommendations") or [])
    if not prev_recommendations:
        return {
            "available": False,
            "reason": "previous snapshot has no recommendation data",
            "summary": {"total": 0},
            "aggregates": {"by_category": {}, "by_status": {}},
            "rows": [],
            "caveats": HISTORY_CORRELATION_CAVEATS,
            "caveat": RECOMMENDATION_OUTCOME_CAVEAT,
        }
    before_lookup = _snapshot_lookup(prev_snapshot)
    after_lookup = _snapshot_lookup(curr_snapshot)
    before_inbound = _inbound_counts(prev_snapshot)
    after_inbound = _inbound_counts(curr_snapshot)
    before_summary = prev_snapshot.get("summary") or {}
    rows: list[dict] = []
    for rec in prev_recommendations[:100]:
        if not isinstance(rec, dict):
            continue
        status, matched_page = _target_change_status(
            rec,
            curr_snapshot,
            before_lookup,
            after_lookup,
            before_inbound,
            after_inbound,
        )
        targets = [str(target) for target in (rec.get("targets") or []) if target]
        primary_url = str(rec.get("primary_url") or (targets[0] if targets else "") or "")
        metric_url = (matched_page or {}).get("url") or primary_url
        before_page = _lookup_url(before_lookup, metric_url)
        after_page = _lookup_url(after_lookup, metric_url)
        metrics = _metric_delta(before_page, after_page) if before_page or after_page else {
            "traffic_delta": None,
            "clicks_delta": None,
            "position_delta": None,
            "confidence": "none",
            "confidence_caveat": "No tracked page change was detected for this URL.",
        }
        rows.append({
            "id": rec.get("id") or "",
            "category": rec.get("category") or "",
            "type": rec.get("type") or "",
            "priority": rec.get("priority") or "",
            "title": rec.get("title") or "",
            "primary_url": primary_url,
            "matched_url": metric_url or "",
            "targets": targets,
            "issued_at": before_summary.get("created_at") or "",
            "issued_snapshot_id": before_summary.get("snapshot_id") or "",
            "change_status": status,
            "estimated_clicks_gain": rec.get("estimated_clicks_gain"),
            **metrics,
        })
    aggregates = _outcome_aggregates(rows)
    return {
        "available": True,
        "summary": {
            "total": len(rows),
            "implemented": aggregates.get("implemented_count", 0),
            "not_implemented": aggregates.get("not_implemented_count", 0),
            "issued_snapshot_id": before_summary.get("snapshot_id") or "",
            "issued_at": before_summary.get("created_at") or "",
        },
        "aggregates": aggregates,
        "rows": rows,
        "caveats": HISTORY_CORRELATION_CAVEATS,
        "caveat": RECOMMENDATION_OUTCOME_CAVEAT,
    }


def _confidence(
    change_count: int,
    traffic_before: float,
    traffic_delta: float,
    clicks_delta: float = 0.0,
    impressions_delta: float = 0.0,
) -> tuple[str, str]:
    if change_count <= 0:
        return "none", "No tracked page change was detected for this URL."
    rel = abs(traffic_delta) / max(traffic_before, 1.0)
    has_search_movement = abs(clicks_delta) >= 10 or abs(impressions_delta) >= 250
    if (abs(traffic_delta) >= 50 and rel >= 0.25) or (has_search_movement and rel >= 0.15):
        return "medium", "Observed metric movement is material, but provider noise, seasonality, and unrelated changes may contribute."
    if abs(traffic_delta) >= 10 or rel >= 0.15 or has_search_movement:
        return "low-medium", "Metric movement is visible but should be validated against a longer window."
    return "low", "Metric movement is small; treat as a watch item rather than impact evidence."


def compare_snapshots(domain: str, before: str, after: str, projects_root: Path, *, window_days: int | None = None) -> dict:
    before_payload = _snapshot_payload(domain, projects_root, before)
    after_payload = _snapshot_payload(domain, projects_root, after)
    before_pages = _page_map(before_payload)
    after_pages = _page_map(after_payload)
    urls = sorted(set(before_pages) | set(after_pages))
    changes: list[dict] = []
    impact: list[dict] = []
    totals = Counter()

    for url in urls:
        b = before_pages.get(url, {})
        a = after_pages.get(url, {})
        status = "changed"
        if not b:
            status = "added"
        elif not a:
            status = "removed"
        fields = []
        for label, key in [
            ("title", "title_hash"),
            ("description", "description_hash"),
            ("canonical", "canonical_hash"),
            ("h1", "h1_hash"),
            ("serp_title", "serp_title_hash"),
            ("word_count", "word_count_hash"),
            ("redirect_target", "redirect_target_hash"),
            ("headings", "heading_hash"),
            ("paragraphs", "paragraph_hash"),
            ("links", "link_hash"),
            ("schema", "schema_hash"),
            ("metadata", "metadata_hash"),
            ("freshness", "freshness_hash"),
        ]:
            if b.get(key) != a.get(key):
                fields.append(label)
                totals[f"{label}_changed"] += 1
        headings_added, headings_removed = _set_delta(b.get("headings") or [], a.get("headings") or [], "hash")
        paragraphs_added, paragraphs_removed = _set_delta(b.get("paragraphs") or [], a.get("paragraphs") or [], "hash")
        links_added, links_removed = _set_delta(b.get("links") or [], a.get("links") or [])
        schema_added, schema_removed = _set_delta(b.get("schema_types") or [], a.get("schema_types") or [])
        b_metrics = b.get("metrics") or {}
        a_metrics = a.get("metrics") or {}
        traffic_before = _safe_float(b_metrics.get("traffic"))
        traffic_after = _safe_float(a_metrics.get("traffic"))
        traffic_delta = traffic_after - traffic_before
        clicks_before = _safe_float(b_metrics.get("clicks"))
        clicks_after = _safe_float(a_metrics.get("clicks"))
        clicks_delta = clicks_after - clicks_before
        impressions_before = _safe_float(b_metrics.get("impressions"))
        impressions_after = _safe_float(a_metrics.get("impressions"))
        impressions_delta = impressions_after - impressions_before
        keywords_delta = _safe_int(a_metrics.get("keywords")) - _safe_int(b_metrics.get("keywords"))
        position_delta = _safe_float(a_metrics.get("position")) - _safe_float(b_metrics.get("position"))
        change_count = len(fields) + len(paragraphs_added) + len(paragraphs_removed) + len(links_added) + len(links_removed)
        totals["heading_additions"] += len(headings_added)
        totals["heading_removals"] += len(headings_removed)
        totals["paragraph_additions"] += len(paragraphs_added)
        totals["paragraph_removals"] += len(paragraphs_removed)
        totals["link_additions"] += len(links_added)
        totals["link_removals"] += len(links_removed)
        totals["schema_additions"] += len(schema_added)
        totals["schema_removals"] += len(schema_removed)
        confidence, caveat = _confidence(change_count, traffic_before, traffic_delta, clicks_delta, impressions_delta)
        row = {
            "url": a.get("url") or b.get("url") or url,
            "title_before": b.get("title") or "",
            "title_after": a.get("title") or "",
            "description_before": b.get("description") or "",
            "description_after": a.get("description") or "",
            "section": a.get("section") or b.get("section") or "",
            "status": status if fields or status in {"added", "removed"} else "unchanged",
            "changed_fields": fields,
            "headings_added": len(headings_added),
            "headings_removed": len(headings_removed),
            "paragraphs_added": len(paragraphs_added),
            "paragraphs_removed": len(paragraphs_removed),
            "links_added": len(links_added),
            "links_removed": len(links_removed),
            "schema_added": schema_added,
            "schema_removed": schema_removed,
            "metadata_before": b.get("metadata_issues") or [],
            "metadata_after": a.get("metadata_issues") or [],
            "canonical_before": b.get("canonical_url") or "",
            "canonical_after": a.get("canonical_url") or "",
            "h1_before": b.get("h1") or "",
            "h1_after": a.get("h1") or "",
            "serp_title_before": b.get("serp_title") or "",
            "serp_title_after": a.get("serp_title") or "",
            "word_count_before": _safe_int(b.get("word_count")),
            "word_count_after": _safe_int(a.get("word_count")),
            "redirect_target_before": b.get("redirect_target_url") or "",
            "redirect_target_after": a.get("redirect_target_url") or "",
            "freshness_before": b.get("freshness") or {},
            "freshness_after": a.get("freshness") or {},
            "traffic_before": round(traffic_before, 1),
            "traffic_after": round(traffic_after, 1),
            "traffic_delta": round(traffic_delta, 1),
            "clicks_before": round(clicks_before, 1),
            "clicks_after": round(clicks_after, 1),
            "clicks_delta": round(clicks_delta, 1),
            "impressions_before": round(impressions_before, 1),
            "impressions_after": round(impressions_after, 1),
            "impressions_delta": round(impressions_delta, 1),
            "keywords_before": _safe_int(b_metrics.get("keywords")),
            "keywords_after": _safe_int(a_metrics.get("keywords")),
            "keywords_delta": keywords_delta,
            "position_before": b_metrics.get("position"),
            "position_after": a_metrics.get("position"),
            "position_delta": round(position_delta, 2) if b_metrics.get("position") and a_metrics.get("position") else None,
            "top_keyword": a_metrics.get("top_keyword") or b_metrics.get("top_keyword") or "",
            "confidence": confidence,
            "impact_caveat": caveat,
            "paragraph_samples_added": [
                p for p in (a.get("paragraphs") or []) if p.get("hash") in set(paragraphs_added)
            ][:5],
        }
        if (
            row["status"] != "unchanged"
            or abs(traffic_delta) > 0
            or abs(clicks_delta) > 0
            or abs(impressions_delta) > 0
            or keywords_delta
            or position_delta
        ):
            changes.append(row)
        if row["status"] != "unchanged" and (
            abs(traffic_delta) > 0
            or abs(clicks_delta) > 0
            or abs(impressions_delta) > 0
            or keywords_delta
            or position_delta
            or row["traffic_after"] > 0
        ):
            impact.append(row)

    changes.sort(key=lambda r: (abs(_safe_float(r.get("traffic_delta"))), len(r.get("changed_fields") or [])), reverse=True)
    impact.sort(key=lambda r: (abs(_safe_float(r.get("traffic_delta"))), _safe_float(r.get("traffic_after"))), reverse=True)
    before_summary = before_payload.get("summary") or {}
    after_summary = after_payload.get("summary") or {}
    traffic_before_total = _snapshot_total(before_payload, "total_traffic", "traffic")
    traffic_after_total = _snapshot_total(after_payload, "total_traffic", "traffic")
    clicks_before_total = _snapshot_total(before_payload, "total_clicks", "clicks")
    clicks_after_total = _snapshot_total(after_payload, "total_clicks", "clicks")
    impressions_before_total = _snapshot_total(before_payload, "total_impressions", "impressions")
    impressions_after_total = _snapshot_total(after_payload, "total_impressions", "impressions")
    position_before = _snapshot_avg_position(before_payload)
    position_after = _snapshot_avg_position(after_payload)
    paragraph_changes = totals["paragraph_additions"] + totals["paragraph_removals"]
    link_changes = totals["link_additions"] + totals["link_removals"]
    schema_changes = totals["schema_additions"] + totals["schema_removals"]
    content_changes = paragraph_changes + totals["heading_additions"] + totals["heading_removals"]
    recommendation_outcomes = detect_recommendation_outcomes(before_payload, after_payload)
    return {
        "summary": {
            "status": "ok",
            "model": "historical_change_impact_v1",
            "domain": domain,
            "before": before,
            "after": after,
            "window_days": window_days,
            "before_date": before_summary.get("created_at") or "",
            "after_date": after_summary.get("created_at") or "",
            "pages_before": before_summary.get("pages", len(before_pages)),
            "pages_after": after_summary.get("pages", len(after_pages)),
            "changed_pages": sum(1 for row in changes if row.get("status") == "changed"),
            "added_pages": sum(1 for row in changes if row.get("status") == "added"),
            "removed_pages": sum(1 for row in changes if row.get("status") == "removed"),
            "traffic_before": round(traffic_before_total, 1),
            "traffic_after": round(traffic_after_total, 1),
            "traffic_delta": round(traffic_after_total - traffic_before_total, 1),
            "clicks_before": round(clicks_before_total, 1),
            "clicks_after": round(clicks_after_total, 1),
            "clicks_delta": round(clicks_after_total - clicks_before_total, 1),
            "impressions_before": round(impressions_before_total, 1),
            "impressions_after": round(impressions_after_total, 1),
            "impressions_delta": round(impressions_after_total - impressions_before_total, 1),
            "avg_position_before": round(position_before, 2),
            "avg_position_after": round(position_after, 2),
            "avg_position_delta": round(position_after - position_before, 2) if position_before and position_after else None,
            "content_changes": content_changes,
            "paragraph_changes": paragraph_changes,
            "link_changes": link_changes,
            "schema_changes": schema_changes,
            "change_counts": dict(totals),
            "caveats": HISTORY_CORRELATION_CAVEATS,
        },
        "timeline": [
            {
                "snapshot_id": before,
                "date": before_summary.get("created_at") or "",
                "traffic": round(traffic_before_total, 1),
                "clicks": round(clicks_before_total, 1),
                "impressions": round(impressions_before_total, 1),
                "avg_position": round(position_before, 2),
                "pages": before_summary.get("pages", 0),
                "changed_pages": 0,
                "content_changes": 0,
                "paragraph_changes": 0,
                "link_changes": 0,
                "schema_changes": 0,
            },
            {
                "snapshot_id": after,
                "date": after_summary.get("created_at") or "",
                "traffic": round(traffic_after_total, 1),
                "clicks": round(clicks_after_total, 1),
                "impressions": round(impressions_after_total, 1),
                "avg_position": round(position_after, 2),
                "pages": after_summary.get("pages", 0),
                "changed_pages": sum(1 for row in changes if row.get("status") == "changed"),
                "content_changes": content_changes,
                "paragraph_changes": paragraph_changes,
                "link_changes": link_changes,
                "schema_changes": schema_changes,
            },
        ],
        "changes": changes[:1200],
        "impact_table": impact[:600],
        "recommendation_outcomes": recommendation_outcomes,
    }


def write_history_html(template_path: Path, payload: dict, out_path: Path) -> None:
    template = template_path.read_text(encoding="utf-8")
    rendered = template.replace("__HISTORY_JSON__", json.dumps(payload, separators=(",", ":")).replace("</", "<\\/"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered, encoding="utf-8")
