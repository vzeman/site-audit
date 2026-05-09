"""Historical report snapshots and before/after impact diffs."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from .cache import domain_slug


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
        }, score_key="traffic")
    return out


def _row_lookup(rows: list[dict], keys: tuple[str, ...] = ("url",), score_key: str = "") -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in rows or []:
        for key in keys:
            if row.get(key):
                _store_url(out, row.get(key), row, score_key=score_key)
    return out


def build_history_snapshot(
    domain: str,
    pages: list,
    extracted_pages: list | None = None,
    *,
    outlinks_map: dict[str, list[tuple[str, str]]] | None = None,
    structured_data: dict | None = None,
    freshness: dict | None = None,
    metadata_quality: dict | None = None,
    search_payload: dict | None = None,
    snapshot_id: str | None = None,
) -> dict:
    extracted_pages = extracted_pages or []
    outlinks_map = outlinks_map or {}
    structured = _row_lookup((structured_data or {}).get("per_page") or [], ("url",))
    fresh = _row_lookup((freshness or {}).get("per_page") or [], ("url",))
    metadata = _row_lookup((metadata_quality or {}).get("per_page") or [], ("url",))
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
        search_row = _lookup_url(search, url)
        schema_types = sorted(set(structured_row.get("types") or getattr(ext, "schema_types", []) or []))
        title = getattr(page, "title", "") or getattr(ext, "title", "") or ""
        description = getattr(page, "description", "") or getattr(ext, "description", "") or ""
        page_rows.append({
            "url": url,
            "title": title,
            "section": getattr(page, "section", "") or "",
            "word_count": _safe_int(getattr(page, "word_count", 0)),
            "title_hash": _hash(title),
            "description_hash": _hash(description),
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
        },
        "pages": page_rows,
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
    if payload:
        return payload
    report = path / "report"
    pages = _load_json(report / "pages.json", [])
    search = _load_json(report / "ahrefs.json", {}) or _load_json(report / "search.json", {})
    structured = _load_json(report / "structured_data.json", {})
    freshness = _load_json(report / "freshness.json", {})
    metadata = _load_json(report / "metadata_quality.json", {})
    page_objs = [type("Page", (), row)() for row in pages]
    return build_history_snapshot(
        domain,
        page_objs,
        [],
        structured_data=structured,
        freshness=freshness,
        metadata_quality=metadata,
        search_payload=search,
        snapshot_id=snapshot_id,
    )


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
            "caveats": [
                "Impact rows are before/after associations, not causal proof.",
                "The configured comparison window is used as the observation window; confirm the same window in GSC or analytics when available.",
                "Validate against seasonality, provider sampling noise, ranking volatility, and unrelated site changes.",
                "Use wider windows for low-traffic pages before treating a delta as meaningful.",
            ],
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
    }


def write_history_html(template_path: Path, payload: dict, out_path: Path) -> None:
    template = template_path.read_text(encoding="utf-8")
    rendered = template.replace("__HISTORY_JSON__", json.dumps(payload, separators=(",", ":")).replace("</", "<\\/"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered, encoding="utf-8")
