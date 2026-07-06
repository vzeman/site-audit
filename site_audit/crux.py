"""Chrome UX Report field data integration.

The CrUX API returns real-user Core Web Vitals distributions for URLs and
origins. This module keeps the network boundary cache-first and normalizes the
small subset of metrics the report needs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import date
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse, urlunparse

import requests

from .ahrefs import _url_keys
from .analyzer import PageInfo, section_for_url
from .config_env import load_dotenv

LOG = logging.getLogger(__name__)

CRUX_ENDPOINT = "https://chromeuxreport.googleapis.com/v1/records:queryRecord"
CRUX_METRICS = (
    "largest_contentful_paint",
    "interaction_to_next_paint",
    "cumulative_layout_shift",
)
METRIC_LABELS = {
    "largest_contentful_paint": "LCP",
    "interaction_to_next_paint": "INP",
    "cumulative_layout_shift": "CLS",
}
METRIC_SHORT_NAMES = {
    "largest_contentful_paint": "lcp",
    "interaction_to_next_paint": "inp",
    "cumulative_layout_shift": "cls",
}
THRESHOLDS = {
    "largest_contentful_paint": {"good": 2500.0, "poor": 4000.0},
    "interaction_to_next_paint": {"good": 200.0, "poor": 500.0},
    "cumulative_layout_shift": {"good": 0.1, "poor": 0.25},
}
REQUEST_SLEEP_SECONDS = 0.4
# Cap for URL-level poor-CWV recommendations; recommendations.py imports this
# so the payload's truncation flag and the emitted rec count cannot drift.
CWV_REC_CAP = 15
# Only these statuses are persisted to the cache: 200 (data) and 404 (CrUX has
# no record for the URL/origin, stable enough to reuse). Transient 429/5xx and
# network errors must be retried on the next run instead of poisoning the cache.
CACHEABLE_STATUSES = (200, 404)


def crux_api_key() -> str:
    """Return the configured CrUX API key from .env/environment."""
    load_dotenv()
    return os.environ.get("CRUX_API_KEY", "").strip()


def assess(metric: str, p75: float) -> str:
    """Assess a CrUX p75 value using Google's CWV thresholds."""
    thresholds = THRESHOLDS.get(metric)
    if not thresholds:
        return "needs_improvement"
    value = _safe_float(p75, default=None)
    if value is None:
        return "needs_improvement"
    if value <= thresholds["good"]:
        return "good"
    if value > thresholds["poor"]:
        return "poor"
    return "needs_improvement"


def fetch_crux(
    urls: Iterable[str],
    api_key: str,
    cache: Path | str,
    *,
    form_factors: tuple[str, ...] = ("PHONE", "DESKTOP"),
    max_urls: int = 100,
    refresh: bool = False,
) -> dict:
    """Fetch URL-level CrUX rows with origin fallback.

    Cached 404s are treated as real cache entries so repeated runs can stay
    offline once the URL/origin availability is known. ``refresh=True``
    bypasses cached responses (they are rewritten) so the 28-day rolling
    window can be re-sampled without deleting cache files.
    """
    if not api_key:
        return _unavailable("CRUX_API_KEY not configured", max_urls=max_urls)

    selected, truncated = _dedupe_urls(urls, max_urls=max_urls)
    if not selected:
        return _unavailable("no URLs selected", max_urls=max_urls)

    cache_root = Path(cache) / "crux"
    rows: list[dict] = []
    errors: list[dict] = []
    origin_cache: dict[tuple[str, str], dict | None] = {}
    calls = 0
    cache_hits = 0

    for url in selected:
        origin = _origin(url)
        for form_factor in form_factors:
            payload = _request_payload("url", url, form_factor)
            response = _load_or_fetch(cache_root, payload, api_key, refresh=refresh)
            calls += 0 if response.get("cache_status") == "hit" else 1
            cache_hits += 1 if response.get("cache_status") == "hit" else 0
            status = int(response.get("status_code") or 0)
            if status == 200:
                row = _row_from_response(url, form_factor, "url", response.get("body") or {}, origin=origin)
                if row:
                    rows.append(row)
                    continue
                # 200 without any usable metric: fall through to the origin
                # fallback, same as a 404.
            elif status != 404:
                errors.append({"url": url, "form_factor": form_factor, "status_code": status})
                continue
            if not origin:
                continue

            origin_key = (origin, form_factor)
            if origin_key not in origin_cache:
                origin_payload = _request_payload("origin", origin, form_factor)
                origin_response = _load_or_fetch(cache_root, origin_payload, api_key, refresh=refresh)
                calls += 0 if origin_response.get("cache_status") == "hit" else 1
                cache_hits += 1 if origin_response.get("cache_status") == "hit" else 0
                if int(origin_response.get("status_code") or 0) == 200:
                    origin_cache[origin_key] = origin_response
                else:
                    origin_cache[origin_key] = None
                    if int(origin_response.get("status_code") or 0) != 404:
                        errors.append({
                            "url": url,
                            "origin": origin,
                            "form_factor": form_factor,
                            "status_code": int(origin_response.get("status_code") or 0),
                        })
            origin_response = origin_cache.get(origin_key)
            if origin_response:
                row = _row_from_response(url, form_factor, "origin", origin_response.get("body") or {}, origin=origin)
                if row:
                    rows.append(row)

    return {
        "available": bool(rows),
        "reason": "" if rows else "no CrUX data returned",
        "rows": rows,
        "requested_urls": selected,
        "max_urls": max_urls,
        "truncated": truncated,
        "errors": errors,
        "meta": {
            "provider": "crux",
            "provider_label": "Chrome UX Report",
            "status": "ok" if rows else "empty",
            "cache_hits": cache_hits,
            "api_calls": calls,
            "fetched_at": date.today().isoformat(),
        },
    }


def build_crux_payload(rows: dict | list[dict] | None, pages: Iterable[PageInfo], search_payload: dict | None) -> dict:
    """Build the report-facing Core Web Vitals payload."""
    if isinstance(rows, dict):
        if rows.get("available") is False and not rows.get("rows"):
            return _payload_unavailable(str(rows.get("reason") or "no CrUX data returned"), rows)
        raw_rows = list(rows.get("rows") or [])
        meta = dict(rows.get("meta") or {})
        truncated = bool(rows.get("truncated"))
        max_urls = int(rows.get("max_urls") or 100)
    else:
        raw_rows = list(rows or [])
        meta = {}
        truncated = False
        max_urls = 100

    if not raw_rows:
        return _payload_unavailable("no CrUX data returned", rows if isinstance(rows, dict) else {})

    page_lookup = _page_context_lookup(pages)
    traffic_lookup = _traffic_lookup(search_payload)
    payload_rows = [_payload_row(row, page_lookup, traffic_lookup) for row in raw_rows]
    payload_rows.sort(key=lambda row: (_safe_int(row.get("traffic")), row.get("url") or "", row.get("form_factor") or ""), reverse=True)

    summary = _summary(payload_rows)
    failing = _failing_urls(payload_rows)
    rec_candidates = [
        item for item in failing
        if item.get("level") == "url" and _safe_int(item.get("traffic")) > 0
    ]
    return {
        "available": True,
        "reason": "",
        "summary": summary,
        "rows": payload_rows,
        "origin_summary": _origin_summary(payload_rows),
        "counts_by_assessment": _counts_by_assessment(payload_rows),
        "failing_urls": failing,
        "recommendations": {
            "candidate_count": len(rec_candidates),
            "cap": CWV_REC_CAP,
            "truncated": len(rec_candidates) > CWV_REC_CAP,
        },
        "collection_period": _collection_period(payload_rows),
        "truncated": truncated,
        "max_urls": max_urls,
        "meta": {
            "provider": "crux",
            "provider_label": "Chrome UX Report",
            **meta,
        },
    }


def _unavailable(reason: str, *, max_urls: int = 100) -> dict:
    return {
        "available": False,
        "reason": reason,
        "rows": [],
        "requested_urls": [],
        "max_urls": max_urls,
        "truncated": False,
        "errors": [],
        "meta": {"provider": "crux", "provider_label": "Chrome UX Report", "status": "unavailable"},
    }


def _payload_unavailable(reason: str, source: dict | None) -> dict:
    source = source or {}
    return {
        "available": False,
        "reason": reason,
        "summary": {
            "total_rows": 0,
            "phone_rows": 0,
            "phone_good_share": {},
            "url_rows": 0,
            "origin_rows": 0,
        },
        "rows": [],
        "origin_summary": [],
        "counts_by_assessment": {},
        "failing_urls": [],
        "recommendations": {"candidate_count": 0, "cap": CWV_REC_CAP, "truncated": False},
        "collection_period": {},
        "truncated": bool(source.get("truncated")),
        "max_urls": int(source.get("max_urls") or 100),
        "meta": source.get("meta") or {"provider": "crux", "provider_label": "Chrome UX Report"},
    }


def _dedupe_urls(urls: Iterable[str], *, max_urls: int) -> tuple[list[str], bool]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in urls:
        url = _normalize_url(str(raw or ""))
        if not url or url in seen:
            continue
        seen.add(url)
        out.append(url)
    return out[:max_urls], len(out) > max_urls


def _normalize_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    parsed = urlparse(url if "://" in url else f"https://{url}")
    if not parsed.netloc:
        return ""
    return urlunparse((parsed.scheme or "https", parsed.netloc.lower(), parsed.path or "/", "", parsed.query, ""))


def _origin(url: str) -> str:
    parsed = urlparse(url if "://" in url else f"https://{url}")
    if not parsed.netloc:
        return ""
    return urlunparse((parsed.scheme or "https", parsed.netloc.lower(), "", "", "", ""))


def _request_payload(level: str, target: str, form_factor: str) -> dict:
    key = "origin" if level == "origin" else "url"
    return {
        key: target,
        "formFactor": form_factor,
        "metrics": list(CRUX_METRICS),
    }


def _cache_path(cache_root: Path, payload: dict) -> Path:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]
    return cache_root / f"record_{digest}.json"


def _load_or_fetch(cache_root: Path, payload: dict, api_key: str, *, refresh: bool = False) -> dict:
    path = _cache_path(cache_root, payload)
    if not refresh:
        cached = _load_json(path)
        if cached:
            cached["cache_status"] = "hit"
            return cached
    if REQUEST_SLEEP_SECONDS > 0:
        time.sleep(REQUEST_SLEEP_SECONDS)
    url = f"{CRUX_ENDPOINT}?key={api_key}"
    try:
        resp = requests.post(url, json=payload, timeout=60)
    except requests.RequestException as exc:
        # Log the exception TYPE only: requests error messages can embed the
        # request URL including the ?key= query parameter.
        LOG.warning(
            "CrUX request failed for %s (%s): %s",
            payload.get("url") or payload.get("origin") or "?",
            payload.get("formFactor") or "?",
            type(exc).__name__,
        )
        return {
            "status_code": 0,
            "payload": payload,
            "body": {},
            "cache_status": "error",
            "fetched_at": date.today().isoformat(),
        }
    try:
        body = resp.json()
    except Exception:
        body = {"text": resp.text[:1000]}
    data = {
        "status_code": int(resp.status_code),
        "payload": payload,
        "body": body,
        "cache_status": "miss",
        "fetched_at": date.today().isoformat(),
    }
    if data["status_code"] in CACHEABLE_STATUSES:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _row_from_response(url: str, form_factor: str, level: str, body: dict, *, origin: str) -> dict | None:
    record = body.get("record") or {}
    metrics_payload = record.get("metrics") or {}
    metrics: dict[str, dict] = {}
    for metric in CRUX_METRICS:
        parsed = _metric_payload(metric, metrics_payload.get(metric) or {})
        if parsed:
            metrics[metric] = parsed
    if not metrics:
        return None
    return {
        "url": url,
        "form_factor": form_factor,
        "level": level,
        "origin": origin,
        "collection_period": _parse_collection_period(record.get("collectionPeriod") or body.get("collectionPeriod") or {}),
        "metrics": metrics,
    }


def _metric_payload(metric: str, payload: dict) -> dict:
    p75 = _safe_float((payload.get("percentiles") or {}).get("p75"), default=None)
    if p75 is None:
        return {}
    return {
        "p75": p75,
        "assessment": assess(metric, p75),
        "densities": _histogram_densities(metric, payload.get("histogram") or []),
    }


def _histogram_densities(metric: str, histogram: list[dict]) -> dict:
    densities = {"good": 0.0, "needs_improvement": 0.0, "poor": 0.0}
    if len(histogram) == 3:
        for name, bucket in zip(("good", "needs_improvement", "poor"), histogram):
            densities[name] = round(_safe_float(bucket.get("density")), 6)
        return densities
    thresholds = THRESHOLDS.get(metric, {})
    good = thresholds.get("good", 0.0)
    poor = thresholds.get("poor", 0.0)
    for bucket in histogram:
        density = _safe_float(bucket.get("density"))
        start = _safe_float(bucket.get("start"), default=0.0)
        end = _safe_float(bucket.get("end"), default=None)
        if end is not None and end <= good:
            densities["good"] += density
        elif start > poor:
            densities["poor"] += density
        else:
            densities["needs_improvement"] += density
    return {key: round(value, 6) for key, value in densities.items()}


def _parse_collection_period(payload: dict) -> dict:
    return {
        "first_date": _date_value(payload.get("firstDate") or {}),
        "last_date": _date_value(payload.get("lastDate") or {}),
    }


def _date_value(payload: dict) -> str:
    try:
        year = int(payload.get("year") or 0)
        month = int(payload.get("month") or 0)
        day = int(payload.get("day") or 0)
        if year and month and day:
            return date(year, month, day).isoformat()
    except (TypeError, ValueError):
        return ""
    return ""


def _page_context_lookup(pages: Iterable[PageInfo]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for page in pages:
        context = {"title": page.title, "section": page.section or section_for_url(page.url)}
        for key in _url_keys(page.url):
            out.setdefault(key, context)
    return out


def _traffic_lookup(search_payload: dict | None) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in (search_payload or {}).get("top_pages") or []:
        url = row.get("matched_url") or row.get("url") or ""
        traffic = _safe_int(row.get("traffic"))
        if not url or traffic <= 0:
            continue
        for key in _url_keys(str(url)):
            out[key] = max(out.get(key, 0), traffic)
    return out


def _payload_row(row: dict, page_lookup: dict[str, dict], traffic_lookup: dict[str, int]) -> dict:
    url = str(row.get("url") or "")
    context = next((page_lookup.get(key) for key in _url_keys(url) if page_lookup.get(key)), {})
    traffic = max((traffic_lookup.get(key, 0) for key in _url_keys(url)), default=0)
    out = {
        "url": url,
        "title": context.get("title") or "",
        "section": context.get("section") or section_for_url(url),
        "form_factor": row.get("form_factor") or "",
        "level": row.get("level") or "",
        "origin": row.get("origin") or _origin(url),
        "traffic": traffic,
        "collection_period": row.get("collection_period") or {},
        "metrics": row.get("metrics") or {},
    }
    for metric, metric_row in out["metrics"].items():
        short = METRIC_SHORT_NAMES.get(metric, metric)
        out[f"{short}_p75"] = metric_row.get("p75")
        out[f"{short}_assessment"] = metric_row.get("assessment")
        out[f"{short}_densities"] = metric_row.get("densities") or {}
    return out


def _summary(rows: list[dict]) -> dict:
    phone_rows = [row for row in rows if row.get("form_factor") == "PHONE"]
    phone_good_share: dict[str, float] = {}
    for metric in CRUX_METRICS:
        assessed = [row for row in phone_rows if (row.get("metrics") or {}).get(metric)]
        if not assessed:
            phone_good_share[metric] = 0.0
            continue
        good = sum(1 for row in assessed if row["metrics"][metric].get("assessment") == "good")
        phone_good_share[metric] = round(good / len(assessed), 4)
    return {
        "total_rows": len(rows),
        "phone_rows": len(phone_rows),
        "url_rows": sum(1 for row in rows if row.get("level") == "url"),
        "origin_rows": sum(1 for row in rows if row.get("level") == "origin"),
        "phone_good_share": phone_good_share,
    }


def _counts_by_assessment(rows: list[dict]) -> dict:
    out: dict[str, dict[str, dict[str, int]]] = {}
    for row in rows:
        form_factor = str(row.get("form_factor") or "")
        for metric, metric_row in (row.get("metrics") or {}).items():
            assessment = str(metric_row.get("assessment") or "")
            out.setdefault(form_factor, {}).setdefault(metric, {"good": 0, "needs_improvement": 0, "poor": 0})
            if assessment:
                out[form_factor][metric][assessment] = out[form_factor][metric].get(assessment, 0) + 1
    return out


def _origin_summary(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], dict] = {}
    for row in rows:
        key = (str(row.get("origin") or ""), str(row.get("form_factor") or ""))
        group = grouped.setdefault(key, {
            "origin": key[0],
            "form_factor": key[1],
            "rows": 0,
            "url_rows": 0,
            "origin_rows": 0,
            "metrics": {},
        })
        group["rows"] += 1
        group["url_rows" if row.get("level") == "url" else "origin_rows"] += 1
        for metric, metric_row in (row.get("metrics") or {}).items():
            group["metrics"].setdefault(metric, {"poor": 0, "needs_improvement": 0, "good": 0})
            assessment = metric_row.get("assessment")
            if assessment:
                group["metrics"][metric][assessment] = group["metrics"][metric].get(assessment, 0) + 1
    return sorted(grouped.values(), key=lambda row: (row["origin"], row["form_factor"]))


def _failing_urls(rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    for row in rows:
        if row.get("form_factor") != "PHONE":
            continue
        for metric, metric_row in (row.get("metrics") or {}).items():
            if metric_row.get("assessment") != "poor":
                continue
            out.append({
                "url": row.get("url") or "",
                "title": row.get("title") or "",
                "section": row.get("section") or "",
                "form_factor": row.get("form_factor") or "",
                "level": row.get("level") or "",
                "origin": row.get("origin") or "",
                "metric": metric,
                "metric_label": METRIC_LABELS.get(metric, metric),
                "p75": metric_row.get("p75"),
                "value_label": format_metric_value(metric, metric_row.get("p75")),
                "threshold_label": threshold_label(metric),
                "traffic": _safe_int(row.get("traffic")),
            })
    out.sort(key=lambda row: (_safe_int(row.get("traffic")), row.get("url") or "", row.get("metric") or ""), reverse=True)
    return out


def _collection_period(rows: list[dict]) -> dict:
    first = sorted({
        str((row.get("collection_period") or {}).get("first_date") or "")
        for row in rows
        if (row.get("collection_period") or {}).get("first_date")
    })
    last = sorted({
        str((row.get("collection_period") or {}).get("last_date") or "")
        for row in rows
        if (row.get("collection_period") or {}).get("last_date")
    })
    return {
        "first_date": first[0] if first else "",
        "last_date": last[-1] if last else "",
    }


def format_metric_value(metric: str, value: object) -> str:
    number = _safe_float(value)
    if metric == "largest_contentful_paint":
        return f"{number / 1000:.1f}s"
    if metric == "interaction_to_next_paint":
        return f"{number:.0f}ms"
    if metric == "cumulative_layout_shift":
        return f"{number:.2f}".rstrip("0").rstrip(".")
    return str(value)


def threshold_label(metric: str) -> str:
    poor = THRESHOLDS.get(metric, {}).get("poor", 0.0)
    if metric == "largest_contentful_paint":
        return f"{poor / 1000:.1f}s"
    if metric == "interaction_to_next_paint":
        return f"{poor:.0f}ms"
    if metric == "cumulative_layout_shift":
        return f"{poor:.2f}".rstrip("0").rstrip(".")
    return str(poor)


def _safe_float(value: object, default: float | None = 0.0) -> float | None:
    if value in (None, ""):
        return default
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return default


def _safe_int(value: object, default: int = 0) -> int:
    parsed = _safe_float(value, default=float(default))
    return int(parsed or 0)
