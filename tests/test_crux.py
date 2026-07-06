from __future__ import annotations

import logging
from pathlib import Path

import requests

from site_audit.analyzer import PageInfo
from site_audit.crux import assess, build_crux_payload, fetch_crux
from site_audit.recommendations import synthesize, to_payload as recommendations_payload


class _Response:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self) -> dict:
        return self._payload


def _record(*, lcp: int = 4101, inp: int = 501, cls: str = "0.26") -> dict:
    return {
        "record": {
            "collectionPeriod": {
                "firstDate": {"year": 2026, "month": 6, "day": 1},
                "lastDate": {"year": 2026, "month": 6, "day": 28},
            },
            "metrics": {
                "largest_contentful_paint": {
                    "percentiles": {"p75": lcp},
                    "histogram": [
                        {"start": 0, "end": 2500, "density": 0.4},
                        {"start": 2500, "end": 4000, "density": 0.2},
                        {"start": 4000, "density": 0.4},
                    ],
                },
                "interaction_to_next_paint": {
                    "percentiles": {"p75": inp},
                    "histogram": [
                        {"start": 0, "end": 200, "density": 0.7},
                        {"start": 200, "end": 500, "density": 0.2},
                        {"start": 500, "density": 0.1},
                    ],
                },
                "cumulative_layout_shift": {
                    "percentiles": {"p75": cls},
                    "histogram": [
                        {"start": "0.00", "end": "0.10", "density": 0.8},
                        {"start": "0.10", "end": "0.25", "density": 0.1},
                        {"start": "0.25", "density": 0.1},
                    ],
                },
            },
        }
    }


def _pages() -> list[PageInfo]:
    return [
        PageInfo(url="https://example.com/a", title="A", description="", section="root", word_count=100, language="en"),
        PageInfo(url="https://example.com/b", title="B", description="", section="root", word_count=100, language="en"),
    ]


def test_fetch_crux_parses_url_success_and_caches(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict] = []
    monkeypatch.setattr("site_audit.crux.REQUEST_SLEEP_SECONDS", 0)

    def post(url: str, *, json: dict, timeout: int) -> _Response:
        calls.append(json)
        return _Response(200, _record(lcp=2500, inp=200, cls="0.1"))

    monkeypatch.setattr("site_audit.crux.requests.post", post)
    first = fetch_crux(["https://example.com/a"], "key", tmp_path, form_factors=("PHONE",))

    assert first["available"] is True
    assert len(first["rows"]) == 1
    row = first["rows"][0]
    assert row["level"] == "url"
    assert row["metrics"]["largest_contentful_paint"]["p75"] == 2500
    assert row["metrics"]["largest_contentful_paint"]["assessment"] == "good"
    assert row["metrics"]["cumulative_layout_shift"]["p75"] == 0.1
    assert row["metrics"]["cumulative_layout_shift"]["densities"]["good"] == 0.8
    assert len(calls) == 1

    second = fetch_crux(["https://example.com/a"], "key", tmp_path, form_factors=("PHONE",))
    assert second["available"] is True
    assert len(calls) == 1
    assert second["meta"]["cache_hits"] == 1


def test_fetch_crux_404_falls_back_to_one_origin_call(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict] = []
    monkeypatch.setattr("site_audit.crux.REQUEST_SLEEP_SECONDS", 0)

    def post(url: str, *, json: dict, timeout: int) -> _Response:
        calls.append(json)
        if "url" in json:
            return _Response(404, {"error": {"code": 404}})
        return _Response(200, _record())

    monkeypatch.setattr("site_audit.crux.requests.post", post)

    result = fetch_crux(
        ["https://example.com/a", "https://example.com/b"],
        "key",
        tmp_path,
        form_factors=("PHONE",),
    )

    assert [row["level"] for row in result["rows"]] == ["origin", "origin"]
    assert [row["origin"] for row in result["rows"]] == ["https://example.com", "https://example.com"]
    assert len(calls) == 3
    assert sum(1 for call in calls if "origin" in call) == 1


def test_fetch_crux_missing_key_and_truncation(monkeypatch, tmp_path: Path) -> None:
    missing = fetch_crux(["https://example.com/a"], "", tmp_path)
    assert missing["available"] is False
    assert missing["reason"] == "CRUX_API_KEY not configured"

    monkeypatch.setattr("site_audit.crux.REQUEST_SLEEP_SECONDS", 0)

    def post(url: str, *, json: dict, timeout: int) -> _Response:
        return _Response(200, _record())

    monkeypatch.setattr("site_audit.crux.requests.post", post)
    result = fetch_crux(
        [f"https://example.com/{index}" for index in range(4)],
        "key",
        tmp_path,
        form_factors=("PHONE",),
        max_urls=2,
    )
    assert result["truncated"] is True
    assert len(result["requested_urls"]) == 2
    assert len(result["rows"]) == 2


def test_assess_boundaries() -> None:
    assert assess("largest_contentful_paint", 2500) == "good"
    assert assess("largest_contentful_paint", 2501) == "needs_improvement"
    assert assess("largest_contentful_paint", 4000) == "needs_improvement"
    assert assess("largest_contentful_paint", 4001) == "poor"

    assert assess("interaction_to_next_paint", 200) == "good"
    assert assess("interaction_to_next_paint", 201) == "needs_improvement"
    assert assess("interaction_to_next_paint", 500) == "needs_improvement"
    assert assess("interaction_to_next_paint", 501) == "poor"

    assert assess("cumulative_layout_shift", 0.1) == "good"
    assert assess("cumulative_layout_shift", 0.1001) == "needs_improvement"
    assert assess("cumulative_layout_shift", 0.25) == "needs_improvement"
    assert assess("cumulative_layout_shift", 0.2501) == "poor"


def test_build_crux_payload_orders_failures_counts_and_period() -> None:
    rows = {
        "available": True,
        "truncated": False,
        "rows": [
            {
                "url": "https://example.com/a",
                "form_factor": "PHONE",
                "level": "url",
                "origin": "https://example.com",
                "collection_period": {"first_date": "2026-06-01", "last_date": "2026-06-28"},
                "metrics": {"largest_contentful_paint": {"p75": 4101, "assessment": "poor", "densities": {}}},
            },
            {
                "url": "https://example.com/b",
                "form_factor": "PHONE",
                "level": "url",
                "origin": "https://example.com",
                "collection_period": {"first_date": "2026-06-02", "last_date": "2026-06-29"},
                "metrics": {"largest_contentful_paint": {"p75": 4101, "assessment": "poor", "densities": {}}},
            },
            {
                "url": "https://example.com/a",
                "form_factor": "DESKTOP",
                "level": "origin",
                "origin": "https://example.com",
                "collection_period": {"first_date": "2026-06-01", "last_date": "2026-06-28"},
                "metrics": {"largest_contentful_paint": {"p75": 2400, "assessment": "good", "densities": {}}},
            },
        ],
    }
    search = {
        "meta": {"status": "ok"},
        "summary": {"top_pages": 2},
        "top_pages": [
            {"url": "https://example.com/a", "traffic": 10},
            {"url": "https://example.com/b", "traffic": 200},
        ],
    }

    payload = build_crux_payload(rows, _pages(), search)

    assert payload["available"] is True
    assert payload["summary"]["total_rows"] == 3
    assert payload["summary"]["phone_good_share"]["largest_contentful_paint"] == 0.0
    assert payload["counts_by_assessment"]["PHONE"]["largest_contentful_paint"]["poor"] == 2
    assert payload["counts_by_assessment"]["DESKTOP"]["largest_contentful_paint"]["good"] == 1
    assert payload["failing_urls"][0]["url"] == "https://example.com/b"
    assert payload["failing_urls"][0]["traffic"] == 200
    assert payload["collection_period"] == {"first_date": "2026-06-01", "last_date": "2026-06-29"}


def test_crux_recommendations_filter_stable_cap_and_synthesize_wiring() -> None:
    failing = []
    for index in range(18):
        failing.append({
            "url": f"https://example.com/page-{index}",
            "form_factor": "PHONE",
            "level": "url",
            "metric": "largest_contentful_paint",
            "metric_label": "LCP",
            "p75": 4101,
            "value_label": "4.1s",
            "threshold_label": "4.0s",
            "traffic": 100 - index,
        })
    failing.append({
        "url": "https://example.com/origin",
        "form_factor": "PHONE",
        "level": "origin",
        "metric": "largest_contentful_paint",
        "metric_label": "LCP",
        "p75": 4101,
        "traffic": 1000,
    })
    failing.append({
        "url": "https://example.com/no-traffic",
        "form_factor": "PHONE",
        "level": "url",
        "metric": "largest_contentful_paint",
        "metric_label": "LCP",
        "p75": 4101,
        "traffic": 0,
    })
    payload = {
        "available": True,
        "summary": {"total_rows": 20},
        "failing_urls": failing,
        "recommendations": {"candidate_count": 18, "cap": 15, "truncated": True},
    }

    first = synthesize(crux_payload=payload)
    second = synthesize(crux_payload=payload)
    first_ids = [rec.id for rec in first if rec.id.startswith("tech-cwv")]
    second_ids = [rec.id for rec in second if rec.id.startswith("tech-cwv")]

    assert first_ids == second_ids
    assert len(first_ids) == 15
    assert all("origin" not in rec.id and "no-traffic" not in rec.id for rec in first)
    assert first_ids[0] == "tech-cwv-lcp-https-example-com-page-0"

    rec_payload = recommendations_payload(first)
    item = rec_payload["items"][0]
    assert item["category"] == "technical"
    assert item["type"] == "core_web_vitals"
    assert "CrUX field data does not name it" in item["instruction"]
    assert payload["recommendations"]["truncated"] is True


def test_fetch_crux_429_not_cached_and_retried_next_run(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict] = []
    monkeypatch.setattr("site_audit.crux.REQUEST_SLEEP_SECONDS", 0)

    def post(url: str, *, json: dict, timeout: int) -> _Response:
        calls.append(json)
        if len(calls) == 1:
            return _Response(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED"}})
        return _Response(200, _record())

    monkeypatch.setattr("site_audit.crux.requests.post", post)

    first = fetch_crux(["https://example.com/a"], "key", tmp_path, form_factors=("PHONE",))
    assert first["available"] is False
    assert first["errors"] == [{"url": "https://example.com/a", "form_factor": "PHONE", "status_code": 429}]
    assert not list((tmp_path / "crux").glob("*.json"))

    second = fetch_crux(["https://example.com/a"], "key", tmp_path, form_factors=("PHONE",))
    assert len(calls) == 2
    assert second["available"] is True
    assert second["errors"] == []


def test_fetch_crux_request_exception_degrades_and_never_logs_key(monkeypatch, tmp_path: Path, caplog) -> None:
    monkeypatch.setattr("site_audit.crux.REQUEST_SLEEP_SECONDS", 0)

    def post(url: str, *, json: dict, timeout: int) -> _Response:
        raise requests.exceptions.ConnectionError(
            "Max retries exceeded with url: /v1/records:queryRecord?key=SECRET-KEY-123"
        )

    monkeypatch.setattr("site_audit.crux.requests.post", post)

    with caplog.at_level(logging.WARNING, logger="site_audit.crux"):
        result = fetch_crux(["https://example.com/a"], "SECRET-KEY-123", tmp_path, form_factors=("PHONE",))

    assert result["available"] is False
    assert result["errors"] == [{"url": "https://example.com/a", "form_factor": "PHONE", "status_code": 0}]
    assert not list((tmp_path / "crux").glob("*.json"))
    assert "ConnectionError" in caplog.text
    assert "SECRET-KEY-123" not in caplog.text


def test_fetch_crux_malformed_200_falls_back_to_origin(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict] = []
    monkeypatch.setattr("site_audit.crux.REQUEST_SLEEP_SECONDS", 0)

    def post(url: str, *, json: dict, timeout: int) -> _Response:
        calls.append(json)
        if "url" in json:
            return _Response(200, {"unexpected": True})
        return _Response(200, _record())

    monkeypatch.setattr("site_audit.crux.requests.post", post)

    result = fetch_crux(["https://example.com/a"], "key", tmp_path, form_factors=("PHONE",))
    assert result["available"] is True
    assert [row["level"] for row in result["rows"]] == ["origin"]
    assert len(calls) == 2


def test_fetch_crux_malformed_200_everywhere_degrades(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("site_audit.crux.REQUEST_SLEEP_SECONDS", 0)

    def post(url: str, *, json: dict, timeout: int) -> _Response:
        return _Response(200, {"unexpected": True})

    monkeypatch.setattr("site_audit.crux.requests.post", post)

    result = fetch_crux(["https://example.com/a"], "key", tmp_path, form_factors=("PHONE",))
    assert result["available"] is False
    assert result["rows"] == []
    assert result["reason"] == "no CrUX data returned"


def test_fetch_crux_refresh_refetches_cached_responses(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict] = []
    monkeypatch.setattr("site_audit.crux.REQUEST_SLEEP_SECONDS", 0)

    def post(url: str, *, json: dict, timeout: int) -> _Response:
        calls.append(json)
        return _Response(200, _record())

    monkeypatch.setattr("site_audit.crux.requests.post", post)

    fetch_crux(["https://example.com/a"], "key", tmp_path, form_factors=("PHONE",))
    fetch_crux(["https://example.com/a"], "key", tmp_path, form_factors=("PHONE",))
    assert len(calls) == 1

    refreshed = fetch_crux(["https://example.com/a"], "key", tmp_path, form_factors=("PHONE",), refresh=True)
    assert len(calls) == 2
    assert refreshed["meta"]["cache_hits"] == 0


def test_missing_key_pipeline_wiring_makes_no_http_and_stays_safe(monkeypatch, tmp_path: Path) -> None:
    def bomb(*args, **kwargs):
        raise AssertionError("HTTP attempted without CRUX_API_KEY")

    monkeypatch.setattr("site_audit.crux.requests.post", bomb)

    raw = fetch_crux(["https://example.com/a"], "", tmp_path)
    assert raw["available"] is False
    assert raw["reason"] == "CRUX_API_KEY not configured"

    # Mirror the pipeline wiring: unavailable raw payload -> report payload ->
    # recommendation synthesis must stay inert.
    payload = build_crux_payload(raw, _pages(), None)
    assert payload["available"] is False
    recs = synthesize(crux_payload=payload)
    assert [rec.id for rec in recs if rec.id.startswith("tech-cwv")] == []
