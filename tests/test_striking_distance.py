import numpy as np

from site_audit.analyzer import PageInfo
from site_audit.search_fusion import build_combined_search_analysis
from site_audit.striking_distance import build_striking_distance


def _page(url: str, title: str = "Guide") -> PageInfo:
    return PageInfo(url, title, "", "blog", 500, "en")


def test_striking_distance_filters_scores_and_groups_by_url() -> None:
    page = _page("https://example.com/guide", "Support guide")
    payload = {
        "meta": {"params": {"start_date": "2026-04-01", "end_date": "2026-04-30"}},
        "query_pages": [
            {"query": "support automation", "url": page.url, "impressions": 1000, "clicks": 30, "ctr": 0.03, "position": 8, "source": "gsc"},
            {"query": "ticket workflow", "url": page.url, "impressions": 500, "clicks": 12, "ctr": 0.024, "position": 12, "source": "gsc"},
            {"query": "too high", "url": page.url, "impressions": 500, "position": 3.5, "source": "gsc"},
            {"query": "too few", "url": page.url, "impressions": 9, "position": 8, "source": "gsc"},
            {"query": "branded", "url": page.url, "impressions": 1000, "position": 9, "source": "ahrefs", "intents": ["branded"]},
        ],
    }

    result = build_striking_distance(payload, [page])

    assert result["available"] is True
    assert result["summary"]["opportunities"] == 2
    assert result["model"]["target_position"] == 3.0
    assert result["model"]["curve"] == "site-audit-v1"
    assert result["model"]["period_days"] == 30
    assert [row["query"] for row in result["rows"]] == ["support automation", "ticket workflow"]
    assert result["rows"][0]["page_title"] == "Support guide"
    assert result["rows"][0]["expected_ctr_target"] == 0.1
    assert len(result["pages"]) == 1
    assert result["pages"][0]["total_estimated_gain"] == round(
        sum(row["estimated_clicks_gain"] for row in result["rows"]),
        2,
    )


def test_striking_distance_unavailable_without_query_pages() -> None:
    result = build_striking_distance({"organic_keywords": []}, [_page("https://example.com/a")])

    assert result["available"] is False
    assert "query+page" in result["reason"]


def test_combined_search_fusion_prefers_gsc_query_page_duplicates() -> None:
    gsc_payload = {
        "meta": {"provider": "gsc", "provider_label": "Google Search Console", "status": "ok"},
        "summary": {"provider": "gsc", "provider_label": "Google Search Console", "organic_keywords": 1},
        "top_pages": [],
        "organic_keywords": [{"keyword": "support automation", "traffic": 3}],
        "query_pages": [
            {"query": "support automation", "url": "https://example.com/a", "impressions": 100, "clicks": 3, "position": 8, "source": "gsc"}
        ],
    }
    ahrefs_payload = {
        "meta": {"provider": "ahrefs", "provider_label": "Ahrefs", "status": "ok"},
        "summary": {"provider": "ahrefs", "provider_label": "Ahrefs", "organic_keywords": 1},
        "top_pages": [],
        "organic_keywords": [{"keyword": "support automation", "traffic": 4}],
        "query_pages": [
            {"query": "support automation", "url": "https://example.com/a", "impressions": 1000, "clicks": 4, "position": 7, "source": "ahrefs"}
        ],
    }

    analysis = build_combined_search_analysis([ahrefs_payload, gsc_payload], [], np.zeros((0, 3), dtype=np.float32))

    assert len(analysis.payload["query_pages"]) == 1
    assert analysis.payload["query_pages"][0]["source"] == "gsc"
    assert analysis.payload["query_pages"][0]["impressions"] == 100


def test_combined_search_fusion_dedupes_url_variants_and_keeps_window_dates() -> None:
    gsc_payload = {
        "meta": {"provider": "gsc", "provider_label": "Google Search Console", "status": "ok"},
        "summary": {
            "provider": "gsc",
            "provider_label": "Google Search Console",
            "organic_keywords": 1,
            "start_date": "2026-04-01",
            "end_date": "2026-04-30",
        },
        "top_pages": [],
        "organic_keywords": [{"keyword": "support automation", "traffic": 3}],
        "query_pages": [
            {"query": "support automation", "url": "https://example.com/a/", "impressions": 100, "clicks": 3, "position": 8, "source": "gsc"}
        ],
    }
    ahrefs_payload = {
        "meta": {"provider": "ahrefs", "provider_label": "Ahrefs", "status": "ok"},
        "summary": {"provider": "ahrefs", "provider_label": "Ahrefs", "organic_keywords": 1},
        "top_pages": [],
        "organic_keywords": [{"keyword": "support automation", "traffic": 4}],
        "query_pages": [
            {"query": "Support Automation", "url": "https://example.com/a", "impressions": 1000, "clicks": 4, "position": 7, "source": "ahrefs"}
        ],
    }

    analysis = build_combined_search_analysis([ahrefs_payload, gsc_payload], [], np.zeros((0, 3), dtype=np.float32))

    assert len(analysis.payload["query_pages"]) == 1
    assert analysis.payload["query_pages"][0]["source"] == "gsc"
    assert analysis.payload["summary"]["start_date"] == "2026-04-01"
    assert analysis.payload["summary"]["end_date"] == "2026-04-30"

    striking = build_striking_distance(analysis.payload, [])
    assert striking["summary"]["start_date"] == "2026-04-01"
    assert striking["model"]["period_days"] == 30


def test_striking_distance_page_totals_include_rows_beyond_emit_cap() -> None:
    page = _page("https://example.com/guide", "Support guide")
    payload = {
        "query_pages": [
            {"query": "support automation", "url": page.url, "impressions": 1000, "position": 8, "source": "gsc"},
            {"query": "ticket workflow", "url": page.url, "impressions": 500, "position": 12, "source": "gsc"},
        ],
    }

    result = build_striking_distance(payload, [page], max_rows=1)

    assert len(result["rows"]) == 1
    assert len(result["pages"]) == 1
    assert result["pages"][0]["total_estimated_gain"] == result["summary"]["total_modeled_click_gain"]
    assert len(result["pages"][0]["queries"]) == 2
