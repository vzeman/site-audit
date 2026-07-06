import numpy as np

from site_audit.analyzer import PageInfo
from site_audit.gsc import GSCClient, GSCConfig, build_analysis


def _page() -> PageInfo:
    return PageInfo(
        url="https://example.com/a",
        title="Support automation",
        description="",
        section="blog",
        word_count=100,
        language="en",
    )


def test_gsc_client_uses_cache_before_fetching_again(tmp_path) -> None:
    calls = []

    def requester(site_url: str, body: dict, token: str) -> dict:
        calls.append((site_url, body["dimensions"], token))
        return {"rows": []}

    config = GSCConfig(
        property_url="sc-domain:example.com",
        start_date="2026-04-01",
        end_date="2026-04-30",
        top_pages_limit=3,
        keywords_limit=5,
    )
    client = GSCClient("token", tmp_path, requester=requester)

    first = client.load_or_fetch("example.com", config)
    second = client.load_or_fetch("example.com", config)

    assert first["meta"]["provider"] == "gsc"
    assert first["meta"]["cache_status"] == "miss"
    assert second["meta"]["cache_status"] == "hit"
    assert len(calls) == 8
    assert calls[0][0] == "sc-domain:example.com"


def test_gsc_client_reports_missing_credentials_after_cache_miss(tmp_path) -> None:
    config = GSCConfig(property_url="sc-domain:example.com", start_date="2026-04-01", end_date="2026-04-30")
    client = GSCClient("", tmp_path, credential_status="missing_credentials", credential_message="missing")

    snapshot = client.load_or_fetch("example.com", config)

    assert snapshot["meta"]["status"] == "missing_credentials"
    assert snapshot["meta"]["provider_label"] == "Google Search Console"
    assert snapshot["raw"] == {}


def test_gsc_build_analysis_normalizes_clicks_impressions_queries_and_pages() -> None:
    page = _page()
    snapshot = {
        "meta": {
            "status": "ok",
            "provider": "gsc",
            "provider_label": "Google Search Console",
            "params": {"property_url": "sc-domain:example.com", "start_date": "2026-04-01", "end_date": "2026-04-30"},
        },
        "raw": {
            "totals": {"rows": [{"clicks": 120, "impressions": 1200, "ctr": 0.1, "position": 4.2}]},
            "daily": {
                "rows": [
                    {"keys": ["2026-04-01"], "clicks": 40, "impressions": 400, "ctr": 0.1, "position": 4.8},
                    {"keys": ["2026-04-02"], "clicks": 80, "impressions": 800, "ctr": 0.1, "position": 3.9},
                ]
            },
            "pages": {"rows": [{"keys": [page.url], "clicks": 100, "impressions": 1000, "ctr": 0.1, "position": 3.5}]},
            "queries": {"rows": [{"keys": ["support automation"], "clicks": 90, "impressions": 900, "ctr": 0.1, "position": 2.5}]},
            "query_page": {
                "rows": [
                    {"keys": ["support automation", page.url], "clicks": 90, "impressions": 900, "ctr": 0.1, "position": 2.5},
                    {"keys": ["helpdesk automation", page.url], "clicks": 10, "impressions": 100, "ctr": 0.1, "position": 8.0},
                ]
            },
            "countries": {"rows": [{"keys": ["usa"], "clicks": 60, "impressions": 600, "ctr": 0.1, "position": 4.0}]},
            "devices": {"rows": [{"keys": ["DESKTOP"], "clicks": 70, "impressions": 700, "ctr": 0.1, "position": 3.8}]},
            "search_appearances": {"rows": [{"keys": ["FAQ rich results"], "clicks": 20, "impressions": 200, "ctr": 0.1, "position": 2.0}]},
        },
    }

    analysis = build_analysis(snapshot, [page], np.eye(1, dtype=np.float32))
    payload = analysis.payload
    top_page = payload["top_pages"][0]
    keyword = payload["organic_keywords"][0]
    query_page = payload["query_pages"][0]

    assert payload["meta"]["provider"] == "gsc"
    assert payload["summary"]["provider_label"] == "Google Search Console"
    assert payload["metrics"]["org_traffic"] == 120
    assert payload["metrics"]["gsc_impressions"] == 1200
    assert payload["summary"]["top_pages_traffic"] == 100
    assert top_page["matched_url"] == page.url
    assert top_page["traffic"] == 100
    assert top_page["clicks"] == 100
    assert top_page["impressions"] == 1000
    assert top_page["top_keyword"] == "support automation"
    assert top_page["keywords"] == 2
    assert keyword["keyword"] == "support automation"
    assert keyword["traffic"] == 90
    assert keyword["volume"] == 900
    assert keyword["position"] == 2.5
    assert query_page["query"] == "support automation"
    assert query_page["url"] == page.url
    assert query_page["source"] == "gsc"
    assert query_page["impressions"] == 900
    assert query_page["position"] == 2.5
    assert payload["daily"][1]["clicks"] == 80
    assert payload["countries"][0]["country"] == "usa"
    assert payload["devices"][0]["device"] == "DESKTOP"
    assert payload["search_appearances"][0]["search_appearance"] == "FAQ rich results"
