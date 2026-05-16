import numpy as np

from site_audit.search_fusion import build_combined_search_analysis


def test_combined_search_analysis_tags_and_summarizes_providers():
    gsc_payload = {
        "meta": {"provider": "gsc", "provider_label": "Google Search Console", "status": "ok"},
        "summary": {"provider": "gsc", "provider_label": "Google Search Console", "top_pages": 1, "organic_keywords": 1, "total_clicks": 12},
        "top_pages": [{"url": "https://example.com/a", "matched_url": "https://example.com/a", "traffic": 12, "matched": True}],
        "organic_keywords": [{"keyword": "help desk software", "traffic": 12, "volume": 100, "position": 4}],
        "clusters": [{"key": "1", "cluster": 1, "label": "help desk", "traffic": 12, "keyword_rows": 1, "top_keywords": [{"keyword": "help desk software"}]}],
    }
    ads_payload = {
        "meta": {"provider": "google_ads", "provider_label": "Google Ads", "status": "ok"},
        "summary": {"provider": "google_ads", "provider_label": "Google Ads", "top_pages": 0, "organic_keywords": 1, "paid_cost": 250.5},
        "top_pages": [],
        "organic_keywords": [{"keyword": "customer support software", "paid_cost": 250.5, "paid_conversions": 4, "paid_conversion_value": 800, "volume": 80}],
        "clusters": [{"key": "1", "cluster": 1, "label": "help desk", "traffic": 3, "paid_traffic": 2, "keyword_rows": 1, "top_keywords": [{"keyword": "customer support software"}]}],
    }

    analysis = build_combined_search_analysis(
        [gsc_payload, ads_payload],
        [],
        np.zeros((0, 3), dtype=np.float32),
    )

    payload = analysis.payload
    assert payload["meta"]["provider"] == "combined"
    assert payload["summary"]["providers"] == 2
    assert payload["summary"]["organic_keywords"] == 2
    assert payload["summary"]["paid_cost"] == 250.5
    assert payload["summary"]["paid_conversions"] == 4
    assert payload["summary"]["paid_conversion_value"] == 800
    assert payload["summary"]["traffic_clusters"] == 1
    assert payload["clusters"][0]["traffic"] == 15
    assert {row["provider"] for row in payload["clusters"][0]["providers"]} == {"gsc", "google_ads"}
    assert {row["provider"] for row in payload["organic_keywords"]} == {"gsc", "google_ads"}
    assert payload["provider_payloads"]["gsc"]["organic_keywords"][0]["provider_label"] == "Google Search Console"
