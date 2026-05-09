from datetime import date
from types import SimpleNamespace

import numpy as np

from site_audit.analyzer import PageInfo
from site_audit.freshness_impact import SUPERFICIAL_WARNING, build_freshness_impact


def test_freshness_impact_prioritizes_traffic_and_reports_evidence():
    pages = [
        PageInfo("https://example.com/pricing", "Pricing comparison", "", "blog", 180),
        PageInfo("https://example.com/archive", "Old archive", "", "blog", 180),
    ]
    pricing_text = (
        "As of 2021, our pricing comparison lists the latest plan limits and version 1 API setup. "
        "The current Slack integration steps show the old interface."
    )
    archive_text = "This 2021 company event recap mentions an old webinar and legacy notes."
    extracted = [
        SimpleNamespace(
            body=pricing_text,
            paragraphs=[pricing_text],
            headers_rich=[{"level": 2, "order": 1, "text": "Pricing comparison"}],
            h1="Pricing comparison",
        ),
        SimpleNamespace(
            body=archive_text,
            paragraphs=[archive_text],
            headers_rich=[{"level": 2, "order": 1, "text": "Archive notes"}],
            h1="Archive notes",
        ),
    ]
    paragraph_records = [
        (0, 0, pricing_text, np.array([1.0, 0.0], dtype=np.float32)),
        (1, 0, archive_text, np.array([0.0, 1.0], dtype=np.float32)),
    ]
    freshness = {
        "per_page": [
            {
                "url": "https://example.com/pricing",
                "date": "2021-01-01",
                "date_source": "jsonld:Article.dateModified",
                "date_kind": "modified",
                "age_days": 1955,
                "bucket": "very_stale",
                "issues": ["very_stale"],
            },
            {
                "url": "https://example.com/archive",
                "date": "2021-01-01",
                "date_source": "time",
                "date_kind": "visible",
                "age_days": 1955,
                "bucket": "very_stale",
                "issues": ["very_stale"],
            },
        ]
    }
    search_payload = {
        "top_pages": [
            {
                "matched_url": "https://example.com/pricing",
                "traffic": 500,
                "keywords": 12,
                "top_keyword": "support software pricing comparison",
                "top_keyword_position": 6,
            },
            {
                "matched_url": "https://example.com/archive",
                "traffic": 1,
                "keywords": 1,
                "top_keyword": "company event 2021",
                "top_keyword_position": 30,
            },
        ]
    }

    payload = build_freshness_impact(
        pages,
        extracted,
        freshness,
        search_payload=search_payload,
        paragraph_records=paragraph_records,
        cluster_labels=[2, 2],
        coords=np.array([[1.0, 2.0], [2.0, 3.0]], dtype=np.float32),
        today=date(2026, 5, 9),
    )

    first = payload["sections"][0]
    assert payload["summary"]["status"] == "ok"
    assert first["url"].endswith("/pricing")
    assert first["traffic"] == 500
    assert first["stale_year_evidence"]
    assert first["product_evidence"]
    assert "pricing" in first["recommended_update_type"].lower()
    assert first["superficial_update_warning"] == SUPERFICIAL_WARNING
    assert payload["clusters"][0]["traffic_at_risk"] >= 501
    assert payload["scatter"]["points"][0]["traffic"] == 500


def test_freshness_impact_returns_no_data_without_freshness_payload():
    pages = [PageInfo("https://example.com/a", "A", "", "root", 10)]
    payload = build_freshness_impact(pages, [], None)

    assert payload["summary"]["status"] == "no_freshness_data"
    assert payload["sections"] == []
