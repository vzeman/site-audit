from types import SimpleNamespace

import numpy as np

from site_audit.analyzer import PageInfo
from site_audit.heading_impact import build_heading_impact


def test_heading_impact_groups_paragraph_demand_under_heading_sections():
    pages = [
        PageInfo(
            url="https://example.com/blog/a",
            title="Support tickets",
            description="",
            section="/blog/",
            word_count=180,
        )
    ]
    paragraphs = [
        "Support ticket software helps teams route requests.",
        "Teams can prioritize urgent conversations and measure resolution time.",
        "Ticket software reporting connects support queues with team performance.",
        "Managers use dashboards to understand support workload.",
    ]
    extracted = [
        SimpleNamespace(
            paragraphs=paragraphs,
            h1="Support tickets",
            headers_rich=[
                {"level": 1, "order": 1, "text": "Support tickets"},
                {"level": 2, "order": 5, "text": "Overview"},
                {"level": 3, "order": 9, "text": "Empty details"},
            ],
        )
    ]
    paragraph_records = [
        (0, i, text, np.array([1.0, 0.0], dtype=np.float32))
        for i, text in enumerate(paragraphs)
    ]
    paragraph_impact = {
        "top_paragraphs": [
            {
                "url": "https://example.com/blog/a",
                "paragraph_index": 2,
                "impact_score": 90,
                "attributed_traffic": 30,
            }
        ]
    }
    keyword_attribution = {
        "keywords": [
            {
                "url": "https://example.com/blog/a",
                "best_paragraph_index": 2,
                "best_heading": "Overview",
                "keyword": "support ticket software",
                "traffic": 30,
                "position": 4,
                "status": "matched",
            }
        ]
    }

    payload = build_heading_impact(
        pages,
        extracted,
        paragraph_records,
        paragraph_impact=paragraph_impact,
        keyword_attribution=keyword_attribution,
        cluster_labels=[3],
    )

    overview = next(row for row in payload["rows"] if row["heading"] == "Overview")
    empty = next(row for row in payload["rows"] if row["heading"] == "Empty details")
    assert payload["summary"]["status"] == "ok"
    assert payload["summary"]["pages_with_heading_metrics"] == 1
    assert overview["attributed_traffic"] == 30
    assert overview["keyword_count"] == 1
    assert overview["cluster"] == 3
    assert "rename" in overview["issue_codes"]
    assert empty["paragraph_count"] == 0
    assert "no_body" in empty["issue_codes"]


def test_heading_impact_creates_synthetic_heading_when_page_has_paragraphs_without_headers():
    pages = [
        PageInfo(
            url="https://example.com/no-headers",
            title="No headers",
            description="",
            section="/",
            word_count=60,
        )
    ]
    extracted = [SimpleNamespace(paragraphs=["Plain body paragraph with useful Product Entity context."], h1="", headers_rich=[])]
    paragraph_records = [(0, 0, extracted[0].paragraphs[0], np.array([1.0, 0.0], dtype=np.float32))]

    payload = build_heading_impact(pages, extracted, paragraph_records)

    assert payload["rows"][0]["synthetic"] is True
    assert payload["rows"][0]["heading"] == "No headers"
    assert payload["per_page"][0]["heading_count"] == 1
