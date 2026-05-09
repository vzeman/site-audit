from types import SimpleNamespace

from site_audit.analyzer import PageInfo
from site_audit.entity_coverage import build_entity_coverage


def test_entity_coverage_scores_missing_expected_cluster_entities():
    pages = [
        PageInfo("https://example.com/a", "Helpdesk Software", "", "support", 100),
        PageInfo("https://example.com/b", "Helpdesk Automation", "", "support", 100),
    ]
    extracted = [
        SimpleNamespace(
            h1="Helpdesk Software",
            headers_rich=[{"text": "Helpdesk Software", "level": 1, "order": 1}],
            body="Helpdesk Software includes Ticket Routing, SLA Compliance, and Salesforce Integration.",
            paragraphs=[],
        ),
        SimpleNamespace(
            h1="Helpdesk Automation",
            headers_rich=[{"text": "Helpdesk Automation", "level": 1, "order": 1}],
            body="Helpdesk Automation includes Ticket Routing only.",
            paragraphs=[],
        ),
    ]
    search = {"top_pages": [{"matched_url": "https://example.com/a", "traffic": 100}]}

    payload = build_entity_coverage(
        pages,
        extracted,
        search_payload=search,
        cluster_labels=[0, 0],
    )

    lower = next(row for row in payload["pages"] if row["url"].endswith("/b"))
    assert payload["summary"]["status"] == "ok"
    assert lower["coverage"] < 1.0
    missing_names = {row["entity"] for row in lower["missing_core_entities"] + lower["missing_supporting_entities"]}
    assert "Salesforce Integration" in missing_names
    assert lower["recommendations"]


def test_entity_coverage_works_without_search_data():
    pages = [PageInfo("https://example.com/a", "Product Platform", "", "root", 80)]
    extracted = [
        SimpleNamespace(
            h1="Product Platform",
            headers_rich=[],
            body="Product Platform supports Customer Support and Workflow Automation.",
            paragraphs=[],
        )
    ]

    payload = build_entity_coverage(pages, extracted, cluster_labels=[0])

    assert payload["summary"]["status"] == "ok"
    assert payload["pages"][0]["coverage"] == 1.0
