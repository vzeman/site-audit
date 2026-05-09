from site_audit.anchor_relevance import build_anchor_relevance
from site_audit.extractor import ExtractedPage


def _page(url: str, title: str, links=None) -> ExtractedPage:
    return ExtractedPage(
        url=url,
        title=title,
        description="",
        body=title,
        word_count=20,
        language="en",
        h1=title,
        h1_count=1,
        headings=[title],
        link_audit_rows=links or [],
    )


def test_anchor_relevance_labels_weak_and_suggests_anchor() -> None:
    target_url = "https://example.com/support-automation"
    source = _page(
        "https://example.com/blog/source",
        "Source",
        links=[
            {"is_internal": True, "target_url": target_url, "anchor": "click here", "context": "Learn about support automation", "has_text": True},
            {"is_internal": True, "target_url": target_url, "anchor": "", "context": "", "is_empty": True, "is_image_only": True},
        ],
    )
    target = _page(target_url, "Support Automation Platform")
    payload = build_anchor_relevance(
        [source, target],
        search_payload={"top_pages": [{"matched_url": target_url, "top_keyword": "support automation platform"}]},
        entities_payload={"per_page": [{"url": target_url, "top_entities": [{"entity": "Support Automation"}]}]},
    )

    labels = {row["label"] for row in payload["links"]}
    assert "vague" in labels
    assert "empty" in labels or "image_only" in labels
    weak = payload["weak_links"][0]
    assert weak["suggested_anchor"] in {"support automation platform", "Support Automation", "Support Automation Platform"}
    assert payload["summary"]["weak_links"] == 2


def test_anchor_relevance_handles_no_links() -> None:
    payload = build_anchor_relevance([_page("https://example.com/", "Home")])

    assert payload["summary"]["status"] == "no_links"
    assert payload["links"] == []
