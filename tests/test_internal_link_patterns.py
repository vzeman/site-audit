from site_audit.analyzer import PageInfo
from site_audit.extractor import ExtractedPage
from site_audit.internal_link_patterns import build_internal_link_patterns


def _page(url: str, title: str, section: str = "blog") -> PageInfo:
    return PageInfo(url=url, title=title, description="", section=section, word_count=500, language="en")


def _extracted(url: str, title: str, links: list[dict] | None = None) -> ExtractedPage:
    return ExtractedPage(
        url=url,
        title=title,
        description="",
        body="",
        word_count=500,
        language="en",
        link_audit_rows=links or [],
    )


def test_internal_link_patterns_mine_rules_and_recommend_missing_links() -> None:
    high_a = _page("https://example.com/blog/a", "Workflow guide A")
    high_b = _page("https://example.com/blog/b", "Workflow guide B")
    weak = _page("https://example.com/blog/c", "Workflow guide C")
    product = _page("https://example.com/product/platform", "Automation platform", "product")
    pages = [high_a, high_b, weak, product]
    link_rows = [
        {
            "anchor": "automation platform",
            "target_url": product.url,
            "context": "Use the automation platform to implement this workflow.",
            "is_internal": True,
        }
    ]
    extracted = [
        _extracted(high_a.url, high_a.title, link_rows),
        _extracted(high_b.url, high_b.title, link_rows),
        _extracted(weak.url, weak.title, []),
        _extracted(product.url, product.title, []),
    ]
    page_types = {
        "per_page": [
            {"url": high_a.url, "page_type": "blog_post"},
            {"url": high_b.url, "page_type": "blog_post"},
            {"url": weak.url, "page_type": "blog_post"},
            {"url": product.url, "page_type": "product"},
        ]
    }
    linkgraph = {
        "traffic_weighted_pagerank": {
            "pages": [
                {"url": high_a.url, "cluster": "workflow", "traffic": 100, "keywords": 8, "weighted_pagerank_percentile": 0.9, "pagerank_percentile": 0.9},
                {"url": high_b.url, "cluster": "workflow", "traffic": 100, "keywords": 8, "weighted_pagerank_percentile": 0.9, "pagerank_percentile": 0.9},
                {"url": weak.url, "cluster": "workflow", "traffic": 5, "keywords": 1, "weighted_pagerank_percentile": 0.2, "pagerank_percentile": 0.2},
                {"url": product.url, "cluster": "product", "traffic": 50, "keywords": 5, "weighted_pagerank_percentile": 0.8, "pagerank_percentile": 0.8},
            ]
        },
        "contextual_link_impact": {
            "links": [
                {"source_url": high_a.url, "target_url": product.url, "anchor": "automation platform", "context_type": "main_content", "contextual_similarity": 0.82},
                {"source_url": high_b.url, "target_url": product.url, "anchor": "automation platform", "context_type": "main_content", "contextual_similarity": 0.8},
            ]
        },
    }

    payload = build_internal_link_patterns(pages, extracted, page_types=page_types, linkgraph=linkgraph, min_segment_size=3)

    assert payload["patterns"]
    pattern = payload["patterns"][0]
    assert pattern["source_page_type"] == "blog_post"
    assert pattern["target_page_type"] == "product"
    assert pattern["support_count"] == 2
    assert pattern["sample_links"]
    assert pattern["confidence"] > 0.5
    assert payload["recommendations"][0]["source_url"] == weak.url
    assert payload["recommendations"][0]["suggested_targets"][0]["url"] == product.url
    assert payload["recommendations"][0]["suggested_anchor"] == "automation platform"
