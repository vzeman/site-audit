from site_audit.analyzer import PageInfo
from site_audit.extractor import ExtractedPage
from site_audit.performance_explainer import build_performance_explainer


def _page(i: int) -> tuple[PageInfo, ExtractedPage]:
    url = f"https://example.com/page-{i}"
    words = 300 + i * 80
    page = PageInfo(
        url=url,
        title=f"Support automation page {i}",
        description="Guide to support automation workflows",
        section="blog" if i % 2 else "features",
        word_count=words,
        language="en",
    )
    extracted = ExtractedPage(
        url=url,
        title=page.title,
        description=page.description,
        body="Support automation workflow. " * (words // 3),
        word_count=words,
        language="en",
        h1=f"Support automation {i}",
        h1_count=1,
        headers_rich=[{"level": 2, "text": "Workflow"}, {"level": 2, "text": "Examples"}],
        list_count=i % 4,
        table_count=1 if i >= 8 else 0,
        schema_types=["Article"] if i >= 5 else [],
        external_link_count=i // 3,
        has_dates=i >= 4,
        stat_count=i // 2,
        paragraphs=["Support automation routes tickets.", "It improves response time."] * max(1, i // 3),
    )
    return page, extracted


def test_performance_explainer_trains_and_returns_page_contributions() -> None:
    pairs = [_page(i) for i in range(1, 15)]
    pages = [p for p, _ in pairs]
    extracted = [e for _, e in pairs]
    search = {
        "top_pages": [
            {
                "matched_url": page.url,
                "traffic": i * i * 8,
                "keywords": i,
                "top_keyword": f"support automation {i}",
                "top_keyword_position": max(1, 15 - i),
            }
            for i, page in enumerate(pages, 1)
        ]
    }
    linkgraph = {
        "page_link_counts": [
            {"url": page.url, "in_degree": i, "out_degree": 3 + i % 4, "click_depth": max(1, 5 - i // 4)}
            for i, page in enumerate(pages, 1)
        ],
        "traffic_weighted_pagerank": {
            "pages": [
                {"url": page.url, "pagerank": i / 1000, "authority_traffic_gap": max(0, 0.8 - i / 20)}
                for i, page in enumerate(pages, 1)
            ]
        },
    }
    performance = {
        "per_page": [
            {"url": page.url, "html_weight_bytes": 10000 + i * 1000, "estimated_weight_bytes": 80000 + i * 5000, "resource_tag_count": i, "render_blocking_count": i % 3, "image_count": i % 5}
            for i, page in enumerate(pages, 1)
        ]
    }

    payload = build_performance_explainer(
        pages,
        extracted,
        search_payload=search,
        linkgraph=linkgraph,
        performance=performance,
    )

    assert payload["summary"]["status"] == "ok"
    assert payload["summary"]["sample_size"] == 14
    assert payload["summary"]["validation_metric"] == "cross_validated_log_traffic_r2"
    assert payload["features"]
    assert payload["feature_definitions"]
    assert payload["pages"]
    assert payload["pages"][0]["top_positive"] or payload["pages"][0]["top_negative"]
    assert any("not direct Google ranking factors" in warning for warning in payload["summary"]["warnings"])
    assert any(row["feature"] == "links_in_degree_log" for row in payload["features"])


def test_performance_explainer_reports_insufficient_labels() -> None:
    pairs = [_page(i) for i in range(1, 5)]
    payload = build_performance_explainer([p for p, _ in pairs], [e for _, e in pairs], search_payload={})

    assert payload["summary"]["status"] == "insufficient_labels"
    assert payload["summary"]["positive_label_pages"] == 0
    assert payload["features"] == []
