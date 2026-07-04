from types import SimpleNamespace

from site_audit.report import write_technical_seo_exports
from site_audit.technical_seo import build_technical_seo


def test_technical_seo_model_merges_existing_page_signals() -> None:
    pages = [
        SimpleNamespace(
            url="https://example.com/a",
            title="A",
            section="blog",
            word_count=500,
            language="en",
        )
    ]
    indexability = {"skipped": [], "noindex_pages": []}
    metadata = {
        "per_page": [
            {
                "url": "https://example.com/a",
                "title": "A",
                "canonical_url": "",
                "robots_content": "",
                "issues": ["missing_canonical", "missing_description"],
            }
        ]
    }
    performance = {
        "per_page": [
            {
                "url": "https://example.com/a",
                "status": 200,
                "weight_bucket": "very_heavy",
                "html_weight_bytes": 700000,
                "estimated_weight_bytes": 3500000,
                "resource_tag_count": 90,
                "render_blocking_count": 4,
            }
        ]
    }
    search = {
        "top_pages": [
            {"matched_url": "https://example.com/a", "traffic": 1200, "keywords": 9, "top_keyword": "example keyword"}
        ]
    }
    page_types = {
        "per_page": [
            {
                "url": "https://example.com/a",
                "page_type": "article",
                "template_family": "content_template",
                "template_signature": "sig",
            }
        ]
    }

    payload = build_technical_seo(
        pages,
        indexability=indexability,
        metadata_quality=metadata,
        performance=performance,
        search_payload=search,
        page_types=page_types,
    )

    assert payload["summary"]["total_pages"] == 1
    assert payload["summary"]["high_issues"] >= 1
    page = payload["pages"][0]
    assert page["url"] == "https://example.com/a"
    assert page["traffic"] == 1200
    assert page["page_type"] == "article"
    assert page["fix_scope"] == "template"
    assert page["technical_issue_count"] >= 4
    issue_types = {row["issue_type"] for row in payload["issues"]}
    assert "missing_canonical" in issue_types
    assert "very_heavy_page" in issue_types
    assert "render_blocking_resources" in issue_types


def test_technical_seo_model_includes_skipped_pages() -> None:
    payload = build_technical_seo(
        [],
        indexability={
            "skipped": [
                {
                    "url": "https://example.com/noindex",
                    "title": "Noindex",
                    "reason": "noindex",
                    "http_status": 200,
                }
            ],
            "noindex_pages": [],
        },
    )

    assert payload["summary"]["total_pages"] == 1
    assert payload["pages"][0]["indexability_status"] == "noindex"
    assert payload["issues"][0]["category"] == "indexability"


def test_technical_seo_model_flags_internal_4xx_and_5xx_pages() -> None:
    payload = build_technical_seo(
        [],
        indexability={
            "skipped": [
                {
                    "url": "https://example.com/missing",
                    "title": "",
                    "reason": "non_2xx_status",
                    "http_status": 404,
                },
                {
                    "url": "https://example.com/error",
                    "title": "",
                    "reason": "non_2xx_status",
                    "http_status": 500,
                },
            ],
            "noindex_pages": [],
        },
    )

    by_url = {}
    for issue in payload["issues"]:
        by_url.setdefault(issue["url"], set()).add(issue["issue_type"])
    assert {"404_page", "4xx_page"} <= by_url["https://example.com/missing"]
    assert {"500_page", "5xx_page"} <= by_url["https://example.com/error"]
    assert payload["issue_counts"]["404_page"] == 1
    assert payload["issue_counts"]["5xx_page"] == 1


def test_technical_seo_model_flags_timed_out_pages() -> None:
    payload = build_technical_seo(
        [],
        indexability={
            "skipped": [
                {
                    "url": "https://example.com/slow",
                    "title": "",
                    "reason": "timed_out",
                    "http_status": 0,
                },
            ],
            "noindex_pages": [],
        },
    )

    issue_types = {row["issue_type"] for row in payload["issues"]}
    assert "timed_out" in issue_types
    assert payload["issue_counts"]["timed_out"] == 1


def test_technical_seo_model_flags_https_http_mixed_content() -> None:
    payload = build_technical_seo(
        [SimpleNamespace(url="https://example.com/a", title="A", section="", word_count=100, language="en")],
        performance={
            "per_page": [
                {
                    "url": "https://example.com/a",
                    "status": 200,
                    "mixed_content_url_count": 2,
                    "mixed_content_urls": ["http://cdn.example.com/app.css", "http://cdn.example.com/hero.jpg"],
                }
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "https_http_mixed_content"]
    assert len(issues) == 1
    assert issues[0]["issue_name"] == "HTTPS/HTTP mixed content"
    assert issues[0]["importance"] == "Warning"
    assert payload["pages"][0]["mixed_content_url_count"] == 2


def test_technical_seo_model_flags_canonical_points_to_4xx() -> None:
    payload = build_technical_seo(
        [SimpleNamespace(url="https://example.com/source", title="Source", section="", word_count=100, language="en")],
        canonical_consistency={
            "rows": [
                {
                    "url": "https://example.com/source",
                    "canonical_url": "https://example.com/broken",
                    "canonical_target_http_status": 404,
                    "canonical_target_indexability_status": "not_analyzed",
                    "issues": ["canonical_points_to_4xx"],
                }
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "canonical_points_to_4xx"]
    assert len(issues) == 1
    assert issues[0]["category"] == "indexability"
    assert issues[0]["issue_name"] == "Canonical points to 4XX"
    assert payload["pages"][0]["canonical_target_http_status"] == 404


def test_technical_seo_model_flags_canonical_points_to_5xx() -> None:
    payload = build_technical_seo(
        [SimpleNamespace(url="https://example.com/source", title="Source", section="", word_count=100, language="en")],
        canonical_consistency={
            "rows": [
                {
                    "url": "https://example.com/source",
                    "canonical_url": "https://example.com/error",
                    "canonical_target_http_status": 503,
                    "canonical_target_indexability_status": "not_analyzed",
                    "issues": ["canonical_points_to_5xx"],
                }
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "canonical_points_to_5xx"]
    assert len(issues) == 1
    assert issues[0]["category"] == "indexability"
    assert issues[0]["issue_name"] == "Canonical points to 5XX"
    assert payload["pages"][0]["canonical_target_http_status"] == 503


def test_technical_seo_model_flags_canonical_points_to_redirect() -> None:
    payload = build_technical_seo(
        [SimpleNamespace(url="https://example.com/source", title="Source", section="", word_count=100, language="en")],
        canonical_consistency={
            "rows": [
                {
                    "url": "https://example.com/source",
                    "canonical_url": "https://example.com/redirecting",
                    "canonical_redirect_target_url": "https://example.com/final",
                    "issues": ["canonical_points_to_redirect"],
                }
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "canonical_points_to_redirect"]
    assert len(issues) == 1
    assert issues[0]["category"] == "indexability"
    assert issues[0]["issue_name"] == "Canonical points to redirect"
    assert payload["pages"][0]["canonical_redirect_target_url"] == "https://example.com/final"


def test_technical_seo_model_flags_non_canonical_target() -> None:
    payload = build_technical_seo(
        [SimpleNamespace(url="https://example.com/source", title="Source", section="", word_count=100, language="en")],
        canonical_consistency={
            "rows": [
                {
                    "url": "https://example.com/source",
                    "canonical_url": "https://example.com/variant",
                    "canonical_target_canonical_url": "https://example.com/final",
                    "issues": ["non_canonical_page_specified_as_canonical_one"],
                }
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "non_canonical_page_specified_as_canonical_one"]
    assert len(issues) == 1
    assert issues[0]["category"] == "indexability"
    assert issues[0]["issue_name"] == "Non-canonical page specified as canonical one"
    assert issues[0]["importance"] == "Warning"
    assert payload["pages"][0]["canonical_target_canonical_url"] == "https://example.com/final"


def test_technical_seo_model_flags_canonical_from_http_to_https() -> None:
    payload = build_technical_seo(
        [SimpleNamespace(url="http://example.com/page", title="HTTP Page", section="", word_count=100, language="en")],
        canonical_consistency={
            "rows": [
                {
                    "url": "http://example.com/page",
                    "canonical_url": "https://example.com/page",
                    "issues": ["canonical_from_http_to_https"],
                }
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "canonical_from_http_to_https"]
    assert len(issues) == 1
    assert issues[0]["issue_name"] == "Canonical from HTTP to HTTPS"
    assert issues[0]["importance"] == "Notice"
    assert issues[0]["severity"] == "low"


def test_technical_seo_model_flags_canonical_from_https_to_http() -> None:
    payload = build_technical_seo(
        [SimpleNamespace(url="https://example.com/page", title="HTTPS Page", section="", word_count=100, language="en")],
        canonical_consistency={
            "rows": [
                {
                    "url": "https://example.com/page",
                    "canonical_url": "http://example.com/page",
                    "issues": ["canonical_from_https_to_http"],
                }
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "canonical_from_https_to_http"]
    assert len(issues) == 1
    assert issues[0]["issue_name"] == "Canonical from HTTPS to HTTP"
    assert issues[0]["importance"] == "Notice"
    assert issues[0]["severity"] == "low"


def test_technical_seo_model_flags_canonical_url_changed() -> None:
    payload = build_technical_seo(
        [SimpleNamespace(url="https://example.com/page", title="Page", section="", word_count=100, language="en")],
        history_changes={
            "changes": [
                {
                    "url": "https://example.com/page",
                    "changed_fields": ["canonical"],
                    "canonical_before": "https://example.com/old",
                    "canonical_after": "https://example.com/page",
                }
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "canonical_url_changed"]
    assert len(issues) == 1
    assert issues[0]["issue_name"] == "Canonical URL changed"
    assert issues[0]["importance"] == "Notice"
    assert payload["pages"][0]["previous_canonical_url"] == "https://example.com/old"


def test_technical_seo_model_flags_indexable_page_became_non_indexable() -> None:
    payload = build_technical_seo(
        [],
        indexability={
            "skipped": [
                {
                    "url": "https://example.com/page",
                    "title": "Page",
                    "reason": "noindex",
                    "http_status": 200,
                    "noindex_source": "meta",
                }
            ],
            "noindex_pages": [],
        },
        history_changes={
            "changes": [
                {
                    "url": "https://example.com/page",
                    "changed_fields": ["indexability"],
                    "indexability_before": "indexable",
                    "indexability_after": "noindex",
                }
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "indexable_page_became_non_indexable"]
    assert len(issues) == 1
    assert issues[0]["issue_name"] == "Indexable page became non-indexable"
    assert issues[0]["importance"] == "Notice"
    assert payload["pages"][0]["previous_indexability_status"] == "indexable"


def test_technical_seo_model_flags_noindex_page_became_indexable() -> None:
    payload = build_technical_seo(
        [SimpleNamespace(url="https://example.com/page", title="Page", section="", word_count=100, language="en")],
        history_changes={
            "changes": [
                {
                    "url": "https://example.com/page",
                    "changed_fields": ["indexability"],
                    "indexability_before": "noindex",
                    "indexability_after": "indexable",
                }
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "noindex_page_became_indexable"]
    assert len(issues) == 1
    assert issues[0]["issue_name"] == "Noindex page became indexable"
    assert issues[0]["importance"] == "Notice"
    assert payload["pages"][0]["current_indexability_status"] == "indexable"


def test_technical_seo_model_flags_self_canonical_with_no_incoming_links() -> None:
    payload = build_technical_seo(
        [SimpleNamespace(url="https://example.com/page", title="Page", section="", word_count=100, language="en")],
        metadata_quality={
            "per_page": [
                {
                    "url": "https://example.com/page",
                    "title": "Page",
                    "canonical_url": "https://example.com/page",
                    "issues": [],
                }
            ]
        },
        linkgraph={
            "page_link_counts": [
                {"url": "https://example.com/page", "in_degree": 0, "out_degree": 2, "click_depth": 2}
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "indexable_canonical_url_has_no_incoming_internal_links"]
    assert len(issues) == 1
    assert issues[0]["issue_name"] == "Canonical URL has no incoming internal links"
    assert issues[0]["importance"] == "Error"
    assert payload["pages"][0]["in_degree"] == 0


def test_technical_seo_model_flags_indexable_orphan_pages() -> None:
    pages = [
        SimpleNamespace(url="https://example.com/orphan", title="Orphan", section="", word_count=100, language="en"),
        SimpleNamespace(url="https://example.com/linked", title="Linked", section="", word_count=100, language="en"),
    ]
    payload = build_technical_seo(
        pages,
        indexability={
            "skipped": [
                {
                    "url": "https://example.com/non-indexable-orphan",
                    "title": "Noindex Orphan",
                    "reason": "noindex",
                    "http_status": 200,
                    "nofollow": False,
                }
            ],
            "noindex_pages": [],
        },
        linkgraph={
            "page_link_counts": [
                {"url": "https://example.com/orphan", "in_degree": 0, "out_degree": 1},
                {"url": "https://example.com/linked", "in_degree": 2, "out_degree": 1},
                {"url": "https://example.com/non-indexable-orphan", "in_degree": 0, "out_degree": 1},
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "indexable_orphan_page_has_no_incoming_internal_links"]
    assert len(issues) == 1
    assert issues[0]["url"] == "https://example.com/orphan"
    assert issues[0]["issue_name"] == "Orphan page (has no incoming internal links)"
    assert issues[0]["importance"] == "Error"


def test_technical_seo_model_flags_https_pages_linking_to_http() -> None:
    pages = [
        SimpleNamespace(url="https://example.com/bad", title="Bad", section="", word_count=100, language="en"),
        SimpleNamespace(url="https://example.com/clean", title="Clean", section="", word_count=100, language="en"),
        SimpleNamespace(url="http://example.com/http-source", title="HTTP Source", section="", word_count=100, language="en"),
    ]
    payload = build_technical_seo(
        pages,
        linkgraph={
            "page_link_counts": [
                {
                    "url": "https://example.com/bad",
                    "in_degree": 2,
                    "out_degree": 3,
                    "internal_http_link_count": 2,
                    "internal_http_links": ["http://example.com/legacy", "http://example.com/sale"],
                },
                {
                    "url": "https://example.com/clean",
                    "in_degree": 2,
                    "out_degree": 1,
                    "internal_http_link_count": 0,
                    "internal_http_links": [],
                },
                {
                    "url": "http://example.com/http-source",
                    "in_degree": 2,
                    "out_degree": 1,
                    "internal_http_link_count": 1,
                    "internal_http_links": ["http://example.com/legacy"],
                },
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "indexable_https_page_has_internal_links_to_http"]
    assert len(issues) == 1
    assert issues[0]["url"] == "https://example.com/bad"
    assert issues[0]["issue_name"] == "HTTPS page has internal links to HTTP"
    assert issues[0]["importance"] == "Error"
    bad_page = next(row for row in payload["pages"] if row["url"] == "https://example.com/bad")
    assert bad_page["internal_http_link_count"] == 2
    assert bad_page["internal_http_links"] == ["http://example.com/legacy", "http://example.com/sale"]


def test_technical_seo_model_flags_indexable_pages_linking_to_broken_pages() -> None:
    pages = [
        SimpleNamespace(url="https://example.com/bad", title="Bad", section="", word_count=100, language="en"),
        SimpleNamespace(url="https://example.com/clean", title="Clean", section="", word_count=100, language="en"),
    ]
    payload = build_technical_seo(
        pages,
        indexability={
            "skipped": [
                {
                    "url": "https://example.com/non-indexable-bad",
                    "title": "Noindex Bad",
                    "reason": "noindex",
                    "http_status": 200,
                    "nofollow": False,
                }
            ],
            "noindex_pages": [],
        },
        linkgraph={
            "page_link_counts": [
                {
                    "url": "https://example.com/bad",
                    "in_degree": 2,
                    "out_degree": 3,
                    "broken_internal_link_count": 2,
                    "broken_internal_links": [
                        {"url": "https://example.com/missing", "http_status": 404},
                        {"url": "https://example.com/error", "http_status": 500},
                    ],
                },
                {"url": "https://example.com/clean", "in_degree": 2, "out_degree": 1, "broken_internal_link_count": 0},
                {
                    "url": "https://example.com/non-indexable-bad",
                    "in_degree": 0,
                    "out_degree": 1,
                    "broken_internal_link_count": 1,
                    "broken_internal_links": [{"url": "https://example.com/missing", "http_status": 404}],
                },
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "indexable_page_has_links_to_broken_page"]
    assert len(issues) == 1
    assert issues[0]["url"] == "https://example.com/bad"
    assert issues[0]["issue_name"] == "Page has links to broken page"
    assert issues[0]["importance"] == "Error"
    bad_page = next(row for row in payload["pages"] if row["url"] == "https://example.com/bad")
    assert bad_page["broken_internal_link_count"] == 2


def test_technical_seo_model_flags_indexable_pages_with_no_outgoing_links() -> None:
    pages = [
        SimpleNamespace(url="https://example.com/dead-end", title="Dead End", section="", word_count=100, language="en"),
        SimpleNamespace(url="https://example.com/linked", title="Linked", section="", word_count=100, language="en"),
    ]
    payload = build_technical_seo(
        pages,
        indexability={
            "skipped": [
                {
                    "url": "https://example.com/non-indexable-dead-end",
                    "title": "Noindex Dead End",
                    "reason": "noindex",
                    "http_status": 200,
                    "nofollow": False,
                }
            ],
            "noindex_pages": [],
        },
        linkgraph={
            "page_link_counts": [
                {"url": "https://example.com/dead-end", "in_degree": 2, "out_degree": 0, "raw_internal_link_count": 0},
                {"url": "https://example.com/linked", "in_degree": 2, "out_degree": 0, "raw_internal_link_count": 1},
                {"url": "https://example.com/non-indexable-dead-end", "in_degree": 0, "out_degree": 0, "raw_internal_link_count": 0},
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "indexable_page_has_no_outgoing_links"]
    assert len(issues) == 1
    assert issues[0]["url"] == "https://example.com/dead-end"
    assert issues[0]["issue_name"] == "Page has no outgoing links"
    assert issues[0]["importance"] == "Error"


def test_technical_seo_model_flags_googlebot_html_size_limit() -> None:
    payload = build_technical_seo(
        [SimpleNamespace(url="https://example.com/large", title="Large", section="", word_count=100, language="en")],
        performance={
            "per_page": [
                {
                    "url": "https://example.com/large",
                    "status": 200,
                    "html_weight_bytes": 2 * 1024 * 1024 + 1,
                    "weight_bucket": "very_heavy",
                }
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "page_size_exceeds_googlebot_s_2_mb_crawl_limit"]
    assert len(issues) == 1
    assert issues[0]["category"] == "indexability"
    assert issues[0]["issue_name"] == "Page size exceeds Googlebot's 2 MB crawl limit"


def test_technical_seo_model_flags_nofollow_in_html_and_header() -> None:
    payload = build_technical_seo(
        [SimpleNamespace(url="https://example.com/nofollow", title="Nofollow", section="", word_count=100, language="en")],
        metadata_quality={
            "per_page": [
                {
                    "url": "https://example.com/nofollow",
                    "title": "Nofollow",
                    "robots_content": "index,nofollow",
                    "nofollow": True,
                    "nofollow_source": "meta+header",
                    "issues": [],
                }
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "nofollow_in_html_and_http_header"]
    assert len(issues) == 1
    assert issues[0]["category"] == "indexability"
    assert issues[0]["issue_name"] == "Nofollow in HTML and HTTP header"
    assert issues[0]["importance"] == "Warning"


def test_technical_seo_model_flags_nofollow_page() -> None:
    payload = build_technical_seo(
        [SimpleNamespace(url="https://example.com/nofollow", title="Nofollow", section="", word_count=100, language="en")],
        metadata_quality={
            "per_page": [
                {
                    "url": "https://example.com/nofollow",
                    "title": "Nofollow",
                    "robots_content": "index,nofollow",
                    "nofollow": True,
                    "nofollow_source": "meta",
                    "issues": [],
                }
            ]
        },
    )

    issue_types = {row["issue_type"] for row in payload["issues"]}
    issues = [row for row in payload["issues"] if row["issue_type"] == "nofollow_page"]
    assert len(issues) == 1
    assert "nofollow_in_html_and_http_header" not in issue_types
    assert issues[0]["issue_name"] == "Nofollow page"
    assert issues[0]["importance"] == "Warning"


def test_technical_seo_model_flags_noindex_in_html_and_header() -> None:
    payload = build_technical_seo(
        [],
        indexability={
            "skipped": [
                {
                    "url": "https://example.com/noindex",
                    "title": "Noindex",
                    "reason": "noindex",
                    "http_status": 200,
                    "noindex_source": "meta+header",
                }
            ],
            "noindex_pages": [],
        },
    )

    issue_types = {row["issue_type"] for row in payload["issues"]}
    issues = [row for row in payload["issues"] if row["issue_type"] == "noindex_in_html_and_http_header"]
    assert len(issues) == 1
    assert "noindex_page" in issue_types
    assert issues[0]["issue_name"] == "Noindex in HTML and HTTP header"
    assert issues[0]["importance"] == "Warning"


def test_technical_seo_model_flags_noindex_page() -> None:
    payload = build_technical_seo(
        [],
        indexability={
            "skipped": [
                {
                    "url": "https://example.com/noindex",
                    "title": "Noindex",
                    "reason": "noindex",
                    "http_status": 200,
                    "noindex_source": "meta",
                }
            ],
            "noindex_pages": [],
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "noindex_page"]
    assert len(issues) == 1
    assert issues[0]["issue_name"] == "Noindex page"
    assert issues[0]["importance"] == "Warning"
    assert payload["pages"][0]["indexability_status"] == "noindex"


def test_technical_seo_model_flags_noindex_and_nofollow_page() -> None:
    payload = build_technical_seo(
        [],
        indexability={
            "skipped": [
                {
                    "url": "https://example.com/noindex-nofollow",
                    "title": "Noindex Nofollow",
                    "reason": "noindex",
                    "http_status": 200,
                    "noindex_source": "meta",
                    "nofollow": True,
                    "nofollow_source": "meta",
                }
            ],
            "noindex_pages": [],
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "noindex_and_nofollow_page"]
    assert len(issues) == 1
    assert issues[0]["issue_name"] == "Noindex and nofollow page"
    assert issues[0]["importance"] == "Notice"


def test_technical_seo_model_flags_noindex_follow_page() -> None:
    payload = build_technical_seo(
        [],
        indexability={
            "skipped": [
                {
                    "url": "https://example.com/noindex-follow",
                    "title": "Noindex Follow",
                    "reason": "noindex",
                    "http_status": 200,
                    "noindex_source": "meta",
                    "nofollow": False,
                }
            ],
            "noindex_pages": [],
        },
    )

    issue_types = {row["issue_type"] for row in payload["issues"]}
    issues = [row for row in payload["issues"] if row["issue_type"] == "noindex_follow_page"]
    assert len(issues) == 1
    assert "noindex_and_nofollow_page" not in issue_types
    assert issues[0]["issue_name"] == "Noindex follow page"
    assert issues[0]["importance"] == "Notice"


def test_technical_seo_model_includes_full_issue_catalog() -> None:
    payload = build_technical_seo([])

    assert payload["summary"]["catalog_issue_types"] >= 150
    names = {row["name"] for row in payload["issue_catalog"]}
    assert "Canonical points to 4XX" in names
    assert "Duplicate pages without canonical" in names
    assert "Structured data has schema.org validation error" in names


def test_write_technical_seo_exports_writes_json_and_csv(tmp_path) -> None:
    payload = build_technical_seo(
        [SimpleNamespace(url="https://example.com/a", title="A", section="", word_count=100, language="en")],
        metadata_quality={"per_page": [{"url": "https://example.com/a", "issues": ["missing_title"]}]},
    )

    write_technical_seo_exports(tmp_path, payload)

    assert (tmp_path / "technical_pages.json").is_file()
    assert (tmp_path / "technical_issues.json").is_file()
    assert (tmp_path / "technical_issue_catalog.json").is_file()
    assert (tmp_path / "technical_pages.csv").is_file()
    assert (tmp_path / "technical_issues.csv").is_file()
    assert (tmp_path / "technical_issue_catalog.csv").is_file()
    csv_text = (tmp_path / "technical_pages.csv").read_text(encoding="utf-8")
    assert "technical_severity_score" in csv_text
    assert "missing_title" in csv_text
    issue_json = (tmp_path / "technical_issues.json").read_text(encoding="utf-8")
    assert "issue_catalog" in issue_json
