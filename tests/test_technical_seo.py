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


def test_technical_seo_model_flags_redirect_loops() -> None:
    payload = build_technical_seo(
        [],
        indexability={
            "skipped": [
                {
                    "url": "https://example.com/loop",
                    "title": "",
                    "reason": "redirect_loop",
                    "http_status": 0,
                    "requested_url": "https://example.com/loop",
                },
            ],
            "noindex_pages": [],
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "redirect_loop"]
    assert len(issues) == 1
    assert issues[0]["url"] == "https://example.com/loop"
    assert issues[0]["issue_name"] == "Redirect loop"
    assert issues[0]["category"] == "redirects"
    assert issues[0]["importance"] == "Error"


def test_technical_seo_model_flags_broken_redirects() -> None:
    payload = build_technical_seo(
        [],
        indexability={
            "skipped": [
                {
                    "url": "https://example.com/missing",
                    "title": "",
                    "reason": "non_2xx_status",
                    "http_status": 404,
                    "requested_url": "https://example.com/old",
                    "redirect_target_url": "https://example.com/missing",
                },
                {
                    "url": "https://example.com/direct-missing",
                    "title": "",
                    "reason": "non_2xx_status",
                    "http_status": 404,
                    "requested_url": "https://example.com/direct-missing",
                    "redirect_target_url": "",
                },
            ],
            "per_page": [
                {
                    "url": "https://example.com/live",
                    "indexability_status": "indexable",
                    "http_status": 200,
                    "requested_url": "https://example.com/old-live",
                    "redirect_target_url": "https://example.com/live",
                }
            ],
            "noindex_pages": [],
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "broken_redirect"]
    assert len(issues) == 1
    assert issues[0]["url"] == "https://example.com/missing"
    assert issues[0]["issue_name"] == "Broken redirect"
    assert issues[0]["category"] == "redirects"
    assert issues[0]["importance"] == "Error"


def test_technical_seo_model_flags_redirect_chains_that_are_too_long() -> None:
    payload = build_technical_seo(
        [],
        indexability={
            "skipped": [
                {
                    "url": "https://example.com/final",
                    "title": "Final",
                    "reason": "noindex",
                    "http_status": 200,
                    "requested_url": "https://example.com/start",
                    "redirect_target_url": "https://example.com/final",
                    "redirect_chain": [
                        "https://example.com/start",
                        "https://example.com/a",
                        "https://example.com/b",
                        "https://example.com/c",
                        "https://example.com/d",
                        "https://example.com/e",
                        "https://example.com/final",
                    ],
                    "redirect_hop_count": 6,
                    "nofollow": False,
                },
                {
                    "url": "https://example.com/short-final",
                    "title": "Short Final",
                    "reason": "noindex",
                    "http_status": 200,
                    "requested_url": "https://example.com/short-start",
                    "redirect_target_url": "https://example.com/short-final",
                    "redirect_chain": [
                        "https://example.com/short-start",
                        "https://example.com/short-final",
                    ],
                    "redirect_hop_count": 1,
                    "nofollow": False,
                },
            ],
            "noindex_pages": [],
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "redirect_chain_too_long"]
    assert len(issues) == 1
    assert issues[0]["url"] == "https://example.com/final"
    assert issues[0]["issue_name"] == "Redirect chain too long"
    assert issues[0]["category"] == "redirects"
    assert issues[0]["importance"] == "Error"
    page = next(row for row in payload["pages"] if row["url"] == "https://example.com/final")
    assert page["redirect_hop_count"] == 6


def test_technical_seo_model_flags_redirect_chains() -> None:
    payload = build_technical_seo(
        [],
        indexability={
            "skipped": [
                {
                    "url": "https://example.com/final",
                    "title": "Final",
                    "reason": "noindex",
                    "http_status": 200,
                    "requested_url": "https://example.com/start",
                    "redirect_target_url": "https://example.com/final",
                    "redirect_hop_count": 2,
                    "redirect_chain": [
                        "https://example.com/start",
                        "https://example.com/middle",
                        "https://example.com/final",
                    ],
                    "nofollow": False,
                },
                {
                    "url": "https://example.com/one-hop-final",
                    "title": "One Hop Final",
                    "reason": "noindex",
                    "http_status": 200,
                    "requested_url": "https://example.com/one-hop-start",
                    "redirect_target_url": "https://example.com/one-hop-final",
                    "redirect_hop_count": 1,
                    "redirect_chain": [
                        "https://example.com/one-hop-start",
                        "https://example.com/one-hop-final",
                    ],
                    "nofollow": False,
                },
            ],
            "noindex_pages": [],
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "redirect_chain"]
    assert len(issues) == 1
    assert issues[0]["url"] == "https://example.com/final"
    assert issues[0]["issue_name"] == "Redirect chain"
    assert issues[0]["category"] == "redirects"
    assert issues[0]["importance"] == "Notice"


def test_technical_seo_model_flags_302_redirects() -> None:
    payload = build_technical_seo(
        [],
        indexability={
            "skipped": [
                {
                    "url": "https://example.com/final",
                    "title": "Final",
                    "reason": "noindex",
                    "http_status": 200,
                    "requested_url": "https://example.com/start",
                    "redirect_target_url": "https://example.com/final",
                    "redirect_status_codes": [301, 302],
                    "nofollow": False,
                },
                {
                    "url": "https://example.com/permanent-final",
                    "title": "Permanent Final",
                    "reason": "noindex",
                    "http_status": 200,
                    "requested_url": "https://example.com/permanent-start",
                    "redirect_target_url": "https://example.com/permanent-final",
                    "redirect_status_codes": [301],
                    "nofollow": False,
                },
            ],
            "noindex_pages": [],
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "302_redirect"]
    assert len(issues) == 1
    assert issues[0]["url"] == "https://example.com/final"
    assert issues[0]["issue_name"] == "302 redirect"
    assert issues[0]["category"] == "redirects"
    assert issues[0]["importance"] == "Warning"


def test_technical_seo_model_flags_3xx_redirects() -> None:
    payload = build_technical_seo(
        [],
        indexability={
            "skipped": [
                {
                    "url": "https://example.com/final",
                    "title": "Final",
                    "reason": "noindex",
                    "http_status": 200,
                    "requested_url": "https://example.com/start",
                    "redirect_target_url": "https://example.com/final",
                    "redirect_status_codes": [307],
                    "nofollow": False,
                },
                {
                    "url": "https://example.com/direct",
                    "title": "Direct",
                    "reason": "noindex",
                    "http_status": 200,
                    "requested_url": "https://example.com/direct",
                    "redirect_target_url": "",
                    "redirect_status_codes": [],
                    "nofollow": False,
                },
            ],
            "noindex_pages": [],
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "3xx_redirect"]
    assert len(issues) == 1
    assert issues[0]["url"] == "https://example.com/final"
    assert issues[0]["issue_name"] == "3XX redirect"
    assert issues[0]["category"] == "redirects"
    assert issues[0]["importance"] == "Warning"


def test_technical_seo_model_flags_https_to_http_redirects() -> None:
    payload = build_technical_seo(
        [],
        indexability={
            "skipped": [
                {
                    "url": "http://example.com/final",
                    "title": "Final",
                    "reason": "noindex",
                    "http_status": 200,
                    "requested_url": "https://example.com/start",
                    "redirect_target_url": "http://example.com/final",
                    "redirect_status_codes": [301],
                    "nofollow": False,
                },
                {
                    "url": "https://example.com/secure-final",
                    "title": "Secure Final",
                    "reason": "noindex",
                    "http_status": 200,
                    "requested_url": "https://example.com/secure-start",
                    "redirect_target_url": "https://example.com/secure-final",
                    "redirect_status_codes": [301],
                    "nofollow": False,
                },
            ],
            "noindex_pages": [],
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "https_to_http_redirect"]
    assert len(issues) == 1
    assert issues[0]["url"] == "http://example.com/final"
    assert issues[0]["issue_name"] == "HTTPS to HTTP redirect"
    assert issues[0]["category"] == "redirects"
    assert issues[0]["importance"] == "Warning"


def test_technical_seo_model_flags_http_to_https_redirects() -> None:
    payload = build_technical_seo(
        [],
        indexability={
            "skipped": [
                {
                    "url": "https://example.com/final",
                    "title": "Final",
                    "reason": "noindex",
                    "http_status": 200,
                    "requested_url": "http://example.com/start",
                    "redirect_target_url": "https://example.com/final",
                    "redirect_status_codes": [301],
                    "nofollow": False,
                },
                {
                    "url": "http://example.com/plain-final",
                    "title": "Plain Final",
                    "reason": "noindex",
                    "http_status": 200,
                    "requested_url": "http://example.com/plain-start",
                    "redirect_target_url": "http://example.com/plain-final",
                    "redirect_status_codes": [301],
                    "nofollow": False,
                },
            ],
            "noindex_pages": [],
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "http_to_https_redirect"]
    assert len(issues) == 1
    assert issues[0]["url"] == "https://example.com/final"
    assert issues[0]["issue_name"] == "HTTP to HTTPS redirect"
    assert issues[0]["category"] == "redirects"
    assert issues[0]["importance"] == "Notice"


def test_technical_seo_model_flags_meta_refresh_redirects() -> None:
    payload = build_technical_seo(
        [SimpleNamespace(url="https://example.com/source", title="Source", section="", word_count=100, language="en")],
        metadata_quality={
            "per_page": [
                {
                    "url": "https://example.com/source",
                    "title": "Source",
                    "meta_refresh_redirect": True,
                    "meta_refresh_target_url": "https://example.com/target",
                    "issues": [],
                }
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "meta_refresh_redirect"]
    assert len(issues) == 1
    assert issues[0]["url"] == "https://example.com/source"
    assert issues[0]["issue_name"] == "Meta refresh redirect"
    assert issues[0]["category"] == "redirects"
    assert issues[0]["importance"] == "Notice"
    page = payload["pages"][0]
    assert page["meta_refresh_target_url"] == "https://example.com/target"


def test_technical_seo_model_flags_redirect_target_changes() -> None:
    payload = build_technical_seo(
        [SimpleNamespace(url="https://example.com/source", title="Source", section="", word_count=100, language="en")],
        history_changes={
            "changes": [
                {
                    "url": "https://example.com/source",
                    "changed_fields": ["redirect_target"],
                    "redirect_target_before": "https://example.com/old",
                    "redirect_target_after": "https://example.com/new",
                }
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "redirect_target_changed"]
    assert len(issues) == 1
    assert issues[0]["url"] == "https://example.com/source"
    assert issues[0]["issue_name"] == "Redirect target changed"
    assert issues[0]["category"] == "redirects"
    assert issues[0]["importance"] == "Notice"
    page = payload["pages"][0]
    assert page["previous_redirect_target_url"] == "https://example.com/old"
    assert page["current_redirect_target_url"] == "https://example.com/new"


def test_technical_seo_model_flags_indexable_pages_with_multiple_meta_description_tags() -> None:
    pages = [
        SimpleNamespace(url="https://example.com/multiple", title="Multiple", section="", word_count=100, language="en"),
        SimpleNamespace(url="https://example.com/one", title="One", section="", word_count=100, language="en"),
    ]
    payload = build_technical_seo(
        pages,
        indexability={
            "skipped": [
                {
                    "url": "https://example.com/noindex",
                    "title": "Noindex",
                    "reason": "noindex",
                    "http_status": 200,
                    "meta_description_tag_count": 2,
                    "nofollow": False,
                }
            ],
            "noindex_pages": [],
        },
        metadata_quality={
            "per_page": [
                {
                    "url": "https://example.com/multiple",
                    "title": "Multiple",
                    "meta_description_tag_count": 2,
                    "issues": [],
                },
                {
                    "url": "https://example.com/one",
                    "title": "One",
                    "meta_description_tag_count": 1,
                    "issues": [],
                },
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "indexable_multiple_meta_description_tags"]
    assert len(issues) == 1
    assert issues[0]["url"] == "https://example.com/multiple"
    assert issues[0]["issue_name"] == "Multiple meta description tags"
    assert issues[0]["category"] == "content"
    assert issues[0]["importance"] == "Error"


def test_technical_seo_model_flags_not_indexable_pages_with_multiple_meta_description_tags() -> None:
    payload = build_technical_seo(
        [SimpleNamespace(url="https://example.com/indexable", title="Indexable", section="", word_count=100, language="en")],
        indexability={
            "skipped": [
                {
                    "url": "https://example.com/noindex-multiple",
                    "title": "Noindex Multiple",
                    "reason": "noindex",
                    "http_status": 200,
                    "meta_description_tag_count": 2,
                    "nofollow": False,
                },
                {
                    "url": "https://example.com/noindex-one",
                    "title": "Noindex One",
                    "reason": "noindex",
                    "http_status": 200,
                    "meta_description_tag_count": 1,
                    "nofollow": False,
                },
            ],
            "noindex_pages": [],
        },
        metadata_quality={
            "per_page": [
                {
                    "url": "https://example.com/indexable",
                    "title": "Indexable",
                    "meta_description_tag_count": 2,
                    "issues": [],
                },
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "not_indexable_multiple_meta_description_tags"]
    assert len(issues) == 1
    assert issues[0]["url"] == "https://example.com/noindex-multiple"
    assert issues[0]["issue_name"] == "Multiple meta description tags"
    assert issues[0]["category"] == "content"
    assert issues[0]["importance"] == "Warning"


def test_technical_seo_model_flags_indexable_pages_with_multiple_title_tags() -> None:
    pages = [
        SimpleNamespace(url="https://example.com/multiple", title="Multiple", section="", word_count=100, language="en"),
        SimpleNamespace(url="https://example.com/one", title="One", section="", word_count=100, language="en"),
    ]
    payload = build_technical_seo(
        pages,
        indexability={
            "skipped": [
                {
                    "url": "https://example.com/noindex",
                    "title": "Noindex",
                    "reason": "noindex",
                    "http_status": 200,
                    "title_tag_count": 2,
                    "nofollow": False,
                }
            ],
            "noindex_pages": [],
        },
        metadata_quality={
            "per_page": [
                {
                    "url": "https://example.com/multiple",
                    "title": "Multiple",
                    "title_tag_count": 2,
                    "issues": [],
                },
                {
                    "url": "https://example.com/one",
                    "title": "One",
                    "title_tag_count": 1,
                    "issues": [],
                },
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "indexable_multiple_title_tags"]
    assert len(issues) == 1
    assert issues[0]["url"] == "https://example.com/multiple"
    assert issues[0]["issue_name"] == "Multiple title tags"
    assert issues[0]["category"] == "content"
    assert issues[0]["importance"] == "Error"


def test_technical_seo_model_flags_not_indexable_pages_with_multiple_title_tags() -> None:
    payload = build_technical_seo(
        [SimpleNamespace(url="https://example.com/indexable", title="Indexable", section="", word_count=100, language="en")],
        indexability={
            "skipped": [
                {
                    "url": "https://example.com/noindex-multiple",
                    "title": "Noindex Multiple",
                    "reason": "noindex",
                    "http_status": 200,
                    "title_tag_count": 2,
                    "nofollow": False,
                },
                {
                    "url": "https://example.com/noindex-one",
                    "title": "Noindex One",
                    "reason": "noindex",
                    "http_status": 200,
                    "title_tag_count": 1,
                    "nofollow": False,
                },
            ],
            "noindex_pages": [],
        },
        metadata_quality={
            "per_page": [
                {
                    "url": "https://example.com/indexable",
                    "title": "Indexable",
                    "title_tag_count": 2,
                    "issues": [],
                },
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "not_indexable_multiple_title_tags"]
    assert len(issues) == 1
    assert issues[0]["url"] == "https://example.com/noindex-multiple"
    assert issues[0]["issue_name"] == "Multiple title tags"
    assert issues[0]["category"] == "content"
    assert issues[0]["importance"] == "Warning"


def test_technical_seo_model_flags_indexable_pages_with_missing_or_empty_title_tag() -> None:
    pages = [
        SimpleNamespace(url="https://example.com/missing", title="", section="", word_count=100, language="en"),
        SimpleNamespace(url="https://example.com/no-tag", title="Fallback title", section="", word_count=100, language="en"),
        SimpleNamespace(url="https://example.com/ok", title="Useful title", section="", word_count=100, language="en"),
    ]
    payload = build_technical_seo(
        pages,
        indexability={
            "skipped": [
                {
                    "url": "https://example.com/noindex",
                    "title": "",
                    "reason": "noindex",
                    "http_status": 200,
                    "title_tag_count": 0,
                    "nofollow": False,
                }
            ],
            "noindex_pages": [],
        },
        metadata_quality={
            "per_page": [
                {
                    "url": "https://example.com/missing",
                    "title": "",
                    "title_tag_count": 0,
                    "issues": ["missing_title"],
                },
                {
                    "url": "https://example.com/no-tag",
                    "title": "Fallback title",
                    "title_tag_count": 0,
                    "issues": [],
                },
                {
                    "url": "https://example.com/ok",
                    "title": "Useful title",
                    "title_tag_count": 1,
                    "issues": [],
                },
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "indexable_title_tag_missing_or_empty"]
    assert {row["url"] for row in issues} == {"https://example.com/missing", "https://example.com/no-tag"}
    assert all(row["issue_name"] == "Title tag missing or empty" for row in issues)
    assert all(row["category"] == "content" for row in issues)
    assert all(row["importance"] == "Error" for row in issues)


def test_technical_seo_model_flags_not_indexable_pages_with_missing_or_empty_title_tag() -> None:
    payload = build_technical_seo(
        [SimpleNamespace(url="https://example.com/indexable", title="", section="", word_count=100, language="en")],
        indexability={
            "skipped": [
                {
                    "url": "https://example.com/noindex-missing",
                    "title": "",
                    "reason": "noindex",
                    "http_status": 200,
                    "title_tag_count": 0,
                    "nofollow": False,
                },
                {
                    "url": "https://example.com/noindex-ok",
                    "title": "Noindex OK",
                    "reason": "noindex",
                    "http_status": 200,
                    "title_tag_count": 1,
                    "nofollow": False,
                },
            ],
            "noindex_pages": [],
        },
        metadata_quality={
            "per_page": [
                {
                    "url": "https://example.com/indexable",
                    "title": "",
                    "title_tag_count": 0,
                    "issues": ["missing_title"],
                },
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "not_indexable_title_tag_missing_or_empty"]
    assert len(issues) == 1
    assert issues[0]["url"] == "https://example.com/noindex-missing"
    assert issues[0]["issue_name"] == "Title tag missing or empty"
    assert issues[0]["category"] == "content"
    assert issues[0]["importance"] == "Warning"


def test_technical_seo_model_flags_indexable_pages_with_title_too_long() -> None:
    pages = [
        SimpleNamespace(url="https://example.com/long", title="Long", section="", word_count=250, language="en"),
        SimpleNamespace(url="https://example.com/length", title="Length", section="", word_count=250, language="en"),
        SimpleNamespace(url="https://example.com/ok", title="OK", section="", word_count=250, language="en"),
    ]
    payload = build_technical_seo(
        pages,
        metadata_quality={
            "per_page": [
                {
                    "url": "https://example.com/long",
                    "title": "Long",
                    "title_length": 80,
                    "issues": ["long_title"],
                },
                {
                    "url": "https://example.com/length",
                    "title": "Length",
                    "title_length": 70,
                    "issues": [],
                },
                {
                    "url": "https://example.com/ok",
                    "title": "OK",
                    "title_length": 45,
                    "issues": [],
                },
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "indexable_title_too_long"]
    assert {row["url"] for row in issues} == {"https://example.com/long", "https://example.com/length"}
    assert all(row["issue_name"] == "Title too long" for row in issues)
    assert all(row["category"] == "content" for row in issues)
    assert all(row["importance"] == "Warning" for row in issues)


def test_technical_seo_model_flags_not_indexable_pages_with_title_too_long() -> None:
    payload = build_technical_seo(
        [SimpleNamespace(url="https://example.com/indexable", title="Indexable", section="", word_count=250, language="en")],
        indexability={
            "skipped": [
                {
                    "url": "https://example.com/noindex-long",
                    "title": "Noindex Long",
                    "reason": "noindex",
                    "http_status": 200,
                    "nofollow": False,
                },
                {
                    "url": "https://example.com/noindex-length",
                    "title": "Noindex Length",
                    "reason": "noindex",
                    "http_status": 200,
                    "nofollow": False,
                },
                {
                    "url": "https://example.com/noindex-ok",
                    "title": "Noindex OK",
                    "reason": "noindex",
                    "http_status": 200,
                    "nofollow": False,
                },
            ],
            "noindex_pages": [],
        },
        metadata_quality={
            "per_page": [
                {"url": "https://example.com/indexable", "title": "Indexable", "title_length": 80, "issues": ["long_title"]},
                {"url": "https://example.com/noindex-long", "title": "Noindex Long", "title_length": 80, "issues": ["long_title"]},
                {"url": "https://example.com/noindex-length", "title": "Noindex Length", "title_length": 70, "issues": []},
                {"url": "https://example.com/noindex-ok", "title": "Noindex OK", "title_length": 45, "issues": []},
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "not_indexable_title_too_long"]
    assert {row["url"] for row in issues} == {"https://example.com/noindex-long", "https://example.com/noindex-length"}
    assert all(row["issue_name"] == "Title too long" for row in issues)
    assert all(row["category"] == "content" for row in issues)
    assert all(row["importance"] == "Notice" for row in issues)


def test_technical_seo_model_flags_indexable_pages_with_title_too_short() -> None:
    pages = [
        SimpleNamespace(url="https://example.com/short", title="Short", section="", word_count=250, language="en"),
        SimpleNamespace(url="https://example.com/length", title="Length", section="", word_count=250, language="en"),
        SimpleNamespace(url="https://example.com/missing", title="", section="", word_count=250, language="en"),
        SimpleNamespace(url="https://example.com/ok", title="Useful page title", section="", word_count=250, language="en"),
    ]
    payload = build_technical_seo(
        pages,
        metadata_quality={
            "per_page": [
                {
                    "url": "https://example.com/short",
                    "title": "Short",
                    "title_length": 12,
                    "issues": ["short_title"],
                },
                {
                    "url": "https://example.com/length",
                    "title": "Length",
                    "title_length": 18,
                    "issues": [],
                },
                {
                    "url": "https://example.com/missing",
                    "title": "",
                    "title_length": 0,
                    "issues": ["missing_title"],
                },
                {
                    "url": "https://example.com/ok",
                    "title": "Useful page title",
                    "title_length": 40,
                    "issues": [],
                },
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "indexable_title_too_short"]
    assert {row["url"] for row in issues} == {"https://example.com/short", "https://example.com/length"}
    assert all(row["issue_name"] == "Title too short" for row in issues)
    assert all(row["category"] == "content" for row in issues)
    assert all(row["importance"] == "Warning" for row in issues)


def test_technical_seo_model_flags_not_indexable_pages_with_title_too_short() -> None:
    payload = build_technical_seo(
        [SimpleNamespace(url="https://example.com/indexable", title="Indexable", section="", word_count=250, language="en")],
        indexability={
            "skipped": [
                {
                    "url": "https://example.com/noindex-short",
                    "title": "Noindex Short",
                    "reason": "noindex",
                    "http_status": 200,
                    "nofollow": False,
                },
                {
                    "url": "https://example.com/noindex-length",
                    "title": "Noindex Length",
                    "reason": "noindex",
                    "http_status": 200,
                    "nofollow": False,
                },
                {
                    "url": "https://example.com/noindex-missing",
                    "title": "",
                    "reason": "noindex",
                    "http_status": 200,
                    "nofollow": False,
                },
                {
                    "url": "https://example.com/noindex-ok",
                    "title": "Useful noindex page title",
                    "reason": "noindex",
                    "http_status": 200,
                    "nofollow": False,
                },
            ],
            "noindex_pages": [],
        },
        metadata_quality={
            "per_page": [
                {"url": "https://example.com/indexable", "title": "Indexable", "title_length": 12, "issues": ["short_title"]},
                {"url": "https://example.com/noindex-short", "title": "Noindex Short", "title_length": 12, "issues": ["short_title"]},
                {"url": "https://example.com/noindex-length", "title": "Noindex Length", "title_length": 18, "issues": []},
                {"url": "https://example.com/noindex-missing", "title": "", "title_length": 0, "issues": ["missing_title"]},
                {"url": "https://example.com/noindex-ok", "title": "Useful noindex page title", "title_length": 40, "issues": []},
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "not_indexable_title_too_short"]
    assert {row["url"] for row in issues} == {"https://example.com/noindex-short", "https://example.com/noindex-length"}
    assert all(row["issue_name"] == "Title too short" for row in issues)
    assert all(row["category"] == "content" for row in issues)
    assert all(row["importance"] == "Notice" for row in issues)


def test_technical_seo_model_flags_indexable_pages_with_missing_or_empty_h1() -> None:
    pages = [
        SimpleNamespace(url="https://example.com/missing", title="Missing", section="", word_count=100, language="en"),
        SimpleNamespace(url="https://example.com/ok", title="OK", section="", word_count=100, language="en"),
    ]
    payload = build_technical_seo(
        pages,
        indexability={
            "skipped": [
                {
                    "url": "https://example.com/noindex",
                    "title": "Noindex",
                    "reason": "noindex",
                    "http_status": 200,
                    "nofollow": False,
                }
            ],
            "noindex_pages": [],
        },
        header_analysis={
            "per_page": [
                {"url": "https://example.com/missing", "h1": "", "h1_count": 0, "header_count": 2},
                {"url": "https://example.com/ok", "h1": "Useful H1", "h1_count": 1, "header_count": 3},
                {"url": "https://example.com/noindex", "h1": "", "h1_count": 0, "header_count": 1},
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "indexable_h1_tag_missing_or_empty"]
    assert len(issues) == 1
    assert issues[0]["url"] == "https://example.com/missing"
    assert issues[0]["issue_name"] == "H1 tag missing or empty"
    assert issues[0]["category"] == "content"
    assert issues[0]["importance"] == "Warning"
    missing_page = next(row for row in payload["pages"] if row["url"] == "https://example.com/missing")
    assert missing_page["h1_count"] == 0


def test_technical_seo_model_flags_not_indexable_pages_with_missing_or_empty_h1() -> None:
    payload = build_technical_seo(
        [SimpleNamespace(url="https://example.com/indexable", title="Indexable", section="", word_count=100, language="en")],
        indexability={
            "skipped": [
                {
                    "url": "https://example.com/noindex-missing",
                    "title": "Noindex Missing",
                    "reason": "noindex",
                    "http_status": 200,
                    "nofollow": False,
                },
                {
                    "url": "https://example.com/noindex-ok",
                    "title": "Noindex OK",
                    "reason": "noindex",
                    "http_status": 200,
                    "nofollow": False,
                },
            ],
            "noindex_pages": [],
        },
        header_analysis={
            "per_page": [
                {"url": "https://example.com/indexable", "h1": "", "h1_count": 0, "header_count": 2},
                {"url": "https://example.com/noindex-missing", "h1": "", "h1_count": 0, "header_count": 1},
                {"url": "https://example.com/noindex-ok", "h1": "Noindex H1", "h1_count": 1, "header_count": 2},
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "not_indexable_h1_tag_missing_or_empty"]
    assert len(issues) == 1
    assert issues[0]["url"] == "https://example.com/noindex-missing"
    assert issues[0]["issue_name"] == "H1 tag missing or empty"
    assert issues[0]["category"] == "content"
    assert issues[0]["importance"] == "Notice"


def test_technical_seo_model_flags_indexable_pages_with_multiple_h1_tags() -> None:
    pages = [
        SimpleNamespace(url="https://example.com/multiple", title="Multiple", section="", word_count=250, language="en"),
        SimpleNamespace(url="https://example.com/ok", title="OK", section="", word_count=250, language="en"),
    ]
    payload = build_technical_seo(
        pages,
        indexability={
            "skipped": [
                {
                    "url": "https://example.com/noindex",
                    "title": "Noindex",
                    "reason": "noindex",
                    "http_status": 200,
                    "nofollow": False,
                }
            ],
            "noindex_pages": [],
        },
        header_analysis={
            "per_page": [
                {"url": "https://example.com/multiple", "h1": "Primary H1", "h1_count": 2, "header_count": 4},
                {"url": "https://example.com/ok", "h1": "Useful H1", "h1_count": 1, "header_count": 3},
                {"url": "https://example.com/noindex", "h1": "Noindex H1", "h1_count": 2, "header_count": 3},
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "indexable_multiple_h1_tags"]
    assert len(issues) == 1
    assert issues[0]["url"] == "https://example.com/multiple"
    assert issues[0]["issue_name"] == "Multiple H1 tags"
    assert issues[0]["category"] == "content"
    assert issues[0]["importance"] == "Notice"


def test_technical_seo_model_flags_not_indexable_pages_with_multiple_h1_tags() -> None:
    payload = build_technical_seo(
        [SimpleNamespace(url="https://example.com/indexable", title="Indexable", section="", word_count=250, language="en")],
        indexability={
            "skipped": [
                {
                    "url": "https://example.com/noindex-multiple",
                    "title": "Noindex Multiple",
                    "reason": "noindex",
                    "http_status": 200,
                    "nofollow": False,
                },
                {
                    "url": "https://example.com/noindex-ok",
                    "title": "Noindex OK",
                    "reason": "noindex",
                    "http_status": 200,
                    "nofollow": False,
                },
            ],
            "noindex_pages": [],
        },
        header_analysis={
            "per_page": [
                {"url": "https://example.com/indexable", "h1": "Primary H1", "h1_count": 2, "header_count": 4},
                {"url": "https://example.com/noindex-multiple", "h1": "Noindex H1", "h1_count": 2, "header_count": 3},
                {"url": "https://example.com/noindex-ok", "h1": "Noindex H1", "h1_count": 1, "header_count": 2},
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "not_indexable_multiple_h1_tags"]
    assert len(issues) == 1
    assert issues[0]["url"] == "https://example.com/noindex-multiple"
    assert issues[0]["issue_name"] == "Multiple H1 tags"
    assert issues[0]["category"] == "content"
    assert issues[0]["importance"] == "Notice"


def test_technical_seo_model_flags_indexable_h1_changes() -> None:
    payload = build_technical_seo(
        [SimpleNamespace(url="https://example.com/page", title="Page", section="", word_count=250, language="en")],
        history_changes={
            "changes": [
                {
                    "url": "https://example.com/page",
                    "changed_fields": ["h1"],
                    "h1_before": "Old H1",
                    "h1_after": "New H1",
                }
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "indexable_h1_tag_changed"]
    assert len(issues) == 1
    assert issues[0]["url"] == "https://example.com/page"
    assert issues[0]["issue_name"] == "H1 tag changed"
    assert issues[0]["category"] == "content"
    assert issues[0]["importance"] == "Notice"
    page = payload["pages"][0]
    assert page["previous_h1"] == "Old H1"
    assert page["current_h1"] == "New H1"


def test_technical_seo_model_flags_indexable_meta_description_changes() -> None:
    payload = build_technical_seo(
        [SimpleNamespace(url="https://example.com/page", title="Page", section="", word_count=250, language="en")],
        history_changes={
            "changes": [
                {
                    "url": "https://example.com/page",
                    "changed_fields": ["description"],
                    "description_before": "Old meta description",
                    "description_after": "New meta description",
                }
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "indexable_meta_description_changed"]
    assert len(issues) == 1
    assert issues[0]["url"] == "https://example.com/page"
    assert issues[0]["issue_name"] == "Meta description changed"
    assert issues[0]["category"] == "content"
    assert issues[0]["importance"] == "Notice"
    page = payload["pages"][0]
    assert page["previous_description"] == "Old meta description"
    assert page["current_description"] == "New meta description"


def test_technical_seo_model_flags_indexable_page_and_serp_title_mismatch() -> None:
    pages = [
        SimpleNamespace(url="https://example.com/mismatch", title="Buy Tablets Online", section="", word_count=250, language="en"),
        SimpleNamespace(url="https://example.com/normalized", title="Buy Tablets | Example", section="", word_count=250, language="en"),
        SimpleNamespace(url="https://example.com/no-serp", title="No SERP Title", section="", word_count=250, language="en"),
    ]
    payload = build_technical_seo(
        pages,
        indexability={
            "skipped": [
                {
                    "url": "https://example.com/noindex",
                    "title": "Noindex Title",
                    "reason": "noindex",
                    "http_status": 200,
                    "nofollow": False,
                }
            ],
            "noindex_pages": [],
        },
        search_payload={
            "top_pages": [
                {
                    "matched_url": "https://example.com/mismatch",
                    "traffic": 100,
                    "keywords": 5,
                    "top_keyword": "tablets",
                    "top_keyword_title": "Cheap Tablets - Example",
                },
                {
                    "matched_url": "https://example.com/normalized",
                    "traffic": 80,
                    "keywords": 3,
                    "top_keyword": "buy tablets",
                    "top_keyword_title": "buy tablets - example",
                },
                {
                    "matched_url": "https://example.com/no-serp",
                    "traffic": 10,
                    "keywords": 1,
                    "top_keyword": "no serp",
                },
                {
                    "matched_url": "https://example.com/noindex",
                    "traffic": 5,
                    "keywords": 1,
                    "top_keyword": "noindex",
                    "top_keyword_title": "Different Noindex SERP Title",
                },
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "indexable_page_and_serp_titles_do_not_match"]
    assert len(issues) == 1
    assert issues[0]["url"] == "https://example.com/mismatch"
    assert issues[0]["issue_name"] == "Page and SERP titles do not match"
    assert issues[0]["category"] == "content"
    assert issues[0]["importance"] == "Notice"
    page = next(row for row in payload["pages"] if row["url"] == "https://example.com/mismatch")
    assert page["serp_title"] == "Cheap Tablets - Example"


def test_technical_seo_model_flags_indexable_serp_title_changes() -> None:
    payload = build_technical_seo(
        [SimpleNamespace(url="https://example.com/page", title="Page", section="", word_count=250, language="en")],
        history_changes={
            "changes": [
                {
                    "url": "https://example.com/page",
                    "changed_fields": ["serp_title"],
                    "serp_title_before": "Old SERP title",
                    "serp_title_after": "New SERP title",
                }
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "indexable_serp_title_changed"]
    assert len(issues) == 1
    assert issues[0]["url"] == "https://example.com/page"
    assert issues[0]["issue_name"] == "SERP title changed"
    assert issues[0]["category"] == "content"
    assert issues[0]["importance"] == "Notice"
    page = payload["pages"][0]
    assert page["previous_serp_title"] == "Old SERP title"
    assert page["current_serp_title"] == "New SERP title"


def test_technical_seo_model_flags_indexable_title_tag_changes() -> None:
    payload = build_technical_seo(
        [SimpleNamespace(url="https://example.com/page", title="New title", section="", word_count=250, language="en")],
        history_changes={
            "changes": [
                {
                    "url": "https://example.com/page",
                    "changed_fields": ["title"],
                    "title_before": "Old title",
                    "title_after": "New title",
                }
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "indexable_title_tag_changed"]
    assert len(issues) == 1
    assert issues[0]["url"] == "https://example.com/page"
    assert issues[0]["issue_name"] == "Title tag changed"
    assert issues[0]["category"] == "content"
    assert issues[0]["importance"] == "Notice"
    page = payload["pages"][0]
    assert page["previous_title"] == "Old title"
    assert page["current_title"] == "New title"


def test_technical_seo_model_flags_indexable_word_count_changes() -> None:
    pages = [
        SimpleNamespace(url="https://example.com/page", title="Page", section="", word_count=260, language="en"),
        SimpleNamespace(url="https://example.com/stable", title="Stable", section="", word_count=260, language="en"),
    ]
    payload = build_technical_seo(
        pages,
        indexability={
            "skipped": [
                {
                    "url": "https://example.com/noindex",
                    "title": "Noindex",
                    "reason": "noindex",
                    "http_status": 200,
                    "word_count": 260,
                    "nofollow": False,
                }
            ],
            "noindex_pages": [],
        },
        history_changes={
            "changes": [
                {
                    "url": "https://example.com/page",
                    "changed_fields": ["word_count"],
                    "word_count_before": 120,
                    "word_count_after": 260,
                },
                {
                    "url": "https://example.com/noindex",
                    "changed_fields": ["word_count"],
                    "word_count_before": 120,
                    "word_count_after": 260,
                },
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "indexable_word_count_changed"]
    assert len(issues) == 1
    assert issues[0]["url"] == "https://example.com/page"
    assert issues[0]["issue_name"] == "Word count changed"
    assert issues[0]["category"] == "content"
    assert issues[0]["importance"] == "Notice"
    page = next(row for row in payload["pages"] if row["url"] == "https://example.com/page")
    assert page["previous_word_count"] == 120
    assert page["current_word_count"] == 260


def test_technical_seo_model_flags_indexable_pages_with_high_ai_content_levels() -> None:
    pages = [
        SimpleNamespace(url="https://example.com/high-level", title="High Level", section="", word_count=250, language="en"),
        SimpleNamespace(url="https://example.com/high-score", title="High Score", section="", word_count=250, language="en"),
        SimpleNamespace(url="https://example.com/medium", title="Medium", section="", word_count=250, language="en"),
    ]
    payload = build_technical_seo(
        pages,
        indexability={
            "skipped": [
                {
                    "url": "https://example.com/noindex",
                    "title": "Noindex",
                    "reason": "noindex",
                    "http_status": 200,
                    "nofollow": False,
                }
            ],
            "noindex_pages": [],
        },
        content_quality={
            "per_page": [
                {"url": "https://example.com/high-level", "ai_content_level": "high", "ai_content_score": 0.6},
                {"url": "https://example.com/high-score", "ai_content_level": "medium", "ai_content_score": 0.86},
                {"url": "https://example.com/medium", "ai_content_level": "medium", "ai_content_score": 0.5},
                {"url": "https://example.com/noindex", "ai_content_level": "very_high", "ai_content_score": 0.95},
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "indexable_pages_have_high_ai_content_levels"]
    assert {row["url"] for row in issues} == {"https://example.com/high-level", "https://example.com/high-score"}
    assert all(row["issue_name"] == "Pages have high AI content levels" for row in issues)
    assert all(row["category"] == "content" for row in issues)
    assert all(row["importance"] == "Notice" for row in issues)
    high_score_page = next(row for row in payload["pages"] if row["url"] == "https://example.com/high-score")
    assert high_score_page["ai_content_score"] == 0.86


def test_technical_seo_model_flags_indexable_pages_with_low_word_count() -> None:
    pages = [
        SimpleNamespace(url="https://example.com/thin", title="Thin", section="", word_count=80, language="en"),
        SimpleNamespace(url="https://example.com/full", title="Full", section="", word_count=250, language="en"),
        SimpleNamespace(url="https://example.com/unknown", title="Unknown", section="", word_count=0, language="en"),
    ]
    payload = build_technical_seo(
        pages,
        indexability={
            "skipped": [
                {
                    "url": "https://example.com/noindex",
                    "title": "Noindex",
                    "reason": "noindex",
                    "http_status": 200,
                    "word_count": 80,
                    "nofollow": False,
                }
            ],
            "noindex_pages": [],
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "indexable_low_word_count"]
    assert len(issues) == 1
    assert issues[0]["url"] == "https://example.com/thin"
    assert issues[0]["issue_name"] == "Low word count"
    assert issues[0]["category"] == "content"
    assert issues[0]["importance"] == "Warning"


def test_technical_seo_model_flags_not_indexable_pages_with_low_word_count() -> None:
    payload = build_technical_seo(
        [SimpleNamespace(url="https://example.com/indexable-thin", title="Thin", section="", word_count=80, language="en")],
        indexability={
            "skipped": [
                {
                    "url": "https://example.com/noindex-thin",
                    "title": "Noindex Thin",
                    "reason": "noindex",
                    "http_status": 200,
                    "word_count": 80,
                    "nofollow": False,
                },
                {
                    "url": "https://example.com/noindex-full",
                    "title": "Noindex Full",
                    "reason": "noindex",
                    "http_status": 200,
                    "word_count": 250,
                    "nofollow": False,
                },
                {
                    "url": "https://example.com/noindex-unknown",
                    "title": "Noindex Unknown",
                    "reason": "noindex",
                    "http_status": 200,
                    "word_count": 0,
                    "nofollow": False,
                },
            ],
            "noindex_pages": [],
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "not_indexable_low_word_count"]
    assert len(issues) == 1
    assert issues[0]["url"] == "https://example.com/noindex-thin"
    assert issues[0]["issue_name"] == "Low word count"
    assert issues[0]["category"] == "content"
    assert issues[0]["importance"] == "Notice"
    page = next(row for row in payload["pages"] if row["url"] == "https://example.com/noindex-thin")
    assert page["word_count"] == 80


def test_technical_seo_model_flags_indexable_pages_with_missing_or_empty_meta_description() -> None:
    pages = [
        SimpleNamespace(url="https://example.com/missing", title="Missing", section="", word_count=250, language="en"),
        SimpleNamespace(url="https://example.com/no-tag", title="No Tag", section="", word_count=250, language="en"),
        SimpleNamespace(url="https://example.com/ok", title="OK", section="", word_count=250, language="en"),
    ]
    payload = build_technical_seo(
        pages,
        indexability={
            "skipped": [
                {
                    "url": "https://example.com/noindex",
                    "title": "Noindex",
                    "reason": "noindex",
                    "http_status": 200,
                    "meta_description_tag_count": 0,
                    "nofollow": False,
                }
            ],
            "noindex_pages": [],
        },
        metadata_quality={
            "per_page": [
                {
                    "url": "https://example.com/missing",
                    "title": "Missing",
                    "meta_description_tag_count": 0,
                    "issues": ["missing_description"],
                },
                {
                    "url": "https://example.com/no-tag",
                    "title": "No Tag",
                    "meta_description_tag_count": 0,
                    "issues": [],
                },
                {
                    "url": "https://example.com/ok",
                    "title": "OK",
                    "description": "A useful search snippet for the OK page.",
                    "meta_description_tag_count": 1,
                    "issues": [],
                },
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "indexable_meta_description_tag_missing_or_empty"]
    assert {row["url"] for row in issues} == {"https://example.com/missing", "https://example.com/no-tag"}
    assert all(row["issue_name"] == "Meta description tag missing or empty" for row in issues)
    assert all(row["category"] == "content" for row in issues)
    assert all(row["importance"] == "Warning" for row in issues)


def test_technical_seo_model_flags_not_indexable_pages_with_missing_or_empty_meta_description() -> None:
    payload = build_technical_seo(
        [SimpleNamespace(url="https://example.com/indexable", title="Indexable", section="", word_count=250, language="en")],
        indexability={
            "skipped": [
                {
                    "url": "https://example.com/noindex-missing",
                    "title": "Noindex Missing",
                    "reason": "noindex",
                    "http_status": 200,
                    "meta_description_tag_count": 0,
                    "nofollow": False,
                },
                {
                    "url": "https://example.com/noindex-ok",
                    "title": "Noindex OK",
                    "reason": "noindex",
                    "http_status": 200,
                    "meta_description_tag_count": 1,
                    "nofollow": False,
                },
            ],
            "noindex_pages": [],
        },
        metadata_quality={
            "per_page": [
                {
                    "url": "https://example.com/indexable",
                    "title": "Indexable",
                    "meta_description_tag_count": 0,
                    "issues": ["missing_description"],
                },
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "not_indexable_meta_description_tag_missing_or_empty"]
    assert len(issues) == 1
    assert issues[0]["url"] == "https://example.com/noindex-missing"
    assert issues[0]["issue_name"] == "Meta description tag missing or empty"
    assert issues[0]["category"] == "content"
    assert issues[0]["importance"] == "Warning"


def test_technical_seo_model_flags_indexable_pages_with_meta_description_too_long() -> None:
    pages = [
        SimpleNamespace(url="https://example.com/long", title="Long", section="", word_count=250, language="en"),
        SimpleNamespace(url="https://example.com/length", title="Length", section="", word_count=250, language="en"),
        SimpleNamespace(url="https://example.com/ok", title="OK", section="", word_count=250, language="en"),
    ]
    payload = build_technical_seo(
        pages,
        metadata_quality={
            "per_page": [
                {
                    "url": "https://example.com/long",
                    "title": "Long",
                    "description_length": 180,
                    "issues": ["long_description"],
                },
                {
                    "url": "https://example.com/length",
                    "title": "Length",
                    "description_length": 170,
                    "issues": [],
                },
                {
                    "url": "https://example.com/ok",
                    "title": "OK",
                    "description_length": 120,
                    "issues": [],
                },
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "indexable_meta_description_too_long"]
    assert {row["url"] for row in issues} == {"https://example.com/long", "https://example.com/length"}
    assert all(row["issue_name"] == "Meta description too long" for row in issues)
    assert all(row["category"] == "content" for row in issues)
    assert all(row["importance"] == "Warning" for row in issues)


def test_technical_seo_model_flags_not_indexable_pages_with_meta_description_too_long() -> None:
    payload = build_technical_seo(
        [SimpleNamespace(url="https://example.com/indexable", title="Indexable", section="", word_count=250, language="en")],
        indexability={
            "skipped": [
                {
                    "url": "https://example.com/noindex-long",
                    "title": "Noindex Long",
                    "reason": "noindex",
                    "http_status": 200,
                    "nofollow": False,
                },
                {
                    "url": "https://example.com/noindex-length",
                    "title": "Noindex Length",
                    "reason": "noindex",
                    "http_status": 200,
                    "nofollow": False,
                },
                {
                    "url": "https://example.com/noindex-ok",
                    "title": "Noindex OK",
                    "reason": "noindex",
                    "http_status": 200,
                    "nofollow": False,
                },
            ],
            "noindex_pages": [],
        },
        metadata_quality={
            "per_page": [
                {"url": "https://example.com/indexable", "title": "Indexable", "description_length": 180, "issues": ["long_description"]},
                {"url": "https://example.com/noindex-long", "title": "Noindex Long", "description_length": 180, "issues": ["long_description"]},
                {"url": "https://example.com/noindex-length", "title": "Noindex Length", "description_length": 170, "issues": []},
                {"url": "https://example.com/noindex-ok", "title": "Noindex OK", "description_length": 120, "issues": []},
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "not_indexable_meta_description_too_long"]
    assert {row["url"] for row in issues} == {"https://example.com/noindex-long", "https://example.com/noindex-length"}
    assert all(row["issue_name"] == "Meta description too long" for row in issues)
    assert all(row["category"] == "content" for row in issues)
    assert all(row["importance"] == "Notice" for row in issues)


def test_technical_seo_model_flags_indexable_pages_with_meta_description_too_short() -> None:
    pages = [
        SimpleNamespace(url="https://example.com/short", title="Short", section="", word_count=250, language="en"),
        SimpleNamespace(url="https://example.com/length", title="Length", section="", word_count=250, language="en"),
        SimpleNamespace(url="https://example.com/missing", title="Missing", section="", word_count=250, language="en"),
        SimpleNamespace(url="https://example.com/ok", title="OK", section="", word_count=250, language="en"),
    ]
    payload = build_technical_seo(
        pages,
        metadata_quality={
            "per_page": [
                {
                    "url": "https://example.com/short",
                    "title": "Short",
                    "description_length": 40,
                    "issues": ["short_description"],
                },
                {
                    "url": "https://example.com/length",
                    "title": "Length",
                    "description_length": 20,
                    "issues": [],
                },
                {
                    "url": "https://example.com/missing",
                    "title": "Missing",
                    "description_length": 0,
                    "issues": ["missing_description"],
                },
                {
                    "url": "https://example.com/ok",
                    "title": "OK",
                    "description_length": 120,
                    "issues": [],
                },
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "indexable_meta_description_too_short"]
    assert {row["url"] for row in issues} == {"https://example.com/short", "https://example.com/length"}
    assert all(row["issue_name"] == "Meta description too short" for row in issues)
    assert all(row["category"] == "content" for row in issues)
    assert all(row["importance"] == "Warning" for row in issues)


def test_technical_seo_model_flags_not_indexable_pages_with_meta_description_too_short() -> None:
    payload = build_technical_seo(
        [SimpleNamespace(url="https://example.com/indexable", title="Indexable", section="", word_count=250, language="en")],
        indexability={
            "skipped": [
                {
                    "url": "https://example.com/noindex-short",
                    "title": "Noindex Short",
                    "reason": "noindex",
                    "http_status": 200,
                    "nofollow": False,
                },
                {
                    "url": "https://example.com/noindex-length",
                    "title": "Noindex Length",
                    "reason": "noindex",
                    "http_status": 200,
                    "nofollow": False,
                },
                {
                    "url": "https://example.com/noindex-missing",
                    "title": "Noindex Missing",
                    "reason": "noindex",
                    "http_status": 200,
                    "nofollow": False,
                },
                {
                    "url": "https://example.com/noindex-ok",
                    "title": "Noindex OK",
                    "reason": "noindex",
                    "http_status": 200,
                    "nofollow": False,
                },
            ],
            "noindex_pages": [],
        },
        metadata_quality={
            "per_page": [
                {"url": "https://example.com/indexable", "title": "Indexable", "description_length": 40, "issues": ["short_description"]},
                {"url": "https://example.com/noindex-short", "title": "Noindex Short", "description_length": 40, "issues": ["short_description"]},
                {"url": "https://example.com/noindex-length", "title": "Noindex Length", "description_length": 20, "issues": []},
                {"url": "https://example.com/noindex-missing", "title": "Noindex Missing", "description_length": 0, "issues": ["missing_description"]},
                {"url": "https://example.com/noindex-ok", "title": "Noindex OK", "description_length": 120, "issues": []},
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "not_indexable_meta_description_too_short"]
    assert {row["url"] for row in issues} == {"https://example.com/noindex-short", "https://example.com/noindex-length"}
    assert all(row["issue_name"] == "Meta description too short" for row in issues)
    assert all(row["category"] == "content" for row in issues)
    assert all(row["importance"] == "Notice" for row in issues)


def test_technical_seo_model_flags_open_graph_tags_incomplete() -> None:
    payload = build_technical_seo(
        [
            SimpleNamespace(url="https://example.com/incomplete", title="Incomplete", section="", word_count=250, language="en"),
            SimpleNamespace(url="https://example.com/complete", title="Complete", section="", word_count=250, language="en"),
            SimpleNamespace(url="https://example.com/missing", title="Missing", section="", word_count=250, language="en"),
        ],
        indexability={
            "skipped": [
                {
                    "url": "https://example.com/noindex-incomplete",
                    "title": "Noindex Incomplete",
                    "reason": "noindex",
                    "http_status": 200,
                    "nofollow": False,
                }
            ],
            "noindex_pages": [],
        },
        metadata_quality={
            "per_page": [
                {
                    "url": "https://example.com/incomplete",
                    "title": "Incomplete",
                    "og_title": "Incomplete OG",
                    "og_description": "",
                    "og_tag_count": 1,
                    "og_missing_fields": ["og_description"],
                    "issues": ["incomplete_open_graph"],
                },
                {
                    "url": "https://example.com/noindex-incomplete",
                    "title": "Noindex Incomplete",
                    "og_title": "",
                    "og_description": "Noindex description",
                    "og_tag_count": 1,
                    "og_missing_fields": ["og_title"],
                    "issues": ["incomplete_open_graph"],
                },
                {
                    "url": "https://example.com/complete",
                    "title": "Complete",
                    "og_title": "Complete OG",
                    "og_description": "Complete OG description",
                    "og_tag_count": 2,
                    "og_missing_fields": [],
                    "issues": [],
                },
                {
                    "url": "https://example.com/missing",
                    "title": "Missing",
                    "og_tag_count": 0,
                    "og_missing_fields": ["og_title", "og_description"],
                    "issues": [],
                },
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "open_graph_tags_incomplete"]
    assert {row["url"] for row in issues} == {"https://example.com/incomplete", "https://example.com/noindex-incomplete"}
    assert all(row["issue_name"] == "Open Graph tags incomplete" for row in issues)
    assert all(row["category"] == "social_tags" for row in issues)
    assert all(row["importance"] == "Warning" for row in issues)
    page = next(row for row in payload["pages"] if row["url"] == "https://example.com/incomplete")
    assert page["og_missing_fields"] == ["og_description"]


def test_technical_seo_model_flags_open_graph_url_not_matching_canonical() -> None:
    payload = build_technical_seo(
        [
            SimpleNamespace(url="https://example.com/mismatch", title="Mismatch", section="", word_count=250, language="en"),
            SimpleNamespace(url="https://example.com/normalized", title="Normalized", section="", word_count=250, language="en"),
            SimpleNamespace(url="https://example.com/missing-og-url", title="Missing OG URL", section="", word_count=250, language="en"),
        ],
        indexability={
            "skipped": [
                {
                    "url": "https://example.com/noindex-mismatch",
                    "title": "Noindex Mismatch",
                    "reason": "noindex",
                    "http_status": 200,
                    "nofollow": False,
                }
            ],
            "noindex_pages": [],
        },
        metadata_quality={
            "per_page": [
                {
                    "url": "https://example.com/mismatch",
                    "title": "Mismatch",
                    "canonical_url": "https://example.com/canonical",
                    "og_url": "https://example.com/other",
                    "issues": [],
                },
                {
                    "url": "https://example.com/noindex-mismatch",
                    "title": "Noindex Mismatch",
                    "canonical_url": "https://example.com/noindex-canonical",
                    "og_url": "https://example.com/noindex-other",
                    "issues": [],
                },
                {
                    "url": "https://example.com/normalized",
                    "title": "Normalized",
                    "canonical_url": "https://www.example.com/normalized/",
                    "og_url": "https://example.com/normalized",
                    "issues": [],
                },
                {
                    "url": "https://example.com/missing-og-url",
                    "title": "Missing OG URL",
                    "canonical_url": "https://example.com/missing-og-url",
                    "og_url": "",
                    "issues": [],
                },
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "open_graph_url_not_matching_canonical"]
    assert {row["url"] for row in issues} == {"https://example.com/mismatch", "https://example.com/noindex-mismatch"}
    assert all(row["issue_name"] == "Open Graph URL not matching canonical" for row in issues)
    assert all(row["category"] == "social_tags" for row in issues)
    assert all(row["importance"] == "Warning" for row in issues)


def test_technical_seo_model_flags_twitter_card_incomplete() -> None:
    payload = build_technical_seo(
        [
            SimpleNamespace(url="https://example.com/incomplete", title="Incomplete", section="", word_count=250, language="en"),
            SimpleNamespace(url="https://example.com/complete", title="Complete", section="", word_count=250, language="en"),
            SimpleNamespace(url="https://example.com/missing", title="Missing", section="", word_count=250, language="en"),
        ],
        indexability={
            "skipped": [
                {
                    "url": "https://example.com/noindex-incomplete",
                    "title": "Noindex Incomplete",
                    "reason": "noindex",
                    "http_status": 200,
                    "nofollow": False,
                }
            ],
            "noindex_pages": [],
        },
        metadata_quality={
            "per_page": [
                {
                    "url": "https://example.com/incomplete",
                    "title": "Incomplete",
                    "twitter_card": "summary",
                    "twitter_title": "",
                    "twitter_description": "Preview description",
                    "twitter_tag_count": 2,
                    "twitter_missing_fields": ["twitter_title"],
                    "issues": ["incomplete_twitter_card"],
                },
                {
                    "url": "https://example.com/noindex-incomplete",
                    "title": "Noindex Incomplete",
                    "twitter_card": "summary",
                    "twitter_title": "Noindex title",
                    "twitter_description": "",
                    "twitter_tag_count": 2,
                    "twitter_missing_fields": ["twitter_description"],
                    "issues": ["incomplete_twitter_card"],
                },
                {
                    "url": "https://example.com/complete",
                    "title": "Complete",
                    "twitter_card": "summary",
                    "twitter_title": "Complete title",
                    "twitter_description": "Complete description",
                    "twitter_tag_count": 3,
                    "twitter_missing_fields": [],
                    "issues": [],
                },
                {
                    "url": "https://example.com/missing",
                    "title": "Missing",
                    "twitter_tag_count": 0,
                    "twitter_missing_fields": ["twitter_card", "twitter_title", "twitter_description"],
                    "issues": ["missing_twitter_card"],
                },
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "twitter_card_incomplete"]
    assert {row["url"] for row in issues} == {"https://example.com/incomplete", "https://example.com/noindex-incomplete"}
    assert all(row["issue_name"] == "X (Twitter) card incomplete" for row in issues)
    assert all(row["category"] == "social_tags" for row in issues)
    assert all(row["importance"] == "Warning" for row in issues)
    page = next(row for row in payload["pages"] if row["url"] == "https://example.com/incomplete")
    assert page["twitter_missing_fields"] == ["twitter_title"]


def test_technical_seo_model_flags_open_graph_tags_missing() -> None:
    payload = build_technical_seo(
        [
            SimpleNamespace(url="https://example.com/missing", title="Missing", section="", word_count=250, language="en"),
            SimpleNamespace(url="https://example.com/incomplete", title="Incomplete", section="", word_count=250, language="en"),
            SimpleNamespace(url="https://example.com/complete", title="Complete", section="", word_count=250, language="en"),
        ],
        indexability={
            "skipped": [
                {
                    "url": "https://example.com/noindex-missing",
                    "title": "Noindex Missing",
                    "reason": "noindex",
                    "http_status": 200,
                    "nofollow": False,
                }
            ],
            "noindex_pages": [],
        },
        metadata_quality={
            "per_page": [
                {
                    "url": "https://example.com/missing",
                    "title": "Missing",
                    "og_tag_count": 0,
                    "og_missing_fields": ["og_title", "og_description"],
                    "issues": ["missing_open_graph"],
                },
                {
                    "url": "https://example.com/noindex-missing",
                    "title": "Noindex Missing",
                    "og_tag_count": 0,
                    "og_missing_fields": ["og_title", "og_description"],
                    "issues": ["missing_open_graph"],
                },
                {
                    "url": "https://example.com/incomplete",
                    "title": "Incomplete",
                    "og_title": "Incomplete OG",
                    "og_tag_count": 1,
                    "og_missing_fields": ["og_description"],
                    "issues": ["incomplete_open_graph"],
                },
                {
                    "url": "https://example.com/complete",
                    "title": "Complete",
                    "og_title": "Complete OG",
                    "og_description": "Complete OG description",
                    "og_tag_count": 2,
                    "og_missing_fields": [],
                    "issues": [],
                },
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "open_graph_tags_missing"]
    assert {row["url"] for row in issues} == {"https://example.com/missing", "https://example.com/noindex-missing"}
    assert all(row["issue_name"] == "Open Graph tags missing" for row in issues)
    assert all(row["category"] == "social_tags" for row in issues)
    assert all(row["importance"] == "Notice" for row in issues)


def test_technical_seo_model_flags_twitter_card_missing() -> None:
    payload = build_technical_seo(
        [
            SimpleNamespace(url="https://example.com/missing", title="Missing", section="", word_count=250, language="en"),
            SimpleNamespace(url="https://example.com/incomplete", title="Incomplete", section="", word_count=250, language="en"),
            SimpleNamespace(url="https://example.com/complete", title="Complete", section="", word_count=250, language="en"),
        ],
        indexability={
            "skipped": [
                {
                    "url": "https://example.com/noindex-missing",
                    "title": "Noindex Missing",
                    "reason": "noindex",
                    "http_status": 200,
                    "nofollow": False,
                }
            ],
            "noindex_pages": [],
        },
        metadata_quality={
            "per_page": [
                {
                    "url": "https://example.com/missing",
                    "title": "Missing",
                    "twitter_tag_count": 0,
                    "twitter_missing_fields": ["twitter_card", "twitter_title", "twitter_description"],
                    "issues": ["missing_twitter_card"],
                },
                {
                    "url": "https://example.com/noindex-missing",
                    "title": "Noindex Missing",
                    "twitter_tag_count": 0,
                    "twitter_missing_fields": ["twitter_card", "twitter_title", "twitter_description"],
                    "issues": ["missing_twitter_card"],
                },
                {
                    "url": "https://example.com/incomplete",
                    "title": "Incomplete",
                    "twitter_card": "summary",
                    "twitter_tag_count": 1,
                    "twitter_missing_fields": ["twitter_title", "twitter_description"],
                    "issues": ["incomplete_twitter_card"],
                },
                {
                    "url": "https://example.com/complete",
                    "title": "Complete",
                    "twitter_card": "summary",
                    "twitter_title": "Complete title",
                    "twitter_description": "Complete description",
                    "twitter_tag_count": 3,
                    "twitter_missing_fields": [],
                    "issues": [],
                },
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "twitter_card_missing"]
    assert {row["url"] for row in issues} == {"https://example.com/missing", "https://example.com/noindex-missing"}
    assert all(row["issue_name"] == "X (Twitter) card missing" for row in issues)
    assert all(row["category"] == "social_tags" for row in issues)
    assert all(row["importance"] == "Notice" for row in issues)


def test_technical_seo_model_flags_duplicate_pages_without_canonical() -> None:
    pages = [
        SimpleNamespace(url="https://example.com/a", title="A", section="", word_count=250, language="en"),
        SimpleNamespace(url="https://example.com/b", title="B", section="", word_count=250, language="en"),
        SimpleNamespace(url="https://example.com/c", title="C", section="", word_count=250, language="en"),
        SimpleNamespace(url="https://example.com/d", title="D", section="", word_count=250, language="en"),
        SimpleNamespace(url="https://example.com/clean", title="Clean", section="", word_count=250, language="en"),
    ]
    payload = build_technical_seo(
        pages,
        metadata_quality={
            "per_page": [
                {"url": "https://example.com/a", "title": "A", "canonical_url": "https://example.com/a", "issues": []},
                {"url": "https://example.com/b", "title": "B", "canonical_url": "", "issues": []},
                {"url": "https://example.com/c", "title": "C", "canonical_url": "https://example.com/d", "issues": []},
                {"url": "https://example.com/d", "title": "D", "canonical_url": "https://example.com/d", "issues": []},
                {"url": "https://example.com/clean", "title": "Clean", "canonical_url": "https://example.com/clean", "issues": []},
            ]
        },
        duplicate_rows=[
            {
                "url_a": "https://example.com/a",
                "title_a": "A",
                "url_b": "https://example.com/b",
                "title_b": "B",
                "similarity": 0.98,
            },
            {
                "url_a": "https://example.com/c",
                "title_a": "C",
                "url_b": "https://example.com/d",
                "title_b": "D",
                "similarity": 0.99,
            },
        ],
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "duplicate_pages_without_canonical"]
    assert {row["url"] for row in issues} == {"https://example.com/a", "https://example.com/b"}
    assert all(row["issue_name"] == "Duplicate pages without canonical" for row in issues)
    assert all(row["category"] == "duplicates" for row in issues)
    assert all(row["importance"] == "Error" for row in issues)
    flagged_page = next(row for row in payload["pages"] if row["url"] == "https://example.com/a")
    consolidated_page = next(row for row in payload["pages"] if row["url"] == "https://example.com/c")
    assert flagged_page["duplicate_partner_urls"] == ["https://example.com/b"]
    assert flagged_page["duplicate_without_canonical"] is True
    assert consolidated_page["duplicate_without_canonical"] is False


def test_technical_seo_model_flags_hreflang_and_html_lang_mismatch() -> None:
    payload = build_technical_seo(
        [
            SimpleNamespace(url="https://example.com/sk", title="SK", section="", word_count=250, language="sk"),
            SimpleNamespace(url="https://example.com/compatible", title="Compatible", section="", word_count=250, language="sk-SK"),
            SimpleNamespace(url="https://example.com/no-self", title="No Self", section="", word_count=250, language="sk"),
        ],
        indexability={
            "skipped": [
                {
                    "url": "https://example.com/noindex",
                    "title": "Noindex",
                    "reason": "noindex",
                    "http_status": 200,
                    "nofollow": False,
                }
            ],
            "noindex_pages": [],
        },
        metadata_quality={
            "per_page": [
                {
                    "url": "https://example.com/sk",
                    "title": "SK",
                    "html_lang": "sk",
                    "canonical_url": "https://example.com/sk",
                    "hreflang": [
                        {"hreflang": "cs", "href": "https://example.com/sk"},
                        {"hreflang": "x-default", "href": "https://example.com/"},
                    ],
                    "issues": [],
                },
                {
                    "url": "https://example.com/noindex",
                    "title": "Noindex",
                    "html_lang": "en",
                    "canonical_url": "https://example.com/noindex",
                    "hreflang": [{"hreflang": "de", "href": "https://example.com/noindex"}],
                    "issues": [],
                },
                {
                    "url": "https://example.com/compatible",
                    "title": "Compatible",
                    "html_lang": "sk-SK",
                    "canonical_url": "https://example.com/compatible",
                    "hreflang": [{"hreflang": "sk", "href": "https://example.com/compatible/"}],
                    "issues": [],
                },
                {
                    "url": "https://example.com/no-self",
                    "title": "No Self",
                    "html_lang": "sk",
                    "canonical_url": "https://example.com/no-self",
                    "hreflang": [{"hreflang": "cs", "href": "https://example.com/cs"}],
                    "issues": [],
                },
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "hreflang_and_html_lang_mismatch"]
    assert {row["url"] for row in issues} == {"https://example.com/sk", "https://example.com/noindex"}
    assert all(row["issue_name"] == "Hreflang and HTML lang mismatch" for row in issues)
    assert all(row["category"] == "localization" for row in issues)
    assert all(row["importance"] == "Error" for row in issues)


def test_technical_seo_model_flags_invalid_hreflang_annotations() -> None:
    payload = build_technical_seo(
        [
            SimpleNamespace(url="https://example.com/invalid-code", title="Invalid Code", section="", word_count=250, language="en"),
            SimpleNamespace(url="https://example.com/invalid-href", title="Invalid Href", section="", word_count=250, language="en"),
            SimpleNamespace(url="https://example.com/ok", title="OK", section="", word_count=250, language="en"),
        ],
        indexability={
            "skipped": [
                {
                    "url": "https://example.com/noindex-invalid",
                    "title": "Noindex Invalid",
                    "reason": "noindex",
                    "http_status": 200,
                    "nofollow": False,
                }
            ],
            "noindex_pages": [],
        },
        metadata_quality={
            "per_page": [
                {
                    "url": "https://example.com/invalid-code",
                    "title": "Invalid Code",
                    "hreflang": [{"hreflang": "english", "href": "https://example.com/invalid-code"}],
                    "issues": [],
                },
                {
                    "url": "https://example.com/invalid-href",
                    "title": "Invalid Href",
                    "hreflang": [{"hreflang": "en-US", "href": "/relative"}],
                    "issues": [],
                },
                {
                    "url": "https://example.com/noindex-invalid",
                    "title": "Noindex Invalid",
                    "hreflang": [{"hreflang": "", "href": ""}],
                    "issues": [],
                },
                {
                    "url": "https://example.com/ok",
                    "title": "OK",
                    "hreflang": [
                        {"hreflang": "en-US", "href": "https://example.com/ok"},
                        {"hreflang": "x-default", "href": "https://example.com/"},
                    ],
                    "issues": [],
                },
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "hreflang_annotation_invalid"]
    assert {row["url"] for row in issues} == {
        "https://example.com/invalid-code",
        "https://example.com/invalid-href",
        "https://example.com/noindex-invalid",
    }
    assert all(row["issue_name"] == "Hreflang annotation invalid" for row in issues)
    assert all(row["category"] == "localization" for row in issues)
    assert all(row["importance"] == "Error" for row in issues)
    page = next(row for row in payload["pages"] if row["url"] == "https://example.com/invalid-href")
    assert page["invalid_hreflang_annotations"] == [{"hreflang": "en-US", "href": "/relative", "reasons": ["invalid_href"]}]


def test_technical_seo_model_flags_invalid_html_lang_attribute() -> None:
    pages = [
        SimpleNamespace(url="https://example.com/word", title="Word", section="", word_count=250, language="en"),
        SimpleNamespace(url="https://example.com/x-default", title="X Default", section="", word_count=250, language="en"),
        SimpleNamespace(url="https://example.com/underscore", title="Underscore", section="", word_count=250, language="en"),
        SimpleNamespace(url="https://example.com/missing", title="Missing", section="", word_count=250, language="en"),
        SimpleNamespace(url="https://example.com/ok", title="OK", section="", word_count=250, language="sk"),
    ]
    payload = build_technical_seo(
        pages,
        metadata_quality={
            "per_page": [
                {
                    "url": "https://example.com/word",
                    "title": "Word",
                    "html_lang": "english",
                    "hreflang": [{"hreflang": "en", "href": "https://example.com/word"}],
                    "issues": [],
                },
                {
                    "url": "https://example.com/x-default",
                    "title": "X Default",
                    "html_lang": "x-default",
                    "hreflang": [],
                    "issues": [],
                },
                {
                    "url": "https://example.com/underscore",
                    "title": "Underscore",
                    "html_lang": "en_US",
                    "hreflang": [],
                    "issues": [],
                },
                {
                    "url": "https://example.com/missing",
                    "title": "Missing",
                    "html_lang": "",
                    "hreflang": [],
                    "issues": [],
                },
                {
                    "url": "https://example.com/ok",
                    "title": "OK",
                    "html_lang": "sk-SK",
                    "hreflang": [{"hreflang": "sk", "href": "https://example.com/ok"}],
                    "issues": [],
                },
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "html_lang_attribute_invalid"]
    assert {row["url"] for row in issues} == {
        "https://example.com/word",
        "https://example.com/x-default",
        "https://example.com/underscore",
    }
    assert all(row["issue_name"] == "HTML lang attribute invalid" for row in issues)
    assert all(row["category"] == "localization" for row in issues)
    assert all(row["importance"] == "Error" for row in issues)
    word = next(row for row in payload["pages"] if row["url"] == "https://example.com/word")
    assert word["invalid_html_lang"] == "english"
    mismatch_urls = {row["url"] for row in payload["issues"] if row["issue_type"] == "hreflang_and_html_lang_mismatch"}
    assert "https://example.com/word" not in mismatch_urls


def test_technical_seo_model_flags_hreflang_defined_but_html_lang_missing() -> None:
    pages = [
        SimpleNamespace(url="https://example.com/missing-with-hreflang", title="Missing With Hreflang", section="", word_count=250, language="en"),
        SimpleNamespace(url="https://example.com/missing-without-hreflang", title="Missing Without Hreflang", section="", word_count=250, language="en"),
        SimpleNamespace(url="https://example.com/ok", title="OK", section="", word_count=250, language="sk"),
    ]
    payload = build_technical_seo(
        pages,
        metadata_quality={
            "per_page": [
                {
                    "url": "https://example.com/missing-with-hreflang",
                    "title": "Missing With Hreflang",
                    "html_lang": "",
                    "hreflang": [{"hreflang": "en", "href": "https://example.com/missing-with-hreflang"}],
                    "issues": [],
                },
                {
                    "url": "https://example.com/missing-without-hreflang",
                    "title": "Missing Without Hreflang",
                    "html_lang": "",
                    "hreflang": [],
                    "issues": [],
                },
                {
                    "url": "https://example.com/ok",
                    "title": "OK",
                    "html_lang": "sk-SK",
                    "hreflang": [{"hreflang": "sk", "href": "https://example.com/ok"}],
                    "issues": [],
                },
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "hreflang_defined_but_html_lang_missing"]
    assert len(issues) == 1
    assert issues[0]["url"] == "https://example.com/missing-with-hreflang"
    assert issues[0]["issue_name"] == "Hreflang defined but HTML lang missing"
    assert issues[0]["category"] == "localization"
    assert issues[0]["importance"] == "Warning"
    assert issues[0]["severity"] == "medium"
    page = next(row for row in payload["pages"] if row["url"] == "https://example.com/missing-with-hreflang")
    assert page["html_lang_missing"] is True


def test_technical_seo_model_flags_html_lang_attribute_missing() -> None:
    pages = [
        SimpleNamespace(url="https://example.com/missing-with-hreflang", title="Missing With Hreflang", section="", word_count=250, language="en"),
        SimpleNamespace(url="https://example.com/missing-without-hreflang", title="Missing Without Hreflang", section="", word_count=250, language="en"),
        SimpleNamespace(url="https://example.com/legacy", title="Legacy", section="", word_count=250, language="en"),
        SimpleNamespace(url="https://example.com/ok", title="OK", section="", word_count=250, language="sk"),
    ]
    payload = build_technical_seo(
        pages,
        metadata_quality={
            "per_page": [
                {
                    "url": "https://example.com/missing-with-hreflang",
                    "title": "Missing With Hreflang",
                    "html_lang": "",
                    "hreflang": [{"hreflang": "en", "href": "https://example.com/missing-with-hreflang"}],
                    "issues": [],
                },
                {
                    "url": "https://example.com/missing-without-hreflang",
                    "title": "Missing Without Hreflang",
                    "html_lang": "",
                    "hreflang": [],
                    "issues": [],
                },
                {
                    "url": "https://example.com/legacy",
                    "title": "Legacy",
                    "hreflang": [],
                    "issues": [],
                },
                {
                    "url": "https://example.com/ok",
                    "title": "OK",
                    "html_lang": "sk",
                    "hreflang": [],
                    "issues": [],
                },
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "html_lang_attribute_missing"]
    assert {row["url"] for row in issues} == {
        "https://example.com/missing-with-hreflang",
        "https://example.com/missing-without-hreflang",
    }
    assert all(row["issue_name"] == "HTML lang attribute missing" for row in issues)
    assert all(row["category"] == "localization" for row in issues)
    assert all(row["importance"] == "Warning" for row in issues)
    assert all(row["severity"] == "medium" for row in issues)


def test_technical_seo_model_flags_hreflang_to_non_canonical() -> None:
    pages = [
        SimpleNamespace(url="https://example.com/source", title="Source", section="", word_count=250, language="en"),
        SimpleNamespace(url="https://example.com/non-canonical", title="Non Canonical", section="", word_count=250, language="sk"),
        SimpleNamespace(url="https://example.com/canonical", title="Canonical", section="", word_count=250, language="sk"),
        SimpleNamespace(url="https://example.com/ok-alt", title="OK Alt", section="", word_count=250, language="cs"),
        SimpleNamespace(url="https://example.com/clean-source", title="Clean Source", section="", word_count=250, language="en"),
    ]
    payload = build_technical_seo(
        pages,
        metadata_quality={
            "per_page": [
                {
                    "url": "https://example.com/source",
                    "title": "Source",
                    "canonical_url": "https://example.com/source",
                    "hreflang": [
                        {"hreflang": "sk", "href": "https://example.com/non-canonical"},
                        {"hreflang": "de", "href": "https://example.com/not-crawled"},
                    ],
                    "issues": [],
                },
                {
                    "url": "https://example.com/non-canonical",
                    "title": "Non Canonical",
                    "canonical_url": "https://example.com/canonical",
                    "hreflang": [],
                    "issues": [],
                },
                {
                    "url": "https://example.com/canonical",
                    "title": "Canonical",
                    "canonical_url": "https://example.com/canonical",
                    "hreflang": [],
                    "issues": [],
                },
                {
                    "url": "https://example.com/ok-alt",
                    "title": "OK Alt",
                    "canonical_url": "https://example.com/ok-alt",
                    "hreflang": [],
                    "issues": [],
                },
                {
                    "url": "https://example.com/clean-source",
                    "title": "Clean Source",
                    "canonical_url": "https://example.com/clean-source",
                    "hreflang": [{"hreflang": "cs", "href": "https://example.com/ok-alt"}],
                    "issues": [],
                },
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "hreflang_to_non_canonical"]
    assert len(issues) == 1
    assert issues[0]["url"] == "https://example.com/source"
    assert issues[0]["issue_name"] == "Hreflang to non-canonical"
    assert issues[0]["category"] == "localization"
    assert issues[0]["importance"] == "Error"
    source = next(row for row in payload["pages"] if row["url"] == "https://example.com/source")
    assert source["hreflang_non_canonical_targets"] == [
        {
            "hreflang": "sk",
            "href": "https://example.com/non-canonical",
            "target_canonical_url": "https://example.com/canonical",
        }
    ]


def test_technical_seo_model_flags_hreflang_to_redirect_or_broken_page() -> None:
    pages = [
        SimpleNamespace(url="https://example.com/source", title="Source", section="", word_count=250, language="en"),
        SimpleNamespace(url="https://example.com/broken", title="Broken", section="", word_count=250, language="sk"),
        SimpleNamespace(url="https://example.com/redirecting", title="Redirecting", section="", word_count=250, language="cs"),
        SimpleNamespace(url="https://example.com/final", title="Final", section="", word_count=250, language="cs"),
        SimpleNamespace(url="https://example.com/ok-alt", title="OK Alt", section="", word_count=250, language="de"),
        SimpleNamespace(url="https://example.com/clean-source", title="Clean Source", section="", word_count=250, language="en"),
    ]
    payload = build_technical_seo(
        pages,
        metadata_quality={
            "per_page": [
                {
                    "url": "https://example.com/source",
                    "title": "Source",
                    "canonical_url": "https://example.com/source",
                    "hreflang": [
                        {"hreflang": "sk", "href": "https://example.com/broken"},
                        {"hreflang": "cs", "href": "https://example.com/redirecting"},
                        {"hreflang": "de", "href": "https://example.com/ok-alt"},
                        {"hreflang": "pl", "href": "https://example.com/not-crawled"},
                    ],
                    "issues": [],
                },
                {
                    "url": "https://example.com/broken",
                    "title": "Broken",
                    "canonical_url": "https://example.com/broken",
                    "hreflang": [],
                    "issues": [],
                },
                {
                    "url": "https://example.com/redirecting",
                    "title": "Redirecting",
                    "canonical_url": "https://example.com/redirecting",
                    "hreflang": [],
                    "issues": [],
                },
                {
                    "url": "https://example.com/final",
                    "title": "Final",
                    "canonical_url": "https://example.com/final",
                    "hreflang": [],
                    "issues": [],
                },
                {
                    "url": "https://example.com/ok-alt",
                    "title": "OK Alt",
                    "canonical_url": "https://example.com/ok-alt",
                    "hreflang": [],
                    "issues": [],
                },
                {
                    "url": "https://example.com/clean-source",
                    "title": "Clean Source",
                    "canonical_url": "https://example.com/clean-source",
                    "hreflang": [{"hreflang": "de", "href": "https://example.com/ok-alt"}],
                    "issues": [],
                },
            ]
        },
        performance={
            "per_page": [
                {"url": "https://example.com/broken", "status": 404},
                {"url": "https://example.com/redirecting", "status": 200},
                {"url": "https://example.com/ok-alt", "status": 200},
            ]
        },
        indexability={
            "per_page": [
                {
                    "url": "https://example.com/redirecting",
                    "requested_url": "https://example.com/redirecting",
                    "redirect_target_url": "https://example.com/final",
                    "redirect_chain": ["https://example.com/redirecting", "https://example.com/final"],
                    "redirect_hop_count": 1,
                    "redirect_status_codes": [301],
                }
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "hreflang_to_redirect_or_broken_page"]
    assert len(issues) == 1
    assert issues[0]["url"] == "https://example.com/source"
    assert issues[0]["issue_name"] == "Hreflang to redirect or broken page"
    assert issues[0]["category"] == "localization"
    assert issues[0]["importance"] == "Error"
    source = next(row for row in payload["pages"] if row["url"] == "https://example.com/source")
    assert source["hreflang_redirect_or_broken_targets"] == [
        {
            "hreflang": "sk",
            "href": "https://example.com/broken",
            "http_status": 404,
            "redirect_target_url": "",
            "issue": "broken",
        },
        {
            "hreflang": "cs",
            "href": "https://example.com/redirecting",
            "http_status": 200,
            "redirect_target_url": "https://example.com/final",
            "issue": "redirect",
        },
    ]
    clean = next(row for row in payload["pages"] if row["url"] == "https://example.com/clean-source")
    assert "hreflang_redirect_or_broken_targets" not in clean


def test_technical_seo_model_flags_missing_reciprocal_hreflang() -> None:
    pages = [
        SimpleNamespace(url="https://example.com/source", title="Source", section="", word_count=250, language="en"),
        SimpleNamespace(url="https://example.com/missing-return", title="Missing Return", section="", word_count=250, language="sk"),
        SimpleNamespace(url="https://example.com/return-url", title="Return URL", section="", word_count=250, language="cs"),
        SimpleNamespace(url="https://example.com/return-canonical", title="Return Canonical", section="", word_count=250, language="de"),
        SimpleNamespace(url="https://example.com/clean-source", title="Clean Source", section="", word_count=250, language="en"),
        SimpleNamespace(url="https://example.com/clean-target", title="Clean Target", section="", word_count=250, language="pl"),
    ]
    payload = build_technical_seo(
        pages,
        metadata_quality={
            "per_page": [
                {
                    "url": "https://example.com/source",
                    "title": "Source",
                    "canonical_url": "https://example.com/source/",
                    "hreflang": [
                        {"hreflang": "en", "href": "https://example.com/source"},
                        {"hreflang": "sk", "href": "https://example.com/missing-return"},
                        {"hreflang": "cs", "href": "https://example.com/return-url"},
                        {"hreflang": "de", "href": "https://example.com/return-canonical"},
                        {"hreflang": "hu", "href": "https://example.com/not-crawled"},
                    ],
                    "issues": [],
                },
                {
                    "url": "https://example.com/missing-return",
                    "title": "Missing Return",
                    "canonical_url": "https://example.com/missing-return",
                    "hreflang": [{"hreflang": "sk", "href": "https://example.com/missing-return"}],
                    "issues": [],
                },
                {
                    "url": "https://example.com/return-url",
                    "title": "Return URL",
                    "canonical_url": "https://example.com/return-url",
                    "hreflang": [{"hreflang": "en", "href": "https://example.com/source"}],
                    "issues": [],
                },
                {
                    "url": "https://example.com/return-canonical",
                    "title": "Return Canonical",
                    "canonical_url": "https://example.com/return-canonical",
                    "hreflang": [{"hreflang": "en", "href": "https://example.com/source/"}],
                    "issues": [],
                },
                {
                    "url": "https://example.com/clean-source",
                    "title": "Clean Source",
                    "canonical_url": "https://example.com/clean-source",
                    "hreflang": [{"hreflang": "pl", "href": "https://example.com/clean-target"}],
                    "issues": [],
                },
                {
                    "url": "https://example.com/clean-target",
                    "title": "Clean Target",
                    "canonical_url": "https://example.com/clean-target",
                    "hreflang": [{"hreflang": "en", "href": "https://example.com/clean-source"}],
                    "issues": [],
                },
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "missing_reciprocal_hreflang_no_return_tag"]
    assert len(issues) == 1
    assert issues[0]["url"] == "https://example.com/source"
    assert issues[0]["issue_name"] == "Missing reciprocal hreflang (no return-tag)"
    assert issues[0]["category"] == "localization"
    assert issues[0]["importance"] == "Error"
    source = next(row for row in payload["pages"] if row["url"] == "https://example.com/source")
    assert source["missing_reciprocal_hreflang_targets"] == [
        {
            "hreflang": "sk",
            "href": "https://example.com/missing-return",
            "target_url": "https://example.com/missing-return",
        }
    ]
    clean = next(row for row in payload["pages"] if row["url"] == "https://example.com/clean-source")
    assert "missing_reciprocal_hreflang_targets" not in clean


def test_technical_seo_model_flags_more_than_one_page_for_same_language_in_hreflang() -> None:
    pages = [
        SimpleNamespace(url="https://example.com/duplicate", title="Duplicate", section="", word_count=250, language="en"),
        SimpleNamespace(url="https://example.com/clean", title="Clean", section="", word_count=250, language="en"),
    ]
    payload = build_technical_seo(
        pages,
        metadata_quality={
            "per_page": [
                {
                    "url": "https://example.com/duplicate",
                    "title": "Duplicate",
                    "hreflang": [
                        {"hreflang": "sk", "href": "https://example.com/sk-one"},
                        {"hreflang": "sk", "href": "https://example.com/sk-two"},
                        {"hreflang": "de", "href": "https://example.com/de"},
                        {"hreflang": "de", "href": "https://example.com/de/"},
                        {"hreflang": "en-US", "href": "https://example.com/us"},
                        {"hreflang": "en-GB", "href": "https://example.com/uk"},
                    ],
                    "issues": [],
                },
                {
                    "url": "https://example.com/clean",
                    "title": "Clean",
                    "hreflang": [
                        {"hreflang": "sk", "href": "https://example.com/sk"},
                        {"hreflang": "cs", "href": "https://example.com/cs"},
                    ],
                    "issues": [],
                },
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "more_than_one_page_for_same_language_in_hreflang"]
    assert len(issues) == 1
    assert issues[0]["url"] == "https://example.com/duplicate"
    assert issues[0]["issue_name"] == "More than one page for same language in hreflang"
    assert issues[0]["category"] == "localization"
    assert issues[0]["importance"] == "Error"
    duplicate = next(row for row in payload["pages"] if row["url"] == "https://example.com/duplicate")
    assert duplicate["duplicate_hreflang_language_targets"] == [
        {
            "hreflang": "sk",
            "hrefs": ["https://example.com/sk-one", "https://example.com/sk-two"],
        }
    ]


def test_technical_seo_model_flags_page_referenced_for_more_than_one_language_in_hreflang() -> None:
    pages = [
        SimpleNamespace(url="https://example.com/source-a", title="Source A", section="", word_count=250, language="en"),
        SimpleNamespace(url="https://example.com/source-b", title="Source B", section="", word_count=250, language="en"),
        SimpleNamespace(url="https://example.com/target", title="Target", section="", word_count=250, language="sk"),
        SimpleNamespace(url="https://example.com/clean-target", title="Clean Target", section="", word_count=250, language="pl"),
    ]
    payload = build_technical_seo(
        pages,
        metadata_quality={
            "per_page": [
                {
                    "url": "https://example.com/source-a",
                    "title": "Source A",
                    "hreflang": [
                        {"hreflang": "sk", "href": "https://example.com/target"},
                        {"hreflang": "pl", "href": "https://example.com/clean-target"},
                        {"hreflang": "x-default", "href": "https://example.com/target"},
                    ],
                    "issues": [],
                },
                {
                    "url": "https://example.com/source-b",
                    "title": "Source B",
                    "hreflang": [
                        {"hreflang": "cs", "href": "https://example.com/target/"},
                        {"hreflang": "pl", "href": "https://example.com/clean-target"},
                    ],
                    "issues": [],
                },
                {
                    "url": "https://example.com/target",
                    "title": "Target",
                    "hreflang": [],
                    "issues": [],
                },
                {
                    "url": "https://example.com/clean-target",
                    "title": "Clean Target",
                    "hreflang": [],
                    "issues": [],
                },
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "page_referenced_for_more_than_one_language_in_hreflang"]
    assert len(issues) == 1
    assert issues[0]["url"] == "https://example.com/target"
    assert issues[0]["issue_name"] == "Page referenced for more than one language in hreflang"
    assert issues[0]["category"] == "localization"
    assert issues[0]["importance"] == "Error"
    target = next(row for row in payload["pages"] if row["url"] == "https://example.com/target")
    assert target["hreflang_multi_language_references"] == [
        {
            "hreflang": "cs",
            "source_urls": ["https://example.com/source-b"],
            "hrefs": ["https://example.com/target/"],
        },
        {
            "hreflang": "sk",
            "source_urls": ["https://example.com/source-a"],
            "hrefs": ["https://example.com/target"],
        },
    ]
    clean = next(row for row in payload["pages"] if row["url"] == "https://example.com/clean-target")
    assert "hreflang_multi_language_references" not in clean


def test_technical_seo_model_flags_self_reference_hreflang_annotation_missing() -> None:
    pages = [
        SimpleNamespace(url="https://example.com/missing", title="Missing", section="", word_count=250, language="en"),
        SimpleNamespace(url="https://example.com/self-url", title="Self URL", section="", word_count=250, language="sk"),
        SimpleNamespace(url="https://example.com/self-canonical", title="Self Canonical", section="", word_count=250, language="cs"),
        SimpleNamespace(url="https://example.com/no-hreflang", title="No Hreflang", section="", word_count=250, language="de"),
    ]
    payload = build_technical_seo(
        pages,
        metadata_quality={
            "per_page": [
                {
                    "url": "https://example.com/missing",
                    "title": "Missing",
                    "canonical_url": "https://example.com/missing",
                    "hreflang": [
                        {"hreflang": "x-default", "href": "https://example.com/missing"},
                        {"hreflang": "sk", "href": "https://example.com/sk"},
                    ],
                    "issues": [],
                },
                {
                    "url": "https://example.com/self-url",
                    "title": "Self URL",
                    "canonical_url": "https://example.com/self-url",
                    "hreflang": [{"hreflang": "sk", "href": "https://example.com/self-url"}],
                    "issues": [],
                },
                {
                    "url": "https://example.com/self-canonical",
                    "title": "Self Canonical",
                    "canonical_url": "https://example.com/self-canonical/",
                    "hreflang": [{"hreflang": "cs", "href": "https://example.com/self-canonical/"}],
                    "issues": [],
                },
                {
                    "url": "https://example.com/no-hreflang",
                    "title": "No Hreflang",
                    "canonical_url": "https://example.com/no-hreflang",
                    "hreflang": [],
                    "issues": [],
                },
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "self_reference_hreflang_annotation_missing"]
    assert len(issues) == 1
    assert issues[0]["url"] == "https://example.com/missing"
    assert issues[0]["issue_name"] == "Self-reference hreflang annotation missing"
    assert issues[0]["category"] == "localization"
    assert issues[0]["importance"] == "Warning"
    assert issues[0]["severity"] == "medium"
    page = next(row for row in payload["pages"] if row["url"] == "https://example.com/missing")
    assert page["self_reference_hreflang_missing"] is True


def test_technical_seo_model_flags_not_all_pages_from_hreflang_group_were_crawled() -> None:
    pages = [
        SimpleNamespace(url="https://example.com/source", title="Source", section="", word_count=250, language="en"),
        SimpleNamespace(url="https://example.com/crawled-alt", title="Crawled Alt", section="", word_count=250, language="sk"),
        SimpleNamespace(url="https://example.com/clean-source", title="Clean Source", section="", word_count=250, language="en"),
        SimpleNamespace(url="https://example.com/clean-alt", title="Clean Alt", section="", word_count=250, language="cs"),
    ]
    payload = build_technical_seo(
        pages,
        metadata_quality={
            "per_page": [
                {
                    "url": "https://example.com/source",
                    "title": "Source",
                    "canonical_url": "https://example.com/source",
                    "hreflang": [
                        {"hreflang": "en", "href": "https://example.com/source"},
                        {"hreflang": "sk", "href": "https://example.com/crawled-alt"},
                        {"hreflang": "cs", "href": "https://example.com/not-crawled"},
                        {"hreflang": "x-default", "href": "https://example.com/default-not-crawled"},
                        {"hreflang": "english", "href": "https://example.com/invalid-code"},
                        {"hreflang": "de", "href": "/relative"},
                    ],
                    "issues": [],
                },
                {
                    "url": "https://example.com/crawled-alt",
                    "title": "Crawled Alt",
                    "canonical_url": "https://example.com/crawled-alt",
                    "hreflang": [{"hreflang": "en", "href": "https://example.com/source"}],
                    "issues": [],
                },
                {
                    "url": "https://example.com/clean-source",
                    "title": "Clean Source",
                    "canonical_url": "https://example.com/clean-source",
                    "hreflang": [
                        {"hreflang": "en", "href": "https://example.com/clean-source"},
                        {"hreflang": "cs", "href": "https://example.com/clean-alt"},
                    ],
                    "issues": [],
                },
                {
                    "url": "https://example.com/clean-alt",
                    "title": "Clean Alt",
                    "canonical_url": "https://example.com/clean-alt",
                    "hreflang": [{"hreflang": "en", "href": "https://example.com/clean-source"}],
                    "issues": [],
                },
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "not_all_pages_from_hreflang_group_were_crawled"]
    assert len(issues) == 1
    assert issues[0]["url"] == "https://example.com/source"
    assert issues[0]["issue_name"] == "Not all pages from hreflang group were crawled"
    assert issues[0]["category"] == "localization"
    assert issues[0]["importance"] == "Notice"
    assert issues[0]["severity"] == "low"
    source = next(row for row in payload["pages"] if row["url"] == "https://example.com/source")
    assert source["uncrawled_hreflang_targets"] == [
        {"hreflang": "cs", "href": "https://example.com/not-crawled"},
        {"hreflang": "x-default", "href": "https://example.com/default-not-crawled"},
    ]
    clean = next(row for row in payload["pages"] if row["url"] == "https://example.com/clean-source")
    assert "uncrawled_hreflang_targets" not in clean


def test_technical_seo_model_flags_x_default_hreflang_annotation_missing() -> None:
    pages = [
        SimpleNamespace(url="https://example.com/missing", title="Missing", section="", word_count=250, language="en"),
        SimpleNamespace(url="https://example.com/ok", title="OK", section="", word_count=250, language="sk"),
        SimpleNamespace(url="https://example.com/no-hreflang", title="No Hreflang", section="", word_count=250, language="cs"),
    ]
    payload = build_technical_seo(
        pages,
        metadata_quality={
            "per_page": [
                {
                    "url": "https://example.com/missing",
                    "title": "Missing",
                    "hreflang": [
                        {"hreflang": "en", "href": "https://example.com/missing"},
                        {"hreflang": "sk", "href": "https://example.com/sk"},
                    ],
                    "issues": [],
                },
                {
                    "url": "https://example.com/ok",
                    "title": "OK",
                    "hreflang": [
                        {"hreflang": "sk", "href": "https://example.com/ok"},
                        {"hreflang": "x-default", "href": "https://example.com/"},
                    ],
                    "issues": [],
                },
                {
                    "url": "https://example.com/no-hreflang",
                    "title": "No Hreflang",
                    "hreflang": [],
                    "issues": [],
                },
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "x_default_hreflang_annotation_missing"]
    assert len(issues) == 1
    assert issues[0]["url"] == "https://example.com/missing"
    assert issues[0]["issue_name"] == "X-default hreflang annotation missing"
    assert issues[0]["category"] == "localization"
    assert issues[0]["importance"] == "Notice"
    assert issues[0]["severity"] == "low"
    page = next(row for row in payload["pages"] if row["url"] == "https://example.com/missing")
    assert page["x_default_hreflang_missing"] is True


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


def test_technical_seo_model_flags_content_is_not_sized_correctly() -> None:
    pages = [
        SimpleNamespace(url="https://example.com/bad", title="Bad", section="", word_count=100, language="en"),
        SimpleNamespace(url="https://example.com/clean", title="Clean", section="", word_count=100, language="en"),
    ]
    payload = build_technical_seo(
        pages,
        performance={
            "per_page": [
                {
                    "url": "https://example.com/bad",
                    "status": 200,
                    "content_sized_correctly": False,
                    "content_width_exceeds_viewport": True,
                    "max_fixed_width_px": 960,
                    "content_sizing_issues": [
                        {"source": "style_block", "tag": "style", "property": "min-width", "width_px": 960},
                    ],
                },
                {
                    "url": "https://example.com/clean",
                    "status": 200,
                    "content_sized_correctly": True,
                    "content_width_exceeds_viewport": False,
                    "max_fixed_width_px": 0,
                    "content_sizing_issues": [],
                },
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "content_is_not_sized_correctly"]
    assert len(issues) == 1
    assert issues[0]["url"] == "https://example.com/bad"
    assert issues[0]["issue_name"] == "Content is not sized correctly"
    assert issues[0]["category"] == "performance"
    assert issues[0]["importance"] == "Warning"
    assert issues[0]["severity"] == "medium"
    page = next(row for row in payload["pages"] if row["url"] == "https://example.com/bad")
    assert page["content_width_exceeds_viewport"] is True
    assert page["max_fixed_width_px"] == 960
    assert page["content_sizing_issues"] == [
        {"source": "style_block", "tag": "style", "property": "min-width", "width_px": 960},
    ]


def test_technical_seo_model_flags_document_uses_plugins() -> None:
    pages = [
        SimpleNamespace(url="https://example.com/bad", title="Bad", section="", word_count=100, language="en"),
        SimpleNamespace(url="https://example.com/clean", title="Clean", section="", word_count=100, language="en"),
    ]
    payload = build_technical_seo(
        pages,
        performance={
            "per_page": [
                {
                    "url": "https://example.com/bad",
                    "status": 200,
                    "plugin_element_count": 2,
                    "plugin_elements": [
                        {"tag": "object", "type": "application/x-shockwave-flash", "source": "/legacy.swf"},
                        {"tag": "embed", "type": "application/x-shockwave-flash", "source": "/movie.swf"},
                    ],
                },
                {
                    "url": "https://example.com/clean",
                    "status": 200,
                    "plugin_element_count": 0,
                    "plugin_elements": [],
                },
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "document_uses_plugins"]
    assert len(issues) == 1
    assert issues[0]["url"] == "https://example.com/bad"
    assert issues[0]["issue_name"] == "Document uses plugins"
    assert issues[0]["category"] == "performance"
    assert issues[0]["importance"] == "Warning"
    assert issues[0]["severity"] == "medium"
    page = next(row for row in payload["pages"] if row["url"] == "https://example.com/bad")
    assert page["plugin_element_count"] == 2
    assert page["plugin_elements"] == [
        {"tag": "object", "type": "application/x-shockwave-flash", "source": "/legacy.swf"},
        {"tag": "embed", "type": "application/x-shockwave-flash", "source": "/movie.swf"},
    ]


def test_technical_seo_model_flags_font_size_too_small() -> None:
    pages = [
        SimpleNamespace(url="https://example.com/bad", title="Bad", section="", word_count=100, language="en"),
        SimpleNamespace(url="https://example.com/clean", title="Clean", section="", word_count=100, language="en"),
    ]
    payload = build_technical_seo(
        pages,
        performance={
            "per_page": [
                {
                    "url": "https://example.com/bad",
                    "status": 200,
                    "small_font_size_count": 1,
                    "small_font_size_issues": [
                        {"source": "style_block", "tag": "style", "font_size": "11px", "font_size_px": 11.0},
                    ],
                },
                {
                    "url": "https://example.com/clean",
                    "status": 200,
                    "small_font_size_count": 0,
                    "small_font_size_issues": [],
                },
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "font_size_too_small"]
    assert len(issues) == 1
    assert issues[0]["url"] == "https://example.com/bad"
    assert issues[0]["issue_name"] == "Font size too small"
    assert issues[0]["category"] == "performance"
    assert issues[0]["importance"] == "Warning"
    assert issues[0]["severity"] == "medium"
    page = next(row for row in payload["pages"] if row["url"] == "https://example.com/bad")
    assert page["small_font_size_count"] == 1
    assert page["small_font_size_issues"] == [
        {"source": "style_block", "tag": "style", "font_size": "11px", "font_size_px": 11.0},
    ]


def test_technical_seo_model_flags_html_file_size_too_large() -> None:
    pages = [
        SimpleNamespace(url="https://example.com/large", title="Large", section="", word_count=100, language="en"),
        SimpleNamespace(url="https://example.com/clean", title="Clean", section="", word_count=100, language="en"),
    ]
    payload = build_technical_seo(
        pages,
        performance={
            "per_page": [
                {"url": "https://example.com/large", "status": 200, "html_weight_bytes": 1_200_000},
                {"url": "https://example.com/clean", "status": 200, "html_weight_bytes": 900_000},
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "html_file_size_too_large"]
    assert len(issues) == 1
    assert issues[0]["url"] == "https://example.com/large"
    assert issues[0]["issue_name"] == "HTML file size too large"
    assert issues[0]["category"] == "performance"
    assert issues[0]["importance"] == "Warning"
    assert issues[0]["severity"] == "medium"
    page = next(row for row in payload["pages"] if row["url"] == "https://example.com/large")
    assert page["html_weight_bytes"] == 1_200_000


def test_technical_seo_model_flags_not_compressed() -> None:
    pages = [
        SimpleNamespace(url="https://example.com/plain", title="Plain", section="", word_count=100, language="en"),
        SimpleNamespace(url="https://example.com/gzip", title="Gzip", section="", word_count=100, language="en"),
    ]
    payload = build_technical_seo(
        pages,
        performance={
            "per_page": [
                {
                    "url": "https://example.com/plain",
                    "status": 200,
                    "content_type": "text/html",
                    "content_encoding": "",
                    "compressed": False,
                    "not_compressed": True,
                },
                {
                    "url": "https://example.com/gzip",
                    "status": 200,
                    "content_type": "text/html",
                    "content_encoding": "gzip",
                    "compressed": True,
                    "not_compressed": False,
                },
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "not_compressed"]
    assert len(issues) == 1
    assert issues[0]["url"] == "https://example.com/plain"
    assert issues[0]["issue_name"] == "Not compressed"
    assert issues[0]["category"] == "performance"
    assert issues[0]["importance"] == "Warning"
    assert issues[0]["severity"] == "medium"
    page = next(row for row in payload["pages"] if row["url"] == "https://example.com/plain")
    assert page["content_encoding"] == ""
    assert page["not_compressed"] is True


def test_technical_seo_model_flags_page_stopped_passing_cwv_requirements() -> None:
    pages = [
        SimpleNamespace(url="https://example.com/regressed", title="Regressed", section="", word_count=100, language="en"),
        SimpleNamespace(url="https://example.com/still-passing", title="Still Passing", section="", word_count=100, language="en"),
        SimpleNamespace(url="https://example.com/already-failing", title="Already Failing", section="", word_count=100, language="en"),
    ]
    payload = build_technical_seo(
        pages,
        history_changes={
            "changes": [
                {
                    "url": "https://example.com/regressed",
                    "cwv_passed_before": True,
                    "cwv_passed_after": False,
                },
                {
                    "url": "https://example.com/still-passing",
                    "cwv_status_before": "good",
                    "cwv_status_after": "good",
                },
                {
                    "url": "https://example.com/already-failing",
                    "cwv_status_before": "poor",
                    "cwv_status_after": "poor",
                },
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "page_stopped_passing_cwv_requirements"]
    assert len(issues) == 1
    assert issues[0]["url"] == "https://example.com/regressed"
    assert issues[0]["issue_name"] == "Page stopped passing CWV requirements"
    assert issues[0]["category"] == "performance"
    assert issues[0]["importance"] == "Warning"
    assert issues[0]["severity"] == "medium"
    page = next(row for row in payload["pages"] if row["url"] == "https://example.com/regressed")
    assert page["previous_cwv_passed"] is True
    assert page["current_cwv_passed"] is False


def test_technical_seo_model_flags_pages_with_poor_cls() -> None:
    pages = [
        SimpleNamespace(url="https://example.com/high", title="High", section="", word_count=100, language="en"),
        SimpleNamespace(url="https://example.com/rating", title="Rating", section="", word_count=100, language="en"),
        SimpleNamespace(url="https://example.com/needs", title="Needs", section="", word_count=100, language="en"),
        SimpleNamespace(url="https://example.com/good", title="Good", section="", word_count=100, language="en"),
    ]
    payload = build_technical_seo(
        pages,
        performance={
            "per_page": [
                {"url": "https://example.com/high", "status": 200, "cls_score": 0.31},
                {"url": "https://example.com/rating", "status": 200, "cls_rating": "poor", "cls_score": 0.05},
                {"url": "https://example.com/needs", "status": 200, "cls_score": 0.2},
                {"url": "https://example.com/good", "status": 200, "cls_rating": "good", "cls_score": 0.4},
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "pages_with_poor_cls"]
    assert {row["url"] for row in issues} == {
        "https://example.com/high",
        "https://example.com/rating",
    }
    assert all(row["issue_name"] == "Pages with poor CLS" for row in issues)
    assert all(row["category"] == "performance" for row in issues)
    assert all(row["importance"] == "Warning" for row in issues)
    assert all(row["severity"] == "medium" for row in issues)
    page = next(row for row in payload["pages"] if row["url"] == "https://example.com/high")
    assert page["cls_score"] == 0.31


def test_technical_seo_model_flags_pages_with_poor_fid() -> None:
    pages = [
        SimpleNamespace(url="https://example.com/high", title="High", section="", word_count=100, language="en"),
        SimpleNamespace(url="https://example.com/rating", title="Rating", section="", word_count=100, language="en"),
        SimpleNamespace(url="https://example.com/needs", title="Needs", section="", word_count=100, language="en"),
        SimpleNamespace(url="https://example.com/good", title="Good", section="", word_count=100, language="en"),
    ]
    payload = build_technical_seo(
        pages,
        performance={
            "per_page": [
                {"url": "https://example.com/high", "status": 200, "fid_score": 350},
                {"url": "https://example.com/rating", "status": 200, "fid_rating": "poor", "fid_score": 50},
                {"url": "https://example.com/needs", "status": 200, "fid_score": 220},
                {"url": "https://example.com/good", "status": 200, "fid_rating": "good", "fid_score": 500},
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "pages_with_poor_fid"]
    assert {row["url"] for row in issues} == {
        "https://example.com/high",
        "https://example.com/rating",
    }
    assert all(row["issue_name"] == "Pages with poor FID" for row in issues)
    assert all(row["category"] == "performance" for row in issues)
    assert all(row["importance"] == "Warning" for row in issues)
    assert all(row["severity"] == "medium" for row in issues)
    page = next(row for row in payload["pages"] if row["url"] == "https://example.com/high")
    assert page["fid_score"] == 350


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


def test_technical_seo_model_flags_not_indexable_orphan_pages() -> None:
    pages = [
        SimpleNamespace(url="https://example.com/indexable-orphan", title="Indexable Orphan", section="", word_count=100, language="en"),
    ]
    payload = build_technical_seo(
        pages,
        indexability={
            "skipped": [
                {
                    "url": "https://example.com/non-indexable-orphan",
                    "title": "Non-indexable Orphan",
                    "reason": "noindex",
                    "http_status": 200,
                    "nofollow": False,
                },
                {
                    "url": "https://example.com/non-indexable-linked",
                    "title": "Non-indexable Linked",
                    "reason": "noindex",
                    "http_status": 200,
                    "nofollow": False,
                },
                {
                    "url": "https://example.com/no-linkgraph-signal",
                    "title": "No Linkgraph Signal",
                    "reason": "noindex",
                    "http_status": 200,
                    "nofollow": False,
                },
            ],
            "noindex_pages": [],
        },
        linkgraph={
            "page_link_counts": [
                {"url": "https://example.com/indexable-orphan", "in_degree": 0, "out_degree": 1},
                {"url": "https://example.com/non-indexable-orphan", "in_degree": 0, "out_degree": 1},
                {"url": "https://example.com/non-indexable-linked", "in_degree": 1, "out_degree": 1},
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "not_indexable_orphan_page_has_no_incoming_internal_links"]
    assert len(issues) == 1
    assert issues[0]["url"] == "https://example.com/non-indexable-orphan"
    assert issues[0]["issue_name"] == "Orphan page (has no incoming internal links)"
    assert issues[0]["importance"] == "Warning"


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


def test_technical_seo_model_flags_not_indexable_https_pages_linking_to_http() -> None:
    pages = [
        SimpleNamespace(url="https://example.com/indexable", title="Indexable", section="", word_count=100, language="en"),
    ]
    payload = build_technical_seo(
        pages,
        indexability={
            "skipped": [
                {
                    "url": "https://example.com/noindex",
                    "title": "Noindex",
                    "reason": "noindex",
                    "http_status": 200,
                    "nofollow": False,
                },
                {
                    "url": "https://example.com/clean-noindex",
                    "title": "Clean Noindex",
                    "reason": "noindex",
                    "http_status": 200,
                    "nofollow": False,
                },
            ],
            "noindex_pages": [],
        },
        linkgraph={
            "page_link_counts": [
                {
                    "url": "https://example.com/indexable",
                    "in_degree": 2,
                    "out_degree": 2,
                    "internal_http_link_count": 1,
                    "internal_http_links": ["http://example.com/legacy"],
                },
                {
                    "url": "https://example.com/noindex",
                    "in_degree": 1,
                    "out_degree": 2,
                    "internal_http_link_count": 1,
                    "internal_http_links": ["http://example.com/legacy"],
                },
                {
                    "url": "https://example.com/clean-noindex",
                    "in_degree": 1,
                    "out_degree": 1,
                    "internal_http_link_count": 0,
                    "internal_http_links": [],
                },
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "not_indexable_https_page_has_internal_links_to_http"]
    assert len(issues) == 1
    assert issues[0]["url"] == "https://example.com/noindex"
    assert issues[0]["issue_name"] == "HTTPS page has internal links to HTTP"
    assert issues[0]["importance"] == "Warning"


def test_technical_seo_model_flags_http_pages_linking_to_https() -> None:
    pages = [
        SimpleNamespace(url="http://example.com/bad", title="Bad", section="", word_count=100, language="en"),
        SimpleNamespace(url="http://example.com/clean", title="Clean", section="", word_count=100, language="en"),
        SimpleNamespace(url="https://example.com/https-source", title="HTTPS Source", section="", word_count=100, language="en"),
    ]
    payload = build_technical_seo(
        pages,
        indexability={
            "skipped": [
                {
                    "url": "http://example.com/noindex-bad",
                    "title": "Noindex Bad",
                    "reason": "noindex",
                    "http_status": 200,
                    "nofollow": False,
                },
                {
                    "url": "http://example.com/noindex-clean",
                    "title": "Noindex Clean",
                    "reason": "noindex",
                    "http_status": 200,
                    "nofollow": False,
                },
            ],
            "noindex_pages": [],
        },
        linkgraph={
            "page_link_counts": [
                {
                    "url": "http://example.com/bad",
                    "in_degree": 2,
                    "out_degree": 3,
                    "internal_https_link_count": 2,
                    "internal_https_links": ["https://example.com/secure", "https://example.com/sale"],
                },
                {
                    "url": "http://example.com/clean",
                    "in_degree": 2,
                    "out_degree": 1,
                    "internal_https_link_count": 0,
                    "internal_https_links": [],
                },
                {
                    "url": "https://example.com/https-source",
                    "in_degree": 2,
                    "out_degree": 1,
                    "internal_https_link_count": 1,
                    "internal_https_links": ["https://example.com/secure"],
                },
                {
                    "url": "http://example.com/noindex-bad",
                    "in_degree": 1,
                    "out_degree": 1,
                    "internal_https_link_count": 1,
                    "internal_https_links": ["https://example.com/secure"],
                },
                {
                    "url": "http://example.com/noindex-clean",
                    "in_degree": 1,
                    "out_degree": 1,
                    "internal_https_link_count": 0,
                    "internal_https_links": [],
                },
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "indexable_http_page_has_internal_links_to_https"]
    assert len(issues) == 1
    assert issues[0]["url"] == "http://example.com/bad"
    assert issues[0]["issue_name"] == "HTTP page has internal links to HTTPS"
    assert issues[0]["importance"] == "Notice"
    assert issues[0]["severity"] == "low"
    not_indexable_issues = [row for row in payload["issues"] if row["issue_type"] == "not_indexable_http_page_has_internal_links_to_https"]
    assert len(not_indexable_issues) == 1
    assert not_indexable_issues[0]["url"] == "http://example.com/noindex-bad"
    assert not_indexable_issues[0]["issue_name"] == "HTTP page has internal links to HTTPS"
    assert not_indexable_issues[0]["importance"] == "Notice"


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
    not_indexable_issues = [
        row
        for row in payload["issues"]
        if row["issue_type"] == "not_indexable_page_has_links_to_broken_page"
    ]
    assert len(not_indexable_issues) == 1
    assert not_indexable_issues[0]["url"] == "https://example.com/non-indexable-bad"
    assert not_indexable_issues[0]["issue_name"] == "Page has links to broken page"
    assert not_indexable_issues[0]["importance"] == "Warning"


def test_technical_seo_model_flags_indexable_pages_linking_to_redirects() -> None:
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
                    "redirect_internal_link_count": 1,
                    "redirect_internal_links": [
                        {"url": "https://example.com/redirecting", "redirect_target_url": "https://example.com/final"},
                    ],
                },
                {"url": "https://example.com/clean", "in_degree": 2, "out_degree": 1, "redirect_internal_link_count": 0},
                {
                    "url": "https://example.com/non-indexable-bad",
                    "in_degree": 0,
                    "out_degree": 1,
                    "redirect_internal_link_count": 1,
                    "redirect_internal_links": [
                        {"url": "https://example.com/redirecting", "redirect_target_url": "https://example.com/final"},
                    ],
                },
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "indexable_page_has_links_to_redirect"]
    assert len(issues) == 1
    assert issues[0]["url"] == "https://example.com/bad"
    assert issues[0]["issue_name"] == "Page has links to redirect"
    assert issues[0]["importance"] == "Warning"
    bad_page = next(row for row in payload["pages"] if row["url"] == "https://example.com/bad")
    assert bad_page["redirect_internal_link_count"] == 1
    not_indexable_issues = [row for row in payload["issues"] if row["issue_type"] == "not_indexable_page_has_links_to_redirect"]
    assert len(not_indexable_issues) == 1
    assert not_indexable_issues[0]["url"] == "https://example.com/non-indexable-bad"
    assert not_indexable_issues[0]["issue_name"] == "Page has links to redirect"
    assert not_indexable_issues[0]["importance"] == "Notice"


def test_technical_seo_model_flags_redirected_indexable_pages_with_no_incoming_links() -> None:
    pages = [
        SimpleNamespace(url="https://example.com/final", title="Final", section="", word_count=100, language="en"),
        SimpleNamespace(url="https://example.com/final-linked", title="Final Linked", section="", word_count=100, language="en"),
        SimpleNamespace(url="https://example.com/direct", title="Direct", section="", word_count=100, language="en"),
    ]
    payload = build_technical_seo(
        pages,
        indexability={
            "per_page": [
                {
                    "url": "https://example.com/final",
                    "indexability_status": "indexable",
                    "requested_url": "https://example.com/redirecting",
                    "redirect_target_url": "https://example.com/final",
                },
                {
                    "url": "https://example.com/final-linked",
                    "indexability_status": "indexable",
                    "requested_url": "https://example.com/redirecting-linked",
                    "redirect_target_url": "https://example.com/final-linked",
                },
                {
                    "url": "https://example.com/direct",
                    "indexability_status": "indexable",
                },
            ],
            "skipped": [
                {
                    "url": "https://example.com/noindex-final",
                    "title": "Noindex Final",
                    "reason": "noindex",
                    "http_status": 200,
                    "requested_url": "https://example.com/noindex-redirecting",
                    "redirect_target_url": "https://example.com/noindex-final",
                    "nofollow": False,
                },
                {
                    "url": "https://example.com/noindex-final-linked",
                    "title": "Noindex Final Linked",
                    "reason": "noindex",
                    "http_status": 200,
                    "requested_url": "https://example.com/noindex-redirecting-linked",
                    "redirect_target_url": "https://example.com/noindex-final-linked",
                    "nofollow": False,
                },
            ],
            "noindex_pages": [],
        },
        linkgraph={
            "page_link_counts": [
                {"url": "https://example.com/final", "in_degree": 0, "out_degree": 1},
                {"url": "https://example.com/final-linked", "in_degree": 2, "out_degree": 1},
                {"url": "https://example.com/direct", "in_degree": 0, "out_degree": 1},
                {"url": "https://example.com/noindex-final", "in_degree": 0, "out_degree": 1},
                {"url": "https://example.com/noindex-final-linked", "in_degree": 2, "out_degree": 1},
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "indexable_redirected_page_has_no_incoming_internal_links"]
    assert len(issues) == 1
    assert issues[0]["url"] == "https://example.com/final"
    assert issues[0]["issue_name"] == "Redirected page has no incoming internal links"
    assert issues[0]["importance"] == "Warning"
    final_page = next(row for row in payload["pages"] if row["url"] == "https://example.com/final")
    assert final_page["requested_url"] == "https://example.com/redirecting"
    not_indexable_issues = [
        row
        for row in payload["issues"]
        if row["issue_type"] == "not_indexable_redirected_page_has_no_incoming_internal_links"
    ]
    assert len(not_indexable_issues) == 1
    assert not_indexable_issues[0]["url"] == "https://example.com/noindex-final"
    assert not_indexable_issues[0]["issue_name"] == "Redirected page has no incoming internal links"
    assert not_indexable_issues[0]["importance"] == "Notice"


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
    not_indexable_issues = [row for row in payload["issues"] if row["issue_type"] == "not_indexable_page_has_no_outgoing_links"]
    assert len(not_indexable_issues) == 1
    assert not_indexable_issues[0]["url"] == "https://example.com/non-indexable-dead-end"
    assert not_indexable_issues[0]["issue_name"] == "Page has no outgoing links"
    assert not_indexable_issues[0]["importance"] == "Warning"


def test_technical_seo_model_flags_indexable_pages_with_only_nofollow_incoming_links() -> None:
    pages = [
        SimpleNamespace(url="https://example.com/nofollow-only", title="Nofollow Only", section="", word_count=100, language="en"),
        SimpleNamespace(url="https://example.com/mixed", title="Mixed", section="", word_count=100, language="en"),
    ]
    payload = build_technical_seo(
        pages,
        indexability={
            "skipped": [
                {
                    "url": "https://example.com/non-indexable",
                    "title": "Non-indexable",
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
                    "url": "https://example.com/nofollow-only",
                    "in_degree": 1,
                    "out_degree": 1,
                    "incoming_nofollow_internal_link_count": 1,
                    "incoming_dofollow_internal_link_count": 0,
                },
                {
                    "url": "https://example.com/mixed",
                    "in_degree": 2,
                    "out_degree": 1,
                    "incoming_nofollow_internal_link_count": 1,
                    "incoming_dofollow_internal_link_count": 1,
                },
                {
                    "url": "https://example.com/non-indexable",
                    "in_degree": 1,
                    "out_degree": 1,
                    "incoming_nofollow_internal_link_count": 1,
                    "incoming_dofollow_internal_link_count": 0,
                },
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "indexable_page_has_nofollow_incoming_internal_links_only"]
    assert len(issues) == 1
    assert issues[0]["url"] == "https://example.com/nofollow-only"
    assert issues[0]["issue_name"] == "Page has nofollow incoming internal links only"
    assert issues[0]["importance"] == "Warning"
    not_indexable_issues = [
        row
        for row in payload["issues"]
        if row["issue_type"] == "not_indexable_page_has_nofollow_incoming_internal_links_only"
    ]
    assert len(not_indexable_issues) == 1
    assert not_indexable_issues[0]["url"] == "https://example.com/non-indexable"
    assert not_indexable_issues[0]["issue_name"] == "Page has nofollow incoming internal links only"
    assert not_indexable_issues[0]["importance"] == "Notice"


def test_technical_seo_model_flags_indexable_pages_with_mixed_nofollow_and_dofollow_incoming_links() -> None:
    pages = [
        SimpleNamespace(url="https://example.com/nofollow-only", title="Nofollow Only", section="", word_count=100, language="en"),
        SimpleNamespace(url="https://example.com/mixed", title="Mixed", section="", word_count=100, language="en"),
    ]
    payload = build_technical_seo(
        pages,
        indexability={
            "skipped": [
                {
                    "url": "https://example.com/non-indexable",
                    "title": "Non-indexable",
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
                    "url": "https://example.com/nofollow-only",
                    "in_degree": 1,
                    "out_degree": 1,
                    "incoming_nofollow_internal_link_count": 1,
                    "incoming_dofollow_internal_link_count": 0,
                },
                {
                    "url": "https://example.com/mixed",
                    "in_degree": 2,
                    "out_degree": 1,
                    "incoming_nofollow_internal_link_count": 1,
                    "incoming_dofollow_internal_link_count": 1,
                },
                {
                    "url": "https://example.com/non-indexable",
                    "in_degree": 2,
                    "out_degree": 1,
                    "incoming_nofollow_internal_link_count": 1,
                    "incoming_dofollow_internal_link_count": 1,
                },
            ]
        },
    )

    issues = [
        row
        for row in payload["issues"]
        if row["issue_type"] == "indexable_page_has_nofollow_and_dofollow_incoming_internal_links"
    ]
    assert len(issues) == 1
    assert issues[0]["url"] == "https://example.com/mixed"
    assert issues[0]["issue_name"] == "Page has nofollow and dofollow incoming internal links"
    assert issues[0]["importance"] == "Notice"
    not_indexable_issues = [
        row
        for row in payload["issues"]
        if row["issue_type"] == "not_indexable_page_has_nofollow_and_dofollow_incoming_internal_links"
    ]
    assert len(not_indexable_issues) == 1
    assert not_indexable_issues[0]["url"] == "https://example.com/non-indexable"
    assert not_indexable_issues[0]["issue_name"] == "Page has nofollow and dofollow incoming internal links"
    assert not_indexable_issues[0]["importance"] == "Notice"


def test_technical_seo_model_flags_indexable_pages_with_nofollow_outgoing_internal_links() -> None:
    pages = [
        SimpleNamespace(url="https://example.com/source", title="Source", section="", word_count=100, language="en"),
        SimpleNamespace(url="https://example.com/clean", title="Clean", section="", word_count=100, language="en"),
    ]
    payload = build_technical_seo(
        pages,
        indexability={
            "skipped": [
                {
                    "url": "https://example.com/non-indexable",
                    "title": "Non-indexable",
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
                    "url": "https://example.com/source",
                    "in_degree": 1,
                    "out_degree": 2,
                    "outgoing_nofollow_internal_link_count": 1,
                    "outgoing_nofollow_internal_links": [
                        {"target_url": "https://example.com/target", "anchor": "nofollow target"}
                    ],
                },
                {
                    "url": "https://example.com/clean",
                    "in_degree": 1,
                    "out_degree": 1,
                    "outgoing_nofollow_internal_link_count": 0,
                },
                {
                    "url": "https://example.com/non-indexable",
                    "in_degree": 1,
                    "out_degree": 1,
                    "outgoing_nofollow_internal_link_count": 1,
                },
            ]
        },
    )

    issues = [row for row in payload["issues"] if row["issue_type"] == "indexable_page_has_nofollow_outgoing_internal_links"]
    assert len(issues) == 1
    assert issues[0]["url"] == "https://example.com/source"
    assert issues[0]["issue_name"] == "Page has nofollow outgoing internal links"
    assert issues[0]["importance"] == "Notice"
    not_indexable_issues = [row for row in payload["issues"] if row["issue_type"] == "not_indexable_page_has_nofollow_outgoing_internal_links"]
    assert len(not_indexable_issues) == 1
    assert not_indexable_issues[0]["url"] == "https://example.com/non-indexable"
    assert not_indexable_issues[0]["issue_name"] == "Page has nofollow outgoing internal links"
    assert not_indexable_issues[0]["importance"] == "Notice"


def test_technical_seo_model_flags_indexable_pages_with_only_one_dofollow_incoming_link() -> None:
    pages = [
        SimpleNamespace(url="https://example.com/one", title="One", section="", word_count=100, language="en"),
        SimpleNamespace(url="https://example.com/two", title="Two", section="", word_count=100, language="en"),
    ]
    payload = build_technical_seo(
        pages,
        indexability={
            "skipped": [
                {
                    "url": "https://example.com/non-indexable",
                    "title": "Non-indexable",
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
                    "url": "https://example.com/one",
                    "in_degree": 1,
                    "out_degree": 1,
                    "incoming_dofollow_internal_link_count": 1,
                    "incoming_nofollow_internal_link_count": 0,
                },
                {
                    "url": "https://example.com/two",
                    "in_degree": 2,
                    "out_degree": 1,
                    "incoming_dofollow_internal_link_count": 2,
                    "incoming_nofollow_internal_link_count": 0,
                },
                {
                    "url": "https://example.com/non-indexable",
                    "in_degree": 1,
                    "out_degree": 1,
                    "incoming_dofollow_internal_link_count": 1,
                    "incoming_nofollow_internal_link_count": 0,
                },
            ]
        },
    )

    issues = [
        row
        for row in payload["issues"]
        if row["issue_type"] == "indexable_page_has_only_one_dofollow_incoming_internal_link"
    ]
    assert len(issues) == 1
    assert issues[0]["url"] == "https://example.com/one"
    assert issues[0]["issue_name"] == "Page has only one dofollow incoming internal link"
    assert issues[0]["importance"] == "Notice"
    not_indexable_issues = [
        row
        for row in payload["issues"]
        if row["issue_type"] == "not_indexable_page_has_only_one_dofollow_incoming_internal_link"
    ]
    assert len(not_indexable_issues) == 1
    assert not_indexable_issues[0]["url"] == "https://example.com/non-indexable"
    assert not_indexable_issues[0]["issue_name"] == "Page has only one dofollow incoming internal link"
    assert not_indexable_issues[0]["importance"] == "Notice"


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
