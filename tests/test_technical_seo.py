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
