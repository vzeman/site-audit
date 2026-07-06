from site_audit.analyzer import PageInfo
from site_audit.ctr_anomalies import build_ctr_anomalies


def _page(url: str, title: str = "Support automation guide", description: str = "A complete guide.") -> PageInfo:
    return PageInfo(url, title, description, "blog", 500, "en")


def test_ctr_anomalies_flags_below_threshold_and_scores_missed_clicks() -> None:
    page = _page("https://example.com/support")
    payload = {
        "meta": {"params": {"start_date": "2026-04-01", "end_date": "2026-04-30"}},
        "query_pages": [
            {"query": "support automation", "url": page.url, "impressions": 1000, "clicks": 50, "ctr": 0.05, "position": 2, "source": "gsc"},
            {"query": "ticket workflow", "url": page.url, "impressions": 1000, "clicks": 100, "ctr": 0.10, "position": 2, "source": "gsc"},
            {"query": "too few", "url": page.url, "impressions": 99, "clicks": 1, "ctr": 0.01, "position": 2, "source": "gsc"},
        ],
    }

    result = build_ctr_anomalies(payload, [page])

    assert result["available"] is True
    assert result["summary"]["anomalies"] == 1
    assert result["model"]["period_days"] == 30
    row = result["rows"][0]
    assert row["query"] == "support automation"
    assert row["actual_ctr"] == 0.05
    assert row["expected_ctr"] == 0.15
    assert row["missed_clicks"] == 100.0
    assert result["summary"]["total_missed_clicks"] == 100.0


def test_ctr_anomalies_classifies_causes_in_order() -> None:
    payload = {
        "organic_keywords": [
            {"keyword": "reviews", "serp_features": ["featured_snippet"]},
        ],
        "query_pages": [
            {"query": "support automation", "url": "https://example.com/missing", "impressions": 1000, "ctr": 0.01, "position": 3, "source": "gsc"},
            {"query": "workflow automation", "url": "https://example.com/long", "impressions": 1000, "ctr": 0.01, "position": 3, "source": "gsc"},
            {"query": "reviews", "url": "https://example.com/reviews", "impressions": 1000, "ctr": 0.01, "position": 3, "source": "gsc"},
            {"query": "unclear", "url": "https://example.com/unclear", "impressions": 1000, "ctr": 0.01, "position": 3, "source": "gsc"},
        ],
    }
    pages = [
        _page("https://example.com/missing", "Generic software page"),
        _page("https://example.com/long", "Workflow automation " + "x" * 47),
        _page("https://example.com/reviews", "Reviews"),
        _page("https://example.com/unclear", "Unclear"),
    ]

    result = build_ctr_anomalies(payload, pages)
    causes = {row["query"]: row["probable_cause"] for row in result["rows"]}

    assert causes["support automation"] == "title_missing_query_terms"
    long_row = next(row for row in result["rows"] if row["url"] == "https://example.com/long")
    assert long_row["title_length"] > 65
    assert long_row["probable_cause"] == "title_too_long_truncated"
    assert causes["reviews"] == "serp_feature_competition"
    assert causes["unclear"] == "unclear"


def test_ctr_anomalies_reuses_metadata_description_flags() -> None:
    page = _page("https://example.com/support")
    payload = {
        "query_pages": [
            {"query": "support automation", "url": page.url, "impressions": 1000, "ctr": 0.01, "position": 3, "source": "gsc"},
        ],
    }
    metadata = {
        "per_page": [
            {
                "url": page.url,
                "title": page.title,
                "description": "",
                "issues": ["missing_description"],
            }
        ],
    }

    result = build_ctr_anomalies(payload, [page], metadata)

    assert result["rows"][0]["probable_cause"] == "description_missing_or_duplicate"


def test_ctr_anomalies_groups_by_url_and_recommendation_inputs() -> None:
    page = _page("https://example.com/support")
    payload = {
        "query_pages": [
            {"query": "support automation", "url": page.url, "impressions": 1000, "ctr": 0.05, "position": 2, "source": "gsc"},
            {"query": "support automation tools", "url": page.url, "impressions": 500, "ctr": 0.02, "position": 3, "source": "gsc"},
        ],
    }

    result = build_ctr_anomalies(payload, [page], max_rows=1)

    assert len(result["rows"]) == 1
    assert len(result["pages"]) == 1
    assert result["pages"][0]["total_missed_clicks"] == result["summary"]["total_missed_clicks"]
    assert len(result["pages"][0]["worst_queries"]) == 2
    rec = result["recommendations"][0]
    assert rec["title"] == 'Title underperforms position #2 for "support automation"'
    assert rec["estimated_clicks_gain"] == result["pages"][0]["total_missed_clicks"]
    assert "Rewrite title/meta of https://example.com/support" in rec["action"]


def test_ctr_anomalies_ignores_rows_without_measured_gsc_ctr() -> None:
    page = _page("https://example.com/support")
    payload = {
        "query_pages": [
            {"query": "support automation", "url": page.url, "impressions": 1000, "position": 2, "source": "gsc"},
            {"query": "support automation", "url": page.url, "impressions": 1000, "ctr": 0.01, "position": 2, "source": "ahrefs"},
        ],
    }

    result = build_ctr_anomalies(payload, [page])

    assert result["available"] is False
    assert result["rows"] == []


def test_ctr_anomalies_keeps_zero_ctr_rows() -> None:
    page = _page("https://example.com/support")
    payload = {
        "query_pages": [
            {"query": "support automation", "url": page.url, "impressions": 1000, "clicks": 0, "ctr": 0.0, "position": 2, "source": "gsc"},
        ],
    }

    result = build_ctr_anomalies(payload, [page])

    assert result["available"] is True
    assert result["summary"]["anomalies"] == 1
    assert result["rows"][0]["missed_clicks"] == 150.0


def test_ctr_anomalies_threshold_boundary_and_inclusive_floors() -> None:
    page = _page("https://example.com/support")
    payload = {
        "query_pages": [
            # actual == ratio * expected(pos 2 → 0.15) == 0.09: not flagged (strictly below only)
            {"query": "at boundary", "url": page.url, "impressions": 1000, "ctr": 0.09, "position": 2, "source": "gsc"},
            # inclusive floors: impressions == 100 and position == 10.0 are both in scope
            {"query": "at floors", "url": page.url, "impressions": 100, "ctr": 0.001, "position": 10.0, "source": "gsc"},
        ],
    }

    result = build_ctr_anomalies(payload, [page])

    queries = [row["query"] for row in result["rows"]]
    assert "at boundary" not in queries
    assert "at floors" in queries


def test_ctr_anomalies_description_cause_wins_over_serp_features() -> None:
    page = _page("https://example.com/reviews", "Reviews of tools")
    payload = {
        "organic_keywords": [
            {"keyword": "reviews", "serp_features": ["featured_snippet"]},
        ],
        "query_pages": [
            {"query": "reviews", "url": page.url, "impressions": 1000, "ctr": 0.01, "position": 3, "source": "gsc"},
        ],
    }
    metadata = {
        "per_page": [
            {"url": page.url, "title": page.title, "description": "", "issues": ["missing_description"]},
        ],
    }

    result = build_ctr_anomalies(payload, [page], metadata)

    assert result["rows"][0]["probable_cause"] == "description_missing_or_duplicate"


def test_ctr_anomalies_available_with_gsc_ctr_but_no_anomalies() -> None:
    page = _page("https://example.com/support")
    payload = {
        "query_pages": [
            {"query": "support automation", "url": page.url, "impressions": 1000, "ctr": 0.20, "position": 2, "source": "gsc"},
        ],
    }

    result = build_ctr_anomalies(payload, [page])

    assert result["available"] is True
    assert result["summary"]["anomalies"] == 0
    assert result["rows"] == []
