import numpy as np

from site_audit.ai_citations import build_ai_citations
from site_audit.analyzer import PageInfo
from site_audit.dataforseo import build_analysis
from site_audit.recommendations import synthesize, to_payload


def _page(url: str, title: str = "Page") -> PageInfo:
    return PageInfo(
        url=url,
        title=title,
        description="",
        section="blog",
        word_count=500,
        language="en",
    )


def _response(items: list[dict]) -> dict:
    return {
        "status_code": 20000,
        "tasks": [{"status_code": 20000, "result": [{"items": items}]}],
    }


def test_ai_citations_join_url_variants_and_compute_top_page_share() -> None:
    pages = [
        _page("https://www.example.com/a/", "Cited page"),
        _page("https://example.com/b/", "Other page"),
    ]
    dataforseo_payload = {
        "ai_overview_citations": [
            {"keyword": "answer engine optimization", "url": "http://example.com/a", "search_volume": 900},
        ],
        "ai_overview_coverage": {"coverage": "own_domain_items_only", "note": "limited"},
    }
    search_payload = {
        "top_pages": [
            {"url": "https://example.com/b/", "traffic": 200},
            {"url": "https://example.com/a/", "traffic": 100, "cluster_label": "AEO"},
        ]
    }

    payload = build_ai_citations(dataforseo_payload, search_payload, pages)

    assert payload["available"] is True
    assert payload["cited_pages"][0]["url"] == "https://www.example.com/a/"
    assert payload["cited_pages"][0]["title"] == "Cited page"
    assert payload["cited_pages"][0]["cluster"] == "AEO"
    assert payload["summary"]["cited_pages"] == 1
    assert payload["summary"]["citing_queries"] == 1
    assert payload["summary"]["citing_query_volume"] == 900
    assert payload["summary"]["top_traffic_pages_cited_share"] == 0.5


def test_ai_citations_marks_stale_cited_pages_and_tolerates_missing_freshness() -> None:
    pages = [_page("https://example.com/a/", "Stale cited page")]
    dataforseo_payload = {
        "ai_overview_citations": [
            {"keyword": "ai seo", "url": "https://example.com/a", "search_volume": 500},
        ]
    }
    freshness_payload = {
        "per_page": [
            {"url": "https://example.com/a/", "title": "Stale cited page", "bucket": "very_stale", "age_days": 900}
        ]
    }

    payload = build_ai_citations(dataforseo_payload, {}, pages, freshness_payload)
    without_freshness = build_ai_citations(dataforseo_payload, {}, pages)

    assert payload["at_risk"][0]["url"] == "https://example.com/a/"
    assert payload["at_risk"][0]["bucket"] == "very_stale"
    assert payload["at_risk"][0]["top_keyword"] == "ai seo"
    assert without_freshness["at_risk"] == []


def test_ai_citations_do_not_infer_opportunities_without_returned_ai_overview_presence() -> None:
    pages = [_page("https://example.com/a/", "Cited"), _page("https://example.com/b/", "Ranking")]
    dataforseo_payload = {
        "ai_overview_citations": [
            {"keyword": "cited query", "url": "https://example.com/a", "search_volume": 100},
        ],
        "organic_keywords": [
            {
                "keyword": "ranking query",
                "url": "https://example.com/b/",
                "matched_url": "https://example.com/b/",
                "position": 8,
                "volume": 700,
            }
        ],
    }

    payload = build_ai_citations(dataforseo_payload, dataforseo_payload, pages)

    assert payload["available"] is True
    assert payload["opportunities"] == []
    assert payload["coverage"] == "own_domain_items_only"
    assert "own SERP items" in payload["coverage_note"]


def test_ai_citations_show_opportunities_when_no_pages_are_cited_but_ai_overview_is_known() -> None:
    pages = [_page("https://example.com/b/", "Ranking")]
    dataforseo_payload = {
        "ai_overview_citations": [],
        "organic_keywords": [
            {
                "keyword": "ranking query",
                "url": "https://example.com/b/",
                "matched_url": "https://example.com/b/",
                "position": 8,
                "volume": 700,
                "has_ai_overview": True,
            }
        ],
    }

    payload = build_ai_citations(dataforseo_payload, dataforseo_payload, pages)

    assert payload["available"] is True
    assert payload["cited_pages"] == []
    assert payload["opportunities"][0]["keyword"] == "ranking query"


def test_ai_citation_recommendations_are_geo_and_stable() -> None:
    pages = [_page("https://example.com/a/", "Stale cited page"), _page("https://example.com/b/", "Opportunity page")]
    dataforseo_payload = {
        "ai_overview_citations": [
            {"keyword": "ai seo", "url": "https://example.com/a", "search_volume": 500},
        ],
        "organic_keywords": [
            {
                "keyword": "answer block examples",
                "url": "https://example.com/b/",
                "matched_url": "https://example.com/b/",
                "position": 9,
                "volume": 1200,
                "has_ai_overview": True,
            }
        ],
    }
    freshness_payload = {
        "per_page": [
            {"url": "https://example.com/a/", "title": "Stale cited page", "bucket": "stale", "age_days": 500}
        ]
    }
    citations = build_ai_citations(dataforseo_payload, dataforseo_payload, pages, freshness_payload)

    first = to_payload(synthesize(ai_citations_payload=citations))
    second = to_payload(synthesize(ai_citations_payload=citations))
    first_ids = [item["id"] for item in first["items"] if item["id"].startswith("geo-aio")]
    second_ids = [item["id"] for item in second["items"] if item["id"].startswith("geo-aio")]

    assert first_ids == second_ids
    assert any(item["category"] == "geo" for item in first["items"] if item["id"].startswith("geo-aio"))
    assert any("is cited in Google AI Overviews" in item["instruction"] for item in first["items"])
    assert any("Strengthen the matching answer block" in item["instruction"] for item in first["items"])


def test_ai_citations_skip_opportunity_keywords_already_cited_via_another_page() -> None:
    pages = [_page("https://example.com/a/", "Cited A"), _page("https://example.com/b/", "Ranking B")]
    dataforseo_payload = {
        "ai_overview_citations": [
            {"keyword": "kw", "url": "https://example.com/a", "search_volume": 1000},
        ],
        "organic_keywords": [
            {
                "keyword": "kw",
                "url": "https://example.com/b/",
                "matched_url": "https://example.com/b/",
                "position": 8,
                "volume": 1000,
                "has_ai_overview": True,
            }
        ],
    }

    payload = build_ai_citations(dataforseo_payload, dataforseo_payload, pages)

    # The AI Overview for "kw" already cites the site (via page A), so page B
    # ranking for the same keyword is not an opportunity.
    assert payload["cited_pages"][0]["url"] == "https://example.com/a/"
    assert payload["opportunities"] == []


def test_ai_citations_reason_distinguishes_missing_provider_from_no_citation_rows() -> None:
    pages = [_page("https://example.com/a/")]

    missing_provider = build_ai_citations({}, {}, pages)
    none_provider = build_ai_citations(None, {}, pages)
    no_rows = build_ai_citations({"ai_overview_citations": []}, {}, pages)

    assert missing_provider["available"] is False
    assert none_provider["available"] is False
    assert no_rows["available"] is False
    assert missing_provider["reason"] == none_provider["reason"]
    assert missing_provider["reason"] != no_rows["reason"]
    assert missing_provider["summary"] == no_rows["summary"]
    assert set(no_rows["summary"]) == {
        "cited_pages",
        "citing_queries",
        "citing_query_volume",
        "top_traffic_pages",
        "top_traffic_pages_cited",
        "top_traffic_pages_cited_share",
    }


def test_unavailable_ai_citations_payload_produces_no_geo_aio_recommendations() -> None:
    unavailable = build_ai_citations(None, {}, [_page("https://example.com/a/")])

    payload = to_payload(synthesize(ai_citations_payload=unavailable))

    assert unavailable["available"] is False
    assert [item["id"] for item in payload["items"] if item["id"].startswith("geo-aio")] == []


def test_old_dataforseo_ranked_keywords_shape_has_empty_citations_and_unavailable_payload() -> None:
    pages = [_page("https://example.com/a/", "Organic page")]
    snapshot = {
        "meta": {"status": "ok", "cache_status": "hit", "provider": "dataforseo"},
        "raw": {
            "domain_rank_overview": _response([]),
            "relevant_pages": _response([]),
            "ranked_keywords": _response([
                {
                    "keyword_data": {
                        "keyword": "organic query",
                        "keyword_info": {"search_volume": 100},
                    },
                    "ranked_serp_element": {
                        "rank_group": 3,
                        "rank_absolute": 3,
                        "serp_item": {
                            "type": "organic",
                            "url": "https://example.com/a/",
                            "title": "Organic page",
                            "etv": 20,
                        },
                    },
                }
            ]),
        },
    }

    dataforseo_payload = build_analysis(snapshot, pages, np.eye(1, dtype=np.float32), embedder=None).payload
    citations = build_ai_citations(dataforseo_payload, dataforseo_payload, pages)

    assert dataforseo_payload["ai_overview_citations"] == []
    assert citations["available"] is False
    assert citations["reason"]
