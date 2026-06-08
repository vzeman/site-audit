import json
from pathlib import Path

from site_audit.serp_gap import (
    SerpGapConfig,
    _add_serp_url_rankings,
    _extract_serp_keyword_suggestions,
    _select_targets_with_budget,
    _serp_url_ranking_rows,
    _targets_from_serp,
    run,
)
from site_audit.competitive_analysis import CompetitiveTarget


def _write_base_report(root: Path) -> None:
    report = root / "example.com" / "report"
    report.mkdir(parents=True)
    homepage = "https://www.example.com/"
    feature = "https://www.example.com/features/live-chat/"
    (report / "pages.json").write_text(
        json.dumps(
            [
                {"url": homepage, "title": "Example Home", "description": "", "section": "/", "word_count": 500},
                {"url": feature, "title": "Live Chat", "description": "", "section": "features", "word_count": 800},
            ]
        ),
        encoding="utf-8",
    )
    (report / "search.json").write_text(
        json.dumps(
            {
                "meta": {"provider": "gsc"},
                "organic_keywords": [
                    {
                        "keyword": "live chat software",
                        "matched_url": feature,
                        "provider": "gsc",
                        "position": 8,
                        "impressions": 1000,
                        "clicks": 20,
                    },
                    {
                        "keyword": "ignored",
                        "matched_url": homepage,
                        "provider": "gsc",
                        "position": 50,
                        "impressions": 1000,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def test_serp_gap_dry_run_selects_pattern_pages_and_keywords(tmp_path: Path) -> None:
    _write_base_report(tmp_path)

    payload = run(
        SerpGapConfig(
            domain="example.com",
            projects_root=tmp_path,
            url_include_patterns=["/features/*"],
            dry_run=True,
        )
    )

    assert payload["status"] == "dry_run"
    assert payload["summary"]["pages_selected"] == 1
    assert payload["summary"]["keywords_selected"] == 1
    assert payload["selected_keywords"][0]["keyword"] == "live chat software"
    assert (tmp_path / "example.com" / "serp_gap" / "report" / "serp_gap.json").is_file()
    assert (tmp_path / "example.com" / "cache" / "serp_gap").is_dir()
    assert not (tmp_path / "example.com" / "serp_gap" / "cache").exists()


def test_serp_gap_manual_keywords_can_target_homepage(tmp_path: Path) -> None:
    _write_base_report(tmp_path)

    payload = run(
        SerpGapConfig(
            domain="example.com",
            projects_root=tmp_path,
            url_include_patterns=["/"],
            keyword_source="file",
            keywords=["helpdesk software", "call center software"],
            keywords_per_page=5,
            dry_run=True,
        )
    )

    assert payload["status"] == "dry_run"
    assert [row["keyword"] for row in payload["selected_keywords"]] == [
        "helpdesk software",
        "call center software",
    ]
    assert all(row["source"] == "manual" for row in payload["selected_keywords"])


def test_serp_gap_budget_cap_stops_before_paid_work(tmp_path: Path) -> None:
    _write_base_report(tmp_path)

    payload = run(
        SerpGapConfig(
            domain="example.com",
            projects_root=tmp_path,
            url_include_patterns=["/features/*"],
            budget_usd=0.0,
        )
    )

    assert payload["status"] == "budget_exceeded"
    assert payload["summary"]["budget_status"] == "over_budget"


def test_serp_gap_serp_targets_skip_ignored_hosts_and_take_next_results() -> None:
    payload = {
        "meta": {"provider": "serper"},
        "raw": {
            "organic": [
                {"link": "https://www.example.com/live-chat/", "position": 1},
                {"link": "https://x.com/livechat", "position": 2},
                {"link": "https://www.youtube.com/watch?v=123", "position": 3},
                {"link": "https://competitor-one.com/live-chat", "position": 4},
                {"link": "https://competitor-two.com/live-chat", "position": 5},
                {"link": "https://competitor-three.com/live-chat", "position": 6},
            ]
        },
    }

    targets = _targets_from_serp(
        "example.com",
        "live chat software",
        payload,
        SerpGapConfig(domain="example.com", results_per_keyword=2),
    )

    assert [target.competitor_url for target in targets] == [
        "https://competitor-one.com/live-chat",
        "https://competitor-two.com/live-chat",
    ]


def test_serp_gap_extracts_serper_keyword_suggestions() -> None:
    payload = {
        "meta": {"provider": "serper"},
        "raw": {
            "peopleAlsoAsk": [
                {"question": "What is live chat software?"},
                {"title": "What is livechat software?"},
            ],
            "relatedSearches": [
                {"query": "best live chat software"},
                {"query": "What is live chat software?"},
            ],
        },
    }

    assert _extract_serp_keyword_suggestions(payload) == [
        ("serp_people_also_ask", "What is live chat software?"),
        ("serp_people_also_ask", "What is livechat software?"),
        ("serp_people_also_search", "best live chat software"),
    ]


def test_serp_gap_extracts_dataforseo_keyword_suggestions() -> None:
    payload = {
        "meta": {"provider": "dataforseo"},
        "raw": {
            "tasks": [
                {
                    "result": [
                        {
                            "items": [
                                {
                                    "type": "people_also_ask",
                                    "items": [
                                        {"title": "How does live chat work?"},
                                        {"title": "  How does live chat work?  "},
                                    ],
                                },
                                {
                                    "type": "people_also_search",
                                    "items": [
                                        {"title": "live chat for website"},
                                        "free live chat software",
                                    ],
                                },
                            ]
                        }
                    ]
                }
            ]
        },
    }

    assert _extract_serp_keyword_suggestions(payload) == [
        ("serp_people_also_ask", "How does live chat work?"),
        ("serp_people_also_search", "live chat for website"),
        ("serp_people_also_search", "free live chat software"),
    ]


def test_serp_gap_url_rankings_count_top_10_repeated_urls() -> None:
    rankings = {}
    _add_serp_url_rankings(
        rankings,
        "example.com",
        "live chat software",
        {
            "meta": {"provider": "serper"},
            "raw": {
                "organic": [
                    {"link": "https://competitor-one.com/live-chat", "position": 1},
                    {"link": "https://x.com/livechat", "position": 2},
                    {"link": "https://competitor-two.com/chat", "position": 8},
                    {"link": "https://outside-top-ten.com/chat", "position": 11},
                ]
            },
        },
    )
    _add_serp_url_rankings(
        rankings,
        "example.com",
        "livechat",
        {
            "meta": {"provider": "serper"},
            "raw": {
                "organic": [
                    {"link": "https://competitor-two.com/chat", "position": 1},
                    {"link": "https://competitor-one.com/live-chat", "position": 4},
                    {"link": "https://www.example.com/live-chat/", "position": 7},
                ]
            },
        },
    )

    rows = _serp_url_ranking_rows(rankings)

    assert rows[0]["url"] == "https://competitor-one.com/live-chat"
    assert rows[0]["top10_count"] == 2
    assert rows[0]["best_rank"] == 1
    assert rows[0]["impressions"] == 0
    assert rows[0]["clicks"] == 0
    assert rows[0]["traffic"] == 0
    assert [row["url"] for row in rows] == [
        "https://competitor-one.com/live-chat",
        "https://competitor-two.com/chat",
        "https://www.example.com/live-chat/",
    ]
    assert rows[-1]["is_selected_domain"] is True


def test_serp_gap_reuses_known_competitors_for_overlapping_keywords() -> None:
    config = SerpGapConfig(domain="example.com", results_per_keyword=3, max_competitor_pages=2)
    known = {"https://competitor-one.com/", "https://competitor-two.com/"}
    targets = [
        CompetitiveTarget("livechat software", "https://competitor-one.com/", "livechat software", 1),
        CompetitiveTarget("livechat software", "https://competitor-two.com/", "livechat software", 2),
        CompetitiveTarget("livechat software", "https://new-competitor.com/", "livechat software", 3),
    ]

    selected = _select_targets_with_budget(targets, known, config)

    assert [target.competitor_url for target in selected] == [
        "https://competitor-one.com/",
        "https://competitor-two.com/",
    ]


def test_serp_gap_html_includes_scatter_and_cluster_sections(tmp_path: Path) -> None:
    _write_base_report(tmp_path)

    run(
        SerpGapConfig(
            domain="example.com",
            projects_root=tmp_path,
            url_include_patterns=["/features/*"],
            dry_run=True,
        )
    )

    html = (tmp_path / "example.com" / "serp_gap" / "report" / "index.html").read_text(encoding="utf-8")

    assert "Semantic Scatterplot" in html
    assert "Semantic Clusters" in html
    assert "Topic Relations" in html
    assert "Keyword and Content Semantic Map" in html
    assert "Use this section to see which URLs repeatedly win across the selected keywords" in html
    assert "Color identifies the domain, shape identifies the entity type" in html
    assert "competitors cover more deeply" in html
    assert "far from the target keyword and/or weakly connected to the SERP topic space" in html
    assert "URLs downloaded" in html
    assert "review_paragraphs" in html
    assert "Top-10 URLs Across Selected Keywords" in html
    assert "serp_url_rankings" in html
    assert "serpUrlGraph" in html
    assert "sharedKeywordNames" in html
    assert "serp-url-graph" in html
    assert "Hierarchical edge bundling chart" in html
    assert "bindSerpUrlGraphInteractions" in html
    assert "graphTooltipHtml" in html
    assert "data-graph-detail" in html
    assert "data-graph-edge" in html
    assert "has-active" in html
    assert "#c9c1b6" in html
    assert "SERP-position proxy" in html
    assert "Demand metrics unavailable" in html
    assert "impr" in html
    assert "serpRankingChart" in html
    assert "serp-ranking-chart" in html
    assert "serpRankingList" in html
    assert "All Keywords, URLs, and Content" in html
    assert "h2:'H2s'" in html
    assert "h6:'H6s'" in html
    assert "H1-H6" in html
    assert "unclassified headings" in html
    assert "overview_scatter" in html
    assert "overview-section" in html
    assert "nearest_keyword" in html
    assert "own_paragraphs_to_review" in html
    assert "review_reason" in html
    assert "Wheel to zoom, drag to pan, double-click to reset" in html
    assert "pointTooltip" in html
    assert "scatter-tooltip" in html
    assert "bindScatterInteractions" in html
    assert "click a dot for details" in html
    assert "data-zoom=\"in\"" in html
    assert "Wheel to zoom, drag to pan, double-click to reset" in html
    assert "domainColor" in html
    assert "function markerSvg" in html
    assert "function pointSize" in html
    assert "data-entity-filter" in html
    assert "data-domain-filter" in html
    assert "domainFilters" in html
    assert "pointDomain" in html
    assert "bindScatterFilters" in html
    assert "topicChart" in html
    assert "topic-chart" in html
    assert "data-detail" in html
    assert "tip-badge" in html
    assert "keyword_distance" in html
    assert "keyword_similarity" in html
    assert "function pointLabel" in html
    assert ".scatter-tooltip.open" in html
    assert "page-section" in html
    assert "report-sidebar" in html
    assert 'id="report-nav"' in html
    assert "report-nav-button" in html
    assert "report-nav-label" in html
    assert "--audit-accent" in html
    assert "buildNav" in html
    assert 'target="_blank" rel="noopener noreferrer"' in html
