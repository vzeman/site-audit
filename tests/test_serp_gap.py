import json
from pathlib import Path

import numpy as np

from site_audit.ai_agent import (
    AgentCompletion,
    build_editor_brief_messages,
    harnext_status,
    parse_keyword_candidates,
    parse_language_detection,
)
from site_audit.cache import HttpCache
from site_audit.serp_gap import (
    SerpGapConfig,
    _add_serp_url_rankings,
    _attach_action_points,
    _action_csv_rows,
    _action_points_for_analysis,
    _ai_agent_state,
    _attach_winnability,
    _content_comparison,
    _content_order_path,
    _dedupe_semantic_row_texts,
    _editorial_guidelines,
    _extract_serp_keyword_suggestions,
    _enrich_keyword_rows,
    _enrich_serp_domain_ratings,
    _is_ugc_host,
    _keyword_metrics_lookup,
    _keyword_priority,
    _keyword_row,
    _load_winnability_cache,
    _page_intent,
    _save_winnability_cache,
    _overview_scatter,
    _paragraph_match_heatmap,
    _resolve_serp_language,
    _select_targets_with_budget,
    _serp_url_ranking_rows,
    _targets_from_serp,
    _todo_markdown,
    _topic_coverage_matrix,
    _intent_assessment,
    _winnability,
    run,
)
from site_audit.analyzer import PageInfo
from site_audit.competitive_analysis import CompetitiveTarget, CompetitorPage
from site_audit.extractor import ExtractedPage


class _StaticEmbedder:
    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        vectors = {
            "live chat software": [1.0, 0.0, 0.0],
            "helpdesk software": [0.0, 1.0, 0.0],
            "Live chat paragraph": [0.8, 0.2, 0.0],
        }
        return np.array([vectors.get(text, [0.0, 0.0, 1.0]) for text in texts], dtype=np.float32)


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
    report_dir = Path(payload["summary"]["report_dir"])
    assert report_dir.parent == tmp_path / "example.com" / "serp_gap" / "report"
    assert report_dir.name == "features-live-chat"
    assert (report_dir / "serp_gap.json").is_file()
    assert (report_dir / "serp_gap_actions.csv").is_file()
    assert (report_dir / "serp_gap_todo.md").is_file()
    assert not (tmp_path / "example.com" / "serp_gap" / "report" / "index.html").exists()
    assert (tmp_path / "example.com" / "cache" / "serp_gap").is_dir()
    assert not (tmp_path / "example.com" / "serp_gap" / "cache").exists()


def test_serp_gap_writes_url_specific_report_directories(tmp_path: Path) -> None:
    _write_base_report(tmp_path)

    feature_payload = run(
        SerpGapConfig(
            domain="example.com",
            projects_root=tmp_path,
            urls=["https://www.example.com/features/live-chat/"],
            keyword_source="file",
            keywords=["live chat software"],
            dry_run=True,
        )
    )
    home_payload = run(
        SerpGapConfig(
            domain="example.com",
            projects_root=tmp_path,
            urls=["https://www.example.com/"],
            keyword_source="file",
            keywords=["helpdesk software"],
            dry_run=True,
        )
    )

    feature_dir = Path(feature_payload["summary"]["report_dir"])
    home_dir = Path(home_payload["summary"]["report_dir"])
    assert feature_dir != home_dir
    assert feature_dir.name == "features-live-chat"
    assert home_dir.name == "home"
    assert (feature_dir / "index.html").is_file()
    assert (home_dir / "index.html").is_file()


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


def test_serp_gap_url_only_dry_run_uses_ai_keyword_fallback(tmp_path: Path) -> None:
    _write_base_report(tmp_path)

    payload = run(
        SerpGapConfig(
            domain="example.com",
            projects_root=tmp_path,
            urls=["https://www.example.com/"],
            dry_run=True,
        )
    )

    assert payload["status"] == "dry_run"
    assert payload["ai_agent"]["status"] == "dry_run"
    assert payload["selected_keywords"][0]["keyword"] == "Example Home"
    assert payload["selected_keywords"][0]["source"] == "ai_agent_fallback"
    assert "no API demand metric match" in payload["selected_keywords"][0]["metrics_source"]


def test_serp_gap_ai_agent_state_reports_missing_openrouter_key(monkeypatch) -> None:
    monkeypatch.setattr("site_audit.serp_gap.openrouter_api_key", lambda: "")

    state = _ai_agent_state(SerpGapConfig(domain="example.com", dry_run=False, ai_agent=True))

    assert state["status"] == "missing_openrouter_api_key"
    assert "OPENROUTER_API_KEY" in state["notes"][0]


def test_serp_gap_ai_agent_state_falls_back_to_openrouter_without_harnext(monkeypatch) -> None:
    monkeypatch.setattr("site_audit.serp_gap.openrouter_api_key", lambda: "sk-test")
    monkeypatch.setattr("site_audit.serp_gap.harnext_status", lambda: (False, "Install Harnext CLI"))

    config = SerpGapConfig(domain="example.com", dry_run=False, ai_agent=True)
    state = _ai_agent_state(config)

    assert state["status"] == "ready"
    assert state["provider"] == "openrouter"
    assert config.ai_agent_provider == "openrouter"
    assert "Install Harnext CLI" in state["notes"][0]


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


def test_serp_gap_dedupes_duplicate_semantic_chart_items() -> None:
    rows = [
        {"url": "https://example.com/a", "source": "ours", "entity_type": "h1", "text": "Live Chat"},
        {"url": "https://example.com/a", "source": "ours", "entity_type": "h1", "text": " Live   Chat "},
        {"url": "https://example.com/a", "source": "ours", "entity_type": "title", "text": "Live Chat"},
        {"url": "https://example.com/b", "source": "competitor", "entity_type": "h1", "text": "Live Chat"},
    ]
    texts = [row["text"] for row in rows]

    deduped_rows, deduped_texts, removed = _dedupe_semantic_row_texts(rows, texts)

    assert removed == 1
    assert len(deduped_rows) == 3
    assert deduped_texts == ["Live Chat", "Live Chat", "Live Chat"]
    assert [row["entity_type"] for row in deduped_rows] == ["h1", "title", "h1"]


def test_overview_scatter_adds_demand_weighted_keyword_centroid() -> None:
    rows = [
        {"entity_type": "keyword", "source": "keyword", "text": "live chat software", "impressions": 100, "clicks": 5, "traffic": 2.5, "volume": 1000},
        {"entity_type": "keyword", "source": "keyword", "text": "helpdesk software", "impressions": 300, "clicks": 8, "traffic": 4.0, "volume": 2000},
        {"entity_type": "paragraph", "source": "ours", "text": "Live chat paragraph", "url": "https://example.com/live-chat"},
    ]

    scatter = _overview_scatter(rows, [row["text"] for row in rows], _StaticEmbedder())
    centroid = next(point for point in scatter["points"] if point["entity_type"] == "keyword_centroid")

    assert centroid["text"] == "Demand-weighted keyword centroid (2 keywords)"
    assert centroid["keyword_count"] == 2
    assert centroid["impressions"] == 400
    assert centroid["clicks"] == 13
    assert centroid["traffic"] == 6.5
    assert centroid["volume"] == 3000
    ridges = scatter["keyword_url_ridges"]
    assert [row["keyword"] for row in ridges["keywords"]] == ["live chat software", "helpdesk software"]
    assert ridges["rows"][0]["url"] == "https://example.com/live-chat"
    assert ridges["rows"][0]["cells"][0]["max_similarity"] > ridges["rows"][0]["cells"][1]["max_similarity"]


def test_serp_gap_builds_action_points_for_ai_content_agents() -> None:
    page = {"url": "https://example.com/live-chat", "title": "Live Chat"}
    analysis = {
        "status": "ok",
        "query": "live chat software",
        "keyword": {"keyword": "live chat software", "impressions": 1200, "clicks": 40, "traffic": 8.5, "volume": 6000},
        "missing_topics": [
            {
                "label": "pricing and implementation details",
                "coverage": "missing",
                "priority": "high",
                "competitor_coverage": 3,
                "competitor_prevalence": 0.75,
                "best_competitor_rank": 2,
                "our_best_similarity": 0.41,
                "competitor_urls": ["https://competitor.example/pricing"],
                "examples": [
                    {
                        "url": "https://competitor.example/pricing",
                        "rank": 2,
                        "paragraph": "Competitor explains pricing and setup.",
                    }
                ],
            }
        ],
        "weak_topics": [],
        "off_intent_paragraphs": [
            {
                "paragraph_index": 4,
                "similarity_to_serp_topics": 0.39,
                "paragraph": "Unrelated operational history.",
                "review_reason": "below off-intent threshold",
            }
        ],
    }

    actions = _action_points_for_analysis(page, analysis)

    assert actions[0]["type"] == "add_topic"
    assert actions[0]["priority"] == "high"
    assert "Add a concise section" in actions[0]["instruction"]
    assert actions[0]["content_brief"]["recommended_format"] == "answer block plus comparison table"
    assert actions[0]["placement"]
    assert actions[0]["acceptance_criteria"]
    assert "direct answer" in " ".join(actions[0]["content_brief"]["paragraph_plan"])
    assert "AI" not in actions[0]["content_brief"]["paragraph_plan"][0]
    assert "Use the SERP evidence to infer intent" in actions[0]["ai_agent_prompt"]
    assert actions[0]["evidence"]["keyword_impressions"] == 1200
    assert actions[0]["evidence"]["keyword_volume"] == 6000
    assert actions[0]["evidence"]["example_url"] == "https://competitor.example/pricing"
    review_action = next(action for action in actions if action["type"] == "review_paragraph")
    assert "keep, rewrite, move, merge, or remove" in review_action["ai_agent_prompt"]
    assert review_action["content_brief"]["quality_profile"]["word_count"] == 3


def _intent_page(url: str = "https://example.com/product") -> ExtractedPage:
    return ExtractedPage(
        url=url,
        title="Widget Platform",
        description="",
        body="",
        word_count=250,
        language="en",
        h1="Widget Platform",
        headers_rich=[{"level": 1, "text": "Widget Platform"}],
        paragraphs=["A product page."],
    )


def test_intent_assessment_respects_provider_intent() -> None:
    intent = _intent_assessment(
        {"keyword": "widget software", "intents": ["commercial"]},
        _intent_page(),
        {"page_type": "listing"},
        [{"title": "How to use widgets", "url": "https://competitor.example/how", "rank": 1}],
        {"people_also_ask": [{"question": "How?"}]},
        [],
    )

    assert intent["serp_intent"] == "commercial-investigation"
    assert "provider_intent:commercial-investigation" in intent["evidence"]


def test_intent_assessment_falls_back_to_serp_title_patterns() -> None:
    intent = _intent_assessment(
        {"keyword": "widget tools", "intents": []},
        _intent_page(),
        {"page_type": "listing"},
        [
            {"title": "Best widget tools vs alternatives", "url": "https://competitor.example/best", "rank": 1},
            {"title": "Top 10 widget reviews", "url": "https://competitor.example/top", "rank": 2},
        ],
        {"people_also_ask": []},
        [],
    )

    assert intent["serp_intent"] == "commercial-investigation"
    assert any("commercial SERP title/query pattern" in row for row in intent["evidence"])


def test_intent_mismatch_adds_retarget_action_first() -> None:
    page = {"url": "https://example.com/product", "title": "Widget Platform", "h1": "Widget Platform"}
    analysis = _analysis_base("how to use widgets")
    analysis["intent"] = {
        "serp_intent": "informational",
        "page_intent": "transactional",
        "match": "mismatch",
        "evidence": ["audit_page_type:product"],
    }
    analysis["missing_topics"] = [{
        "label": "setup steps",
        "coverage": "missing",
        "priority": "high",
        "competitor_prevalence": 0.9,
        "best_competitor_rank": 1,
        "examples": [{"url": "https://competitor.example/how", "rank": 1, "paragraph": "Steps."}],
    }]

    actions = _action_points_for_analysis(page, analysis)

    assert actions[0]["type"] == "retarget_or_new_page"
    assert "SERP intent is informational" in actions[0]["instruction"]


def test_winnability_bands_from_domain_rating_distribution() -> None:
    assert _winnability([
        {"rank": 1, "url": "https://a.example/", "domain_rating": 40},
        {"rank": 2, "url": "https://b.example/", "domain_rating": 42},
        {"rank": 3, "url": "https://c.example/", "domain_rating": 45},
    ], 35)["band"] == "winnable"
    hard = _winnability([
        {"rank": 1, "url": "https://a.example/", "domain_rating": 65},
        {"rank": 2, "url": "https://b.example/", "domain_rating": 70},
        {"rank": 3, "url": "https://c.example/", "domain_rating": 75},
    ], 50)
    assert hard["band"] == "hard"
    assert hard["top10_dr_median"] == 70
    unlikely = _winnability([
        {"rank": 1, "url": "https://a.example/", "domain_rating": 70},
        {"rank": 2, "url": "https://b.example/", "domain_rating": 82},
    ], 35)
    assert unlikely["band"] == "unlikely"
    assert unlikely["factor"] == 0.25
    assert _winnability([{"rank": 1, "url": "https://reddit.com/r/widgets", "domain_rating": 91}], 35)["band"] == "winnable"


def test_winnability_missing_dr_is_unknown_and_does_not_gate() -> None:
    result = _winnability([{"rank": 1, "url": "https://a.example/"}], None)

    assert result["band"] == "unknown"
    assert result["factor"] == 1.0
    assert "Missing own DR" in result["evidence"][0]


def test_unlikely_winnability_changes_header_and_suggests_alternative_keyword() -> None:
    page = {"url": "https://example.com/product", "title": "Widget Platform", "h1": "Widget Platform"}
    analysis = _analysis_base("enterprise widget platform")
    analysis["serp"] = {"top10": [
        {"rank": 1, "url": "https://strong-a.example/", "domain_rating": 80},
        {"rank": 2, "url": "https://strong-b.example/", "domain_rating": 84},
        {"rank": 3, "url": "https://strong-c.example/", "domain_rating": 90},
    ]}
    page_results = [{**page, "analyses": [analysis]}]
    keyword_rows = [
        {"url": page["url"], "keyword": "enterprise widget platform", "impressions": 1000},
        {
            "url": page["url"],
            "keyword": "widget platform for small teams",
            "impressions": 300,
            "winnability_band": "winnable",
            "winnability_factor": 1.0,
        },
    ]

    _attach_winnability(page_results, keyword_rows, 35)
    actions = _action_points_for_analysis(page, analysis)

    assert analysis["winnability"]["band"] == "unlikely"
    assert "content changes alone are unlikely" in analysis["recommendation_header"].lower()
    assert analysis["alternative_keyword"]["keyword"] == "widget platform for small teams"
    assert actions[0]["type"] == "winnability_prerequisite"
    assert "link acquisition" in actions[0]["instruction"]


def test_keyword_priority_uses_winnability_factor() -> None:
    base = {"source": "gsc", "position": 8, "impressions": 1000}
    hard = {**base, "keyword": "hard", "winnability_band": "hard"}
    winnable = {**base, "keyword": "winnable", "winnability_band": "winnable"}
    unlikely = {**base, "keyword": "unlikely", "winnability_band": "unlikely"}

    rows = sorted([hard, unlikely, winnable], key=_keyword_priority, reverse=True)

    assert [row["keyword"] for row in rows] == ["winnable", "hard", "unlikely"]


def _dr_rows(ratings: list[float]) -> list[dict]:
    return [
        {"rank": index + 1, "url": f"https://serp-{index}.example/", "domain_rating": value}
        for index, value in enumerate(ratings)
    ]


def test_winnability_gap_over_30_is_never_winnable() -> None:
    # Nothing within reach and a median gap > 30: hopeless even though own DR
    # is not below min(top10) - 30.
    assert _winnability(_dr_rows([70, 75, 80]), 40)["band"] == "unlikely"
    assert _winnability(_dr_rows([65, 95, 95, 95, 95]), 50)["band"] == "unlikely"
    # One result within reach keeps the same gap merely hard, not winnable.
    assert _winnability(_dr_rows([55, 95, 95]), 50)["band"] == "hard"
    assert _winnability(_dr_rows([45] + [95] * 9), 40)["band"] == "hard"


def test_winnability_boundary_at_min_minus_30() -> None:
    rows = _dr_rows([70, 70, 70])

    assert _winnability(rows, 40)["band"] == "hard"  # own == min - 30: strict "<" per spec
    assert _winnability(rows, 39)["band"] == "unlikely"


def test_winnability_small_median_gap_is_winnable() -> None:
    assert _winnability(_dr_rows([58, 59, 66]), 50)["band"] == "winnable"


def test_winnability_empty_serp_and_missing_competitor_dr_are_unknown() -> None:
    assert _winnability([], 55)["band"] == "unknown"
    assert _winnability([{"rank": 1, "url": "https://a.example/"}], 55)["band"] == "unknown"


def test_weak_result_hosts_match_exact_domains_and_whole_labels() -> None:
    assert _is_ugc_host("https://old.reddit.com/r/widgets")
    assert _is_ugc_host("https://forum.acme.com/thread")
    assert _is_ugc_host("https://community.acme.com/q")
    assert not _is_ugc_host("https://notmedium.com/post")
    assert not _is_ugc_host("https://performance-community.io/blog")
    assert not _is_ugc_host("https://example.com/reddit.com-review")


def test_weak_result_low_absolute_dr_is_winnable_with_honest_evidence() -> None:
    win = _winnability(_dr_rows([25, 90, 92]), 40)

    assert win["band"] == "winnable"
    assert any("low-authority" in row for row in win["evidence"])
    # A DR within own reach but above the absolute threshold is not "weak".
    hard = _winnability(_dr_rows([45] + [95] * 9), 40)
    assert not any("Weak result signal" in row for row in hard["evidence"])


def test_page_intent_prefers_audit_page_type_over_title_wording() -> None:
    page = ExtractedPage(
        url="https://shop.example/widget",
        title="Best Widget for Teams | Acme",
        description="",
        body="",
        word_count=120,
        language="en",
        h1="Best Widget",
        headers_rich=[],
        paragraphs=[],
    )

    intent, evidence = _page_intent(page, {"page_type": "product"})

    assert intent == "transactional"
    assert "audit_page_type:product" in evidence


def test_product_page_vs_informational_serp_yields_mismatch_and_retarget_first() -> None:
    page_ext = ExtractedPage(
        url="https://shop.example/widget",
        title="Acme Widget Pro",
        description="Enterprise widget appliance.",
        body="",
        word_count=300,
        language="en",
        h1="Acme Widget Pro",
        headers_rich=[{"level": 2, "text": "Features"}, {"level": 2, "text": "Specifications"}],
        paragraphs=["Buy the widget."],
    )
    serp_rows = [
        {"title": "How to choose a widget: complete guide", "url": "https://a.example/guide", "rank": 1},
        {"title": "What is a widget? Definition and examples", "url": "https://b.example/what", "rank": 2},
        {"title": "Widget tutorial for beginners", "url": "https://c.example/tut", "rank": 3},
    ]

    intent = _intent_assessment(
        {"keyword": "widget", "intents": []},
        page_ext,
        {"page_type": "product"},
        serp_rows,
        {"people_also_ask": [{"question": "How do widgets work?"}]},
        [],
    )

    assert intent["serp_intent"] == "informational"
    assert intent["page_intent"] == "transactional"
    assert intent["match"] == "mismatch"

    analysis = _analysis_base("widget")
    analysis["intent"] = intent
    analysis["missing_topics"] = [{
        "label": "setup steps",
        "coverage": "missing",
        "priority": "high",
        "competitor_prevalence": 0.9,
        "best_competitor_rank": 1,
        "examples": [{"url": "https://a.example/guide", "rank": 1, "paragraph": "Steps."}],
    }]

    actions = _action_points_for_analysis({"url": "https://shop.example/widget"}, analysis)

    assert actions[0]["type"] == "retarget_or_new_page"


def test_winnability_cache_round_trip_feeds_keyword_selection(tmp_path: Path) -> None:
    _save_winnability_cache(tmp_path, [
        {"url": "https://example.com/p", "keyword": "Hard KW", "winnability_band": "hard"},
        {"url": "https://example.com/p", "keyword": "no band yet", "winnability_band": "unknown"},
    ])

    lookup = _load_winnability_cache(tmp_path)
    page = PageInfo("https://example.com/p", "Widget Page", "", "", 100, "en")
    metrics = {"impressions": 500, "position": 8}
    cached_row = _keyword_row(page, "hard kw", "gsc", row=metrics, winnability_lookup=lookup)
    fresh_row = _keyword_row(page, "no band yet", "gsc", row=metrics, winnability_lookup=lookup)

    assert cached_row["winnability_band"] == "hard"
    assert cached_row["winnability_factor"] == 0.6
    assert "winnability_band" not in fresh_row  # unknown bands are not persisted
    assert _keyword_priority(cached_row) < _keyword_priority({**cached_row, "winnability_band": "winnable", "winnability_factor": 1.0})


def test_serp_gap_attaches_page_content_briefs() -> None:
    page = {"url": "https://example.com/live-chat", "title": "Live Chat"}
    analysis = {
        "status": "ok",
        "query": "live chat software",
        "keyword": {"keyword": "live chat software", "impressions": 1200},
        "missing_topics": [
            {
                "label": "implementation checklist",
                "coverage": "missing",
                "priority": "critical",
                "competitor_coverage": 4,
                "competitor_prevalence": 0.8,
                "best_competitor_rank": 1,
                "our_best_similarity": 0.3,
                "examples": [{"url": "https://competitor.example/setup", "rank": 1, "paragraph": "Setup details."}],
            }
        ],
        "weak_topics": [],
        "off_intent_paragraphs": [],
    }
    pages = [{**page, "analyses": [analysis]}]

    aggregate = _attach_action_points(pages)

    assert aggregate
    assert pages[0]["content_brief"]["target_url"] == page["url"]
    assert pages[0]["content_brief"]["primary_keywords"] == ["live chat software"]
    assert pages[0]["content_brief"]["paragraph_rules"] == _editorial_guidelines()["paragraph_rules"]
    assert pages[0]["content_brief"]["next_actions"][0]["task_summary"] == "Create a focused section about implementation checklist."


def test_serp_gap_action_csv_rows_include_editorial_brief_fields() -> None:
    page = {"url": "https://example.com/live-chat", "title": "Live Chat"}
    analysis = {
        "status": "ok",
        "query": "live chat software",
        "keyword": {"keyword": "live chat software", "impressions": 1200},
        "missing_topics": [
            {
                "label": "pricing details",
                "coverage": "missing",
                "priority": "high",
                "competitor_coverage": 3,
                "competitor_prevalence": 0.75,
                "best_competitor_rank": 2,
                "our_best_similarity": 0.41,
                "examples": [{"url": "https://competitor.example/pricing", "rank": 2, "paragraph": "Pricing details."}],
            }
        ],
        "weak_topics": [],
        "off_intent_paragraphs": [],
    }
    pages = [{**page, "analyses": [analysis]}]
    actions = _attach_action_points(pages)

    rows = _action_csv_rows({"action_points": actions})

    assert rows[0]["target_url"] == page["url"]
    assert rows[0]["recommended_format"] == "answer block plus comparison table"
    assert "direct answer" in rows[0]["paragraph_plan"]
    assert "The section clearly covers" in rows[0]["acceptance_criteria"]
    assert rows[0]["ai_agent_prompt"]


def test_serp_gap_builds_page_difference_visual_payload() -> None:
    page = PageInfo(
        url="https://example.com/live-chat",
        title="Live Chat",
        description="",
        section="/",
        word_count=320,
        language="en",
    )
    own_ext = ExtractedPage(
        url=page.url,
        title="Live Chat",
        description="",
        body="",
        word_count=320,
        language="en",
        h1="Live Chat",
        headers_rich=[
            {"level": 1, "text": "Live Chat"},
            {"level": 2, "text": "Support workflows"},
        ],
        paragraphs=["Live chat overview.", "Support workflow details."],
        content_sequence=[
            {"order": 0, "entity_type": "h1", "level": 1, "text": "Live Chat"},
            {"order": 1, "entity_type": "paragraph", "level": 0, "text": "Live chat overview."},
            {"order": 2, "entity_type": "h2", "level": 2, "text": "Support workflows"},
            {"order": 3, "entity_type": "paragraph", "level": 0, "text": "Support workflow details."},
        ],
    )
    competitor_one = CompetitorPage(
        target=CompetitiveTarget("live chat software", "https://competitor.example/live-chat", rank=1),
        title="Competitor Live Chat",
        paragraphs=["Pricing details", "Implementation checklist", "Routing rules"],
        paragraph_embeddings=np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        structural_gaps=[],
        answerability=0.0,
        paragraph_count=3,
        h1="Competitor Live Chat",
        headers_rich=[
            {"level": 1, "text": "Competitor Live Chat"},
            {"level": 2, "text": "Pricing"},
            {"level": 2, "text": "Implementation"},
        ],
        content_sequence=[
            {"order": 0, "entity_type": "h1", "level": 1, "text": "Competitor Live Chat"},
            {"order": 1, "entity_type": "h2", "level": 2, "text": "Pricing"},
            {"order": 2, "entity_type": "paragraph", "level": 0, "text": "Pricing details"},
            {"order": 3, "entity_type": "h2", "level": 2, "text": "Implementation"},
            {"order": 4, "entity_type": "paragraph", "level": 0, "text": "Implementation checklist"},
        ],
    )
    competitor_two = CompetitorPage(
        target=CompetitiveTarget("live chat software", "https://another.example/chat", rank=2),
        title="Another Chat",
        paragraphs=["Automation routing", "Reporting"],
        paragraph_embeddings=np.array(
            [
                [0.0, 1.0, 0.0],
                [0.2, 0.0, 0.98],
            ],
            dtype=np.float32,
        ),
        structural_gaps=[],
        answerability=0.0,
        paragraph_count=2,
        h1="Another Chat",
        headers_rich=[{"level": 2, "text": "Automation"}],
        content_sequence=[
            {"order": 0, "entity_type": "h2", "level": 2, "text": "Automation"},
            {"order": 1, "entity_type": "paragraph", "level": 0, "text": "Automation routing"},
            {"order": 2, "entity_type": "paragraph", "level": 0, "text": "Reporting"},
        ],
    )
    topics = [
        {
            "label": "pricing and implementation details",
            "coverage": "missing",
            "priority": "high",
            "competitor_urls": [competitor_one.target.competitor_url],
            "competitor_coverage": 1,
            "competitor_prevalence": 0.5,
            "best_competitor_rank": 1,
            "competitor_paragraphs": 3,
            "our_best_similarity": 0.31,
            "examples": [
                {
                    "url": competitor_one.target.competitor_url,
                    "rank": 1,
                    "paragraph": "Pricing details and implementation checklist",
                }
            ],
        },
        {
            "label": "routing workflow",
            "coverage": "partial",
            "priority": "high",
            "competitor_urls": [competitor_one.target.competitor_url, competitor_two.target.competitor_url],
            "competitor_coverage": 2,
            "competitor_prevalence": 1.0,
            "best_competitor_rank": 1,
            "competitor_paragraphs": 4,
            "our_best_similarity": 0.67,
            "examples": [
                {
                    "url": competitor_one.target.competitor_url,
                    "rank": 1,
                    "paragraph": "Routing rules",
                },
                {
                    "url": competitor_two.target.competitor_url,
                    "rank": 2,
                    "paragraph": "Automation routing",
                },
            ],
        },
        {
            "label": "live chat overview",
            "coverage": "covered",
            "priority": "covered",
            "competitor_urls": [competitor_two.target.competitor_url],
            "competitor_coverage": 1,
            "competitor_prevalence": 0.5,
            "best_competitor_rank": 2,
            "competitor_paragraphs": 1,
            "our_best_similarity": 0.84,
        },
    ]

    comparison = _content_comparison(page, own_ext, own_ext.paragraphs, [competitor_one, competitor_two], topics)
    matrix = _topic_coverage_matrix(page.url, [competitor_one, competitor_two], topics)
    paragraph_heatmap = _paragraph_match_heatmap(
        own_ext.paragraphs,
        np.array([[1.0, 0.0, 0.0], [0.0, 0.2, 0.98]], dtype=np.float32),
        [competitor_one, competitor_two],
    )
    content_path = _content_order_path(
        "live chat software",
        own_ext,
        [competitor_one, competitor_two],
        _StaticEmbedder(),
        max_competitors=1,
        max_items_per_page=1,
    )

    assert comparison["summary"]["missing_topics"] == 1
    assert comparison["ours"]["topic_count"] == 1
    assert comparison["ours"]["partial_topics"] == 1
    assert comparison["competitors"][0]["domain"] == "competitor.example"
    assert comparison["competitors"][0]["missing_topics_covered"] == 1
    assert matrix["columns"][0]["source"] == "ours"
    assert len(matrix["columns"]) == 3
    assert matrix["rows"][0]["label"] == "routing workflow"
    assert matrix["rows"][0]["cells"][0]["status"] == "partial"
    assert matrix["rows"][0]["cells"][1]["status"] == "covered"
    assert matrix["rows"][0]["examples"][0]["paragraph"] == "Routing rules"
    assert matrix["rows"][0]["examples"][0]["domain"] == "competitor.example"
    assert len(paragraph_heatmap["columns"]) == 2
    assert paragraph_heatmap["rows"][0]["cells"][0]["status"] == "strong"
    assert paragraph_heatmap["rows"][0]["cells"][0]["rank_impact"] == 1.0
    assert paragraph_heatmap["rows"][0]["max_rank_impact"] == 1.0
    assert paragraph_heatmap["rows"][1]["cells"][0]["paragraph"] == "Routing rules"
    assert content_path["summary"]["page_count"] == 2
    assert content_path["summary"]["item_count"] == 2
    assert content_path["items"][0]["source"] == "ours"
    assert "cluster" in content_path["items"][0]
    assert isinstance(content_path["unmatched_clusters_by_url"], list)


def test_serp_gap_todo_markdown_is_actionable_and_deduped() -> None:
    payload = {
        "status": "ok",
        "domain": "example.com",
        "summary": {"pages_analyzed": 1, "keywords_selected": 1, "action_points": 2},
        "editorial_guidelines": {
            "paragraph_rules": [
                "One paragraph should answer one concrete question or make one concrete point.",
                "Every paragraph should contain at least one useful detail.",
            ]
        },
        "pages": [
            {
                "url": "https://example.com/live-chat",
                "title": "Live Chat",
                "action_points": [
                    {
                        "priority": "high",
                        "type": "add_topic",
                        "keyword": "live chat software",
                        "topic": "pricing details",
                        "task_summary": "Create a focused pricing section.",
                        "instruction": "Add a direct answer about pricing details.",
                        "placement": "Add near pricing heading.",
                        "acceptance_criteria": ["The section clearly covers pricing details."],
                        "evidence": {
                            "competitor_coverage": 3,
                            "best_competitor_rank": 2,
                            "our_best_similarity": 0.41,
                            "example_url": "https://competitor.example/pricing",
                        },
                    },
                    {
                        "priority": "high",
                        "type": "add_topic",
                        "keyword": "live chat software",
                        "topic": "pricing details",
                        "task_summary": "Create a focused pricing section.",
                        "instruction": "Add a direct answer about pricing details.",
                        "placement": "Add near pricing heading.",
                        "acceptance_criteria": ["The section clearly covers pricing details."],
                        "evidence": {"competitor_coverage": 3},
                    },
                    {
                        "priority": "medium",
                        "type": "review_paragraph",
                        "keyword": "live chat software",
                        "task_summary": "Review paragraph 4 for intent drift or filler.",
                        "instruction": "Rewrite or remove paragraph 4.",
                    },
                ],
                "analyses": [
                    {
                        "query": "live chat software",
                        "keyword": {"keyword": "live chat software"},
                        "visual_summary": ["1 SERP topic group is absent from the target page."],
                        "paragraph_match_heatmap": {
                            "rows": [
                                {
                                    "paragraph_index": 0,
                                    "paragraph": "Generic intro paragraph.",
                                    "max_similarity": 0.5,
                                }
                            ]
                        },
                    }
                ],
            }
        ],
    }

    markdown = _todo_markdown(payload)

    assert "# SERP Gap TODO" in markdown
    assert "- Scope: 1 page(s), 1 keyword(s), 1 content task(s)" in markdown
    assert "### Ordered Content Tasks" in markdown
    assert markdown.count("Create a focused pricing section.") == 1
    assert "Review paragraph 4 for intent drift or filler." not in markdown
    assert "P1 (0.50 best SERP paragraph match" in markdown
    assert "Do not copy competitor wording." in markdown


def test_serp_gap_todo_markdown_embeds_ai_editor_brief_without_duplicate_lines() -> None:
    payload = {
        "status": "ok",
        "domain": "example.com",
        "summary": {"pages_analyzed": 1, "keywords_selected": 1, "action_points": 0},
        "pages": [
            {
                "url": "https://example.com/support",
                "title": "Support",
                "ai_editor_brief": {
                    "status": "ok",
                    "provider": "openrouter",
                    "model": "deepseek/deepseek-v4-pro",
                    "cache_status": "miss",
                    "markdown": "\n".join([
                        "# AI Agent TODO",
                        "## Evidence",
                        "- Demand metrics absent; do not estimate search volume.",
                        "- Demand metrics absent; do not estimate search volume.",
                        "## Paragraph Decisions",
                        "- P1: rewrite to answer the user intent directly.",
                    ]),
                },
                "analyses": [],
            }
        ],
    }

    markdown = _todo_markdown(payload)

    assert "### AI Agent TODO" in markdown
    assert "provider: openrouter" in markdown
    assert markdown.count("Demand metrics absent; do not estimate search volume.") == 1
    assert "- P1: rewrite to answer the user intent directly." in markdown


def test_ai_agent_keyword_parser_and_editor_prompt_are_specific() -> None:
    keywords = parse_keyword_candidates(
        '{"keywords":[{"keyword":"ai customer support paradox","intent":"matches article"},'
        '{"keyword":"ai support handoff","priority":2}]}'
    )
    messages = build_editor_brief_messages({
        "url": "https://example.com/blog/ai-support-paradox/",
        "title": "AI Support Paradox",
        "keywords": [{"keyword": "ai customer support paradox", "impressions": 0}],
        "action_points": [
            {
                "priority": "high",
                "type": "add_topic",
                "keyword": "ai customer support paradox",
                "topic": "human handoff",
                "task_summary": "Add handoff section.",
            }
        ],
        "analyses": [],
    })

    assert keywords == ["ai customer support paradox", "ai support handoff"]
    prompt = "\n".join(message["content"] for message in messages)
    assert "keep, rewrite, move, merge, or remove" in prompt
    assert "Do not copy competitor wording" in prompt
    assert "demand metrics absent" in prompt
    assert "## Final Article Draft" in prompt
    assert "Harnext AI coding/content agent" in prompt


def test_ai_agent_keyword_parser_reads_fenced_json_without_code_tokens() -> None:
    text = """```json
{
  "keywords": [
    {"keyword": "agent ticket scope", "priority": 1},
    {"keyword": "what is agent ticket scope", "priority": 2},
    {"keyword": "ticket access control for support agents", "priority": 3}
  ]
}
```"""

    keywords = parse_keyword_candidates(text, limit=3)

    assert keywords == [
        "agent ticket scope",
        "what is agent ticket scope",
        "ticket access control for support agents",
    ]
    assert "json" not in keywords
    assert "{" not in keywords
    assert "[" not in keywords


def test_ai_agent_language_parser_reads_fenced_json() -> None:
    detected = parse_language_detection(
        """```json
{"language_code":"sk","language_name":"Slovak","confidence":0.93,"reason":"Main paragraphs are Slovak."}
```"""
    )

    assert detected["language_code"] == "sk"
    assert detected["language_name"] == "Slovak"
    assert detected["confidence"] == 0.93


def test_serp_gap_uses_ai_agent_to_detect_missing_language(tmp_path: Path, monkeypatch) -> None:
    class _Client:
        provider = "harnext"

    def fake_cached_completion(cache_dir, *, kind, messages, client, model, refresh, temperature, timeout):
        prompt = "\n".join(message["content"] for message in messages)
        assert "Detect the best Google SERP language code" in prompt
        return AgentCompletion(
            text='{"language_code":"sk","language_name":"Slovak","confidence":0.91,"reason":"Visible body copy is Slovak."}',
            provider="harnext",
            model=model,
            cache_status="miss",
        )

    monkeypatch.setattr("site_audit.serp_gap.build_agent_client", lambda _provider: _Client())
    monkeypatch.setattr("site_audit.serp_gap.cached_completion", fake_cached_completion)

    def fake_fetch_sk(_url, _cache, refresh=False):
        return ExtractedPage(
            url="https://www.example.com/sk/podpora/",
            title="Podpora pre zákazníkov",
            description="",
            body="Ako nastaviť podporu pre zákazníkov.",
            word_count=80,
            language="sk",
            h1="Podpora",
            paragraphs=["Ako nastaviť podporu pre zákazníkov."],
        )

    monkeypatch.setattr("site_audit.serp_gap._fetch_and_extract", fake_fetch_sk)
    config = SerpGapConfig(domain="example.com", language=None, ai_agent=True, ai_agent_model="test-model")
    state = {
        "status": "ready",
        "language_prompts": 0,
        "cache_hits": 0,
        "detected_language": "",
        "errors": [],
        "notes": [],
    }

    info = _resolve_serp_language(
        [PageInfo("https://www.example.com/sk/podpora/", "Podpora", "", "sk", 80, None)],
        {},
        HttpCache(tmp_path / "http.sqlite"),
        tmp_path,
        config,
        state,
    )

    assert info["status"] == "detected"
    assert info["language"] == "sk"
    assert info["source"] == "harnext"
    assert config.language == "sk"
    assert state["language_prompts"] == 1
    assert state["detected_language"] == "sk"


def test_serp_gap_language_detection_falls_back_to_page_language(tmp_path: Path, monkeypatch) -> None:
    def fake_fetch_cs(_url, _cache, refresh=False):
        return ExtractedPage(
            url="https://www.example.com/cs/podpora/",
            title="Zákaznická podpora",
            description="",
            body="Jak nastavit podporu.",
            word_count=80,
            language="cs-CZ",
            h1="Podpora",
            paragraphs=["Jak nastavit podporu."],
        )

    monkeypatch.setattr("site_audit.serp_gap._fetch_and_extract", fake_fetch_cs)
    config = SerpGapConfig(domain="example.com", language=None, ai_agent=True)
    state = {"status": "missing_harnext", "detected_language": "", "errors": [], "notes": []}

    info = _resolve_serp_language(
        [PageInfo("https://www.example.com/cs/podpora/", "Podpora", "", "cs", 80, "cs-CZ")],
        {},
        HttpCache(tmp_path / "http.sqlite"),
        tmp_path,
        config,
        state,
    )

    assert info["status"] == "fallback"
    assert info["language"] == "cs"
    assert config.language == "cs"


def test_harnext_status_reports_missing_cli(monkeypatch) -> None:
    import harnext_sdk

    def fail(_path):
        raise RuntimeError("missing cli")

    monkeypatch.setattr(harnext_sdk, "resolve_cli_invocation", fail)

    ok, detail = harnext_status()

    assert ok is False
    assert "npm install -g harnext" in detail


def test_serp_gap_enriches_manual_keywords_from_ahrefs_metrics() -> None:
    payload = {
        "meta": {"provider": "ahrefs"},
        "organic_keywords": [
            {
                "keyword": "live chat software",
                "matched_url": "https://www.example.com/features/live-chat/",
                "provider": "ahrefs",
                "position": 4,
                "traffic": 25,
                "volume": 1200,
            }
        ],
    }
    rows = [
        {
            "url": "https://example.com/different-page",
            "keyword": "live chat software",
            "source": "manual",
            "position": 0,
            "impressions": 0,
            "clicks": 0,
            "traffic": 0,
            "volume": 0,
        }
    ]

    _enrich_keyword_rows(rows, _keyword_metrics_lookup(payload))

    assert rows[0]["position"] == 4
    assert rows[0]["traffic"] == 25
    assert rows[0]["volume"] == 1200
    assert rows[0]["metrics_source"] == "ahrefs"
    assert rows[0]["metrics_url"] == "https://www.example.com/features/live-chat/"


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
    assert rows[0]["volume"] == 0
    assert [row["url"] for row in rows] == [
        "https://competitor-one.com/live-chat",
        "https://competitor-two.com/chat",
        "https://www.example.com/live-chat/",
    ]
    assert rows[-1]["is_selected_domain"] is True


def test_serp_gap_enriches_serp_domains_with_ahrefs_domain_rating(tmp_path: Path, monkeypatch) -> None:
    def fake_fetch(targets, cache_dir, *, refresh=False):
        assert cache_dir == tmp_path
        assert refresh is True
        return {
            "competitor-one.com": {
                "domain_rating": 71.2,
                "status": "ok",
                "source": "ahrefs_public_domain_rating_free",
                "license": "http://license.example",
                "attribution": "Domain Rating by Ahrefs",
            },
            "example.com": {
                "domain_rating": 42.0,
                "status": "ok",
                "source": "ahrefs_public_domain_rating_free",
                "license": "http://license.example",
                "attribution": "Domain Rating by Ahrefs",
            },
        }

    monkeypatch.setattr("site_audit.serp_gap.fetch_domain_ratings_free", fake_fetch)
    rankings = {
        "https://competitor-one.com/a": {"url": "https://competitor-one.com/a", "domain": "competitor-one.com"},
    }
    page_results = [{
        "analyses": [{
            "competitor_pages": [{"url": "https://competitor-one.com/a", "domain": "competitor-one.com"}],
        }],
    }]
    overview_rows = [{"url": "https://www.example.com/", "domain": "www.example.com"}]

    meta = _enrich_serp_domain_ratings("example.com", rankings, page_results, overview_rows, tmp_path, refresh=True)

    assert meta["domains_enriched"] == 2
    assert meta["own_domain_rating"] == 42.0
    assert rankings["https://competitor-one.com/a"]["domain_rating"] == 71.2
    assert page_results[0]["analyses"][0]["competitor_pages"][0]["domain_rating"] == 71.2
    assert overview_rows[0]["domain_rating"] == 42.0


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

    payload = run(
        SerpGapConfig(
            domain="example.com",
            projects_root=tmp_path,
            url_include_patterns=["/features/*"],
            dry_run=True,
        )
    )

    html = (Path(payload["summary"]["report_dir"]) / "index.html").read_text(encoding="utf-8")

    assert "Semantic Scatterplot" in html
    assert "Semantic Clusters" in html
    assert "Topic Relations" in html
    assert "SERP Content Task Board" in html
    assert "Page Content Briefs" in html
    assert "AI Paragraph Rules" in html
    assert "Content briefs" in html
    assert "contentBriefsSection" in html
    assert "pageBriefCard" in html
    assert "diagnostic-details" in html
    assert "Raw semantic evidence for this keyword" in html
    assert "Why These Edits Matter" in html
    assert "Top ranking page differences" in html
    assert "SERP rank vs content coverage" in html
    assert "Topic coverage heatmap" in html
    assert "content-comparison" in html
    assert "coverage-heatmap" in html
    assert "contentComparisonRows" in html
    assert "topicCoverageHeatmap" in html
    assert "topicExamplesForRow" in html
    assert "topicEvidenceHtml" in html
    assert "Ranking paragraphs missing or weak" in html
    assert "topic-evidence" in html
    assert "topic-snippet" in html
    assert "paragraphMatchHeatmap" in html
    assert "Content order semantic path" in html
    assert "contentOrderPathSection" in html
    assert "contentPathSvg" in html
    assert "contentPathParallelCoordinates" in html
    assert "contentPathUnmatchedClusters" in html
    assert "content-path-chart" in html
    assert "content-path-parallel" in html
    assert "parallel-topic-line" in html
    assert "data-path-cluster" in html
    assert "bindContentPathInteractions" in html
    assert "content-path-wrap.has-active" in html
    assert "Hover, focus, or click a line or node" in html
    assert "content_order_path" in html
    assert "unmatched_clusters_by_url" in html
    assert "path-unmatched-card" in html
    assert "Parallel coordinates compare the order of similar topic clusters" in html
    assert "Axes are ordered by top-10 SERP position" in html
    assert "URL axes are ordered by SERP top-10 position" in html
    assert "URL-only clusters" in html
    assert "unique clusters not matched with other URLs" in html
    assert "collapsiblePanel" in html
    assert "collapsible-panel" in html
    assert "Keyword Content Actions" in html
    assert "collapsible-state" in html
    assert "orderedPathPages" in html
    assert "ordered by SERP top-10 position" in html
    assert "Keyword-to-Paragraph Coverage by URL" in html
    assert "keywordParagraphRidgeline" in html
    assert "keyword_url_ridges" in html
    assert "keyword-ridge-chart" in html
    assert "URL Demand Metrics" in html
    assert "urlDemandMetricsSection" in html
    assert "url-demand-table" in html
    assert "rank impact proxy" in html
    assert "semanticScatterSection" in html
    assert "keywordChartsSection" in html
    assert "visualComparisonSection" in html
    assert "visual_summary" in html
    assert "topic_coverage_matrix" in html
    assert "content_comparison" in html
    assert "paragraph_match_heatmap" in html
    assert "heatmap-cell" in html
    assert "paragraph-heatmap" in html
    assert "paragraph-row" in html
    assert "comparison-row" in html
    assert "Acceptance criteria" in html
    assert "AI agent prompt" in html
    assert "what to change, where to place it, how paragraphs should be structured" in html
    assert "Use filters to isolate entity types and domains" in html
    assert "Repeated winners show which URLs and intents Google currently rewards" in html
    assert "Vector-space chart for keywords" in html
    assert "Vector space for keyword, target page, competitor titles, headings, and paragraphs" in html
    assert "Edit these after reviewing the charts" in html
    # The per-keyword cluster panel was removed (one cluster view lives in the overview);
    # the overview aggregate cluster section must remain.
    assert "competitors cover nearby themes more deeply" not in html
    assert "Aggregate Semantic Clusters" in html
    assert "Review candidates for intent drift, thinness, or filler" in html
    assert "Charts first, then tasks" in html
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
    assert "Aggregate Semantic Clusters" in html
    assert "Broad topic groups across selected keywords and processed URLs" in html
    assert "Topic Traffic Impact" in html
    assert "clusterImpactChart" in html
    assert "demand-weighted keyword centroid" in html
    assert "keyword_centroid" in html
    assert "keyword centroid" in html
    assert "Keyword Frequency Analysis" in html
    assert "Weighted Content Keyword Cloud" in html
    assert "Title Keywords" in html
    assert "H1 Keywords" in html
    assert "H2-H6 Keywords" in html
    assert "Paragraph Keywords" in html
    assert "keywordFrequencySection" in html
    assert "frequencyTokens" in html
    assert "wordCloud" in html
    assert "Keyword Metrics From APIs" in html
    assert "keywordMetricsTable" in html
    assert "API metrics source" in html
    assert "No API metric match" in html
    assert "Content Action Plan For AI Agents" in html
    assert "Keyword Content Actions" in html
    assert "action_points" in html
    assert "actionList" in html
    assert "Prioritized instructions for editors or an AI agent" in html
    assert "SERP URLs" in html
    assert "urlKeywordTable" in html
    assert "url-keyword-table" in html
    assert "Source pos" in html
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
    assert "keyword diamond size = impressions or Ahrefs volume" in html
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


def _norm_rows(rows):
    import numpy as np
    arr = np.asarray(rows, dtype=np.float32)
    denom = np.linalg.norm(arr, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return arr / denom


def test_serp_features_parses_serper_payload() -> None:
    from site_audit.serp_gap import _serp_features

    payload = {
        "meta": {"provider": "serper", "status": "ok"},
        "raw": {
            "peopleAlsoAsk": [
                {"question": "How much does X cost?", "snippet": "s", "title": "t", "link": "https://a"},
            ],
            "relatedSearches": [{"query": "x pricing"}],
            "answerBox": {"title": "t", "answer": "a", "link": "https://b"},
        },
    }
    features = _serp_features(payload)
    assert features["people_also_ask"] == [
        {"question": "How much does X cost?", "snippet": "s", "url": "https://a", "title": "t"}
    ]
    assert features["related_searches"] == ["x pricing"]
    assert features["answer_box"] == {"title": "t", "answer": "a", "url": "https://b", "format": "paragraph", "word_count": 1}
    assert features["ai_overview"] is None


def test_serp_features_handles_missing_blocks() -> None:
    from site_audit.serp_gap import _serp_features

    features = _serp_features({"meta": {"provider": "serper"}, "raw": {}})
    assert features == {"people_also_ask": [], "related_searches": [], "answer_box": {}, "ai_overview": None}


def test_serp_features_parses_dataforseo_payload() -> None:
    from site_audit.serp_gap import _serp_features

    payload = {
        "meta": {"provider": "dataforseo", "status": "ok"},
        "raw": {
            "tasks": [{"result": [{"items": [
                {"type": "people_also_ask", "items": [
                    {"title": "What is X?", "expanded_element": [{"description": "d", "url": "https://c"}]},
                ]},
                {"type": "related_searches", "items": ["x alternatives"]},
                {"type": "featured_snippet", "title": "ft", "description": "fd", "url": "https://d"},
            ]}]}],
        },
    }
    features = _serp_features(payload)
    assert features["people_also_ask"][0]["question"] == "What is X?"
    assert features["people_also_ask"][0]["snippet"] == "d"
    assert features["related_searches"] == ["x alternatives"]
    assert features["answer_box"]["answer"] == "fd"
    assert features["answer_box"]["format"] == "paragraph"


def test_serp_features_parses_serper_ai_overview() -> None:
    from site_audit.serp_gap import _serp_features

    payload = {
        "meta": {"provider": "serper", "status": "ok", "domain": "ours.example"},
        "raw": {
            "organic": [],
            "aiOverview": {
                "text": "Overview answer.",
                "sources": [
                    {"link": "https://ours.example/guide"},
                    {"link": "https://competitor.example/post"},
                ],
            },
        },
    }

    features = _serp_features(payload)

    assert features["ai_overview"] == {
        "present": True,
        "cites_us": True,
        "cited_domains": ["ours.example", "competitor.example"],
    }


def test_serp_features_parses_dataforseo_ai_overview() -> None:
    from site_audit.serp_gap import _serp_features

    payload = {
        "meta": {"provider": "dataforseo", "status": "ok"},
        "raw": {
            "tasks": [{"result": [{"items": [
                {
                    "type": "ai_overview",
                    "items": [{"url": "https://source-a.example/a"}, {"url": "https://source-b.example/b"}],
                },
            ]}]}],
        },
    }

    features = _serp_features(payload, own_domain="ours.example")

    assert features["ai_overview"] == {
        "present": True,
        "cites_us": False,
        "cited_domains": ["source-a.example", "source-b.example"],
    }


def test_paa_coverage_classifies_thresholds() -> None:
    import numpy as np
    from site_audit.serp_gap import _paa_coverage

    class StubEmbedder:
        def encode(self, texts, batch_size=32, show_progress=False):
            vectors = []
            for text in texts:
                if "covered" in text.lower():
                    vectors.append([1.0, 0.0])
                else:
                    vectors.append([0.0, 1.0])
            return _norm_rows(vectors)

    own_paragraphs = ["This paragraph is about the covered topic in detail."]
    own_embeddings = _norm_rows([[1.0, 0.0]])
    features = {
        "people_also_ask": [
            {"question": "Is the covered topic explained?"},
            {"question": "Something entirely different?"},
        ]
    }
    rows = _paa_coverage(features, own_paragraphs, own_embeddings, StubEmbedder())
    assert rows[0]["status"] == "covered"
    assert rows[0]["best_paragraph_index"] == 0
    assert rows[1]["status"] == "missing"
    assert 0.0 <= rows[1]["best_similarity"] <= 1.0


def test_paa_coverage_without_own_paragraphs_marks_missing() -> None:
    import numpy as np
    from site_audit.serp_gap import _paa_coverage

    rows = _paa_coverage(
        {"people_also_ask": [{"question": "Anything?"}]},
        [],
        np.zeros((0, 0), dtype=np.float32),
        embedder=None,
    )
    assert rows == [{"question": "Anything?", "status": "missing", "best_similarity": 0.0, "best_paragraph_index": None, "best_paragraph": ""}]


def test_editor_payload_sorts_paragraph_review_by_weakness() -> None:
    from site_audit.ai_agent import _editor_prompt_payload

    page = {
        "url": "https://ours.example/page",
        "title": "T",
        "h1": "H",
        "own_content": {"headings": [], "paragraphs": [], "word_count": 10},
        "analyses": [{
            "keyword": {"keyword": "kw"},
            "paragraph_match_heatmap": {"rows": [
                {"paragraph_index": 0, "max_similarity": 0.9, "status": "strong", "paragraph": "a",
                 "cells": [{"url": "https://c1", "similarity": 0.9, "rank": 1, "paragraph": "cp1"}]},
                {"paragraph_index": 1, "max_similarity": 0.2, "status": "weak", "paragraph": "b",
                 "cells": [{"url": "https://c2", "similarity": 0.2, "rank": 2, "paragraph": "cp2"},
                           {"url": "https://c3", "similarity": 0.1, "rank": 3, "paragraph": "cp3"}]},
                {"paragraph_index": 2, "max_similarity": 0.5, "status": "weak", "paragraph": "c",
                 "cells": []},
            ]},
        }],
    }
    payload = _editor_prompt_payload(page)
    review = payload["analyses"][0]["paragraph_review"]
    assert [row["paragraph_index"] for row in review] == [1, 2, 0]
    assert review[0]["best_competitor"]["url"] == "https://c2"
    assert review[1]["best_competitor"] == {}


def test_editor_payload_includes_new_evidence_keys() -> None:
    from site_audit.ai_agent import _editor_prompt_payload

    page = {
        "url": "https://ours.example/page",
        "own_content": {"headings": [{"order": 0, "level": 2, "text": "H2"}], "paragraphs": [{"index": 0, "word_count": 3, "text": "x y z"}], "word_count": 3},
        "analyses": [{
            "keyword": {"keyword": "kw"},
            "content_comparison": {"benchmark": {"median_competitor_paragraphs": 20}, "ours": {"paragraph_count": 5, "word_count": 100, "heading_count": 2, "h2_h3_count": 2, "coverage_ratio": 0.4}},
            "structural_patterns": [{"signal": "s", "competitors": 3, "advice": "a"}],
            "paa_coverage": [{"question": "Q?", "status": "missing", "best_similarity": 0.1}],
            "serp_features": {"related_searches": ["alt"]},
            "covered_topics": [{"label": "done topic"}],
            "content_order_path": {"summary": {"order_score": 1.0}, "missing_clusters": [{"label": "m", "competitor_pages": 2, "sample_text": "s"}], "deviations": []},
        }],
    }
    payload = _editor_prompt_payload(page)
    assert payload["own_page"]["paragraphs"][0]["text"] == "x y z"
    analysis = payload["analyses"][0]
    assert analysis["benchmark"]["median_competitor_paragraphs"] == 20
    assert analysis["our_profile"]["paragraph_count"] == 5
    assert analysis["structural_patterns"][0]["signal"] == "s"
    assert analysis["serp_features"]["people_also_ask"][0]["question"] == "Q?"
    assert analysis["serp_features"]["related_searches"] == ["alt"]
    assert analysis["covered_topics"] == ["done topic"]
    assert analysis["content_order"]["missing_clusters"][0]["label"] == "m"


def test_shrink_editor_payload_respects_budget() -> None:
    from site_audit.ai_agent import _shrink_editor_payload
    import json as _json

    payload = {
        "own_page": {"paragraphs": [{"index": i, "text": "word " * 200} for i in range(60)]},
        "analyses": [{
            "paragraph_review": [{"paragraph_index": i, "paragraph": "x" * 400, "best_competitor": {"paragraph": "y" * 320}} for i in range(25)],
            "topics": [{"example_paragraph": "z" * 320} for _ in range(12)],
            "content_order": {"missing_clusters": [{"sample_text": "s" * 200} for _ in range(8)]},
        }],
    }
    out = _shrink_editor_payload(payload, max_chars=20_000)
    assert len(_json.dumps(out, ensure_ascii=False)) <= 20_000
    assert len(out["analyses"][0]["paragraph_review"]) <= 15


def _analysis_base(keyword="widget tool"):
    return {
        "status": "ok",
        "keyword": {"keyword": keyword, "impressions": 100},
        "competitor_pages": [],
        "missing_topics": [],
        "weak_topics": [],
        "off_intent_paragraphs": [],
        "own_paragraphs_to_review": [],
    }


def test_title_gap_action_triggers_on_missing_keyword() -> None:
    from site_audit.serp_gap import _title_gap_action

    page = {"url": "https://ours.example/p", "title": "Our Generic Landing Page Headline", "h1": "Hello"}
    analysis = _analysis_base()
    analysis["competitor_pages"] = [
        {"rank": i, "title": f"Best widget tool option {i}", "error": ""} for i in range(1, 6)
    ]
    action = _title_gap_action(page, analysis, 1)
    assert action is not None
    assert action["type"] == "rewrite_title"
    assert action["priority"] == "high"
    assert len(action["evidence"]["competitor_titles"]) == 5

    page_with_keyword = {"url": "https://ours.example/p", "title": "The Widget Tool for Everyone Everywhere", "h1": "Hello"}
    assert _title_gap_action(page_with_keyword, analysis, 1) is None


def test_depth_action_uses_benchmark() -> None:
    from site_audit.serp_gap import _depth_action

    page = {"url": "https://ours.example/p"}
    analysis = _analysis_base()
    analysis["content_comparison"] = {
        "ours": {"paragraph_count": 5, "heading_count": 2},
        "benchmark": {"median_competitor_paragraphs": 20, "median_competitor_headings": 9},
    }
    action = _depth_action(page, analysis, 1)
    assert action is not None
    assert action["type"] == "expand_depth"
    assert "15" in action["instruction"]

    analysis["content_comparison"]["ours"]["paragraph_count"] = 18
    assert _depth_action(page, analysis, 1) is None


def test_structural_and_paa_actions_emitted() -> None:
    from site_audit.serp_gap import _action_points_for_analysis

    page = {"url": "https://ours.example/p", "title": "T", "h1": "H"}
    analysis = _analysis_base()
    analysis["structural_patterns"] = [
        {"signal": "Comparison / data tables", "competitors": 4, "advice": "Add a table.", "ours": 0, "max_theirs": 3},
        {"signal": "only one competitor", "competitors": 1, "advice": "x", "ours": 0, "max_theirs": 1},
    ]
    analysis["serp_features"] = {"people_also_ask": [{"question": "What is a widget tool?"}]}
    analysis["paa_coverage"] = [
        {"question": "What is a widget tool?", "status": "missing", "best_similarity": 0.2},
        {"question": "Covered question?", "status": "covered", "best_similarity": 0.9},
    ]
    actions = _action_points_for_analysis(page, analysis)
    types = {a["type"] for a in actions}
    assert "structural" in types
    assert "answer_paa" in types
    structural = next(a for a in actions if a["type"] == "structural")
    assert structural["priority"] == "high"
    assert all(a["topic"] != "only one competitor" for a in actions if a["type"] == "structural")
    paa = next(a for a in actions if a["type"] == "answer_paa")
    assert paa["priority"] == "high"


def test_featured_snippet_action_emitted_for_competitor_holder() -> None:
    from site_audit.serp_gap import _action_points_for_analysis

    page = {"url": "https://ours.example/p", "title": "T", "h1": "H"}
    analysis = _analysis_base("what is a widget tool")
    analysis["serp_features"] = {
        "answer_box": {
            "answer": "A widget tool helps teams sort widget tasks and route the next action.",
            "url": "https://competitor.example/snippet",
            "format": "paragraph",
            "word_count": 12,
        }
    }

    actions = _action_points_for_analysis(page, analysis)
    action = next(row for row in actions if row["type"] == "win_featured_snippet")

    assert "https://competitor.example/snippet" in action["instruction"]
    assert "paragraph with 12 words" in action["instruction"]
    assert "40-55-word direct-answer paragraph" in action["instruction"]
    assert action["placement"] == "Immediately under an H2 phrased as 'what is a widget tool'."
    assert any("same snippet format (paragraph)" in row for row in action["acceptance_criteria"])
    assert action["evidence"]["snippet_format"] == "paragraph"


def test_featured_snippet_action_skips_own_holder() -> None:
    from site_audit.serp_gap import _action_points_for_analysis

    page = {"url": "https://ours.example/p", "title": "T", "h1": "H"}
    analysis = _analysis_base("what is a widget tool")
    analysis["serp_features"] = {
        "answer_box": {
            "answer": "A widget tool helps teams sort widget tasks.",
            "url": "https://ours.example/snippet",
            "format": "paragraph",
            "word_count": 8,
        }
    }

    actions = _action_points_for_analysis(page, analysis)

    assert all(row["type"] != "win_featured_snippet" for row in actions)


def test_action_dedupe_across_keywords() -> None:
    from site_audit.serp_gap import _attach_action_points

    def topic(label):
        return {
            "label": label, "coverage": "missing", "priority": "high",
            "competitor_prevalence": 0.9, "best_competitor_rank": 1,
            "competitor_coverage": 4, "competitor_urls": [], "examples": [],
        }

    page = {
        "url": "https://ours.example/p", "title": "T", "h1": "H",
        "analyses": [
            {**_analysis_base("kw one"), "missing_topics": [topic("pricing, free plan")]},
            {**_analysis_base("kw two"), "missing_topics": [topic("pricing free plan")]},
        ],
    }
    out = _attach_action_points([page])
    add_topic_actions = [a for a in out if a["type"] == "add_topic"]
    assert len(add_topic_actions) == 1
    assert add_topic_actions[0].get("merged_duplicates") == 1


def test_recommended_outline_orders_have_and_add() -> None:
    from site_audit.serp_gap import _recommended_outline

    analysis = {
        "content_order_path": {
            "clusters": [
                {"label": "intro", "ours_mean_order": 0.05, "competitor_mean_order": 0.1, "competitor_pages": 5, "sample_text": "s"},
                {"label": "setup", "ours_mean_order": None, "competitor_mean_order": 0.5, "competitor_pages": 4, "sample_text": "s"},
            ],
            "missing_clusters": [
                {"label": "pricing", "competitor_mean_order": 0.3, "competitor_pages": 3, "sample_text": "s"},
            ],
        },
    }
    rows = _recommended_outline(analysis)
    assert [r["label"] for r in rows] == ["intro", "pricing", "setup"]
    assert [r["status"] for r in rows] == ["have", "add", "add"]
    assert [r["position"] for r in rows] == [1, 2, 3]


def test_html_renders_with_minimal_payload() -> None:
    from site_audit.serp_gap import _html

    payload = {
        "status": "ok",
        "domain": "ours.example",
        "summary": {},
        "selected_pages": [],
        "selected_keywords": [],
        "skipped_pages": [],
        "skipped_keywords": [],
        "domain_ratings": {
            "provider": "ahrefs_public_domain_rating_free",
            "attribution": "Domain Rating by Ahrefs",
            "license": "http://ahrefs.com/legal/domain-rating-license",
            "domains_enriched": 1,
        },
        "serp_url_rankings": [
            {
                "url": "https://comp.example/a",
                "domain": "comp.example",
                "domain_rating": 71.2,
                "top10_count": 1,
                "best_rank": 1,
                "average_rank": 1.0,
                "keywords": [{"keyword": "kw", "rank": 1}],
            },
            {
                "url": "https://other.example/a",
                "domain": "other.example",
                "domain_rating": 42.0,
                "top10_count": 1,
                "best_rank": 2,
                "average_rank": 2.0,
                "keywords": [{"keyword": "kw", "rank": 2}],
            },
        ],
        "editorial_guidelines": {"paragraph_rules": ["One idea per paragraph."], "avoid": ["No filler."]},
        "pages": [{
            "url": "https://ours.example/p",
            "title": "T",
            "h1": "H",
            "own_content": {"headings": [], "paragraphs": [{"index": 0, "word_count": 3, "text": "a b c"}], "word_count": 3},
            "ai_editor_brief": {"status": "ok", "provider": "harnext", "cache_status": "miss", "markdown": "# Brief\n\n- do this <script>alert(1)</script>"},
            "ai_recommendation": {
                "status": "ok",
                "errors": [],
                "data": {
                    "page_assessment": {"is_right_target_page": True, "reason": "ok"},
                    "title": {"current": "T", "recommended": "T2", "reason": "kw"},
                    "meta_description": {"recommended": "m"},
                    "h1": {"recommended": "H"},
                    "outline": [{"level": 2, "heading": "X", "status": "new", "maps_to_topic": "x"}],
                    "paragraph_decisions": [{"index": 0, "decision": "keep", "reason": "fine"}],
                    "new_sections": [{"heading": "S", "placement_after_paragraph": -1, "format": "faq", "draft": "d", "covers_paa": ["q"]}],
                    "structured_data": [],
                    "internal_links": [],
                },
                "verification": {
                    "topics": [{"keyword": "kw", "label": "x", "priority": "high", "before": "missing", "after": "covered", "best_similarity": 0.9}],
                    "paa": [],
                    "summary": {"missing_before": 1, "missing_after": 0, "partial_before": 0, "partial_after": 0, "paa_missing_before": 0, "paa_missing_after": 0, "unresolved_critical": []},
                },
            },
            "analyses": [{
                "status": "ok",
                "query": "kw",
                "keyword": {"keyword": "kw"},
                "summary": {"missing": 1, "partial": 0, "covered": 0},
                "topics": [{"label": "x", "coverage": "missing", "priority": "high", "centroid": [1.0, 0.0]}],
                "structural_patterns": [{"signal": "Comparison / data tables", "competitors": 3, "advice": "Add a table.", "ours": 0, "max_theirs": 2}],
                "paa_coverage": [{"question": "Q?", "status": "missing", "best_similarity": 0.1, "best_paragraph": ""}],
                "serp_features": {"related_searches": ["alt"], "people_also_ask": [], "answer_box": {}},
                "recommended_outline": [{"position": 1, "label": "x", "status": "add", "competitor_pages": 3, "sample_text": "s"}],
                "competitor_pages": [],
                "action_points": [],
            }],
        }],
    }
    html = _html(payload)
    assert "mdToHtml" in html
    assert "Structural / GEO Gaps" in html
    assert "People Also Ask Coverage" in html
    assert "Recommended Section Order" in html
    assert "AI Page Recommendation" in html
    assert "Domain Rating by" in html
    assert '"domain_rating": 71.2' in html
    assert "DR ${v.toFixed(1)}" in html
    assert "replace(/\\r\\n/g,'\\n').split('\\n')" in html
    assert "trimmed.match(/^(#{1,4})\\s+(.*)$/)" in html
    assert ".split(/\\s+/).map(Number)" in html
    assert ".split(/\\\\s+/).map(Number)" not in html
    assert "trimmed.match(/^(#{1,4})\n" not in html
    # centroids must be stripped from the HTML payload
    assert '"centroid"' not in html


def test_recommended_article_markdown_assembles_full_page() -> None:
    from site_audit.serp_gap import _recommended_article_markdown

    page = {"url": "https://ours.example/p", "title": "Old", "h1": "Old H1"}
    paragraphs = ["Intro text.", "Weak text.", "Setup step one."]
    rec = {
        "title": {"recommended": "New Title", "reason": "keyword alignment"},
        "meta_description": {"recommended": "New meta."},
        "h1": {"recommended": "New H1"},
        "page_assessment": {"is_right_target_page": True, "reason": "right page"},
        "outline": [
            {"level": 2, "heading": "Setup", "status": "keep", "source_paragraphs": [2]},
        ],
        "paragraph_decisions": [
            {"index": 0, "decision": "keep", "reason": "good"},
            {"index": 1, "decision": "rewrite", "reason": "filler", "rewrite": "Strong rewritten text."},
            {"index": 2, "decision": "keep", "reason": "essential setup"},
        ],
        "new_sections": [
            {"heading": "Pricing", "placement_after_paragraph": 0, "topic": "pricing",
             "format": "paragraphs", "draft": "Original pricing copy.", "covers_paa": ["How much does it cost?"]},
        ],
        "structured_data": [{"type": "FAQPage", "reason": "competitors carry it"}],
        "internal_links": [],
    }
    verification = {"summary": {"missing_before": 2, "missing_after": 0, "partial_before": 1, "partial_after": 0,
                                "paa_missing_before": 1, "paa_missing_after": 0, "unresolved_critical": []}}
    md = _recommended_article_markdown(page, rec, paragraphs, verification)

    assert "**Title:** New Title" in md
    assert "**H1:** New H1" in md
    # reading order: intro, new pricing section, rewritten paragraph, heading + setup step
    assert md.index("Intro text.") < md.index("## Pricing") < md.index("Strong rewritten text.") < md.index("## Setup") < md.index("Setup step one.")
    assert "Original pricing copy." in md
    assert "answers PAA: How much does it cost?" in md
    assert "Why this version should rank better" in md
    assert "missing 2 -> 0" in md
    assert "FAQPage" in md
    assert "**Title change:** keyword alignment" in md
    # removed paragraphs do not appear
    assert "Weak text." not in md


def test_numeric_verification_ignores_numbers_in_untouched_kept_paragraphs() -> None:
    from site_audit.serp_gap import _verification_for

    tail = "Our team resolved 4817 tickets last quarter."
    long_paragraph = ("This paragraph explains our support workflow in detail. " * 9) + tail
    assert len(long_paragraph) > 500 and tail[:-1] in long_paragraph[500:]
    page = {
        # own_content stores paragraphs truncated to 500 chars, like run() does
        "own_content": {"paragraphs": [{"index": 0, "word_count": len(long_paragraph.split()), "text": long_paragraph[:500]}]},
        "analyses": [],
    }
    recommendation = {"paragraph_decisions": [{"index": 0, "decision": "keep"}], "new_sections": []}

    verification = _verification_for(page, recommendation, [long_paragraph], embedder=None)

    assert verification["unverified_numbers"] == []


def _chat_recommendation_text(section_draft: str) -> str:
    rec = {
        "page_assessment": {"is_right_target_page": True, "reason": "ok"},
        "title": {"recommended": "Widget tools guide"},
        "meta_description": {"recommended": "All about widget tools."},
        "h1": {"recommended": "Widget tools"},
        "outline": [{"heading": "Why widgets", "status": "new"}],
        "paragraph_decisions": [{"index": 0, "decision": "keep"}],
        "new_sections": [{"heading": "Why widgets", "draft": section_draft, "placement_after_paragraph": 0}],
        "structured_data": [],
        "internal_links": [],
    }
    return "Brief text here.\n```json\n" + json.dumps(rec) + "\n```"


def _chat_page() -> dict:
    return {
        "url": "https://ours.example/p",
        "title": "T",
        "h1": "H",
        "keywords": ["widget tools"],
        "own_content": {"paragraphs": [{"index": 0, "word_count": 4, "text": "Widgets help teams work."}]},
        "analyses": [],
    }


class _ScriptedChatClient:
    provider = "fake"

    def __init__(self, texts: list[str]) -> None:
        self.texts = texts
        self.calls: list[list[dict]] = []

    def complete(self, messages, *, model, temperature=0.2, timeout=120) -> AgentCompletion:
        self.calls.append(messages)
        return AgentCompletion(text=self.texts[min(len(self.calls), len(self.texts)) - 1], provider="fake", model=model)


def _run_chat_brief(tmp_path: Path, client: _ScriptedChatClient) -> dict:
    from types import SimpleNamespace

    from site_audit.serp_gap import _attach_chat_brief

    page = _chat_page()
    config = SimpleNamespace(ai_agent_model="m", ai_agent_refresh=True, ai_agent_max_turns=12)
    _attach_chat_brief(
        page,
        tmp_path,
        config,
        {"cache_hits": 0, "editor_briefs": 0},
        client=client,
        paragraph_count=1,
        own_paragraphs=["Widgets help teams work."],
        embedder=None,
    )
    return page


def test_chat_brief_numeric_repair_turn_carries_draft_and_rechecks(tmp_path: Path) -> None:
    client = _ScriptedChatClient([
        _chat_recommendation_text("Teams report 73% faster resolution with widgets."),
        _chat_recommendation_text("Teams report faster resolution with widgets [NEEDS DATA]."),
    ])

    page = _run_chat_brief(tmp_path, client)

    assert len(client.calls) == 2
    roles = [message["role"] for message in client.calls[1]]
    assert roles == ["system", "user", "assistant", "user"]
    assistant = [m for m in client.calls[1] if m["role"] == "assistant"]
    assert "73%" in assistant[0]["content"]
    repair_prompt = client.calls[1][-1]["content"]
    assert "Replace or source these numbers" in repair_prompt
    assert "73%" in repair_prompt
    assert "Output the corrected brief and the full recommendation JSON block again." in repair_prompt
    assert "recommendation.json" not in repair_prompt
    rec = page["ai_recommendation"]
    assert rec["status"] == "ok"
    assert rec["verification_repair_attempted"] is True
    assert rec["verification"]["unverified_numbers"] == []


def test_chat_brief_keeps_valid_recommendation_when_repair_candidate_is_invalid(tmp_path: Path) -> None:
    client = _ScriptedChatClient([
        _chat_recommendation_text("Teams report 73% faster resolution with widgets."),
        "Sorry, I cannot modify files.",
    ])

    page = _run_chat_brief(tmp_path, client)

    rec = page["ai_recommendation"]
    assert rec["status"] == "ok"
    assert rec["data"]["title"]["recommended"] == "Widget tools guide"
    assert [claim["text"] for claim in rec["verification"]["unverified_numbers"]] == ["73%"]
    assert "Unverified numeric claims" in rec["article_markdown"]
    assert "73%" in rec["article_markdown"]


def test_recommended_article_markdown_lists_unverified_numbers_without_summary() -> None:
    from site_audit.serp_gap import _recommended_article_markdown

    page = {"url": "https://ours.example/p", "title": "T", "h1": "H"}
    rec = {
        "title": {"recommended": "T2"},
        "h1": {"recommended": "H2"},
        "paragraph_decisions": [{"index": 0, "decision": "keep"}],
        "new_sections": [],
    }
    verification = {
        "summary": {},
        "unverified_numbers": [{"text": "73%", "context": "Teams report 73% faster resolution."}],
    }

    md = _recommended_article_markdown(page, rec, ["Widgets help teams work."], verification)

    assert "**Unverified numeric claims:** 73%" in md


def test_recommendation_header_notes_ai_overview_only_with_known_citations() -> None:
    from site_audit.serp_gap import _recommendation_header

    cited = {"serp_features": {"ai_overview": {"present": True, "cites_us": False, "cited_domains": ["competitor.example"]}}}
    unknown = {"serp_features": {"ai_overview": {"present": True, "cites_us": None, "cited_domains": []}}}
    citing_us = {"serp_features": {"ai_overview": {"present": True, "cites_us": True, "cited_domains": ["ours.example"]}}}

    assert "AI Overview is present but does not cite this domain" in _recommendation_header(cited)
    assert _recommendation_header(unknown) == ""
    assert _recommendation_header(citing_us) == ""


def test_featured_snippet_action_deduped_across_keywords() -> None:
    page = {"url": "https://ours.example/p", "title": "T", "h1": "H"}
    for keyword, impressions in (("what is a widget tool", 100), ("widget tool definition", 900)):
        analysis = _analysis_base(keyword)
        analysis["keyword"]["impressions"] = impressions
        analysis["serp_features"] = {
            "answer_box": {
                "answer": "A widget tool helps teams sort widget tasks.",
                "url": "https://competitor.example/snippet",
                "format": "paragraph",
                "word_count": 8,
            }
        }
        page.setdefault("analyses", []).append(analysis)

    _attach_action_points([page])
    snippet_actions = [row for row in page["action_points"] if row["type"] == "win_featured_snippet"]

    assert len(snippet_actions) == 1
    assert snippet_actions[0]["keyword"] == "widget tool definition"


def test_serp_features_ai_overview_without_sources_reports_unknown_citation() -> None:
    from site_audit.serp_gap import _serp_features

    payload = {"meta": {"provider": "serper"}, "raw": {"organic": [], "aiOverview": {"text": "Overview answer."}}}

    features = _serp_features(payload, "ours.example")

    assert features["ai_overview"] == {"present": True, "cites_us": None, "cited_domains": []}


def test_serp_features_parses_list_shaped_ai_overview() -> None:
    from site_audit.serp_gap import _serp_features

    payload = {
        "meta": {"provider": "serper"},
        "raw": {"organic": [], "aiOverview": [{"text": "block", "link": "https://competitor.example/post"}]},
    }

    features = _serp_features(payload, "ours.example")

    assert features["ai_overview"] == {"present": True, "cites_us": False, "cited_domains": ["competitor.example"]}
