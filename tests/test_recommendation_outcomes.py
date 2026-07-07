from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

from site_audit.extractor import ExtractedPage
from site_audit.history import build_history_snapshot, detect_recommendation_outcomes
from site_audit.recommendations import synthesize, to_payload


def _page(
    url: str,
    *,
    title_hash: str = "title",
    description_hash: str = "desc",
    paragraph_hash: str = "para",
    heading_hash: str = "head",
    link_hash: str = "links",
    links: list[str] | None = None,
    title: str = "",
    position: float = 8.0,
    clicks: float = 10.0,
    traffic: float = 100.0,
    status_code: int = 200,
    redirect_target_url: str = "",
) -> dict:
    return {
        "url": url,
        "title": title or url.rsplit("/", 1)[-1],
        "title_hash": title_hash,
        "description_hash": description_hash,
        "paragraph_hash": paragraph_hash,
        "heading_hash": heading_hash,
        "h1_hash": heading_hash,
        "link_hash": link_hash,
        "schema_hash": "schema",
        "links": links or [],
        "status_code": status_code,
        "redirect_target_url": redirect_target_url,
        "metrics": {
            "position": position,
            "clicks": clicks,
            "traffic": traffic,
            "impressions": clicks * 20,
        },
    }


def _snapshot(pages: list[dict], recommendations: list[dict] | None = None, snapshot_id: str = "snap") -> dict:
    return {
        "summary": {
            "status": "ok",
            "snapshot_id": snapshot_id,
            "created_at": "2026-01-01T00:00:00+00:00",
            "pages": len(pages),
        },
        "pages": pages,
        "recommendations": recommendations or [],
    }


def _rec(rec_id: str, category: str, rec_type: str, target: str, *targets: str, title: str = "Fix page") -> dict:
    all_targets = [target, *targets] if target else list(targets)
    return {
        "id": rec_id,
        "category": category,
        "type": rec_type,
        "priority": "medium",
        "primary_url": target,
        "targets": all_targets,
        "title": title,
        "estimated_clicks_gain": None,
    }


def test_stable_recommendation_ids_are_repeatable_and_distinct() -> None:
    kwargs = {
        "duplicates_rows": [
            {"similarity": 0.98, "url_a": "https://example.com/a", "url_b": "https://example.com/b"},
        ],
        "coverage_payload": [
            {"status": "gap", "query": "support automation tools", "best_similarity": 0.2},
        ],
        "linkgraph_payload": {
            "recommendations": [
                {
                    "source_url": "https://example.com/source",
                    "target_url": "https://example.com/target",
                    "similarity": 0.91,
                }
            ],
        },
    }
    first = to_payload(synthesize(**kwargs))["items"]
    second = to_payload(synthesize(**kwargs))["items"]

    assert [row["id"] for row in first] == [row["id"] for row in second]
    assert len({row["id"] for row in first}) == len(first)

    changed = to_payload(synthesize(
        duplicates_rows=[
            {"similarity": 0.98, "url_a": "https://example.com/a", "url_b": "https://example.com/c"},
        ],
    ))["items"][0]["id"]
    dup_id = next(row["id"] for row in first if row["id"].startswith("dup-"))
    assert changed != dup_id

    swapped = to_payload(synthesize(
        duplicates_rows=[
            {"similarity": 0.98, "url_a": "https://example.com/b", "url_b": "https://example.com/a"},
        ],
    ))["items"][0]["id"]
    assert swapped == dup_id


def test_detect_recommendation_outcomes_classifies_rules_and_metrics() -> None:
    prev = _snapshot(
        [
            _page("https://example.com/title", title_hash="old-title", description_hash="old-desc", position=8, clicks=10, traffic=100),
            _page("https://example.com/unchanged", title_hash="same", description_hash="same", position=7, clicks=5, traffic=30),
            _page("https://example.com/geo", paragraph_hash="old-para"),
            _page("https://example.com/keep"),
            _page("https://example.com/drop"),
            _page("https://example.com/source", paragraph_hash="old-source"),
            _page("https://example.com/target", paragraph_hash="old-target"),
            _page("https://example.com/link-source", link_hash="old-links", links=["https://example.com/old"]),
            _page("https://example.com/removed", paragraph_hash="old-removed"),
        ],
        [
            _rec("title-main", "onpage", "title_rewrite", "https://example.com/title", title="Rewrite title"),
            _rec("title-unchanged", "onpage", "title_rewrite", "https://example.com/unchanged", title="Rewrite unchanged"),
            _rec("geo-main", "geo", "answerability", "https://example.com/geo", title="Improve answerability"),
            _rec("gap-support", "coverage", "coverage_gap", "", title='No page answers "support automation tools"'),
            _rec("dup-pair", "content_debt", "merge_duplicate", "https://example.com/keep", "https://example.com/drop"),
            _rec("wh-move", "content_debt", "move_paragraph", "https://example.com/source", "https://example.com/target"),
            _rec("link-main", "linking", "internal_link", "https://example.com/link-source", "https://example.com/new-target"),
            _rec("geo-missing", "geo", "answerability", "https://example.com/missing"),
            _rec("geo-removed", "geo", "answerability", "https://example.com/removed"),
        ],
        snapshot_id="before",
    )
    curr = _snapshot(
        [
            _page("https://example.com/title", title_hash="new-title", description_hash="old-desc", position=6, clicks=30, traffic=130),
            _page("https://example.com/unchanged", title_hash="same", description_hash="same", position=7, clicks=5, traffic=30),
            _page("https://example.com/geo", paragraph_hash="new-para"),
            _page("https://example.com/keep"),
            _page("https://example.com/source", paragraph_hash="new-source"),
            _page("https://example.com/target", paragraph_hash="old-target"),
            _page("https://example.com/link-source", link_hash="new-links", links=["https://example.com/old", "https://example.com/new-target"]),
            _page("https://example.com/support-automation-tools", title="Support automation tools"),
        ],
        snapshot_id="after",
    )

    outcomes = detect_recommendation_outcomes(prev, curr)
    by_id = {row["id"]: row for row in outcomes["rows"]}

    assert outcomes["available"] is True
    assert by_id["title-main"]["change_status"] == "implemented"
    assert by_id["title-unchanged"]["change_status"] == "not_implemented"
    assert by_id["geo-main"]["change_status"] == "implemented"
    assert by_id["gap-support"]["change_status"] == "implemented"
    assert by_id["dup-pair"]["change_status"] == "implemented"
    assert by_id["wh-move"]["change_status"] == "partially"
    assert by_id["link-main"]["change_status"] == "implemented"
    assert by_id["geo-missing"]["change_status"] == "unknown"
    assert by_id["geo-removed"]["change_status"] == "page_removed"
    assert by_id["title-main"]["position_delta"] == -2
    assert by_id["title-main"]["clicks_delta"] == 20
    assert by_id["title-main"]["traffic_delta"] == 30
    assert by_id["title-main"]["confidence"] in {"low", "low-medium", "medium"}
    assert outcomes["aggregates"]["by_status"]["implemented"]["count"] == 5
    assert outcomes["aggregates"]["by_status"]["not_implemented"]["count"] == 1
    assert outcomes["aggregates"]["avg_position_delta_implemented"] is not None
    assert outcomes["aggregates"]["avg_position_delta_not_implemented"] == 0
    assert "Impact rows are before/after associations, not causal proof." in outcomes["caveats"]


def test_old_snapshot_without_recommendations_is_unavailable() -> None:
    outcomes = detect_recommendation_outcomes(_snapshot([_page("https://example.com/a")]), _snapshot([]))

    assert outcomes["available"] is False
    assert outcomes["rows"] == []


def test_snapshot_recommendation_round_trip_detects_change() -> None:
    page = SimpleNamespace(
        url="https://example.com/page",
        title="Old title",
        description="Description",
        section="blog",
        word_count=100,
    )
    extracted = ExtractedPage(
        url=page.url,
        title=page.title,
        description=page.description,
        body="Old paragraph",
        word_count=100,
        language="en",
        headers_rich=[{"level": 1, "text": "Old title"}],
        paragraphs=["Old paragraph"],
    )
    recommendations = {
        "items": [
            {
                "id": "title-page",
                "category": "onpage",
                "type": "title_rewrite",
                "priority": "medium",
                "targets": [page.url],
                "title": "Rewrite title",
                "estimated_clicks_gain": 12.0,
            }
        ]
    }
    prev = build_history_snapshot(
        "example.com",
        [page],
        [extracted],
        recommendations_payload=recommendations,
        snapshot_id="before",
    )
    curr = deepcopy(prev)
    curr["summary"]["snapshot_id"] = "after"
    curr["pages"][0]["title_hash"] = "changed"

    outcomes = detect_recommendation_outcomes(prev, curr)

    assert prev["recommendations"] == [
        {
            "id": "title-page",
            "category": "onpage",
            "type": "title_rewrite",
            "priority": "medium",
            "primary_url": page.url,
            "targets": [page.url],
            "title": "Rewrite title",
            "estimated_clicks_gain": 12.0,
        }
    ]
    assert outcomes["rows"][0]["change_status"] == "implemented"


def test_duplicate_drop_via_redirect_counts_as_implemented() -> None:
    prev = _snapshot(
        [_page("https://example.com/keep"), _page("https://example.com/drop")],
        [_rec("dup-pair", "content_debt", "merge_duplicate", "https://example.com/keep", "https://example.com/drop")],
    )
    curr = _snapshot([
        _page("https://example.com/keep"),
        _page("https://example.com/drop", status_code=301, redirect_target_url="https://example.com/keep"),
    ])

    outcomes = detect_recommendation_outcomes(prev, curr)

    assert outcomes["rows"][0]["change_status"] == "implemented"


def test_cannibalization_is_partial_when_one_runner_up_remains() -> None:
    prev = _snapshot(
        [
            _page("https://example.com/best"),
            _page("https://example.com/runner-1"),
            _page("https://example.com/runner-2"),
        ],
        [_rec(
            "cann-query", "coverage", "cannibalization",
            "https://example.com/best", "https://example.com/runner-1", "https://example.com/runner-2",
        )],
    )
    curr = _snapshot([
        _page("https://example.com/best"),
        _page("https://example.com/runner-2"),
    ])

    outcomes = detect_recommendation_outcomes(prev, curr)

    assert outcomes["rows"][0]["change_status"] == "partially"


def test_url_normalization_matches_slash_and_www_variants() -> None:
    prev = _snapshot(
        [_page("https://example.com/page")],
        [_rec("title-page", "onpage", "title_rewrite", "http://www.example.com/page/")],
    )
    curr = _snapshot([_page("https://example.com/page", title_hash="changed")])

    outcomes = detect_recommendation_outcomes(prev, curr)

    assert outcomes["rows"][0]["change_status"] == "implemented"


def test_history_compare_cli_prints_scoreboard(tmp_path, capsys) -> None:
    import json

    from site_audit.cli import _history_compare_command, build_parser
    from site_audit.history import save_report_snapshot

    def _write_report(name: str, title: str) -> None:
        page = SimpleNamespace(url="https://example.com/page", title=title, description="Desc", section="blog", word_count=100)
        extracted = ExtractedPage(
            url=page.url,
            title=title,
            description="Desc",
            body="Paragraph",
            word_count=100,
            language="en",
            headers_rich=[{"level": 1, "text": title}],
            paragraphs=["Paragraph"],
        )
        recommendations = {
            "items": [
                {
                    "id": "title-page",
                    "category": "onpage",
                    "type": "title_rewrite",
                    "priority": "medium",
                    "targets": [page.url],
                    "title": "Rewrite title",
                    "estimated_clicks_gain": None,
                }
            ]
        }
        report_dir = tmp_path / name
        report_dir.mkdir()
        snapshot = build_history_snapshot("example.com", [page], [extracted], recommendations_payload=recommendations)
        (report_dir / "history_snapshot.json").write_text(json.dumps(snapshot), encoding="utf-8")
        save_report_snapshot("example.com", tmp_path / "projects", report_dir, snapshot_id=name)

    _write_report("before", "Old title")
    _write_report("after", "New title")
    args = build_parser().parse_args([
        "history", "compare", "example.com", "before", "after",
        "--projects-root", str(tmp_path / "projects"),
    ])

    assert _history_compare_command(args) == 0

    out = capsys.readouterr().out
    assert "Past recommendations scoreboard: 1 issued (implemented 1)" in out
    assert "Position delta: implemented n/a, not implemented n/a" in out
