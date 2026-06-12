import json
from pathlib import Path

from site_audit.agent_workspace import write_agent_workspace
from site_audit.ai_agent import (
    RECOMMENDATION_SCHEMA_DOC,
    AgentCompletion,
    parse_recommendation,
    run_harnext_workspace_session,
    validate_recommendation,
)
from site_audit.extractor import ExtractedPage


def _page() -> dict:
    return {
        "url": "https://ours.example/features/widget/",
        "title": "Widget",
        "h1": "Widget",
        "own_content": {
            "headings": [{"order": 0, "level": 2, "text": "What is a widget?"}],
            "paragraphs": [{"index": 0, "word_count": 5, "text": "Widgets do useful things daily."}],
            "word_count": 5,
        },
        "ai_editor_brief": {"status": "ok", "markdown": "old"},
        "analyses": [{
            "keyword": {"keyword": "widget tool"},
            "scatter": {"points": [1, 2, 3]},
            "visual_summary": ["x"],
            "serp_features": {"people_also_ask": [{"question": "What is a widget?"}], "related_searches": []},
            "paa_coverage": [{"question": "What is a widget?", "status": "covered"}],
            "competitor_pages": [{"url": "https://comp.example/w", "rank": 1, "title": "W", "error": ""}],
        }],
    }


def _own_ext() -> ExtractedPage:
    return ExtractedPage(
        url="https://ours.example/features/widget/",
        title="Widget",
        description="meta",
        body="Widgets do useful things daily.",
        word_count=5,
        language="en",
        h1="Widget",
        headers_rich=[{"level": 2, "text": "What is a widget?", "order": 0}],
        paragraphs=["Widgets do useful things daily."],
    )


def test_write_agent_workspace_creates_all_artifacts(tmp_path: Path) -> None:
    competitor_content = {
        "https://comp.example/w": {
            "rank": 1,
            "title": "W",
            "h1": "W",
            "headings": [{"level": 2, "text": "Widget pricing", "order": 0}],
            "paragraphs": ["Competitor paragraph about widget pricing."],
        }
    }
    workspace = write_agent_workspace(tmp_path, _page(), _own_ext(), competitor_content, schema_doc=RECOMMENDATION_SCHEMA_DOC)

    assert workspace == tmp_path / "agent" / "features-widget"
    evidence = json.loads((workspace / "evidence.json").read_text(encoding="utf-8"))
    assert "scatter" not in evidence["analyses"][0]
    assert "visual_summary" not in evidence["analyses"][0]
    assert "ai_editor_brief" not in evidence
    our_page = (workspace / "our_page.md").read_text(encoding="utf-8")
    assert "[P0]" in our_page and "Widgets do useful things daily." in our_page
    competitor_file = workspace / "competitors" / "01-comp.example.md"
    assert competitor_file.is_file()
    assert "Widget pricing" in competitor_file.read_text(encoding="utf-8")
    serp = json.loads((workspace / "serp.json").read_text(encoding="utf-8"))
    assert serp["widget tool"]["rankings"][0]["url"] == "https://comp.example/w"
    task = (workspace / "TASK.md").read_text(encoding="utf-8")
    assert "recommendation.json contract" in task
    assert "paragraph_decisions" in task


def _valid_recommendation() -> dict:
    return {
        "page_assessment": {"is_right_target_page": True, "reason": "feature page"},
        "title": {"current": "Widget", "recommended": "Widget Tool", "reason": "keyword"},
        "meta_description": {"recommended": "Better meta.", "reason": "ctr"},
        "h1": {"recommended": "Widget", "reason": "fine"},
        "outline": [{"level": 2, "heading": "What is a widget?", "status": "keep", "maps_to_topic": "", "source_paragraphs": [0]}],
        "paragraph_decisions": [{"index": 0, "decision": "keep", "reason": "good", "rewrite": None}],
        "new_sections": [{"heading": "Pricing", "placement_after_paragraph": 0, "topic": "pricing", "format": "paragraphs", "draft": "Original pricing copy.", "covers_paa": []}],
        "structured_data": [{"type": "FAQPage", "reason": "competitors have it"}],
        "internal_links": [{"anchor": "widget tool", "from_hint": "blog posts", "reason": "equity"}],
    }


def test_validate_recommendation_accepts_valid_payload() -> None:
    assert validate_recommendation(_valid_recommendation(), paragraph_count=1) == []


def test_validate_recommendation_catches_gaps() -> None:
    payload = _valid_recommendation()
    payload["paragraph_decisions"] = [
        {"index": 5, "decision": "polish", "reason": "", "rewrite": None},
        {"index": 0, "decision": "rewrite", "reason": "", "rewrite": ""},
    ]
    errors = validate_recommendation(payload, paragraph_count=2)
    assert any("index invalid" in e for e in errors)
    assert any("decision invalid" in e for e in errors)
    assert any("rewrite text is empty" in e for e in errors)
    assert any("missing indexes" in e for e in errors)


def test_validate_recommendation_rejects_empty() -> None:
    assert validate_recommendation({}, paragraph_count=3) == ["recommendation is empty or not a JSON object"]


def test_parse_recommendation_from_fenced_block() -> None:
    text = "# Brief\n\nSome prose.\n\n```json\n" + json.dumps(_valid_recommendation()) + "\n```\n"
    parsed = parse_recommendation(text)
    assert parsed["title"]["recommended"] == "Widget Tool"


def test_run_harnext_workspace_session_reads_files(tmp_path: Path) -> None:
    recommendation = _valid_recommendation()

    def stub_runner(prompt, *, workspace, model, max_turns, api_key=None):
        assert "TASK.md" in prompt
        Path(workspace, "recommendation.json").write_text(json.dumps(recommendation), encoding="utf-8")
        Path(workspace, "brief.md").write_text("# Editorial brief\n\nDo the things.", encoding="utf-8")
        return {"num_turns": 7}

    completion = run_harnext_workspace_session(
        tmp_path, model="test-model", max_turns=5, session_runner=stub_runner,
    )
    assert isinstance(completion, AgentCompletion)
    assert completion.text.startswith("# Editorial brief")
    assert completion.raw["recommendation"]["h1"]["recommended"] == "Widget"
    assert completion.raw["session"]["num_turns"] == 7
