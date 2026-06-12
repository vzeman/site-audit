# Session 04 — Agent workspace on disk + structured recommendation.json (Harnext as a real agent)

## Context

Repo: `site-audit`. Today `site_audit/ai_agent.py::HarnextOpenRouterClient` flattens chat messages into one prompt (`messages_to_prompt`), runs `harnext_sdk.query()` with `max_turns=3`, and the system prompt forbids tool use — Harnext is a glorified chat completion. The brief is markdown-only (`page["ai_editor_brief"]["markdown"]`), so the report cannot render per-paragraph decisions, and nothing is machine-verifiable.

Goal: per analyzed URL, write an evidence workspace to disk, let Harnext (a coding agent with file tools) read it across many turns, and require a structured `recommendation.json` next to the human `brief.md`. OpenRouter provider keeps the current inline single-shot behavior but adopts the same JSON contract.

Relevant code:
- `serp_gap.py::run()` — per-page loop; `own_ext` (ExtractedPage), `competitor_pages` (list[CompetitorPage]) are in memory per keyword; `report_dir` is the run output dir.
- `serp_gap.py::_attach_ai_editor_briefs(page_results, cache_dir, config, state)` (~line 2562) — generates briefs after `_attach_action_points`.
- `ai_agent.py::build_editor_brief_messages` / `_editor_prompt_payload` — rewritten in session 03.
- `SerpGapConfig` (serp_gap.py ~line 74) — has `ai_agent`, `ai_agent_provider` ("harnext" default), `ai_agent_model`, `ai_agent_refresh`.
- `CompetitorPage` dataclass: `competitive_analysis.py` ~line 87 (`target.competitor_url`, `target.rank`, `title`, `h1`, `paragraphs`, `headers_rich`).

## Task

### 1. New module `site_audit/agent_workspace.py`

```python
def write_agent_workspace(report_dir: Path, page: dict, own_ext, competitor_content: dict[str, dict]) -> Path:
```

Creates `report_dir / "agent" / _url_report_slug(page["url"])/` (import `_url_report_slug` from `serp_gap` would be circular — copy the slug logic into this module as `_slug(url)`, same behavior) containing:

- `evidence.json` — the FULL page dict minus bulky render-only keys: deep-copy `page`, drop `analyses[*].scatter`, `analyses[*].visual_summary`, and any `ai_editor_brief`. Keep heatmap with cells, topic_coverage_matrix, content_comparison, structural_patterns, serp_features, paa_coverage, content_order_path, action_points, own_content. `json.dumps(..., ensure_ascii=False, indent=1)`.
- `our_page.md` — title, meta description, H1, then the heading tree (indent by level), then every paragraph as `### [P{index}] ({word_count} words)` + full text (untruncated — take from `own_ext.paragraphs`, not the 500-char copies).
- `competitors/{rank:02d}-{domain}.md` — one file per competitor from `competitor_content`: keyed by url, value `{"rank": int, "title": str, "h1": str, "headings": list[dict], "paragraphs": list[str]}`. Render heading outline + paragraphs numbered `[C{idx}]`.
- `serp.json` — `{keyword: {"features": analysis["serp_features"], "rankings": [{url, rank, title} ...]}}` for each analysis (rankings from `analysis["competitor_pages"]` profiles: url, rank, title).
- `TASK.md` — the agent instructions (see step 3).

Return the workspace path.

### 2. Collect competitor content in `run()` (serp_gap.py)

Inside the keyword loop where `competitor_pages` exist, accumulate per page:

```python
competitor_content.setdefault(cp.target.competitor_url, {
    "rank": cp.target.rank, "title": cp.title, "h1": cp.h1,
    "headings": cp.headers_rich, "paragraphs": cp.paragraphs,
})
```

(only for `cp` without `error`). Keep a `page_competitor_content: dict[str, dict[str, dict]]` keyed by page url at `run()` scope; pass it into `_attach_ai_editor_briefs` (extend its signature: `..., page_competitor_content: dict | None = None, report_dir: Path | None = None, own_exts: dict | None = None`). Also keep `own_exts[page.url] = own_ext`.

Memory note: paragraphs are already capped at `max_paragraphs_per_page` (80) and competitor pages at `max_competitor_pages` (100) — acceptable.

### 3. Recommendation contract

Add to `ai_agent.py`:

```python
RECOMMENDATION_SCHEMA_DOC = """..."""   # human-readable contract embedded in prompts and TASK.md
```

Contract (document every field; the agent must output exactly this):

```json
{
  "page_assessment": {"is_right_target_page": true, "reason": "..."},
  "title": {"current": "...", "recommended": "...", "reason": "..."},
  "meta_description": {"recommended": "...", "reason": "..."},
  "h1": {"recommended": "...", "reason": "..."},
  "outline": [
    {"level": 2, "heading": "...", "status": "keep|rename|new|remove",
     "maps_to_topic": "<topic label or empty>", "source_paragraphs": [0, 1]}
  ],
  "paragraph_decisions": [
    {"index": 0, "decision": "keep|rewrite|move|merge|remove",
     "reason": "...", "rewrite": "<full replacement text, or null unless decision==rewrite>"}
  ],
  "new_sections": [
    {"heading": "...", "placement_after_paragraph": 7, "topic": "<label>",
     "format": "paragraphs|table|faq|steps", "draft": "<full original draft copy>",
     "covers_paa": ["question ..."]}
  ],
  "structured_data": [{"type": "FAQPage", "reason": "..."}],
  "internal_links": [{"anchor": "...", "from_hint": "<what kind of page should link>", "reason": "..."}]
}
```

Add `validate_recommendation(payload: dict, paragraph_count: int) -> list[str]` (pure, returns error strings, empty = valid):
- required top-level keys present and correctly typed;
- every `paragraph_decisions[].index` is an int in `[0, paragraph_count)`; no duplicate indexes; every index `0..paragraph_count-1` present exactly once (when `paragraph_count > 0`);
- `decision`/`status`/`format` values within their enums;
- `rewrite` non-empty exactly when `decision == "rewrite"`; `draft` non-empty for every new section;
- `placement_after_paragraph` is `-1` (top) or a valid index;
- `outline` non-empty when `paragraph_decisions` mark anything `remove` or any `new_sections` exist.

Add `parse_recommendation(text: str) -> dict` reusing `_extract_json`.

### 4. Agentic Harnext path

In `ai_agent.py` add:

```python
def run_harnext_workspace_session(workspace: Path, *, model: str, max_turns: int = 20, timeout: int = 600) -> AgentCompletion:
```

- **First inspect the installed SDK**: run `python -c "import inspect, harnext_sdk; print(inspect.signature(harnext_sdk.HarnextAgentOptions))"` and `python -c "import harnext_sdk; print([x for x in dir(harnext_sdk) if not x.startswith('_')])"` to learn the supported option names (working directory, allowed tools, permission mode). Use the discovered names — likely candidates: `cwd`/`workdir`, `allowed_tools`, `permission_mode`. If no cwd option exists, embed absolute file paths in the prompt instead.
- Options: provider "openrouter", `model`, `max_turns=max_turns`, env as today, working dir = workspace, tools allowing file read/write within the workspace only.
- Prompt: "You are a senior SEO/GEO editor. Read `TASK.md` first, then `evidence.json` and `our_page.md`. Consult `competitors/` and `serp.json` as needed. Write two files into this directory: `recommendation.json` (must satisfy the embedded contract exactly) and `brief.md` (human summary). Do not modify any other file."
- `TASK.md` content (written by `write_agent_workspace`): the same role text + the full `RECOMMENDATION_SCHEMA_DOC` + rules from session 03 (no competitor wording, decide every paragraph listed in evidence, cover missing PAA, respect structural_patterns and benchmark, mark "demand metrics absent" instead of guessing).
- After the session, read `workspace/recommendation.json` and `workspace/brief.md` from disk; the SDK result text is secondary. Return an `AgentCompletion` whose `text` = brief markdown and `raw` = `{"recommendation": <parsed json>, "result": ...}`.

**Repair turn:** in the caller (below), run `validate_recommendation`; if errors, invoke one follow-up session (same function, fresh query) with prompt: "Your `recommendation.json` failed validation: <errors>. Fix recommendation.json only." Re-validate; after the second failure, keep the brief, set status "invalid_recommendation" with the error list.

### 5. Rework `_attach_ai_editor_briefs` (serp_gap.py)

For each page:
- If provider == "harnext" and `harnext_status()[0]` and workspace inputs available: `workspace = write_agent_workspace(report_dir, page, own_exts[url], page_competitor_content.get(url, {}))`, then `run_harnext_workspace_session(...)` with cache: key the cache on `content_hash` of `evidence.json` content + model (reuse the `cached_completion` directory layout — add a sibling helper `cached_workspace_completion(cache_dir, kind, evidence_hash, runner)` that stores `{brief, recommendation}`; on hit, skip the session entirely).
- Else (openrouter, or harnext unavailable): current `build_editor_brief_messages` + `cached_completion` path, BUT append to the user message: "Additionally output a fenced ```json block with `recommendation.json` following this contract: <RECOMMENDATION_SCHEMA_DOC>." Parse it with `parse_recommendation`; validate with `paragraph_count = len(page["own_content"]["paragraphs"])`.
- Store on the page:

```python
page["ai_editor_brief"] = {"status": "ok", "provider", "model", "cache_status", "markdown": brief_md}
page["ai_recommendation"] = {"status": "ok"|"invalid_recommendation"|"error"|..., "errors": [...], "data": <dict or {}>, "workspace": str(workspace) or ""}
```

Failure isolation: any exception per page -> statuses "error" with message, continue with the next page (mirror existing try/except style).

### 6. Render `ai_recommendation` in the report

In `_html()::pageSection` (~line 4566), before the existing brief panel add `recommendationSection(page)`:
- header chips: title/H1 change indicators (`title.recommended !== title.current`);
- table "Outline": level, heading, status chip, maps_to_topic;
- table "Paragraph decisions": [P#], decision chip (keep=covered/rewrite=partial/remove,move,merge=missing colors), reason, rewrite text in a collapsed `<details>`;
- "New sections": heading, placement (`after [P7]` or `top`), format, covers_paa count, draft in collapsed `<details>`;
- "Structured data" + "Internal links" as simple lists;
- when `status === 'invalid_recommendation'` show the error list; when missing, render nothing.

### 7. CLI flag

In `site_audit/cli.py`, the serp-gap subparser already defines `--ai-agent*` flags (see `tests/test_cli.py::test_serp_gap_parser_accepts_budget_and_keyword_options` asserting `ai_agent_*` attrs). Add `--ai-agent-max-turns` (int, default 20) wired to new `SerpGapConfig.ai_agent_max_turns: int = 20`, passed to `run_harnext_workspace_session`. Update the env-name test expectation in `tests/test_config_env.py` if it enumerates serp-gap ai_agent options (check `env_names("serp-gap", "ai_agent")` — extend the expected list accordingly).

### 8. Tests

- `tests/test_agent_workspace.py` (new): `write_agent_workspace` with a minimal page dict + fake ExtractedPage + one competitor -> assert all five artifacts exist, `evidence.json` lacks `scatter`, `our_page.md` contains `[P0]`, competitor file named `01-<domain>.md`.
- In `tests/test_serp_gap.py`: `test_validate_recommendation_catches_gaps` (missing index, bad enum, rewrite without text -> 3+ errors; a fully valid fixture -> []), `test_parse_recommendation_from_fenced_block`.
- Harnext session itself: do NOT test against the real SDK; factor the query loop so the runner accepts an injectable `query_fn` and test with a stub that writes `recommendation.json`/`brief.md` into the workspace.

Run: `python -m pytest tests/test_serp_gap.py tests/test_agent_workspace.py tests/test_cli.py tests/test_config_env.py -q`.

## Constraints

- No new third-party dependencies (harnext_sdk is already optional via `.[agent]`).
- The OpenRouter path must keep working with zero workspace files (CI machines without npm).
- Workspace writes go only under `report_dir / "agent"`. Never write outside it.
- All disk artifacts UTF-8, `ensure_ascii=False`.

## Definition of done

- Harnext runs as a multi-turn agent over an on-disk evidence workspace and produces validated `recommendation.json` + `brief.md`, with one automatic repair turn.
- OpenRouter fallback emits the same contract inline.
- Report renders the structured recommendation. New flag + config field covered by tests. All tests pass.
