# Session 07 — Report polish: render markdown, trim duplicate visuals, slim action boilerplate, docs

## Context

Repo: `site-audit`, file `site_audit/serp_gap.py` — the serp-gap HTML report is one big template in `_html()` (~line 3261, JS functions near the end: `overviewSection` ~4564, `keywordCard` ~4565, `pageSection` ~4566, `aiEditorBriefSection` ~4494). Docs live in `docs/`.

Issues:
1. The AI brief is shown as escaped plain text (`aiEditorBriefSection` wraps `esc(brief.markdown)` in a `prompt-box`) — unreadable for long briefs.
2. Three near-duplicate cluster visualizations: per-keyword `clusterCards` (inside keywordCard's diagnostics), overview `clusterCards`, plus `keywordParagraphRidgelineSection`/`clusterImpactChart`. Specialists need one.
3. Every action JSON repeats identical boilerplate (`avoid` lists, `paragraph_plan`, paragraph rules), inflating report JSON and agent payloads.
4. `visual_summary` strings duplicate the summary chips.
5. Docs don't cover the new features from sessions 01–06.

## Task

### 1. Minimal markdown renderer in the report JS

Add a JS function `mdToHtml(md)` inside `_html()` (next to helpers like `esc`): supports headings (`#`–`####`), unordered/ordered lists, bold/italic, inline code, fenced code blocks, paragraphs. Escape ALL text content through the existing `esc()` BEFORE wrapping in tags (XSS-safe: never inject raw input). ~40 lines, no library, no CDN.

Use it in `aiEditorBriefSection`: replace `<div class="prompt-box">${esc(brief.markdown||'')}</div>` with `<div class="md-body">${mdToHtml(brief.markdown||'')}</div>`, and add CSS for `.md-body` (readable line-height, margins for h2/h3/ul, `code` styling) next to the existing `.prompt-box` rules. Keep `.prompt-box` for the per-action `ai_agent_prompt` snippets.

Also apply `mdToHtml` to rewrite/draft texts inside `recommendationSection` `<details>` blocks (from session 04) if they are currently `esc()`-only.

### 2. One cluster view

- Keep the OVERVIEW `Aggregate Semantic Clusters` panel and `Topic Traffic Impact` chart in `overviewSection`.
- In `keywordCard`, the per-keyword `semanticEvidence` block (Semantic Clusters + Competitor SERP panels inside `diagnosticDetails('Raw semantic evidence for this keyword', ...)`) — keep the Competitor SERP panel, drop the per-keyword `clusterCards` (the scatter already encodes clusters). The `diagnosticDetails` wrapper stays collapsed by default (verify the third argument `false` does that; if not, fix it so diagnostics start collapsed).
- Remove `keywordFrequencySection` from `overviewSection` if it only repeats keyword counts already shown in the keyword metrics table — inspect it first; if it shows demand-weighted frequencies not visible elsewhere, keep it but move it inside a collapsed diagnostics block.

### 3. Drop `visual_summary` from payload and UI

- `serp_gap.py::_build_gap` (~line 1465): stop attaching `gap["visual_summary"]` and delete `_visual_summary` (~line 2108) plus its render usage (search `visual_summary` across the file — also remove from `_editor_prompt_payload` in `ai_agent.py` if still referenced after session 03).
- Check `_todo_markdown` and tests for references; update accordingly.

### 4. Slim per-action boilerplate

In `serp_gap.py`:
- `_editorial_guidelines()` (~line 2212) already centralizes rules. In `_topic_content_brief` and `_paragraph_content_brief`, the `avoid` lists and the static parts of `paragraph_plan`/`paragraph_rules` are identical for every action. Replace the static lists with a `"guidelines_ref": "editorial_guidelines"` key, and emit the shared lists ONCE in the top-level payload: in `run()` where the final `payload` dict is assembled (search for `"summary": _summary(`), add `"editorial_guidelines": _editorial_guidelines()`.
- KEEP dynamic, action-specific bullets (e.g. "Split this paragraph because it is long...", filler-term replacements, the topic-specific first bullet of paragraph_plan).
- Update consumers: `_html` action rendering (`actionList` — render shared guidelines once in `paragraphRulesSection`, which already exists in overviewSection; per-action display should show only dynamic bullets), `_todo_markdown`, `_editor_prompt_payload`/TASK.md (reference the guidelines once at top level of evidence), CSV rows (verify `_action_csv_rows` doesn't export the removed static lists; adjust).
- Acceptance criteria per action: keep the topic-specific ones (they reference the label/keyword), drop only the two generic ones ("No paragraph is generic marketing copy...", "A reader can quote...") into the shared guidelines.

### 5. Docs

- `docs/serp-gap-command.md`: add sections for — Structural/GEO gaps panel; PAA coverage; AI agent workspace mode (harnext: evidence directory layout under `serp_gap/report/<run>/agent/<slug>/`, `recommendation.json` contract summary, `--ai-agent-max-turns`); coverage verification (before/after semantics); new action types (rewrite_title, expand_depth, structural, answer_paa); recommended section order panel.
- `docs/serp-paragraph-gap-analysis.md`: update "How To Read The Report" with the verification strip and the recommendation tables.
- `docs/review-serp-gap-ai-agent.md`: append an "Implementation status" line listing sessions 01–07 done.
- Check `tests/test_report_docs.py` — it may assert that documented sections exist or that doc files mention report section names; run it and satisfy whatever invariants it enforces.

### 6. Full verification

```bash
python -m pytest -q
```
All tests green. Then build a report from any cached project if one exists under `projects/` (`site-audit serp-gap <domain> --dry-run` at minimum) and open the HTML to sanity-check: no JS console errors (the report is static; check that the generated HTML has balanced template literals — a quick `node --check`-style sanity isn't possible, so at least run Python to render `_html` with a synthetic payload in a unit test).

Add `tests/test_serp_gap.py::test_html_renders_with_minimal_payload`: call `_html({minimal payload with one page, one analysis, ai_editor_brief, ai_recommendation, paa_coverage, structural_patterns, recommended_outline, editorial_guidelines})` and assert the output contains `mdToHtml`, `Structural / GEO Gaps`, `People Also Ask Coverage`, `Recommended Section Order`, and does NOT contain `visual_summary`.

## Constraints

- No CDN scripts, no new dependencies — the report must stay a self-contained offline HTML file.
- `mdToHtml` must escape before tagging; add a test-like comment showing why (`<script>` in brief must render inert).
- Do not remove `paragraphRulesSection` — repurpose it as the single shared-guidelines block.
- Locate everything by function name; line numbers approximate.

## Definition of done

- AI brief and drafts render as formatted markdown, XSS-safe.
- One cluster view; per-keyword diagnostics collapsed; `visual_summary` gone.
- Shared editorial guidelines emitted once; actions carry only dynamic content.
- Docs updated; doc tests and full suite pass; minimal-payload HTML render test added.
