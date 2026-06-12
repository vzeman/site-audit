# Session 06 — Deterministic specialist features: title gap, outline, depth target, structural actions, dedupe

## Context

Repo: `site-audit`, file `site_audit/serp_gap.py`. Action points per keyword analysis are built in `_action_points_for_analysis(page, analysis)` (~line 2486): today only `add_topic` (missing topics, max 8), `strengthen_topic` (partial, max 6), `review_paragraph` (off-intent rows, max 6). They are aggregated and globally sorted in `_attach_action_points` (~line 2543) with NO dedupe across keywords — the same topic reachable from two keywords of one page produces two near-identical tasks.

Available evidence per analysis (keys on the analysis dict): `content_comparison` (`ours`, `benchmark` with `median_competitor_paragraphs`, `median_competitor_headings`, `median_competitor_h2_h3`), `competitor_pages` (profiles with `rank`, `title`, `h1`), `structural_patterns` (populated since session 01: `{signal, competitors, advice, ours, max_theirs}`), `content_order_path` (`clusters`, `missing_clusters`, each `{label, competitor_pages, best_competitor_rank, competitor_mean_order, sample_text}`), `paa_coverage` (since session 02). Action dict shape: copy an existing builder (`_topic_action`, ~line 2329) — keep the same keys (`id, order, type, priority, action, task_summary, target_url, keyword, instruction, rationale, topic, content_brief, placement, acceptance_criteria, ai_agent_prompt, impact_score, evidence`).

Demand helper: `_keyword_demand(analysis)` (~line 2181). Priority sort: `_action_priority_score` (~line 2172).

## Task

### 1. Title / H1 gap action — `_title_gap_action(page, analysis, own_ext_title, own_h1, order)`

Build it from data already on the analysis (no new fetches):
- `keyword` from the analysis; competitor titles from `analysis["competitor_pages"]` (top 5 by rank, skip error rows).
- Trigger when ANY of: (a) keyword (case-insensitive, whitespace-normalized) not contained in our title AND at least 3 of the top-5 competitor titles contain it (or a token-overlap >= 0.6 of the keyword's words); (b) our title length < 30 chars; (c) our H1 empty.
- `type: "rewrite_title"`, `priority: "high"` when (a), else "medium".
- `instruction`: state exactly what is wrong ("Title 'X' does not contain 'kw' while 4/5 top competitors do") and what to do ("Rewrite the title (<=60 chars) to lead with the primary keyword intent; align H1 with the title without duplicating it verbatim").
- `evidence`: `{our_title, our_h1, keyword, competitor_titles: [{rank, title}]}`.
- `impact_score`: `60 + demand["score"] * 10`.

Where do we get our title/H1? `page` dict carries `title` and `h1` (assembled in `run()`). Use those. Emit at most ONE rewrite_title action per page (guard in `_action_points_for_analysis` caller — simplest: build it in `_attach_action_points` per page from the first analysis that triggers).

### 2. Depth target action — `_depth_action(page, analysis, order)`

- `ours = analysis["content_comparison"]["ours"]`, `bench = ...["benchmark"]`.
- Trigger when `ours["paragraph_count"] < 0.6 * bench["median_competitor_paragraphs"]` and `bench["median_competitor_paragraphs"] >= 8`.
- `type: "expand_depth"`, priority "medium".
- `instruction`: "Page has {ours.paragraph_count} paragraphs / {ours.heading_count} headings; top-5 competitor median is {median_paragraphs} paragraphs / {median_headings} headings. Add ~{median_paragraphs - ours.paragraph_count} paragraphs by implementing the missing-topic tasks rather than padding existing sections."
- `evidence`: the two profile dicts (ours numbers + benchmark).
- Max one per page (same guard pattern as title action).

### 3. Structural actions — `_structural_action(page, analysis, pattern, order)`

For each `analysis["structural_patterns"]` row with `competitors >= 2`, emit `type: "structural"`:
- `topic` = `pattern["signal"]`; `instruction` = pattern advice + the counts ("ours: {ours}, strongest competitor: {max_theirs}, seen on {competitors} competitors");
- priority: "high" when `competitors >= 4`, else "medium";
- `impact_score`: `20 + pattern["competitors"] * 8 + demand["score"] * 5`;
- evidence: the pattern row.
Cap at 4 per analysis.

### 4. PAA actions

For each `analysis["paa_coverage"]` row with `status == "missing"` (cap 4): `type: "answer_paa"`, priority "high" when also present in `serp_features.people_also_ask` top 4, else "medium"; instruction: "Add a question-form H3 '{question}' with a 40-60 word direct answer first, detail after. Candidate placement: FAQ block or nearest related section."; evidence: the coverage row. `impact_score`: `40 + demand["score"] * 10`.

### 5. Recommended outline (deterministic, rendered, not an action)

New `_recommended_outline(analysis) -> list[dict]` from `content_order_path`:
- take `clusters` (shared ordering clusters, already sorted) and `missing_clusters`;
- output rows `{position, label, status: "have"|"add", competitor_pages, sample_text}` — "have" rows from clusters where our page participates (cluster rows carry our presence via `ours_mean`-style fields; inspect `_content_order_path` cluster dict construction around line 2040 to find which key marks our positions — clusters built from `group["ours_positions"]`; if the cluster rows don't expose it, extend `_content_order_path` to include `"ours_present": bool(ours_positions)` when building each cluster);
- order rows by `competitor_mean_order` (have+add interleaved);
- attach as `analysis["recommended_outline"]` in `_build_gap` (after `content_order_path` is computed).
Render in `_html()::keywordCard` as a numbered list panel "Recommended Section Order" — each row `[have|add chip] label — seen on N competitors`, with sample_text in a `title=` tooltip. Place after the structural panel.

### 6. Wire new actions in

In `_action_points_for_analysis`, after the existing three loops append: structural actions, PAA actions, depth action (page-level guards for depth/title handled in `_attach_action_points` as noted). Keep the existing sort.

### 7. Cross-keyword dedupe in `_attach_action_points`

Before truncating to 30 per page: dedupe `page_actions` with key
`(action["type"], re.sub(r"\W+", " ", str(action.get("topic") or action.get("paragraph_index") or "")).strip().lower())`
keeping the highest `impact_score` per key; additionally, for `add_topic`/`strengthen_topic` pairs whose topic labels have `difflib.SequenceMatcher(None, a, b).ratio() >= 0.85`, keep only the higher-impact one (compare each kept action against already-kept ones; O(n²) is fine, n <= ~60). Record `action["merged_duplicates"] = k` when k > 0 duplicates were dropped. Same dedupe for the global aggregate before `[:80]`.

### 8. CSV / markdown

`_action_csv_rows` (~line 2991) and `_markdown_action_rows` (~line 3046): confirm they are type-agnostic (they map generic fields); if they hardcode type lists anywhere, include the new types. New action types must appear in the Task Board (`aggregateActionsSection` / `actionList` in `_html` are generic — verify, adjust only if they filter by type).

### 9. Tests (`tests/test_serp_gap.py`)

- `test_title_gap_action_triggers_on_missing_keyword`: competitor titles contain the keyword, ours doesn't -> action emitted with priority high and 4/5 evidence; ours contains keyword -> no action.
- `test_depth_action_uses_benchmark`: ours 5 paragraphs vs median 20 -> emitted with delta in instruction; ours 18 vs 20 -> not emitted.
- `test_structural_and_paa_actions_emitted`: analysis fixture with 1 structural pattern (competitors=4) and 1 missing PAA -> two actions, correct priorities.
- `test_action_dedupe_across_keywords`: two analyses with near-identical topic labels ("pricing, free plan" / "pricing free plan") -> one action survives with the higher impact_score and `merged_duplicates == 1`.
- `test_recommended_outline_orders_have_and_add`: synthetic content_order_path -> rows sorted by competitor_mean_order, statuses correct.

Run: `python -m pytest tests/test_serp_gap.py -q`.

## Constraints

- No new dependencies (difflib is stdlib). No changes to existing action types' fields or priorities.
- All new builders must tolerate missing keys (`.get()` everywhere) — analyses from cached older runs may lack `structural_patterns`/`paa_coverage`.
- Locate code by function name; line numbers approximate.

## Definition of done

- New action types: rewrite_title, expand_depth, structural, answer_paa — emitted, deduped, sorted, exported, rendered.
- Recommended outline computed and rendered per keyword.
- Tests pass.
