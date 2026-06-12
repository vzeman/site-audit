# Session 05 — Verification loop: re-score the agent's recommended page against SERP topics

## Context

Repo: `site-audit`. After session 04, each analyzed page may carry `page["ai_recommendation"]["data"]` (the structured recommendation: `paragraph_decisions` with rewrites, `new_sections` with drafts). The deterministic pipeline computed SERP topic clusters in `competitive_analysis.py::build_serp_paragraph_gap` (~line 862): per-topic centroid vectors exist as local `centroids` but are NOT stored in the output topics.

Goal: after the agent produces a recommendation, assemble the "recommended page" text blocks, embed them with the same local embedder, and re-score topic coverage and PAA coverage. Report before/after ("missing 6 -> 0"). If critical/high topics remain missing in harnext mode, run one repair turn.

Thresholds: covered >= 0.78, partial >= 0.62 (same constants as `build_serp_paragraph_gap` args `missing_threshold` / `covered_threshold`).

## Task

### 1. Persist topic centroids

In `competitive_analysis.py::build_serp_paragraph_gap`, when appending each topic dict, add:

```python
"centroid": [round(float(x), 5) for x in centroid.tolist()],
```

Size note: ~12 topics x 384 floats per keyword is fine for report JSON. BUT exclude centroids from places where they would bloat user-facing payloads:
- `serp_gap.py::_topic_coverage_matrix` and `_editor_prompt_payload` (ai_agent.py) map explicit topic fields already — verify they do not copy `centroid` (they map field-by-field; just confirm).
- In `serp_gap.py::_write_outputs` / `_html`: the analysis dicts embed full topics. To avoid doubling report size, strip centroids from the HTML payload only: in `_html()`, before serializing `data` (find where the payload is JSON-embedded into the page, search for `json.dumps` inside `_html`), deep-strip key `"centroid"` from all topic dicts via a small `_strip_centroids(payload)` helper in serp_gap.py. Keep centroids in `serp_gap.json` (machine output) — that is where verification reads them on re-runs.

### 2. New module `site_audit/draft_verification.py`

```python
from typing import Callable
import numpy as np

def assemble_recommended_blocks(own_paragraphs: list[str], recommendation: dict) -> list[dict]:
    """Returns ordered blocks: [{"source": "kept|rewrite|new", "ref": "P3"|"S0", "text": str}]."""
```

Rules:
- iterate paragraphs by index; decision lookup from `recommendation["paragraph_decisions"]`;
- `keep` -> original text; `rewrite` -> the `rewrite` text; `move`/`remove` -> excluded; `merge` -> excluded (its facts are assumed merged into a rewrite);
- `new_sections`: insert at `placement_after_paragraph` position (-1 = before everything); block text = `heading + "\n" + draft`;
- missing/blank decisions default to `keep`.

```python
def verify_recommendation(
    blocks: list[dict],
    analyses: list[dict],
    embed_fn: Callable[[list[str]], np.ndarray],
) -> dict:
```

- Embed all block texts once: `matrix = embed_fn([b["text"] for b in blocks])` (normalized float32 — the project `Embedder.encode` already returns normalized vectors; do not re-normalize).
- For each analysis, for each topic with a `centroid`: `best = float(np.max(matrix @ np.asarray(topic["centroid"], dtype=np.float32)))`; classify after (0.78/0.62); record `{"keyword", "label", "priority", "before": topic["coverage"], "after", "best_similarity", "best_block_ref"}`.
- PAA: for each `analysis["paa_coverage"]` row, re-embed the question? No — questions were embedded in session 02 but not persisted. Re-embed questions here with the same `embed_fn` and score against `matrix`; record before (`row["status"]`) / after.
- Return:

```python
{
  "topics": [...rows...],
  "paa": [...rows...],
  "summary": {
    "missing_before": n, "missing_after": n,
    "partial_before": n, "partial_after": n,
    "paa_missing_before": n, "paa_missing_after": n,
    "unresolved_critical": [labels of priority critical/high topics still not 'covered'],
  },
}
```

### 3. Wire into `_attach_ai_editor_briefs` (serp_gap.py)

After a valid recommendation is obtained for a page:

```python
own_paragraphs = [p["text_full"] ...]
```

— careful: `own_content.paragraphs[*].text` is truncated to 500 chars (session 03). For verification use the untruncated paragraphs: extend the data passed into `_attach_ai_editor_briefs` (session 04 already passes `own_exts`); use `own_exts[url].paragraphs[:config.max_paragraphs_per_page]`.

```python
blocks = assemble_recommended_blocks(own_paragraphs, rec_data)
verification = verify_recommendation(blocks, page.get("analyses") or [], embed_fn=lambda texts: embedder.encode(texts, batch_size=64).astype(np.float32))
page["ai_recommendation"]["verification"] = verification
```

`embedder` is not currently available in `_attach_ai_editor_briefs` — extend its signature to accept `embedder: Embedder | None = None` and pass it from `run()`. When `embedder is None`, skip verification silently.

**Repair turn (harnext mode only):** if `verification["summary"]["unresolved_critical"]` is non-empty AND this was a fresh (non-cache-hit) session, run ONE follow-up workspace session with prompt: "Verification: these critical/high SERP topics are still not covered by your recommendation: <labels with best_similarity>. Update recommendation.json (add or strengthen sections) to cover them. Modify recommendation.json and brief.md only." Then re-parse, re-validate, re-verify, and store the final result plus `"repair_attempted": true`. Never loop more than once.

### 4. Render before/after in the report

In `_html()::recommendationSection` (from session 04), when `page.ai_recommendation?.verification` exists, prepend a summary strip:

```
Coverage check: missing {missing_before} -> {missing_after} · partial {partial_before} -> {partial_after} · PAA missing {paa_missing_before} -> {paa_missing_after}
```

with chips, plus a collapsed table of topic rows (Keyword | Topic | Priority | Before -> After | Best similarity). Highlight rows still missing after (`chip missing`). If `unresolved_critical` non-empty, show a warning line.

Also add the same numbers to `_todo_markdown` in the page's AI brief block (search `_append_ai_editor_brief_markdown`, ~line 3084): append a `Coverage check:` line when verification exists.

### 5. Tests — `tests/test_draft_verification.py` (new)

Use a stub `embed_fn` returning hand-built normalized vectors (e.g. 4-dim one-hot-ish) so similarities are exact:

- `test_assemble_blocks_orders_and_filters`: 4 paragraphs; decisions keep/rewrite/remove/keep + one new section after index 1 -> assert block order, sources, refs, the rewrite text used, removed paragraph absent, default-keep when decision missing.
- `test_verify_marks_topic_covered_after_new_section`: topic centroid aligned with new-section vector -> before "missing", after "covered", `best_block_ref` = the new section.
- `test_verify_summary_counts_and_unresolved`: one critical topic stays missing -> appears in `unresolved_critical`.
- `test_verify_handles_missing_centroids`: topics without `centroid` key are skipped without error.

Run: `python -m pytest tests/test_draft_verification.py tests/test_serp_gap.py tests/test_serp_paragraph_gap.py -q`.

## Constraints

- No new dependencies. Verification must be optional/fail-soft: any exception -> `page["ai_recommendation"]["verification"] = {"status": "error", "message": ...}`, never crash the run.
- Do not alter coverage thresholds; import/duplicate the constants 0.78/0.62 as module constants `COVERED_THRESHOLD` / `PARTIAL_THRESHOLD` in `draft_verification.py` with a comment pointing at `build_serp_paragraph_gap`.
- Locate code by function name; line numbers approximate.

## Definition of done

- Topics persist centroids (stripped from the HTML payload).
- Valid recommendations get a before/after coverage verification, rendered in HTML and TODO markdown.
- One bounded repair turn in harnext mode. All tests pass.
