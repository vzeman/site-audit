# Session 02 — Capture SERP features (People Also Ask, answer box, related searches) and compute PAA coverage

## Context

Repo: `site-audit`. `site_audit/serp_gap.py` fetches SERPs in `_fetch_serp()` (around line 1173). For provider `serper` the full Google response is cached as `payload["raw"]` (the raw Serper JSON). For provider `dataforseo`, `fetch_dataforseo_serp` returns a payload whose items are read via `_serp_items(payload)` (imported from `competitive_analysis.py`, around line 542). Today only organic results are consumed (`_serp_result_rows`, around line 1252). People Also Ask questions, answer boxes, and related searches are fetched, paid for, and thrown away.

The per-keyword analysis dict is built in `run()`: `gap = _build_gap(page, kw, own_ext, competitor_pages, embedder, config)` — find this call inside the keyword loop in `run()`. `_build_gap` (around line 1465) already attaches extra keys to `gap` (e.g. `gap["scatter"]`, `gap["paragraph_match_heatmap"]`). Own paragraph embeddings are computed inside `_build_gap` as `own_embeddings`.

Coverage thresholds used everywhere: similarity >= 0.78 -> "covered", >= 0.62 -> "partial", else "missing" (see `build_serp_paragraph_gap` in `competitive_analysis.py`).

## Task

### 1. Add `_serp_features(payload) -> dict` in `serp_gap.py`

Place it near `_serp_result_rows`. Output shape (always all keys, empty lists/dicts when absent):

```python
{
    "people_also_ask": [{"question": str, "snippet": str, "url": str, "title": str}],
    "related_searches": [str],
    "answer_box": {"title": str, "answer": str, "url": str},  # {} when absent
}
```

Serper raw shape (`payload["raw"]`):
- `peopleAlsoAsk`: list of `{question, snippet, title, link}`
- `relatedSearches`: list of `{query}`
- `answerBox`: `{title, answer | snippet, link}` (use `answer` falling back to `snippet`)

DataForSEO: iterate `_serp_items(payload)` (import is already present or add it next to the existing `from .competitive_analysis import ...`). Inspect `site_audit/dataforseo.py` before coding to confirm exact item shapes; expected types: `people_also_ask` (item has `items` list whose elements have `title` = question and nested `expanded_element` with `description`/`url`), `related_searches` (item has `items` list of strings), `featured_snippet` (`title`, `description`, `url`). Code defensively: every access via `.get()` with defaults; never raise.

Detection of provider: same pattern as `_serp_result_rows` (`provider == "serper" or "organic" in raw`).

Cap lists: `people_also_ask[:10]`, `related_searches[:10]`.

### 2. Compute PAA coverage against our paragraphs

Add to `serp_gap.py`:

```python
def _paa_coverage(features: dict, own_paragraphs: list[str], own_embeddings: np.ndarray, embedder: Embedder) -> list[dict]:
```

For each question in `features["people_also_ask"]`:
- embed the question text with `embedder.encode([question])` (batch all questions in ONE encode call for speed, then iterate),
- if `own_embeddings` is empty -> status "missing", best_similarity 0.0, best_paragraph_index None,
- else `sims = own_embeddings @ q_emb`; best index/sim; status by thresholds 0.78/0.62 ("covered"/"partial"/"missing").

Return rows:
```python
{"question": q, "status": status, "best_similarity": round(best, 4), "best_paragraph_index": idx, "best_paragraph": own_paragraphs[idx][:240] if idx is not None else ""}
```

### 3. Wire into the pipeline

`_build_gap` does not receive the SERP payload. Do NOT change `_build_gap`'s signature. Instead, in `run()` right after `gap = _build_gap(...)` (inside the keyword loop), add:

```python
features = _serp_features(serp)
gap["serp_features"] = features
gap["paa_coverage"] = _paa_coverage(features, own_paragraphs_for_page, own_embeddings_for_page, embedder)
```

Problem: `own_paragraphs`/`own_embeddings` live inside `_build_gap`. Solution: compute them once per page in `run()` before the keyword loop:

```python
own_paragraphs_for_page = (own_ext.paragraphs or [])[:config.max_paragraphs_per_page]
own_embeddings_for_page = embedder.encode(own_paragraphs_for_page, batch_size=64).astype(np.float32) if own_paragraphs_for_page else np.zeros((0, 0), dtype=np.float32)
```

then change `_build_gap` to ACCEPT these as optional parameters to avoid double-encoding:

```python
def _build_gap(page, keyword, own_ext, competitor_pages, embedder, config, own_paragraphs=None, own_embeddings=None) -> dict:
    if own_paragraphs is None:
        own_paragraphs = (own_ext.paragraphs or [])[:config.max_paragraphs_per_page]
    if own_embeddings is None:
        own_embeddings = embedder.encode(...)  # existing line
```

and pass them from `run()`. This keeps the existing behavior for any other caller.

### 4. Render in the report

In `_html()`, in `function keywordCard(a, ...)` add a collapsible panel (after the topics/review two-col block) shown only when `(a.paa_coverage||[]).length`:

- Title: `People Also Ask Coverage`, meta `${covered}/${total} covered`.
- Table columns: Question | Status (chip-style: reuse `chip missing/partial/covered` classes) | Best similarity | Closest paragraph (escaped, truncated by the data already).
- Below the table, if `(a.serp_features?.related_searches||[]).length`, render one line: `Related searches: q1 · q2 · ...` (escaped).

### 5. Persist + export

- `_csv_rows` / `_action_csv_rows` (around lines 2973–3020): leave unchanged.
- `_todo_markdown` (around line 3116): inside the per-keyword section (find where keyword analyses are written), add a `### People Also Ask` block listing `missing`/`partial` questions only, one per line as `- [missing] question`. Skip the block when empty. Follow the `_append_unique` pattern used in that function.

### 6. Tests (`tests/test_serp_gap.py`)

- `test_serp_features_parses_serper_payload`: feed `{"meta":{"provider":"serper","status":"ok"},"raw":{"peopleAlsoAsk":[{"question":"How much does X cost?","snippet":"s","title":"t","link":"https://a"}],"relatedSearches":[{"query":"x pricing"}],"answerBox":{"title":"t","answer":"a","link":"https://b"}}}` and assert all three groups parsed.
- `test_serp_features_handles_missing_blocks`: empty raw -> all keys present, empty.
- `test_paa_coverage_classifies_thresholds`: stub embedder whose `encode` returns predetermined normalized vectors so one question scores ~0.9 vs paragraph 0 (covered) and another ~0.3 (missing). Assert statuses, indexes, rounding.

Run: `python -m pytest tests/test_serp_gap.py -q`.

## Constraints

- No new dependencies. No behavior change for organic-result selection. Defensive parsing only — a malformed SERP payload must never crash `run()`.
- Locate everything by function name; line numbers approximate.

## Definition of done

- Every keyword analysis dict contains `serp_features` and `paa_coverage`.
- Report shows PAA coverage panel; TODO markdown lists uncovered questions.
- New + existing tests pass.
