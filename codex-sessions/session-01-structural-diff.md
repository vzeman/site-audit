# Session 01 — Fix: structural GEO diff is never computed in serp-gap

## Context

Repo: `site-audit` (Python, root = repo root). The `site-audit serp-gap` flow (`site_audit/serp_gap.py`) compares one of our URLs against SERP competitor pages. `site_audit/competitive_analysis.py` contains `_structural_diff(ours_ext, theirs_ext)` (around line 612) which computes GEO-relevant structural gaps: FAQ/HowTo schema, question-form headings, statistics with units, external citations, tables, page depth. Each gap is a dict `{signal, ours, theirs, advice}`.

**Bug:** in `site_audit/serp_gap.py`, `_competitor_page()` (around line 1415) hardcodes `structural_gaps=[]` and `answerability=0.0` in both its return paths. Therefore `build_serp_paragraph_gap()` (in `competitive_analysis.py`, around line 862), which aggregates `page.structural_gaps` into `structural_patterns` (search for `structural_counts`), always returns an empty `structural_patterns` list in serp-gap reports.

## Task

### 1. Expose a public wrapper in `competitive_analysis.py`

Below `_structural_diff`, add:

```python
def structural_diff(ours_ext, theirs_ext) -> list[dict]:
    """Public wrapper used by serp_gap to compute per-competitor structural gaps."""
    return _structural_diff(ours_ext, theirs_ext)
```

Note: `_structural_diff` currently computes `ours_score = score_page(ours_ext)` and `theirs_score = score_page(theirs_ext)` but never uses them — leave that as-is in this session (answerability is wired below via `score_page` directly).

### 2. Wire it into `_competitor_page` in `serp_gap.py`

Current signature:
```python
def _competitor_page(target: CompetitiveTarget, cache: HttpCache, embedder: Embedder, config: SerpGapConfig) -> CompetitorPage:
```

Change to:
```python
def _competitor_page(target: CompetitiveTarget, cache: HttpCache, embedder: Embedder, config: SerpGapConfig, own_ext: "ExtractedPage | None" = None) -> CompetitorPage:
```

In the success path (after `ext = _fetch_and_extract(...)` returns non-None):

```python
structural_gaps: list[dict] = []
if own_ext is not None:
    try:
        structural_gaps = structural_diff(own_ext, ext)
    except Exception:
        structural_gaps = []
try:
    answerability = float(score_page(ext).total)
except Exception:
    answerability = 0.0
```

and pass `structural_gaps=structural_gaps, answerability=answerability` instead of the hardcoded empty values. The error path (fetch failed) keeps `structural_gaps=[]`.

**Before coding, inspect `site_audit/answerability.py`** — `score_page()` returns an `AnswerabilityScore` object; check the actual attribute name for the overall score (likely `total` or `score`) and use that. If it is a 0–100 scale, keep it as-is (do not renormalize).

Add the imports at the top of `serp_gap.py` next to the existing `competitive_analysis` imports (search for `from .competitive_analysis import`): add `structural_diff`, and `from .answerability import score_page`.

### 3. Update the caller

Find the call site of `_competitor_page(` inside `run()` in `serp_gap.py` (search for `_competitor_page(`). It runs inside the per-page loop where `own_ext` already exists (set by `own_ext = _fetch_and_extract(page.url, own_cache, refresh=False)`). Pass `own_ext=own_ext`.

### 4. Render structural patterns in the per-keyword report card

In `_html()` in `serp_gap.py`, locate `function keywordCard(a, pageIndex, keywordIndex)` (around line 4565). The analysis dict `a` now carries a non-empty `a.structural_patterns` (list of `{signal, competitors, advice, ours, max_theirs}` — see the end of `build_serp_paragraph_gap`). Add a collapsible panel after `actionsPanel` mirroring existing panels:

```js
const structRows=(a.structural_patterns||[]).filter(r=>r.competitors>=1);
const structPanel=structRows.length?collapsiblePanel('Structural / GEO Gaps',`${sectionNote('Page-structure signals where ranking competitors beat this page: schema, question headings, statistics, citations, tables, depth.')}<table><thead><tr><th>Signal</th><th>Competitors</th><th>Ours</th><th>Theirs (max)</th><th>Advice</th></tr></thead><tbody>${structRows.map(r=>`<tr><td>${esc(r.signal)}</td><td>${n(r.competitors)}</td><td>${esc(String(r.ours))}</td><td>${esc(String(r.max_theirs))}</td><td>${esc(r.advice)}</td></tr>`).join('')}</tbody></table>`,{meta:`${n(structRows.length)} signals`}):'';
```

and include `${structPanel}` in the returned card markup right after `${actionsPanel}`. Follow the exact helper conventions used by neighboring code (`collapsiblePanel`, `sectionNote`, `esc`, `n`).

Also note: `structural_counts` aggregation in `build_serp_paragraph_gap` overwrites `max_theirs` with the latest value instead of the max — fix it to keep the maximum:
```python
row["max_theirs"] = max(_as_num(row.get("max_theirs")), _as_num(gap.get("theirs")))
```
Use a small local helper that converts bool/int/None safely (`True -> 1`, `None -> 0`). Keep `ours` from the first occurrence.

### 5. Tests

In `tests/test_serp_paragraph_gap.py` add:

- `test_structural_patterns_aggregate_from_competitor_pages`: build two fake `CompetitorPage` objects (import from `site_audit.competitive_analysis`) with `structural_gaps=[{"signal": "Comparison / data tables", "ours": 0, "theirs": 2, "advice": "x"}]` and `[... "theirs": 5 ...]`, minimal paragraphs + embeddings (use small numpy float32 arrays, normalized rows), call `build_serp_paragraph_gap(...)`, assert `result["structural_patterns"][0]["competitors"] == 2` and `max_theirs == 5`.
- `test_competitor_page_computes_structural_gaps`: monkeypatch `site_audit.serp_gap._fetch_and_extract` to return a fake `ExtractedPage` (import from `site_audit.extractor`; construct with minimal required fields: `url, title, description, body, word_count, language`, plus `table_count=3`, `paragraphs=["..."]`). Build an own `ExtractedPage` with `table_count=0`. Call `_competitor_page(target, cache, embedder, config, own_ext=own)` with a stub embedder (object with `encode` returning a normalized numpy array) and a dummy cache; assert returned `CompetitorPage.structural_gaps` contains the `Comparison / data tables` signal.

Run: `python -m pytest tests/test_serp_paragraph_gap.py tests/test_serp_gap.py -q` — all green.

## Constraints

- No new dependencies. Do not reformat untouched code. Line numbers are approximate — locate by name.
- Do not change `_structural_diff` thresholds or messages.
- Do not change the `CompetitorPage` dataclass.

## Definition of done

- `_competitor_page` populates `structural_gaps` (when `own_ext` given) and `answerability`.
- `build_serp_paragraph_gap` aggregates a correct `max_theirs`.
- Report HTML shows the Structural / GEO Gaps panel when data exists.
- New tests pass; existing tests pass.
