# Session 03 — Rebuild the AI editor evidence payload (complete, sorted, honest)

## Context

Repo: `site-audit`. The AI editor brief for each analyzed URL is generated in `site_audit/serp_gap.py::_attach_ai_editor_briefs` (around line 2562), which calls `build_editor_brief_messages(page)` in `site_audit/ai_agent.py` (around line 422). The evidence JSON is assembled by `_editor_prompt_payload(page)` in `ai_agent.py` (around line 461).

Three defects to fix:

1. **Wrong paragraph selection.** `_editor_prompt_payload` takes `(analysis["paragraph_match_heatmap"]["rows"])[:10]` in page order — the agent sees the FIRST ten paragraphs, not the weakest.
2. **Impossible instruction.** The prompt demands keep/rewrite/move/merge/remove decisions for "existing paragraphs" and a "Final Article Draft", but the payload contains neither the full paragraph inventory nor any headings of our page.
3. **Computed evidence dropped.** Not passed today: `structural_patterns` (populated since session 01), `serp_features`/`paa_coverage` (since session 02), `content_comparison.benchmark` + `ours` profile, covered topic labels, `content_order_path` `missing_clusters`/`deviations`, per-row best-competitor info beyond one field.

Heatmap row shape (see `_paragraph_match_heatmap`, serp_gap.py around line 1682): `{paragraph_index, paragraph (<=420 chars), word_count, status, max_similarity, average_similarity, max_rank_impact, cells:[{url, status, similarity, rank, rank_weight, rank_impact, paragraph_index, paragraph (<=360)}]}`.

Topic shape (see `build_serp_paragraph_gap` in `competitive_analysis.py`): `{label, coverage, priority, competitor_paragraphs, competitor_urls, competitor_coverage, competitor_prevalence, best_competitor_rank, our_best_similarity, our_best_paragraph_index, our_best_paragraph, examples:[{url,title,rank,paragraph}]}`.

## Task — all in `site_audit/ai_agent.py` unless stated

### 1. Add full own-page inventory to the page dict (in `serp_gap.py`)

In `run()` where the per-page result dict is assembled (search for the dict that gets keys `"url"`, `"title"`, `"analyses"` appended to `page_results`), add:

```python
"own_content": {
    "headings": [
        {"order": i, "level": _safe_int(h.get("level")), "text": str(h.get("text") or "").strip()}
        for i, h in enumerate(own_ext.headers_rich or []) if str(h.get("text") or "").strip()
    ][:60],
    "paragraphs": [
        {"index": i, "word_count": len(p.split()), "text": p[:500]}
        for i, p in enumerate(own_paragraphs_for_page)
    ],
    "word_count": own_ext.word_count,
},
```

(`own_paragraphs_for_page` exists since session 02; if absent, derive from `own_ext.paragraphs[:config.max_paragraphs_per_page]`.)

### 2. Rewrite `_editor_prompt_payload(page)`

Replace the function body with a payload of this exact structure (preserve the function name and signature):

```python
{
  "url": ..., "title": ..., "h1": ...,                       # as today
  "keywords": page.get("keywords") or [],
  "own_page": page.get("own_content") or {},                 # NEW: full inventory
  "content_brief": page.get("content_brief") or {},
  "actions": [...18 as today, unchanged mapping...],
  "analyses": [per analysis:
    {
      "keyword": ..., "keyword_metrics": {...},              # as today
      "summary": analysis.get("summary") or {},
      "benchmark": (analysis.get("content_comparison") or {}).get("benchmark") or {},   # NEW
      "our_profile": _pick(analysis.get("content_comparison", {}).get("ours") or {},
                           ["paragraph_count","word_count","heading_count","h2_h3_count","coverage_ratio"]),  # NEW
      "structural_patterns": (analysis.get("structural_patterns") or [])[:8],           # NEW
      "serp_features": {                                                                # NEW
          "people_also_ask": [
              {"question": r.get("question"), "status": r.get("status"), "best_similarity": r.get("best_similarity")}
              for r in (analysis.get("paa_coverage") or [])[:10]
          ],
          "related_searches": (analysis.get("serp_features") or {}).get("related_searches") or [],
      },
      "topics": [...up to 12, mapping as today PLUS "our_best_paragraph_index": topic.get("our_best_paragraph_index")...],
      "covered_topics": [t.get("label") for t in analysis.get("covered_topics") or []][:12],  # NEW
      "content_order": {                                                                # NEW (replaces summary-only)
          "summary": (analysis.get("content_order_path") or {}).get("summary") or {},
          "missing_clusters": [
              {"label": c.get("label"), "competitor_pages": c.get("competitor_pages"), "sample_text": str(c.get("sample_text") or "")[:200]}
              for c in (analysis.get("content_order_path") or {}).get("missing_clusters") or []
          ][:8],
          "order_deviations": [
              {"label": c.get("label"), "direction": c.get("direction"), "delta": c.get("delta")}
              for c in (analysis.get("content_order_path") or {}).get("deviations") or []
          ][:8],
      },
      "paragraph_review": [...],                              # REWRITTEN, see below
    }
  ],
}
```

Add a tiny module-level helper `_pick(d: dict, keys: list[str]) -> dict`.

**`paragraph_review` rewrite:** take ALL heatmap rows, sort ascending by `max_similarity`, keep the 25 weakest, and map each to:

```python
{
  "paragraph_index": row.get("paragraph_index"),
  "status": row.get("status"),
  "max_similarity": row.get("max_similarity"),
  "max_rank_impact": row.get("max_rank_impact"),
  "word_count": row.get("word_count"),
  "paragraph": str(row.get("paragraph") or "")[:400],
  "best_competitor": _best_cell(row),   # NEW helper
}
```

`_best_cell(row)`: among `row["cells"]`, pick the cell with the highest `similarity`; return `{"url": cell["url"], "rank": cell["rank"], "similarity": cell["similarity"], "paragraph": str(cell["paragraph"])[:320]}` or `{}` if no cells. (Today's code reads non-existent keys `best_competitor_url` / `best_competitor_paragraph` from the row — they are always empty; this fixes that too.)

### 3. Size guard

Add `_shrink_editor_payload(payload: dict, max_chars: int = 120_000) -> dict`: serialize with `json.dumps(ensure_ascii=False)`; while too large, apply in order: (a) drop `own_page.paragraphs[i].text` down to 240 chars, (b) reduce `paragraph_review` to 15 rows, (c) drop `topics[*].example_paragraph`/`examples` texts to 160 chars, (d) drop `content_order.missing_clusters` sample_text. Call it at the end of `_editor_prompt_payload`. Keep it deterministic and lossless on structure (only truncate strings / shorten lists).

### 4. Update the prompt in `build_editor_brief_messages`

Keep the section structure, change the user message rules to match reality:

- Replace the sentence about existing paragraphs with: "Our page's complete content is in `own_page` (headings in order, paragraphs numbered by `index`). In **Paragraph Decisions**, give a decision (keep, rewrite, move, merge, remove) for every paragraph listed in `paragraph_review`, referencing paragraphs as [P<index>]. For paragraphs not listed, only mention them if they conflict with a new section."
- Keep **Final Article Draft** but add: "Assemble it from `own_page` order: reuse kept paragraphs by reference ([P<index>] + first 6 words), include rewritten/new paragraphs in full, and mark removals explicitly. Use the `benchmark` (median competitor paragraphs/headings) as the size target. Cover every `missing`-status question from `serp_features.people_also_ask` either in a section or a FAQ block. Respect `structural_patterns` advice (tables, question-form headings, statistics, schema). Do not duplicate topics listed in `covered_topics`."
- Keep: no competitor wording, no invented demand metrics, no duplicate instructions, JSON evidence only, no tools/files.

### 5. Tests (`tests/test_serp_gap.py`)

The file already imports from `site_audit.ai_agent` (see top of file). Add:

- `test_editor_payload_sorts_paragraph_review_by_weakness`: page fixture with one analysis whose heatmap has 3 rows with `max_similarity` 0.9 / 0.2 / 0.5 and cells; assert order of `paragraph_review` indexes is [weakest first] and `best_competitor.url` comes from the highest-similarity cell.
- `test_editor_payload_includes_new_evidence_keys`: assert presence of `own_page`, `benchmark`, `structural_patterns`, `serp_features`, `covered_topics`, `content_order` keys.
- `test_shrink_editor_payload_respects_budget`: build an oversized payload (long paragraph texts), assert result serializes under the budget and `paragraph_review` length <= 25.

Run: `python -m pytest tests/test_serp_gap.py -q`.

## Constraints

- No new dependencies. Do not change `_paragraph_match_heatmap`, `build_serp_paragraph_gap`, or thresholds.
- `_editor_prompt_payload` must never raise on missing keys (everything `.get()` with defaults) — page dicts from older cached runs may lack `own_content`/`serp_features`.
- Locate code by function name; line numbers approximate.

## Definition of done

- Agent payload contains the full own-page inventory, weakest-first paragraph review with real best-competitor cells, benchmark, structural patterns, PAA, covered topics, content-order gaps.
- Prompt matches the data it receives. Size guard active. Tests pass.
