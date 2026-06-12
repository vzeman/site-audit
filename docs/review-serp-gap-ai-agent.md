# Review: SERP gap AI agent (Harnext) and report value

Date: 2026-06-12. Scope: `serp_gap.py`, `ai_agent.py`, `competitive_analysis.py`, serp-gap report HTML.

## Verdict

The deterministic pipeline computes rich evidence (~40 metrics per paragraph/topic), but the "AI agent" sees maybe 15% of it, and Harnext is used as a plain one-shot chat completion — not as an agent. The biggest single bug: the GEO structural diff is never computed in the serp-gap flow, so `structural_patterns` is always empty.

---

## 1. Confirmed bugs

**B1. `structural_gaps` hardcoded to `[]`** — `serp_gap.py:1428,1440` (`_competitor_page`). `_structural_diff()` in `competitive_analysis.py:612` computes exactly the GEO signals a specialist wants (FAQ/HowTo schema, question-form headings, stats with units, external citations, tables, depth), but serp-gap never calls it. `build_serp_paragraph_gap` aggregates `structural_patterns` from these — always empty in serp-gap reports. One-line fix: compute `_structural_diff(own_ext, ext)` in `_competitor_page` (needs own_ext passed in).

**B2. Agent sees the first 10 paragraphs, not the worst 10** — `ai_agent.py:479-486`: `paragraph_review` slices heatmap rows `[:10]` in page order. Sort by `max_similarity` ascending (or `max_rank_impact`) before slicing — or better, send all rows (see below).

**B3. "Final Article Draft" is structurally impossible** — the prompt (`build_editor_brief_messages`) demands a full final article with keep/rewrite/move/merge/remove decisions for "existing paragraphs", but the payload contains ≤10 paragraphs truncated to 320 chars, no own headings, no full paragraph inventory. The model must hallucinate the rest of the page. Either send the full page content or drop the final-draft requirement.

**B4. Inconsistent thresholds** — serp-gap coverage uses 0.62/0.78 (`build_serp_paragraph_gap`), the older compare flow uses 0.70 (`_missing_topics`), off-intent uses 0.52. Not calibrated per language or paragraph length; document or unify.

**B5. Brief markdown is `esc()`-escaped, not rendered** — `serp_gap.py:4494` shows the brief as escaped plain text in a `prompt-box`. Render markdown (you already ship D3; a tiny md renderer or pre-rendered HTML server-side).

**B6. SERP features ignored** — Serper/DataForSEO payloads contain People Also Ask, answer boxes, related searches; only organic ranks are consumed. PAA questions are the cheapest high-value GEO evidence you're already paying for.

---

## 2. Harnext agent: what's wrong and how it should work

### Current state
`HarnextOpenRouterClient` flattens messages to a single prompt, sets `max_turns=3`, and the system prompt explicitly forbids tools/file reads ("Do not inspect files, browse, run commands"). So Harnext adds npm CLI + SDK dependency, an asyncio restriction (`_run_async` raises inside running loops), and auto-update overhead — for an OpenRouter call you could make directly. There is no agentic value today.

### Data computed but never given to the agent
- `paragraph_match_heatmap` cells: per-competitor similarity, `rank_weight`, `rank_impact`, best matching competitor paragraph per competitor (only row-level max survives)
- `topic_coverage_matrix` (who covers which topic, ours vs top 6)
- `content_comparison.benchmark` (median competitor paragraphs/headings/h2h3, max topics) — the "how big should this page be" answer
- `competitor_pages` profiles (heading counts, paragraph counts per rank)
- `structural_patterns` (empty anyway, see B1)
- `content_order_path` full sequence (only `summary` passed) — competitor article outlines, the key input for a recommended page structure
- `covered_topics` — agent can't avoid duplicating what's already good
- own `headers_rich`, full paragraph list with indexes, word counts
- topic `examples` beyond the first one (up to 3 computed, 4 stored in matrix)
- SERP features (PAA, answer box) — not even computed

### Recommended redesign (this is where "much more detailed work" comes from)

**Step 1 — write evidence to disk, let Harnext actually be an agent.**
Per analyzed URL, dump a working directory:

```
serp_gap/agent/<url-slug>/
  evidence.json        # the FULL gap dict: topics, heatmap w/ cells, coverage matrix,
                       # benchmark, structural_patterns, content_order_path, action_points
  our_page.md          # full extracted page: title, meta, H1, heading tree, all paragraphs
                       # numbered [P0]..[Pn]
  competitors/<rank>-<domain>.md   # competitor outline + paragraphs
  serp.json            # ranks, PAA, related searches
```

Invoke Harnext with file tools enabled, `max_turns` 15–25, and a task prompt: "Read evidence.json and our_page.md first; consult competitors/ as needed; write recommendation.json + brief.md." This converts truncation limits into agent-driven retrieval — the agent reads exactly what it needs at full fidelity instead of you guessing what fits in one prompt.

**Step 2 — demand structured output, not just markdown.**
`recommendation.json` schema, validated after the run:

```json
{
  "page_assessment": {"is_right_target_page": true, "reason": "..."},
  "title": {"current": "...", "recommended": "...", "reason": "..."},
  "meta_description": {"recommended": "..."},
  "h1": {"recommended": "..."},
  "outline": [
    {"level": 2, "heading": "...", "status": "keep|rename|new|remove",
     "maps_to_topic": "label", "source_paragraphs": [3,4]}
  ],
  "paragraph_decisions": [
    {"index": 0, "decision": "keep|rewrite|move|merge|remove",
     "reason": "...", "rewrite": "full replacement text or null",
     "evidence": {"max_similarity": 0.41, "off_intent": true}}
  ],
  "new_sections": [
    {"heading": "...", "placement_after_paragraph": 7, "topic": "label",
     "draft": "...", "format": "paragraphs|table|faq|steps",
     "covers_paa": ["question 1"]}
  ],
  "structured_data": [{"type": "FAQPage", "reason": "4/5 competitors have it"}],
  "internal_links": [{"anchor": "...", "from_hint": "...", "reason": "..."}]
}
```

Every paragraph index must appear in `paragraph_decisions` — verifiable mechanically. `brief.md` stays as the human-readable rendering.

**Step 3 — close the loop with the local embedder (cheap, high value, nobody else does this).**
After the agent produces drafts, embed the recommended page (existing kept paragraphs + rewrites + new sections) with the same model and re-run coverage against the SERP topic centroids. Report before/after: "missing 6 → 0, partial 4 → 1, off-intent 3 → 0". If critical topics remain uncovered, send the diff back to the agent for one repair turn. This turns the brief from prose into a verified deliverable and directly answers "how should the page look to win".

**Step 4 — fix the fallback story.** If Harnext CLI/SDK is unavailable, fall back to OpenRouter with the same JSON contract (smaller, pre-chunked payloads), so the feature degrades instead of erroring.

---

## 3. Report value for a GEO/SEO specialist

### Genuinely valuable (keep)
- Topic coverage classification with competitor prevalence + rank-based priority — the core, and it's sound
- Paragraph match heatmap with rank_impact — paragraph-level evidence most tools don't have
- Off-intent paragraph detection (dilution) — rare and useful
- Action points with placement + acceptance criteria, CSV/markdown export
- Demand enrichment from GSC/Ahrefs, budget guard, dry-run, caching
- AI keyword selection with evidence-only prompt and honest "demand metrics absent" rule

### Missing (a specialist would ask immediately)
1. **Title/meta/H1 gap** — competitor titles are fetched and embedded but there's no "your title vs top-10 titles" comparison or rewrite recommendation. Cheap, high CTR impact.
2. **Recommended heading outline** — `content_order_path` computes competitor content sequences but renders only as a chart. A specialist wants: "recommended H2/H3 outline, merged from top-3 competitors, mapped to your existing sections." (Step 2 schema covers this.)
3. **SERP intent/format evidence** — `_recommended_content_format()` guesses format from keyword words ("vs" → table). Derive it from what's actually ranking (listicles? product pages? docs?) and from SERP features.
4. **PAA / question coverage** — core GEO input, ignored (B6).
5. **Structural GEO diff** — computed but broken (B1). Once fixed, convert `structural_patterns` into action points (currently topics + paragraphs only generate actions).
6. **Depth target** — benchmark medians exist but never become "add ~600 words / 4 sections to reach the top-5 median".
7. **Entity/term gap** — only 4-word c-TF-IDF topic labels carry term advice. A proper "terms competitors use that you don't" table (you have all the text and embeddings) would strengthen `suggested_terms`.
8. **Internal-link actions** — the main audit computes link recommendations; serp-gap recommends none, yet "add internal links to target page" is step 8 of your own editorial doc.
9. **Cross-keyword dedupe** — `_attach_action_points` concatenates actions from all keywords of a page; near-identical topics across keywords produce duplicate tasks. Dedupe by topic-centroid similarity.

### Low value / noise (trim or demote)
- Three near-duplicate cluster visualizations per report (per-keyword cluster cards, overview cluster cards, treemap-ish ridgeline). Keep one.
- The all-entities scatterplot (keywords + URLs + titles + headings + paragraphs) is impressive but rarely actionable; fine as collapsed diagnostics, which you partly do — make it consistently collapsed-by-default.
- `visual_summary` prose strings duplicate the chips/summary numbers.
- Generic boilerplate in every action (`avoid` lists, paragraph rules repeated per action) inflates JSON and the agent prompt; reference a single shared guidelines block.

---

## 4. Priority order

1. Fix B1 (structural diff) — one-line-ish, unlocks GEO actions already designed
2. Fix B2/B3 (agent payload: sorted/complete paragraphs, drop or fix Final Article Draft)
3. Step 1+2 redesign: evidence-on-disk + Harnext with tools + structured `recommendation.json`
4. Step 3 verification loop (embed agent draft, re-score coverage, show before/after)
5. PAA + title/outline recommendations (B6, missing #1-#2)
6. Action dedupe + benchmark-driven depth targets
7. Report trimming (one cluster view, markdown rendering of brief)
