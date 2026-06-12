# Session 08 — End-to-end test: https://www.liveagent.com/features/ai-answer-improver/

Run AFTER sessions 01–07 are merged and the full test suite is green. This session validates the pipeline on a real URL and produces the artifacts for a human quality review.

## Prerequisites (environment, not code)

- `.env` in repo root: `SERPER_API_KEY` (or DataForSEO credentials + `--provider dataforseo`), `OPENROUTER_API_KEY`, optionally `OPENROUTER_MODEL`.
- Harnext agent installed: `npm install -g harnext` and `python -m pip install -e '.[agent]'`. Verify: `python -c "from site_audit.ai_agent import harnext_status; print(harnext_status())"` → `(True, ...)`.
- serp-gap requires an existing base audit (`projects/liveagent.com/report/pages.json`). If `projects/liveagent.com/` does not exist, first run a limited base audit — check `site-audit run --help` for the page-limit flag and crawl only a few hundred pages (the features section must be included).

## Run

```bash
site-audit serp-gap liveagent.com \
  --urls https://www.liveagent.com/features/ai-answer-improver/ \
  --keywords "ai answer improver" "ai reply assistant for customer support" \
  --results-per-keyword 5 \
  --budget-usd 2 \
  --ai-agent --ai-agent-provider harnext
```

(Adjust flag spellings to whatever `site-audit serp-gap --help` actually shows — do not guess.) First do a `--dry-run` to confirm page/keyword selection, then the real run.

## Pipeline assertions (Codex: verify mechanically and report)

Open the newest `projects/liveagent.com/serp_gap/report/<run>/serp_gap.json` and assert:

1. **Extraction sanity** — `own_content.paragraphs` contains real article paragraphs ("AI Answer Improver is a powerful enhancement tool…", FlowHunt/OpenAI setup steps) and does NOT contain navigation/footer noise (no "Choose your language", no cookie-consent text, no menu link lists). If boilerplate leaks in, file it as a bug with examples — it poisons every downstream similarity score.
2. `analyses[*].structural_patterns` non-empty when competitors have tables/FAQ schema/question headings we lack (session 01).
3. `analyses[*].serp_features.people_also_ask` non-empty for at least one keyword (PAA exists for this query space) and `paa_coverage` rows have statuses (session 02).
4. The cached agent prompt under `projects/liveagent.com/cache/serp_gap/ai_agent/` contains `own_page` with the full paragraph inventory and `paragraph_review` sorted weakest-first (session 03).
5. Agent workspace exists: `.../agent/<slug>/evidence.json`, `our_page.md` (with `[P0]`… markers), `competitors/*.md`, `serp.json`, `TASK.md` (session 04).
6. `recommendation.json` exists, `validate_recommendation` returns no errors, every own paragraph index has exactly one decision (session 04).
7. `ai_recommendation.verification` present with before/after counts, and `missing_after <= missing_before` (session 05).
8. Action points include at least one of: `rewrite_title`, `structural`, `answer_paa`; no near-duplicate add_topic/strengthen_topic pairs across the two keywords (session 06).
9. Open `index.html`: AI brief renders as formatted markdown; recommendation tables, PAA panel, Structural/GEO panel, Recommended Section Order, and the coverage-check strip are all visible.

## Quality rubric (for the human review of "is the recommendation spot on")

Page facts (as of 2026-06): title "AI Answer Improver | LiveAgent - Help Desk Software & Live Chat"; H1 "AI Answer Improver"; sections: what it is, how to use, how it works (Improve / Extend / Simplify / Custom instructions), formality options, setup via FlowHunt and OpenAI (step-by-step), 3-question FAQ, related articles. Plan availability is mentioned (Small/Medium/Large/Enterprise, not Free/Legacy). Notably absent from the page: concrete before/after reply examples, measurable benefits (time saved, CSAT), security/data-privacy of AI processing, supported languages, comparison with generic tools (ChatGPT alone) or competitor helpdesk AI assistants, customer proof/quotes, pricing numbers.

A "spot on" recommendation should:

- **Target page check**: confirm this URL is the right target for "ai answer improver" (it is — branded feature page) and NOT propose retargeting.
- **Title**: keep brand pattern but likely recommend a benefit/intent extension; flag only if competitors' titles show a clearly better pattern. A recommendation that strips "LiveAgent" entirely is a miss.
- **Missing topics**: expect recommendations among — benefits with measurable outcomes, before/after examples of improved replies, data privacy/security of customer data sent to AI, language support, pricing/plan detail, comparison vs writing replies with ChatGPT manually, troubleshooting/limitations. Recommendations to add generic "what is customer support" content = failure (off-intent).
- **Paragraph decisions**: setup steps (FlowHunt/OpenAI) should be `keep` (unique, high-value, original content — exactly what GEO rewards); marketing filler ("Are you ready to add a real wow-factor…") should be `rewrite` or `remove`. If the agent recommends deleting the setup walkthroughs, the evidence weighting is wrong.
- **PAA**: missing questions should map to the FAQ block recommendation; the page already has an FAQ — expect `FAQPage` structured-data advice if competitors carry it, plus expansion of the 3 existing questions.
- **Verification strip**: missing topics after applying the draft should drop to ~0; if not, the repair turn should have fired.
- **No competitor copying**: drafts must be original; spot-check phrases from `competitors/*.md` against drafts.
- **No hallucinated facts**: drafts must not invent pricing numbers, statistics, or capabilities (e.g. languages the product doesn't support). Any invented metric = fail; the prompt mandates "demand metrics absent" honesty.

## Output of this session

Write `codex-sessions/e2e-results-liveagent.md` summarizing: each assertion pass/fail with file paths, screenshots/excerpts of the recommendation, and any bugs found (with the exact JSON snippets). Do not fix bugs in this session — file them.
