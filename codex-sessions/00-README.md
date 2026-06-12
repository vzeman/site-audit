# Codex implementation sessions — SERP gap AI agent upgrade

> **STATUS: Sessions 01–07 are already implemented and committed on branch `serp-gap-ai-agent`** (one commit per session, full test suite green). Do NOT re-run them with Codex. Only `session-08-e2e-test-liveagent.md` remains — it must run on a machine with SERPER/OPENROUTER keys and the Harnext CLI. The briefs below are kept as the implementation record.

Source review: `docs/review-serp-gap-ai-agent.md`. Seven sessions, run in order. Each session file is a complete, self-contained brief for one Codex run — Codex has no memory of previous sessions, so every brief restates needed context.

## How to run each session

```bash
cd ~/work/site-audit
git status              # must be clean
git checkout -b serp-gap-ai-agent   # once, before session 1

# Session N:
codex exec --full-auto "$(cat codex-sessions/session-0N-*.md)"
# or interactively: codex, then paste the file content

# After Codex finishes:
python -m pytest tests/test_serp_gap.py tests/test_serp_paragraph_gap.py tests/test_cli.py -q
git diff                # review every change
git add -A && git commit -m "serp-gap: session 0N <title>"
```

If a session fails its acceptance criteria, re-run Codex in the same branch with the brief plus a note describing what failed. Do not start the next session until tests pass and the diff is reviewed.

After session 7: `python -m pytest -q` (full suite) and an end-to-end smoke test:
```bash
site-audit serp-gap <yourdomain> --urls <one-url> --keywords "<one keyword>" --dry-run
```

## Session order and dependencies

1. `session-01` Fix structural diff (independent)
2. `session-02` Capture SERP features / PAA (independent)
3. `session-03` Rebuild AI agent evidence payload (needs 01 + 02)
4. `session-04` Agent workspace + structured recommendation.json (needs 03)
5. `session-05` Verification loop (needs 04)
6. `session-06` Deterministic specialist features (needs 01; independent of 04/05)
7. `session-07` Report polish + docs (needs all)

## Global constraints (repeated in every brief)

- Python 3.10+, no new third-party dependencies unless the brief explicitly allows it.
- Match existing code style: module-private helpers prefixed `_`, dataclasses, `from __future__ import annotations` where present, no reformatting of untouched code.
- Line numbers in briefs are approximate — always locate code by function/class name.
- Never break the public CLI; all new behavior must be covered by tests in `tests/`.
- All HTML/JS report code lives inside `_html()` in `site_audit/serp_gap.py` as template strings — follow that pattern.
