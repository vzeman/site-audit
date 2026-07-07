"""Shared thresholds for SERP-gap analysis and agent recommendations."""

from __future__ import annotations

COVERED = 0.78
PARTIAL = 0.62
OFF_INTENT = 0.52
PREVALENCE_CRITICAL = 0.8
PREVALENCE_HIGH = 0.6

# Similarity threshold for merging paragraphs into one topic cluster in
# competitive_analysis._cluster_paragraphs. It equals COVERED only by
# coincidence: it is a clustering knob, not a coverage band, so it stays a
# separate constant and tuning the coverage bands never retunes clustering.
CLUSTER_SIMILARITY = 0.78

TITLE_MAX_CHARS = 65
META_MAX_CHARS = 165


def band(score: float) -> str:
    """Return the SERP-gap similarity band for a normalized score."""
    if score >= COVERED:
        return "covered"
    if score >= PARTIAL:
        return "partial"
    return "weak"


def similarity_band_prompt_text() -> str:
    return f">= {COVERED:g} covered, {PARTIAL:g}-{COVERED:g} partial, < {PARTIAL:g} weak"


def similarity_band_task_lines() -> list[str]:
    return [
        f"- How to read similarity scores in evidence.json: >= {COVERED:g} means the topic/question is",
        f"  already covered, {PARTIAL:g}-{COVERED:g} means partially covered, < {PARTIAL:g} means weak or missing.",
        "  `paragraph_review` is sorted weakest-first, so its values can still be high; never",
        f"  call a score above {COVERED:g} 'low'. Cite the actual number and the correct band in reasons.",
    ]
