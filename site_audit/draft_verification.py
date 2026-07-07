"""Verify an AI page recommendation against the computed SERP topic space.

After the agent produces a structured recommendation (paragraph decisions,
rewrites, new sections), we assemble the recommended page as ordered text
blocks, embed them with the same local model, and re-score topic and People
Also Ask coverage. The report then shows before/after coverage so editors can
see whether the recommendation actually closes the gap.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

# Keep in sync with build_serp_paragraph_gap() in competitive_analysis.py.
COVERED_THRESHOLD = 0.78
PARTIAL_THRESHOLD = 0.62


def assemble_recommended_blocks(own_paragraphs: list[str], recommendation: dict) -> list[dict]:
    """Build the recommended page as ordered blocks.

    Returns [{"source": "kept|rewrite|new", "ref": "P3"|"S0", "text": str}].
    keep -> original text; rewrite -> replacement text; move/merge/remove -> excluded.
    New sections are inserted after their placement paragraph (-1 = top).
    Paragraphs without a decision default to keep.
    """
    decisions: dict[int, dict] = {}
    for row in recommendation.get("paragraph_decisions") or []:
        if isinstance(row, dict) and isinstance(row.get("index"), int) and not isinstance(row.get("index"), bool):
            decisions[row["index"]] = row

    sections_by_placement: dict[int, list[tuple[int, dict]]] = {}
    for s_index, section in enumerate(recommendation.get("new_sections") or []):
        if not isinstance(section, dict):
            continue
        placement = section.get("placement_after_paragraph")
        if not isinstance(placement, int) or isinstance(placement, bool):
            placement = len(own_paragraphs) - 1
        placement = max(-1, min(placement, len(own_paragraphs) - 1))
        sections_by_placement.setdefault(placement, []).append((s_index, section))

    def section_blocks(placement: int) -> list[dict]:
        blocks = []
        for s_index, section in sections_by_placement.get(placement, []):
            heading = str(section.get("heading") or "").strip()
            draft = str(section.get("draft") or "").strip()
            text = f"{heading}\n{draft}".strip()
            if text:
                blocks.append({"source": "new", "ref": f"S{s_index}", "text": text})
        return blocks

    out: list[dict] = []
    out.extend(section_blocks(-1))
    for index, paragraph in enumerate(own_paragraphs):
        row = decisions.get(index) or {}
        decision = str(row.get("decision") or "keep")
        if decision == "rewrite":
            rewrite = str(row.get("rewrite") or "").strip()
            if rewrite:
                out.append({"source": "rewrite", "ref": f"P{index}", "text": rewrite})
        elif decision in {"move", "merge", "remove"}:
            pass
        else:
            text = str(paragraph or "").strip()
            if text:
                out.append({"source": "kept", "ref": f"P{index}", "text": text})
        out.extend(section_blocks(index))
    return out


def _classify(similarity: float) -> str:
    if similarity >= COVERED_THRESHOLD:
        return "covered"
    if similarity >= PARTIAL_THRESHOLD:
        return "partial"
    return "missing"


def verify_recommendation(
    blocks: list[dict],
    analyses: list[dict],
    embed_fn: Callable[[list[str]], np.ndarray],
) -> dict:
    """Re-score topic and PAA coverage for the recommended page blocks."""
    texts = [str(block.get("text") or "") for block in blocks]
    matrix = (
        np.asarray(embed_fn(texts), dtype=np.float32)
        if texts
        else np.zeros((0, 0), dtype=np.float32)
    )

    def best_match(vector: np.ndarray) -> tuple[float, str]:
        if not len(matrix):
            return 0.0, ""
        sims = matrix @ vector
        best_i = int(np.argmax(sims))
        return float(np.clip(sims[best_i], -1.0, 1.0)), str(blocks[best_i].get("ref") or "")

    topic_rows: list[dict] = []
    paa_rows: list[dict] = []
    questions_to_score: list[tuple[dict, str, str]] = []

    for analysis in analyses or []:
        keyword = str((analysis.get("keyword") or {}).get("keyword") or analysis.get("query") or "")
        for topic in analysis.get("topics") or []:
            centroid = topic.get("centroid")
            if not centroid:
                continue
            vector = np.asarray(centroid, dtype=np.float32)
            best, ref = best_match(vector)
            topic_rows.append({
                "keyword": keyword,
                "label": topic.get("label", ""),
                "priority": topic.get("priority", ""),
                "before": topic.get("coverage", ""),
                "after": _classify(best),
                "best_similarity": round(best, 4),
                "best_block_ref": ref,
            })
        for row in analysis.get("paa_coverage") or []:
            question = str(row.get("question") or "").strip()
            if question:
                questions_to_score.append((row, keyword, question))

    if questions_to_score and len(matrix):
        question_matrix = np.asarray(
            embed_fn([question for _, _, question in questions_to_score]), dtype=np.float32
        )
        for (row, keyword, question), vector in zip(questions_to_score, question_matrix):
            best, ref = best_match(vector)
            paa_rows.append({
                "keyword": keyword,
                "question": question,
                "before": row.get("status", ""),
                "after": _classify(best),
                "best_similarity": round(best, 4),
                "best_block_ref": ref,
            })

    unresolved_critical = sorted({
        row["label"]
        for row in topic_rows
        if row.get("priority") in {"critical", "high"} and row.get("after") != "covered"
    })
    summary = {
        "missing_before": sum(1 for row in topic_rows if row["before"] == "missing"),
        "missing_after": sum(1 for row in topic_rows if row["after"] == "missing"),
        "partial_before": sum(1 for row in topic_rows if row["before"] == "partial"),
        "partial_after": sum(1 for row in topic_rows if row["after"] == "partial"),
        "paa_missing_before": sum(1 for row in paa_rows if row["before"] == "missing"),
        "paa_missing_after": sum(1 for row in paa_rows if row["after"] == "missing"),
        "unresolved_critical": unresolved_critical,
    }
    return {"topics": topic_rows, "paa": paa_rows, "summary": summary}
