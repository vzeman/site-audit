"""Verify an AI page recommendation against the computed SERP topic space.

After the agent produces a structured recommendation (paragraph decisions,
rewrites, new sections), we assemble the recommended page as ordered text
blocks, embed them with the same local model, and re-score topic and People
Also Ask coverage. The report then shows before/after coverage so editors can
see whether the recommendation actually closes the gap.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np

# Keep in sync with build_serp_paragraph_gap() in competitive_analysis.py.
COVERED_THRESHOLD = 0.78
PARTIAL_THRESHOLD = 0.62


@dataclass(frozen=True)
class Claim:
    """A numeric claim found in draft copy."""

    text: str
    number: str
    normalized: str
    kind: str
    context: str
    needs_data: bool = False


_CURRENCY_CODES = "USD|EUR|GBP|CZK|PLN|HUF|CHF|CAD|AUD|NZD|SEK|NOK|DKK"
_CURRENCY_SYMBOLS = r"$€£"
_NUM_CORE = r"\d+(?:[,\s]\d{3})*(?:[.,]\d+)?|\d+(?:[.,]\d+)?"
_NUMERIC_RE = re.compile(
    rf"""
    (?P<currency_prefix>(?:[{re.escape(_CURRENCY_SYMBOLS)}]|(?:{_CURRENCY_CODES})\b)\s*(?P<currency_prefix_num>{_NUM_CORE}))
    |
    (?P<currency_suffix>(?P<currency_suffix_num>{_NUM_CORE})\s*(?:[{re.escape(_CURRENCY_SYMBOLS)}]|\b(?:{_CURRENCY_CODES})\b))
    |
    (?P<percent>(?P<percent_num>{_NUM_CORE})\s*(?:%|\bpercent\b|\bpercentage\s+points?\b))
    |
    (?P<multiplier>(?P<multiplier_num>\d+(?:[.,]\d+)?)\s*x\b)
    |
    (?P<count>\b(?P<count_num>{_NUM_CORE})\b)
    """,
    re.IGNORECASE | re.VERBOSE,
)
_ORDERED_LIST_RE = re.compile(r"(?m)^\s*\d+[.)]\s+\S")
_BULLET_LIST_RE = re.compile(r"(?m)^\s*[-*+]\s+\S")
_HEADING_COUNT_RE = re.compile(
    r"^\s{0,3}#{1,6}\s+(?P<number>\d+)\s+(?P<noun>ways|tips|steps|reasons|items|examples|ideas|checks)\b",
    re.IGNORECASE,
)
_LIST_POSITION_RE = re.compile(r"^\s*(?:#{1,6}\s*)?\d+[.)]\s+")
_NEEDS_DATA_RE = re.compile(r"\[NEEDS DATA\]", re.IGNORECASE)
_SENTENCE_BOUNDARY_RE = re.compile(r"[.!?\n]")
_MONTHS = (
    r"(?:january|february|march|april|may|june|july|august|september|october|november|december"
    r"|jan|feb|mar|apr|jun|jul|aug|sept?|oct|nov|dec)"
)
_MONTH_BEFORE_DAY_RE = re.compile(rf"\b{_MONTHS}\.?\s+$", re.IGNORECASE)
_DAY_BEFORE_MONTH_RE = re.compile(rf"^(?:st|nd|rd|th)?,?\s+(?:of\s+)?{_MONTHS}\b", re.IGNORECASE)


def _numeric_context(text: str, start: int, end: int, radius: int = 70) -> str:
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    context = re.sub(r"\s+", " ", text[left:right]).strip()
    return context


def _sentence_for_span(text: str, start: int, end: int) -> str:
    """Return the sentence containing ``text[start:end]`` (boundaries: . ! ? or newline)."""
    left = 0
    for boundary in _SENTENCE_BOUNDARY_RE.finditer(text, 0, start):
        left = boundary.end()
    boundary = _SENTENCE_BOUNDARY_RE.search(text, end)
    right = boundary.start() if boundary else len(text)
    return text[left:right]


def _normalized_number(value: str) -> str:
    cleaned = re.sub(r"(?<=\d)[,\s](?=\d{3}\b)", "", str(value or ""))
    cleaned = cleaned.replace(",", ".")
    cleaned = re.sub(r"[^0-9.]", "", cleaned)
    if "." in cleaned:
        cleaned = cleaned.rstrip("0").rstrip(".")
    return cleaned


def _claim_kind(match: re.Match[str]) -> tuple[str, str]:
    if match.group("currency_prefix"):
        return "currency", match.group("currency_prefix_num")
    if match.group("currency_suffix"):
        return "currency", match.group("currency_suffix_num")
    if match.group("percent"):
        return "percent", match.group("percent_num")
    if match.group("multiplier"):
        return "multiplier", match.group("multiplier_num")
    return "count", match.group("count_num")


def _line_for_offset(text: str, offset: int) -> str:
    start = text.rfind("\n", 0, offset) + 1
    end = text.find("\n", offset)
    if end < 0:
        end = len(text)
    return text[start:end]


def _is_heading_item_count(line: str, number: str, draft_markdown: str) -> bool:
    match = _HEADING_COUNT_RE.search(line)
    if not match or match.group("number") != number:
        return False
    expected = int(number)
    list_items = len(_ORDERED_LIST_RE.findall(draft_markdown))
    if list_items == expected:
        return True
    return len(_BULLET_LIST_RE.findall(draft_markdown)) == expected


def _is_decimal_fragment(match: re.Match[str], text: str) -> bool:
    """True when the match is the tail of a larger token split by ``[.,]`` (e.g. "4" in "v2.4")."""
    start = match.start()
    return start >= 2 and text[start - 1] in ".," and text[start - 2].isdigit()


def _is_day_of_month(raw: str, match: re.Match[str], text: str) -> bool:
    if not raw.isdigit() or not 1 <= int(raw) <= 31:
        return False
    if _MONTH_BEFORE_DAY_RE.search(text, 0, match.start()):
        return True
    return bool(_DAY_BEFORE_MONTH_RE.match(text[match.end():]))


def _is_excluded_count(match: re.Match[str], text: str, normalized: str) -> bool:
    line = _line_for_offset(text, match.start())
    raw = match.group(0).strip()
    if re.fullmatch(r"\d{4}", raw):
        year = int(raw)
        if 1990 <= year <= 2030:
            return True
    if _is_decimal_fragment(match, text):
        return True
    if _is_day_of_month(raw, match, text):
        return True
    if _LIST_POSITION_RE.match(line) and line.strip().startswith(raw):
        return True
    if _is_heading_item_count(line, raw, text):
        return True
    return not normalized


def extract_numeric_claims(text: str) -> list[Claim]:
    """Extract numeric claims from draft text with local context."""
    draft = str(text or "")
    claims: list[Claim] = []
    seen: set[tuple[str, int, int]] = set()
    for match in _NUMERIC_RE.finditer(draft):
        kind, raw_number = _claim_kind(match)
        normalized = _normalized_number(raw_number)
        if kind == "count" and _is_excluded_count(match, draft, normalized):
            continue
        context = _numeric_context(draft, match.start(), match.end())
        sentence = _sentence_for_span(draft, match.start(), match.end())
        claim = Claim(
            text=match.group(0).strip(),
            number=raw_number.strip(),
            normalized=normalized,
            kind=kind,
            context=context,
            needs_data=bool(_NEEDS_DATA_RE.search(sentence)),
        )
        key = (claim.text, match.start(), match.end())
        if key not in seen:
            seen.add(key)
            claims.append(claim)
    return claims


def _number_pattern(normalized: str) -> str:
    if not normalized:
        return r"(?!x)x"
    if "." in normalized:
        whole, decimal = normalized.split(".", 1)
        return rf"{_integer_number_pattern(whole)}[\.,]{re.escape(decimal)}"
    return _integer_number_pattern(normalized)


def _integer_number_pattern(normalized: str) -> str:
    digits = re.escape(normalized)
    if len(normalized) > 3:
        groups = []
        rest = normalized
        while rest:
            groups.append(rest[-3:])
            rest = rest[:-3]
        grouped = r"[,\s]?".join(re.escape(group) for group in reversed(groups))
        return rf"(?:{digits}|{grouped})"
    return digits


def _claim_verified(claim: Claim, evidence_text: str) -> bool:
    number = _number_pattern(claim.normalized)
    if claim.kind == "percent":
        pattern = rf"(?<!\d){number}\s*(?:%|percent|percentage\s+points?)(?!\w)"
    elif claim.kind == "multiplier":
        pattern = rf"(?<!\d){number}\s*(?:x|times)(?!\w)"
    elif claim.kind == "currency":
        marker = rf"(?:[{re.escape(_CURRENCY_SYMBOLS)}]|\b(?:{_CURRENCY_CODES})\b)"
        pattern = rf"(?:{marker}\s*{number}|{number}\s*{marker})"
    else:
        pattern = rf"(?<!\d){number}(?!\d)"
    return bool(re.search(pattern, evidence_text, flags=re.IGNORECASE))


def verify_numeric_claims(draft_markdown: str, evidence_texts: list[str]) -> dict:
    """Verify draft numeric claims against supplied evidence text."""
    evidence_text = "\n".join(str(text or "") for text in evidence_texts)
    verified: list[dict] = []
    unverified: list[dict] = []
    for claim in extract_numeric_claims(draft_markdown):
        payload = asdict(claim)
        if claim.needs_data or _claim_verified(claim, evidence_text):
            verified.append(payload)
        else:
            unverified.append(payload)
    return {"verified": verified, "unverified": unverified}


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
