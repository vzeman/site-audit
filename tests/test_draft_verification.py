import numpy as np

from site_audit.draft_verification import (
    assemble_recommended_blocks,
    extract_numeric_claims,
    verify_numeric_claims,
    verify_recommendation,
)


def _norm(rows):
    arr = np.asarray(rows, dtype=np.float32)
    denom = np.linalg.norm(arr, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return arr / denom


def _embed_fn(texts):
    vectors = []
    for text in texts:
        lowered = text.lower()
        if "pricing" in lowered:
            vectors.append([1.0, 0.0, 0.0])
        elif "security" in lowered:
            vectors.append([0.0, 1.0, 0.0])
        else:
            vectors.append([0.0, 0.0, 1.0])
    return _norm(vectors)


def test_assemble_blocks_orders_and_filters() -> None:
    paragraphs = ["Intro paragraph.", "Old details.", "Off topic ramble.", "Closing words."]
    recommendation = {
        "paragraph_decisions": [
            {"index": 0, "decision": "keep"},
            {"index": 1, "decision": "rewrite", "rewrite": "New, sharper details."},
            {"index": 2, "decision": "remove"},
            # index 3 intentionally missing -> defaults to keep
        ],
        "new_sections": [
            {"heading": "Pricing", "draft": "Pricing details here.", "placement_after_paragraph": 1},
            {"heading": "Overview", "draft": "Top overview.", "placement_after_paragraph": -1},
        ],
    }
    blocks = assemble_recommended_blocks(paragraphs, recommendation)
    refs = [(b["source"], b["ref"]) for b in blocks]
    assert refs == [
        ("new", "S1"),
        ("kept", "P0"),
        ("rewrite", "P1"),
        ("new", "S0"),
        ("kept", "P3"),
    ]
    assert blocks[2]["text"] == "New, sharper details."
    assert all("Off topic" not in b["text"] for b in blocks)


def test_verify_marks_topic_covered_after_new_section() -> None:
    paragraphs = ["Generic intro about nothing specific."]
    recommendation = {
        "paragraph_decisions": [{"index": 0, "decision": "keep"}],
        "new_sections": [{"heading": "Pricing", "draft": "Pricing plans and pricing tiers.", "placement_after_paragraph": 0}],
    }
    blocks = assemble_recommended_blocks(paragraphs, recommendation)
    pricing_centroid = _norm([[1.0, 0.0, 0.0]])[0].tolist()
    analyses = [{
        "keyword": {"keyword": "tool pricing"},
        "topics": [{
            "label": "pricing, plans",
            "centroid": pricing_centroid,
            "coverage": "missing",
            "priority": "critical",
        }],
        "paa_coverage": [{"question": "How much does pricing cost?", "status": "missing"}],
    }]
    result = verify_recommendation(blocks, analyses, _embed_fn)
    topic = result["topics"][0]
    assert topic["before"] == "missing"
    assert topic["after"] == "covered"
    assert topic["best_block_ref"] == "S0"
    assert result["paa"][0]["after"] == "covered"
    assert result["summary"]["missing_before"] == 1
    assert result["summary"]["missing_after"] == 0
    assert result["summary"]["unresolved_critical"] == []


def test_verify_summary_counts_and_unresolved() -> None:
    blocks = [{"source": "kept", "ref": "P0", "text": "Nothing relevant at all."}]
    security_centroid = _norm([[0.0, 1.0, 0.0]])[0].tolist()
    analyses = [{
        "keyword": {"keyword": "kw"},
        "topics": [{
            "label": "security, privacy",
            "centroid": security_centroid,
            "coverage": "missing",
            "priority": "high",
        }],
        "paa_coverage": [],
    }]
    result = verify_recommendation(blocks, analyses, _embed_fn)
    assert result["topics"][0]["after"] == "missing"
    assert result["summary"]["unresolved_critical"] == ["security, privacy"]


def test_verify_handles_missing_centroids_and_empty_blocks() -> None:
    analyses = [{
        "keyword": {"keyword": "kw"},
        "topics": [{"label": "no centroid", "coverage": "missing", "priority": "high"}],
        "paa_coverage": [{"question": "Q?", "status": "missing"}],
    }]
    result = verify_recommendation([], analyses, _embed_fn)
    assert result["topics"] == []
    assert result["paa"] == []
    assert result["summary"]["missing_before"] == 0


def test_extract_numeric_claims_handles_common_numbers_and_exclusions() -> None:
    text = """
# 3 ways to improve support
1. Start with triage.
2. Route urgent work.
3. Review weekly.

Results improved by 73%, the plan costs $1,200, and routing is 3x faster.
The 2024 report is background only.
"""
    claims = extract_numeric_claims(text)

    assert [claim.text for claim in claims] == ["73%", "$1,200", "3x"]
    assert [claim.kind for claim in claims] == ["percent", "currency", "multiplier"]
    assert [claim.normalized for claim in claims] == ["73", "1200", "3"]


def test_verify_numeric_claims_matches_evidence_variants_and_needs_data() -> None:
    draft = "Conversion rose 73%. Setup costs 1200 USD. Routing is 3x faster. Churn fell 19% [NEEDS DATA]."
    result = verify_numeric_claims(draft, ["The study says 73 percent. Pricing is $1,200. Routing improved 3 x."])

    assert [claim["text"] for claim in result["verified"]] == ["73%", "1200 USD", "3x", "19%"]
    assert result["unverified"] == []


def test_verify_numeric_claims_flags_absent_numbers() -> None:
    result = verify_numeric_claims("Customers save 73% after launch.", ["Customers save time after launch."])

    assert [claim["text"] for claim in result["unverified"]] == ["73%"]


def test_verification_repair_prompt_and_payload_list_unverified_numbers() -> None:
    from site_audit.serp_gap import _verification_for, _verification_repair_prompt

    page = {
        "own_content": {"paragraphs": [{"text": "Existing evidence mentions setup qualitatively."}]},
        "analyses": [],
    }
    recommendation = {
        "paragraph_decisions": [{"index": 0, "decision": "rewrite", "rewrite": "Setup saves teams 73% of review time."}],
        "new_sections": [],
    }

    verification = _verification_for(page, recommendation, ["Original paragraph."], embedder=None)
    prompt = _verification_repair_prompt(verification)

    assert verification["unverified_numbers"][0]["text"] == "73%"
    assert "Replace or source these numbers" in prompt
    assert "73%" in prompt


def test_needs_data_exemption_is_scoped_to_the_claims_own_sentence() -> None:
    draft = (
        "Support costs fell 40% [NEEDS DATA]. "
        "Meanwhile revenue grew 55% across teams this quarter."
    )
    result = verify_numeric_claims(draft, ["No numbers in the evidence."])

    assert [claim["text"] for claim in result["verified"]] == ["40%"]
    assert [claim["text"] for claim in result["unverified"]] == ["55%"]


def test_extract_numeric_claims_excludes_dates_and_version_fragments() -> None:
    text = (
        "Released on January 15, 2026 as v2.4. "
        "The update shipped on 3 March and improved routing for 87 teams."
    )
    claims = extract_numeric_claims(text)

    assert [claim.text for claim in claims] == ["87"]
