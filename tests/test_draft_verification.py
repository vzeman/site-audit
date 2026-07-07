import numpy as np

from site_audit.draft_verification import (
    assemble_recommended_blocks,
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
