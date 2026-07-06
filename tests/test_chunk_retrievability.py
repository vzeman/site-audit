from __future__ import annotations

import ast
import inspect
import math
from types import SimpleNamespace

import numpy as np

from site_audit.chunk_retrievability import build_chunk_retrievability, build_chunks
from site_audit.recommendations import synthesize, to_payload


def _text(words: int) -> str:
    return " ".join(f"w{i}" for i in range(words))


def _record(url: str, para: int, heading: str, words: int = 10) -> dict:
    return {
        "url": url,
        "paragraph_index": para,
        "heading": heading,
        "text": _text(words),
    }


def test_build_chunks_respects_headings_windows_long_paragraphs_and_empty_pages() -> None:
    assert build_chunks([]) == []

    rows = [
        _record("https://example.com/a", 0, "Intro", 60),
        _record("https://example.com/a", 1, "Intro", 70),
        _record("https://example.com/a", 2, "Intro", 130),
        _record("https://example.com/a", 3, "Details", 40),
        _record("https://example.com/a", 4, "Details", 260),
        _record("https://example.com/a", 5, "Details", 30),
    ]

    assert build_chunks(rows) == [
        {"heading": "Intro", "paragraph_indexes": [0, 1], "word_count": 130},
        {"heading": "Intro", "paragraph_indexes": [2], "word_count": 130},
        # sub-120-word leftover under a *different* heading than its
        # predecessor is kept as-is
        {"heading": "Details", "paragraph_indexes": [3], "word_count": 40},
        # sub-120-word trailing leftover folds into the previous
        # same-heading chunk (soft minimum)
        {"heading": "Details", "paragraph_indexes": [4, 5], "word_count": 290},
    ]
    assert build_chunks(rows) == build_chunks(rows)


def test_build_chunks_soft_minimum_merges_overflow_leftover() -> None:
    rows = [
        _record("https://example.com/a", 0, "Setup", 240),
        _record("https://example.com/a", 1, "Setup", 20),
    ]

    # 240 + 20 overflows 250 during accumulation, but the 20-word leftover
    # shares the heading, so the soft minimum folds it back in.
    assert build_chunks(rows) == [
        {"heading": "Setup", "paragraph_indexes": [0, 1], "word_count": 260},
    ]


def test_chunk_embedding_is_weighted_l2_normalized_mean() -> None:
    url = "https://example.com/a"
    rows = [
        _record(url, 0, "Answer", 1),
        _record(url, 1, "Answer", 3),
    ]
    paragraph_embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    expected = np.asarray([1.0, 3.0], dtype=np.float32) / math.sqrt(10)
    payload = build_chunk_retrievability(
        rows,
        paragraph_embeddings,
        [{"query": "weighted answer", "best_url": url, "source": "manual"}],
        np.asarray([expected], dtype=np.float32),
        strong=0.99,
    )

    assert payload["queries"][0]["status"] == "retrievable"
    assert payload["queries"][0]["best_similarity"] == 1.0
    assert payload["queries"][0]["paragraph_indexes"] == [0, 1]


def test_status_classification_includes_split_margin_boundary_and_floor() -> None:
    url = "https://example.com/a"
    rows = [
        _record(url, 0, "Strong", 10),
        _record(url, 1, "Tie A", 10),
        _record(url, 2, "Tie B", 10),
        _record(url, 3, "Weak B", 10),
    ]
    paragraph_embeddings = np.asarray([
        [1.0, 0.0],
        [0.60, math.sqrt(1 - 0.60**2)],
        [0.55, math.sqrt(1 - 0.55**2)],
        [0.549, math.sqrt(1 - 0.549**2)],
    ], dtype=np.float32)
    queries = [
        {"query": "retrievable", "best_url": url, "source": "manual"},
        {"query": "split boundary", "best_url": url, "source": "manual"},
        {"query": "missing floor", "best_url": url, "source": "manual"},
    ]
    q = np.asarray([
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
    ], dtype=np.float32)
    split_payload = build_chunk_retrievability(
        rows[1:3],
        paragraph_embeddings[1:3],
        [queries[1]],
        q[1:2],
        strong=0.65,
        split_margin=0.05,
    )
    missing_payload = build_chunk_retrievability(
        rows[1:2] + rows[3:4],
        np.asarray([paragraph_embeddings[1], paragraph_embeddings[3]], dtype=np.float32),
        [queries[2]],
        q[2:3],
        strong=0.65,
        split_margin=0.05,
    )
    full_payload = build_chunk_retrievability(
        rows,
        paragraph_embeddings,
        queries,
        q,
        strong=0.65,
        split_margin=0.05,
    )

    assert full_payload["queries"][0]["status"] == "retrievable"
    assert split_payload["queries"][0]["status"] == "split_answer"
    assert split_payload["queries"][0]["second_heading"] == "Tie B"
    assert missing_payload["queries"][0]["status"] == "missing"


def test_rollups_summary_truncation_and_unavailable_path() -> None:
    url = "https://example.com/a"
    rows = [_record(url, 0, "Answer", 10)]
    paragraph_embeddings = np.asarray([[1.0, 0.0]], dtype=np.float32)
    query_rows = [
        {"query": "low", "best_url": url, "source": "manual", "volume": 1},
        {"query": "high", "best_url": url, "source": "manual", "volume": 100},
        {"query": "mid", "best_url": url, "source": "manual", "volume": 50},
    ]
    query_embeddings = np.asarray([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    payload = build_chunk_retrievability(
        rows,
        paragraph_embeddings,
        query_rows,
        query_embeddings,
        strong=0.65,
        max_queries=2,
    )

    assert [row["query"] for row in payload["queries"]] == ["high", "mid"]
    assert payload["summary"]["truncated"] == 1
    assert payload["summary"]["retrievable"] == 1
    assert payload["summary"]["missing"] == 1
    assert payload["pages"][0]["retrievable_share"] == 0.5
    assert payload["pages"][0]["weakest_queries"][0]["query"] == "mid"

    unavailable = build_chunk_retrievability(rows, paragraph_embeddings, query_rows, None)
    assert unavailable["available"] is False
    assert unavailable["reason"] == "query embeddings unavailable"


def test_recommendations_are_stable_capped_and_include_required_text() -> None:
    records = []
    embeddings = []
    query_rows = []
    query_embeddings = []
    for i in range(25):
        url = f"https://example.com/a{i}"
        records.extend([
            _record(url, i * 2, f"Heading A {i}", 10),
            _record(url, i * 2 + 1, f"Heading B {i}", 10),
        ])
        embeddings.extend([
            [0.60, math.sqrt(1 - 0.60**2)],
            [0.56, math.sqrt(1 - 0.56**2)],
        ])
        query_rows.append({
            "query": f"question {i}",
            "best_url": url,
            "source": "manual",
            "top_traffic_page": True,
        })
        query_embeddings.append([1.0, 0.0])

    first = build_chunk_retrievability(
        records,
        np.asarray(embeddings, dtype=np.float32),
        query_rows,
        np.asarray(query_embeddings, dtype=np.float32),
        strong=0.65,
    )
    second = build_chunk_retrievability(
        records,
        np.asarray(embeddings, dtype=np.float32),
        query_rows,
        np.asarray(query_embeddings, dtype=np.float32),
        strong=0.65,
    )

    assert len(first["recommendations"]) == 20
    assert first["summary"]["recommendations_truncated"] == 5
    assert [rec["id"] for rec in first["recommendations"]] == [rec["id"] for rec in second["recommendations"]]
    assert first["recommendations"][0]["id"].startswith("geo-chunk-")
    assert 'between "Heading A 0" and "Heading B 0"' in first["recommendations"][0]["text"]
    assert "best chunk currently scores 0.60" in first["recommendations"][0]["text"]


def test_missing_recommendation_requires_demand_and_mentions_score() -> None:
    url = "https://example.com/a"
    payload = build_chunk_retrievability(
        [_record(url, 0, "Thin", 10)],
        np.asarray([[0.0, 1.0]], dtype=np.float32),
        [
            {"query": "demand query", "best_url": url, "source": "manual", "impressions": 10},
            {"query": "no demand query", "best_url": url, "source": "manual"},
        ],
        np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        strong=0.65,
    )

    assert len(payload["recommendations"]) == 1
    rec = payload["recommendations"][0]
    assert rec["query"] == "demand query"
    assert '"https://example.com/a" has no chunk that answers "demand query" (best 0.00)' in rec["text"]


def test_synthesize_consumes_chunk_payload_and_emits_geo_chunk_recommendations() -> None:
    payload = {
        "available": True,
        "recommendations": [
            {
                "id": "geo-chunk-example",
                "status": "missing",
                "url": "https://example.com/a",
                "query": "what is example",
                "text": '"https://example.com/a" has no chunk that answers "what is example" (best 0.20). Add it.',
                "effort": "medium",
                "best_similarity": 0.2,
            }
        ],
    }

    report = to_payload(synthesize(chunk_retrievability_payload=payload))
    item = next(row for row in report["items"] if row["id"] == "geo-chunk-example")
    assert item["category"] == "geo"
    assert item["type"] == "answerability"
    assert item["instruction"] == payload["recommendations"][0]["text"]


def test_build_chunk_retrievability_accepts_pipeline_tuple_records() -> None:
    """Exercise the exact argument shapes pipeline.run passes.

    Paragraph records are 4-tuples (page_index, para_index, text, embedding)
    with no url/title/heading; those must come from pages[page_index] and the
    extractor's heading fallback.
    """
    from site_audit.pipeline import _chunk_query_rows

    pages = [SimpleNamespace(url="https://example.com/a", title="Page A")]
    ext = SimpleNamespace(
        h1="Guide",
        headers_rich=[{"text": "Guide", "order": 1}],
        paragraphs=[_text(130), _text(130)],
    )
    paragraph_records = [
        (0, 0, _text(130), np.asarray([1.0, 0.0], dtype=np.float32)),
        (0, 1, _text(130), np.asarray([0.0, 1.0], dtype=np.float32)),
    ]
    paragraph_embedding_matrix = np.stack([r[3] for r in paragraph_records]).astype(np.float32)
    coverage_rows = [{
        "query": "guide question",
        "source": "manual",
        "status": "covered",
        "best_similarity": 0.8,
        "best_url": "https://example.com/a",
        "best_title": "Page A",
        "candidates_above_threshold": 1,
        "runner_ups": [],
    }]
    search_payload = {
        "organic_keywords": [{"keyword": "Guide Question", "volume": 50, "impressions": 10}],
        # http scheme + trailing slash on purpose: variants must still match
        "top_pages": [{"url": "http://example.com/a/", "traffic": 100}],
    }

    payload = build_chunk_retrievability(
        {
            "pages": pages,
            "extracted_pages": [ext],
            "paragraph_records": paragraph_records,
        },
        paragraph_embedding_matrix,
        _chunk_query_rows(coverage_rows, search_payload),
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        strong=0.65,
    )

    assert payload["available"] is True
    row = payload["queries"][0]
    assert row["status"] == "retrievable"
    assert row["url"] == "https://example.com/a"
    assert row["title"] == "Page A"
    assert row["chunk_heading"] == "Guide"
    assert row["has_demand"] is True
    assert row["top_traffic_page"] is True


def test_pipeline_calls_chunk_retrievability_after_search_enrichment() -> None:
    """Demand gating needs ahrefs_data populated before the chunk stage runs."""
    from site_audit import pipeline

    tree = ast.parse(inspect.getsource(pipeline))
    run = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run"
    )
    ahrefs_lines = []
    for node in ast.walk(run):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id == "ahrefs_data":
                ahrefs_lines.append(node.lineno)
    call_lines = [
        node.lineno
        for node in ast.walk(run)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "build_chunk_retrievability"
    ]

    assert call_lines, "pipeline.run must call build_chunk_retrievability"
    assert ahrefs_lines, "pipeline.run must define ahrefs_data"
    assert min(ahrefs_lines) < min(call_lines)


def test_split_floor_applies_to_best_score() -> None:
    url = "https://example.com/a"
    rows = [
        _record(url, 0, "Half A", 10),
        _record(url, 1, "Half B", 10),
    ]
    paragraph_embeddings = np.asarray([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float32)

    def classify(component: float) -> str:
        q = np.asarray(
            [[component, component, math.sqrt(max(0.0, 1.0 - 2 * component**2))]],
            dtype=np.float32,
        )
        payload = build_chunk_retrievability(
            rows,
            paragraph_embeddings,
            [{"query": "floor", "best_url": url, "source": "manual"}],
            q,
            strong=0.65,
            split_margin=0.05,
        )
        return payload["queries"][0]["status"]

    # best score exactly at strong - 0.1 with a tied second chunk -> split
    assert classify(0.55) == "split_answer"
    # best score just below the floor -> missing even though chunks are tied
    assert classify(0.549) == "missing"


def test_zero_vector_paragraph_embeddings_are_safe() -> None:
    url = "https://example.com/a"
    payload = build_chunk_retrievability(
        [_record(url, 0, "Zero", 10)],
        np.zeros((1, 2), dtype=np.float32),
        [{"query": "zero vector", "best_url": url, "source": "manual"}],
        np.asarray([[1.0, 0.0]], dtype=np.float32),
    )

    row = payload["queries"][0]
    assert row["status"] == "missing"
    assert row["best_similarity"] == 0.0
    assert not math.isnan(row["best_similarity"])
