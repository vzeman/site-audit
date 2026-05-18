from types import SimpleNamespace

import numpy as np

from site_audit.paragraph_links import build_addition_simulation, recommend, to_payload
from site_audit.report import write_internal_linkbuilding_csv


class CountingEmbedder:
    def __init__(self) -> None:
        self.encoded_count = 0

    def encode(self, texts, batch_size=256, show_progress=True):
        self.encoded_count += len(texts)
        return np.ones((len(texts), 3), dtype=np.float32)


def test_recommend_trims_candidates_before_anchor_embedding() -> None:
    pages = [
        SimpleNamespace(url="https://example.com/source", title="Source"),
        SimpleNamespace(url="https://example.com/target-a", title="Target A"),
        SimpleNamespace(url="https://example.com/target-b", title="Target B"),
    ]
    page_embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    paragraph_records = [
        (
            0,
            index,
            f"Relevant anchor phrase number {index} about target content and internal linking.",
            np.array([0.0, 1.0, 1.0], dtype=np.float32),
        )
        for index in range(20)
    ]
    embedder = CountingEmbedder()

    recs = recommend(
        pages,
        page_embeddings,
        paragraph_records,
        pages_with_outlinks=[],
        embedder=embedder,
        similarity_floor=0.1,
        lift_floor=0.0,
        top_k_per_page=2,
        top_k_total=1,
    )

    assert len(recs) == 1
    assert embedder.encoded_count <= 120


def test_recommend_dedupes_same_anchor_in_same_paragraph() -> None:
    pages = [
        SimpleNamespace(url="https://example.com/source", title="Source"),
        SimpleNamespace(url="https://example.com/target-a", title="Target A"),
        SimpleNamespace(url="https://example.com/target-b", title="Target B"),
    ]
    page_embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    paragraph_records = [
        (
            0,
            4,
            "Use the shared anchor phrase when improving internal links.",
            np.array([0.0, 1.0, 1.0], dtype=np.float32),
        )
    ]

    recs = recommend(
        pages,
        page_embeddings,
        paragraph_records,
        pages_with_outlinks=[],
        embedder=CountingEmbedder(),
        similarity_floor=0.1,
        lift_floor=0.0,
        top_k_per_page=2,
        top_k_total=10,
    )

    keys = {(r.source_url, r.paragraph_index, r.suggested_anchor.lower()) for r in recs}
    assert len(recs) == len(keys)


def test_recommend_skips_canonical_self_link_variants() -> None:
    pages = [
        SimpleNamespace(url="https://www.example.com/source/", title="Source"),
        SimpleNamespace(url="https://example.com/source", title="Source canonical duplicate"),
        SimpleNamespace(url="https://example.com/target", title="Target"),
    ]
    page_embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.99, 0.01, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    paragraph_records = [
        (
            0,
            1,
            "This paragraph strongly matches source duplicate content.",
            np.array([0.99, 0.01, 0.0], dtype=np.float32),
        )
    ]

    recs = recommend(
        pages,
        page_embeddings,
        paragraph_records,
        pages_with_outlinks=[],
        embedder=CountingEmbedder(),
        similarity_floor=0.1,
        lift_floor=-1.0,
        top_k_per_page=3,
        top_k_total=10,
    )

    assert all(r.source_url != "https://www.example.com/source/" or r.target_url != "https://example.com/source" for r in recs)


def test_addition_simulation_scores_components_and_density_cap() -> None:
    pages = [
        SimpleNamespace(url="https://example.com/source", title="Source", section="blog"),
        SimpleNamespace(url="https://example.com/target", title="Target", section="docs"),
    ]
    recs = to_payload([
        SimpleNamespace(
            source_url=pages[0].url,
            paragraph_index=2,
            paragraph_excerpt="Contextual paragraph about target topics.",
            target_url=pages[1].url,
            target_title=pages[1].title,
            fit=0.9,
            lift=0.2,
            suggested_anchor="target topics",
            anchor_confidence=0.8,
        )
    ])
    linkgraph = {
        "page_link_counts": [
            {"url": pages[0].url, "pagerank": 0.6, "out_degree": 4},
            {"url": pages[1].url, "pagerank": 0.1, "in_degree": 1},
        ],
        "traffic_weighted_pagerank": {
            "pages": [
                {"url": pages[0].url, "weighted_pagerank_percentile": 0.9, "traffic_weighted_pagerank": 0.05, "out_degree": 4},
                {"url": pages[1].url, "weighted_pagerank_percentile": 0.2, "traffic": 500, "keywords": 10, "authority_traffic_gap": 0.7, "in_degree": 1},
            ]
        },
    }

    payload = build_addition_simulation(recs, pages, linkgraph, paragraph_saturation={(0, 2): 1.5})

    row = payload["recommendations"][0]
    assert row["expected_benefit_score"] > 70
    assert row["priority"] == "high"
    assert row["current_target_in_degree"] == 1
    assert row["after_target_in_degree"] == 2
    assert row["score_components"]["authority_flow"] > 0
    assert row["respects_density_cap"] is True


def test_internal_linkbuilding_csv_uses_exact_anchor_and_destination_description(tmp_path) -> None:
    pages = [
        SimpleNamespace(url="https://example.com/source", title="Source", description="Source desc"),
        SimpleNamespace(url="https://example.com/target", title="Target", description="Target meta description"),
    ]
    result = SimpleNamespace(pages=pages)
    recs = [{
        "source_url": pages[0].url,
        "source_title": pages[0].title,
        "paragraph_index": 3,
        "paragraph_excerpt": "Mention target topics here.",
        "target_url": pages[1].url,
        "target_title": pages[1].title,
        "suggested_anchor": "target topics",
        "priority": "high",
        "expected_benefit_score": 81.2,
        "fit": 0.91,
        "lift": 0.18,
        "anchor_confidence": 0.8,
    }]

    rows = write_internal_linkbuilding_csv(tmp_path / "links.csv", result, recs)
    text = (tmp_path / "links.csv").read_text(encoding="utf-8")

    assert rows[0]["url_where_to_place_link"] == pages[0].url
    assert rows[0]["exact_keywords_to_link"] == "target topics"
    assert rows[0]["destination_url"] == pages[1].url
    assert rows[0]["link_title"] == "Target meta description"
    assert "exact_keywords_to_link" in text


def test_internal_linkbuilding_csv_prefers_paid_converting_anchor_when_present(tmp_path) -> None:
    pages = [
        SimpleNamespace(url="https://example.com/source", title="Source", description=""),
        SimpleNamespace(url="https://example.com/affiliate-software", title="Affiliate Software", description="Affiliate tracking software for partner programs."),
    ]
    result = SimpleNamespace(pages=pages)
    recs = [{
        "source_url": pages[0].url,
        "paragraph_excerpt": "Compare affiliate software before you choose a partner platform.",
        "target_url": pages[1].url,
        "target_title": pages[1].title,
        "suggested_anchor": "partner platform",
    }]
    search = {
        "organic_keywords": [
            {"provider": "google_ads", "keyword": "affiliate software", "paid_conversions": 12, "paid_conversion_value": 1200, "paid_cost": 300},
            {"provider": "google_ads", "keyword": "unrelated crm", "paid_conversions": 30, "paid_conversion_value": 3000, "paid_cost": 500},
        ]
    }

    rows = write_internal_linkbuilding_csv(tmp_path / "links.csv", result, recs, search)

    assert rows[0]["exact_keywords_to_link"] == "affiliate software"
    assert rows[0]["anchor_source"] == "paid_converting_keyword"
    assert rows[0]["paid_conversions"] == 12


def test_internal_linkbuilding_csv_dedupes_same_anchor_in_same_paragraph(tmp_path) -> None:
    pages = [
        SimpleNamespace(url="https://example.com/source", title="Source", description=""),
        SimpleNamespace(url="https://example.com/target-a", title="Target A", description="A"),
        SimpleNamespace(url="https://example.com/target-b", title="Target B", description="B"),
    ]
    result = SimpleNamespace(pages=pages)
    recs = [
        {
            "source_url": pages[0].url,
            "paragraph_index": 3,
            "paragraph_excerpt": "Mention target topics here.",
            "target_url": pages[1].url,
            "target_title": pages[1].title,
            "suggested_anchor": "target topics",
            "expected_benefit_score": 80,
        },
        {
            "source_url": pages[0].url,
            "paragraph_index": 3,
            "paragraph_excerpt": "Mention target topics here.",
            "target_url": pages[2].url,
            "target_title": pages[2].title,
            "suggested_anchor": "Target   Topics",
            "expected_benefit_score": 70,
        },
    ]

    rows = write_internal_linkbuilding_csv(tmp_path / "links.csv", result, recs)

    assert len(rows) == 1
    assert rows[0]["destination_url"] == pages[1].url


def test_internal_linkbuilding_csv_skips_canonical_self_links(tmp_path) -> None:
    pages = [
        SimpleNamespace(url="https://www.example.com/source/", title="Source", description=""),
        SimpleNamespace(url="https://example.com/target", title="Target", description=""),
    ]
    result = SimpleNamespace(pages=pages)
    recs = [
        {
            "source_url": "https://www.example.com/source/",
            "paragraph_index": 1,
            "paragraph_excerpt": "Self link should not be exported.",
            "target_url": "https://example.com/source",
            "target_title": "Source",
            "suggested_anchor": "source page",
        },
        {
            "source_url": pages[0].url,
            "paragraph_index": 2,
            "paragraph_excerpt": "Valid target link.",
            "target_url": pages[1].url,
            "target_title": pages[1].title,
            "suggested_anchor": "target page",
        },
    ]

    rows = write_internal_linkbuilding_csv(tmp_path / "links.csv", result, recs)

    assert len(rows) == 1
    assert rows[0]["destination_url"] == pages[1].url
