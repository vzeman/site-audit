import numpy as np

from site_audit.contextual_links import build_contextual_link_impact
from site_audit.extractor import ExtractedPage


def _page(url: str, title: str, paragraphs=None) -> ExtractedPage:
    return ExtractedPage(
        url=url,
        title=title,
        description="",
        body=" ".join(paragraphs or [title]),
        word_count=50,
        language="en",
        h1=title,
        h1_count=1,
        paragraphs=paragraphs or [title],
    )


def test_contextual_link_impact_combines_paragraph_and_authority_components() -> None:
    source = _page("https://example.com/source", "Source", ["Support automation paragraph with routing and workflows."])
    target = _page("https://example.com/target", "Support Automation")
    embeddings = np.array([[1.0, 0.0], [0.95, 0.05]], dtype=np.float32)
    paragraph_records = [(0, 0, source.paragraphs[0], embeddings[1])]
    linkgraph = {
        "anchor_relevance": {
            "links": [{
                "source_url": source.url,
                "source_title": source.title,
                "target_url": target.url,
                "target_title": target.title,
                "anchor": "support automation",
                "context": source.paragraphs[0],
                "score": 80,
                "context_overlap": 0.7,
            }]
        },
        "link_removal_simulation": {
            "links": [{"source_url": source.url, "target_url": target.url, "removal_loss_score": 60, "placement": "contextual"}]
        },
        "traffic_weighted_pagerank": {
            "pages": [{"url": target.url, "traffic": 100, "weighted_pagerank_percentile": 0.3}]
        },
    }
    paragraph_impact = {"top_paragraphs": [{"url": source.url, "paragraph_index": 0, "impact_score": 20}]}

    payload = build_contextual_link_impact([source, target], paragraph_records, embeddings, linkgraph=linkgraph, paragraph_impact=paragraph_impact)

    row = payload["top_contextual_links"][0]
    assert row["context_type"] == "main_content"
    assert row["paragraph_index"] == 0
    assert row["contextual_link_impact"] > 60
    assert payload["summary"]["high_impact_contextual_links"] == 1


def test_contextual_link_impact_handles_no_links() -> None:
    payload = build_contextual_link_impact([], [], None, linkgraph={})

    assert payload["summary"]["status"] == "no_links"
    assert payload["links"] == []
