import numpy as np

from site_audit.analyzer import PageInfo
from site_audit.linkgraph import analyze, hub_bottleneck_payload, link_flow_payload, link_removal_simulation_payload, traffic_weighted_pagerank_payload


def test_link_flow_payload_keeps_traffic_nodes_and_weighted_edges() -> None:
    pages = [
        PageInfo(url="https://example.com/", title="Home", description="", section="root", word_count=100, language="en"),
        PageInfo(url="https://example.com/blog/a", title="Guide A", description="", section="blog", word_count=100, language="en"),
        PageInfo(url="https://example.com/blog/b", title="Guide B", description="", section="blog", word_count=100, language="en"),
    ]
    outlinks = [
        (pages[0].url, [(pages[1].url, "guide"), (pages[1].url, "read guide"), (pages[2].url, "guide b")]),
        (pages[1].url, [(pages[2].url, "next")]),
        (pages[2].url, []),
    ]
    result = analyze(pages, np.eye(3, dtype=np.float32), outlinks, home_url=pages[0].url)
    authority = traffic_weighted_pagerank_payload(
        result,
        pages,
        np.eye(3, dtype=np.float32),
        search_payload={"top_pages": [{"matched_url": pages[1].url, "traffic": 1000, "keywords": 20, "top_keyword": "guide a", "cluster_label": "guides"}]},
        page_types={"per_page": [{"url": pages[1].url, "page_type": "article"}]},
    )
    removal = link_removal_simulation_payload(result, pages, np.eye(3, dtype=np.float32), [], traffic_authority=authority)
    payload = link_flow_payload(
        result,
        pages,
        [{"matched_url": pages[1].url, "traffic": 1000, "keywords": 20, "top_keyword": "guide a", "cluster_label": "guides"}],
        page_types={"per_page": [{"url": pages[1].url, "page_type": "article"}]},
        traffic_authority=authority,
        link_removal=removal,
        max_nodes=3,
        max_edges=10,
    )

    assert payload["shown_nodes"] == 3
    traffic_node = next(node for node in payload["nodes"] if node["traffic"] == 1000)
    assert traffic_node["top_keyword"] == "guide a"
    assert traffic_node["directory"] == "/blog/"
    assert traffic_node["cluster"] == "guides"
    assert traffic_node["page_type"] == "article"
    assert traffic_node["traffic_weighted_pagerank"] > 0
    target_edge = next(edge for edge in payload["edges"] if edge["source"] == pages[0].url and edge["target"] == pages[1].url)
    assert target_edge["weight"] == 2
    assert target_edge["removal_loss_score"] >= 0


def test_link_removal_simulation_ranks_contextual_and_template_links() -> None:
    pages = [
        PageInfo(url="https://example.com/", title="Home", description="", section="root", word_count=100, language="en"),
        PageInfo(url="https://example.com/blog/demand", title="Demand guide", description="", section="blog", word_count=100, language="en"),
        PageInfo(url="https://example.com/privacy", title="Privacy", description="", section="legal", word_count=100, language="en"),
    ]
    embeddings = np.array([
        [1.0, 0.0, 0.0],
        [0.95, 0.05, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float32)
    outlinks = [
        (pages[0].url, [(pages[1].url, "demand guide"), (pages[2].url, "Privacy")]),
        (pages[1].url, [(pages[0].url, "home")]),
        (pages[2].url, [(pages[0].url, "home")]),
    ]
    result = analyze(pages, embeddings, outlinks, home_url=pages[0].url)
    authority = traffic_weighted_pagerank_payload(
        result,
        pages,
        embeddings,
        search_payload={"top_pages": [{"matched_url": pages[1].url, "traffic": 1000, "keywords": 15, "top_keyword": "demand guide"}]},
    )
    paragraph_records = [
        (0, 0, "This paragraph explains the demand guide and links to the important demand topic.", embeddings[1]),
        (0, 1, "Footer legal navigation with privacy links.", embeddings[2]),
    ]

    payload = link_removal_simulation_payload(result, pages, embeddings, paragraph_records, traffic_authority=authority)

    critical = payload["critical_links"][0]
    assert critical["target_url"] == pages[1].url
    assert critical["classification"] == "critical"
    assert critical["placement"] == "contextual"
    assert critical["paragraph_excerpt"]
    assert any(row["placement"] == "template_navigation" for row in payload["links"])
    assert payload["edit_warnings"][0]["source_url"] == pages[0].url


def test_traffic_weighted_pagerank_flags_mismatches_and_nonindexable_search_pages() -> None:
    pages = [
        PageInfo(url="https://example.com/", title="Home", description="", section="root", word_count=100, language="en"),
        PageInfo(url="https://example.com/blog/demand", title="Demand guide", description="", section="blog", word_count=100, language="en"),
        PageInfo(url="https://example.com/old", title="Old hub", description="", section="old", word_count=100, language="en"),
    ]
    outlinks = [
        (pages[0].url, [(pages[2].url, "old hub")]),
        (pages[1].url, []),
        (pages[2].url, [(pages[0].url, "home")]),
    ]
    result = analyze(pages, np.eye(3, dtype=np.float32), outlinks, home_url=pages[0].url)
    payload = traffic_weighted_pagerank_payload(
        result,
        pages,
        np.eye(3, dtype=np.float32),
        search_payload={
            "top_pages": [
                {"matched_url": pages[1].url, "traffic": 500, "keywords": 12, "top_keyword": "demand guide", "cluster_label": "guides"},
                {"url": "https://example.com/noindex", "traffic": 80, "keywords": 3, "top_keyword": "blocked page"},
            ]
        },
        page_types={"per_page": [{"url": pages[1].url, "page_type": "article"}]},
        indexability={"skipped": [{"url": "https://example.com/noindex", "title": "Noindex", "status": "skipped", "reason": "noindex"}], "noindex_pages": []},
    )

    demand = next(row for row in payload["pages"] if row["url"] == pages[1].url)
    assert demand["traffic"] == 500
    assert demand["pagerank"] >= 0
    assert demand["weighted_pagerank_percentile"] >= 0
    assert demand["mismatch_label"] in {"ranked_orphan", "high_traffic_low_authority"}
    assert payload["summary"]["high_traffic_low_authority_pages"] >= 1
    assert payload["mismatches"]["non_indexable_search_pages"][0]["url"] == "https://example.com/noindex"


def test_hub_bottleneck_payload_detects_cluster_bridges() -> None:
    pages = [
        PageInfo(url="https://example.com/a", title="A", description="", section="alpha", word_count=100, language="en"),
        PageInfo(url="https://example.com/bridge", title="Bridge", description="", section="hub", word_count=100, language="en"),
        PageInfo(url="https://example.com/b", title="B", description="", section="beta", word_count=100, language="en"),
        PageInfo(url="https://example.com/c", title="C", description="", section="gamma", word_count=100, language="en"),
    ]
    outlinks = [
        (pages[0].url, [(pages[1].url, "bridge")]),
        (pages[1].url, [(pages[2].url, "beta"), (pages[3].url, "gamma")]),
        (pages[2].url, []),
        (pages[3].url, []),
    ]
    result = analyze(pages, np.eye(4, dtype=np.float32), outlinks, home_url=pages[0].url)
    payload = hub_bottleneck_payload(result, pages)

    bridge = next(row for row in payload["pages"] if row["url"] == pages[1].url)
    assert bridge["role"] in {"bottleneck", "cluster_bridge"}
    assert bridge["cluster_bridge_count"] >= 2
    assert payload["cluster_edges"]
