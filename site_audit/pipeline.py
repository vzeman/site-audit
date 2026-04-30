"""Top-level orchestration: domain → cached pages → embeddings → report.

This is the single function the CLI calls; library users can call it
directly too. Everything below it is composable: hand a different
``Crawler`` or ``EmbeddingCache`` and the rest works unchanged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from .analyzer import PageInfo, analyze, section_for_url
from .cache import EmbeddingCache, HttpCache, domain_slug
from .crawler import Crawler, CrawlConfig
from .embedder import DEFAULT_MODEL, EmbedInput, embed_pages
from .extractor import extract
from .html_report import write_html_report
from .report import write_all
from .scatter import project

LOG = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    domain: str
    output_root: Path = Path("output")
    cache_root: Path = Path("cache")
    model: str = DEFAULT_MODEL
    max_pages: int = 2000
    max_workers: int = 8
    duplicate_threshold: float = 0.92
    duplicate_knn: int = 10
    scatter_clusters: int = 30
    follow_subdomains: bool = False
    respect_robots: bool = True
    use_http_cache: bool = True
    use_embedding_cache: bool = True
    skip_scatterplot: bool = False
    max_chars: int = 4000


def _domain_only(domain: str) -> str:
    if "://" in domain:
        host = urlparse(domain).netloc
    else:
        host = domain
    return host.split("/")[0].lower()


def run(config: PipelineConfig) -> dict:
    host = _domain_only(config.domain)
    slug = domain_slug(host)

    cache_dir = Path(config.cache_root) / slug
    output_dir = Path(config.output_root) / slug

    LOG.info("=== site-audit run for %s ===", host)
    LOG.info("  cache: %s", cache_dir)
    LOG.info("  output: %s", output_dir)

    http_cache = HttpCache(cache_dir / "http.sqlite")
    embed_cache = EmbeddingCache(cache_dir / f"embeddings_{config.model.replace('/', '_').replace('-', '_')}.npz")

    crawl_config = CrawlConfig(
        domain=config.domain,
        max_pages=config.max_pages,
        max_workers=config.max_workers,
        follow_subdomains=config.follow_subdomains,
        respect_robots=config.respect_robots,
        use_cache=config.use_http_cache,
    )

    crawler = Crawler(crawl_config, http_cache)
    fetched = crawler.discover_and_crawl()
    LOG.info("  fetched %d pages (cache stats: %s)", len(fetched), http_cache.stats())

    pages: list[PageInfo] = []
    embed_inputs: list[EmbedInput] = []
    for r in fetched:
        ext = extract(r.url, r.body, max_chars=config.max_chars)
        if ext is None:
            continue
        if not ext.title:
            continue
        section = section_for_url(r.url)
        embed_text = ". ".join(part for part in [ext.title, ext.description, ext.body] if part)
        if not embed_text.strip():
            continue
        pages.append(PageInfo(
            url=r.url,
            title=ext.title,
            description=ext.description,
            section=section,
            word_count=ext.word_count,
            language=ext.language,
        ))
        embed_inputs.append(EmbedInput(url=r.url, text=embed_text))

    if not pages:
        LOG.warning("No usable pages — aborting before embedding.")
        return {"pages": 0}

    LOG.info("  prepared %d pages for embedding", len(pages))

    embeddings = embed_pages(
        embed_inputs,
        embed_cache,
        model_name=config.model,
        use_cache=config.use_embedding_cache,
    )

    LOG.info("  embeddings shape: %s", embeddings.shape)

    result = analyze(
        pages,
        embeddings,
        duplicate_threshold=config.duplicate_threshold,
        duplicate_knn=config.duplicate_knn,
    )

    coords = labels = None
    if not config.skip_scatterplot:
        labels, coords = project(embeddings, num_clusters=config.scatter_clusters)

    summary = write_all(
        output_dir,
        result,
        model_name=config.model,
        domain=host,
        coords=coords,
        cluster_labels=labels,
    )

    template_path = Path(__file__).resolve().parent.parent / "ui" / "index.html"
    if template_path.is_file():
        html_path = write_html_report(
            output_dir,
            template_path,
            result,
            model_name=config.model,
            domain=host,
            coords=coords,
            cluster_labels=labels,
        )
        LOG.info("  HTML report: %s", html_path)

    LOG.info("=== summary ===")
    LOG.info("  pages: %d", len(pages))
    LOG.info("  siteFocusScore: %.4f", result.site_metrics["focus_score"])
    LOG.info("  siteRadius:     %.4f", result.site_metrics["radius"])
    LOG.info("  outliers: %d  near-duplicate pairs: %d", summary["outliers"], summary["duplicates"])
    LOG.info("  reports: %s", output_dir)

    return {
        "domain": host,
        "pages": len(pages),
        "site_focus_score": result.site_metrics["focus_score"],
        "site_radius": result.site_metrics["radius"],
        "outliers": summary["outliers"],
        "duplicate_pairs": summary["duplicates"],
        "output_dir": str(output_dir),
        "html_report": str(output_dir / "index.html"),
    }
