import json

import pytest

from site_audit.crawler import FetchResult
from site_audit.pipeline import PipelineConfig, run


HTML = """
<!doctype html>
<html lang="en">
  <head>
    <title>{title}</title>
    <meta name="description" content="{title} description for testing">
    <link rel="canonical" href="{url}">
  </head>
  <body>
    <h1>{title}</h1>
    <p>This page has enough body copy for extraction and technical audit tests.</p>
    <a href="{next_url}">Next</a>
    <a href="https://external.example/page">External</a>
  </body>
</html>
"""


class FakeCrawler:
    sitemap_entries = []
    sitemap_errors = []
    robots_txt_info = {}

    def __init__(self, config, cache):
        self.config = config
        self.cache = cache

    def discover_and_crawl(self):
        base = "https://example.com"
        return [
            FetchResult(
                url=f"{base}/",
                status=200,
                body=HTML.format(title="Home", url=f"{base}/", next_url=f"{base}/two"),
                content_type="text/html",
                from_cache=False,
                outlinks=[(f"{base}/two", "Next")],
                external_links=[("https://external.example/page", "External")],
            ),
            FetchResult(
                url=f"{base}/two",
                status=200,
                body=HTML.format(title="Second", url=f"{base}/two", next_url=f"{base}/"),
                content_type="text/html",
                from_cache=False,
                outlinks=[(f"{base}/", "Home")],
                external_links=[],
            ),
        ]


class ReleasedBodyCrawler(FakeCrawler):
    def discover_and_crawl(self):
        base = "https://example.com"
        rows = []
        for path, title, next_path in [
            ("/", "Home", "/two"),
            ("/two", "Second", "/"),
        ]:
            url = f"{base}{path}"
            next_url = f"{base}{next_path}"
            html = HTML.format(title=title, url=url, next_url=next_url)
            self.cache.put(url, 200, {"Content-Type": "text/html"}, html.encode("utf-8"))
            rows.append(
                FetchResult(
                    url=url,
                    status=200,
                    body="",
                    content_type="text/html",
                    from_cache=True,
                    outlinks=[(next_url, "Next")],
                    external_links=[],
                    body_cache_url=url,
                    body_released=True,
                )
            )
        return rows


class FailingEmbedder:
    def __init__(self, *args, **kwargs):
        raise AssertionError("Embedder should not be loaded")


def test_technical_only_pipeline_writes_bundle_without_embeddings(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("site_audit.pipeline.Crawler", FakeCrawler)
    monkeypatch.setattr("site_audit.pipeline.Embedder", FailingEmbedder)

    summary = run(
        PipelineConfig(
            domain="example.com",
            projects_root=tmp_path,
            technical_only=True,
            save_snapshot=False,
        )
    )

    report_dir = tmp_path / "example.com" / "report"
    run_summary = json.loads((report_dir / "run_summary.json").read_text())

    assert summary["status"] == "technical_only"
    assert summary["pages"] == 2
    assert (report_dir / "technical_issues.json").is_file()
    assert (report_dir / "technical_pages.csv").is_file()
    assert (report_dir / "stage_timings.json").is_file()
    assert run_summary["mode"] == "technical"
    assert run_summary["summary"]["pages"] == 2


def test_large_site_safeguard_writes_technical_bundle_without_embeddings(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("site_audit.pipeline.Crawler", FakeCrawler)
    monkeypatch.setattr("site_audit.pipeline.Embedder", FailingEmbedder)

    summary = run(
        PipelineConfig(
            domain="example.com",
            projects_root=tmp_path,
            large_site_embedding_threshold=1,
            save_snapshot=False,
        )
    )

    report_dir = tmp_path / "example.com" / "report"
    run_summary = json.loads((report_dir / "run_summary.json").read_text())

    assert summary["status"] == "stopped_before_large_embedding"
    assert summary["pages"] == 2
    assert "Stopped before embedding 2 pages" in summary["message"]
    assert (report_dir / "technical_issues.csv").is_file()
    assert run_summary["mode"] == "large_site_embedding_safeguard"


def test_pipeline_reuses_extraction_cache_on_second_run(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("site_audit.pipeline.Crawler", FakeCrawler)
    monkeypatch.setattr("site_audit.pipeline.Embedder", FailingEmbedder)

    config = PipelineConfig(
        domain="example.com",
        projects_root=tmp_path,
        technical_only=True,
        save_snapshot=False,
    )
    run(config)

    def fail_extract(*args, **kwargs):
        raise AssertionError("extract should not run when artifact cache hits")

    monkeypatch.setattr("site_audit.pipeline.extract", fail_extract)
    summary = run(config)

    assert summary["status"] == "technical_only"
    assert summary["pages"] == 2


def test_pipeline_extracts_released_bodies_from_http_cache(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("site_audit.pipeline.Crawler", ReleasedBodyCrawler)
    monkeypatch.setattr("site_audit.pipeline.Embedder", FailingEmbedder)

    summary = run(
        PipelineConfig(
            domain="example.com",
            projects_root=tmp_path,
            technical_only=True,
            save_snapshot=False,
        )
    )

    assert summary["status"] == "technical_only"
    assert summary["pages"] == 2


def test_pipeline_resume_uses_extraction_checkpoint(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("site_audit.pipeline.Crawler", FakeCrawler)
    monkeypatch.setattr("site_audit.pipeline.Embedder", FailingEmbedder)

    config = PipelineConfig(
        domain="example.com",
        projects_root=tmp_path,
        technical_only=True,
        save_snapshot=False,
    )
    run(config)
    checkpoint = tmp_path / "example.com" / "cache" / "checkpoints" / "extraction.json"
    assert checkpoint.is_file()

    class NoExtractionCacheUse:
        def __init__(self, *args, **kwargs):
            pass

        def get(self, *args, **kwargs):
            raise AssertionError("extraction cache should not be read when checkpoint resumes")

        def put(self, *args, **kwargs):
            raise AssertionError("extraction cache should not be written when checkpoint resumes")

        def stats(self):
            return {"hits": 0, "misses": 0, "writes": 0}

    def fail_extract(*args, **kwargs):
        raise AssertionError("extract should not run when checkpoint resumes")

    monkeypatch.setattr("site_audit.pipeline.ExtractionCache", NoExtractionCacheUse)
    monkeypatch.setattr("site_audit.pipeline.extract", fail_extract)

    resumed = run(
        PipelineConfig(
            domain="example.com",
            projects_root=tmp_path,
            technical_only=True,
            resume=True,
            save_snapshot=False,
        )
    )

    assert resumed["status"] == "technical_only"
    assert resumed["pages"] == 2


def test_pipeline_uses_configured_extraction_workers(tmp_path, monkeypatch) -> None:
    calls: list[int] = []

    class RecordingExecutor:
        def __init__(self, max_workers):
            calls.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def map(self, fn, items):
            return [fn(item) for item in items]

    monkeypatch.setattr("site_audit.pipeline.Crawler", FakeCrawler)
    monkeypatch.setattr("site_audit.pipeline.Embedder", FailingEmbedder)
    monkeypatch.setattr("site_audit.pipeline.ThreadPoolExecutor", RecordingExecutor)

    summary = run(
        PipelineConfig(
            domain="example.com",
            projects_root=tmp_path,
            technical_only=True,
            extraction_workers=3,
            analysis_workers=1,
            save_snapshot=False,
        )
    )

    assert summary["status"] == "technical_only"
    assert calls == [3]


def test_pipeline_uses_configured_analysis_workers(tmp_path, monkeypatch) -> None:
    calls: list[int] = []

    class RecordingExecutor:
        def __init__(self, max_workers):
            calls.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def map(self, fn, items):
            return [fn(item) for item in items]

        def submit(self, fn, *args, **kwargs):
            class Result:
                def result(self_inner):
                    return fn(*args, **kwargs)

            return Result()

    monkeypatch.setattr("site_audit.pipeline.Crawler", FakeCrawler)
    monkeypatch.setattr("site_audit.pipeline.Embedder", FailingEmbedder)
    monkeypatch.setattr("site_audit.pipeline.ThreadPoolExecutor", RecordingExecutor)

    summary = run(
        PipelineConfig(
            domain="example.com",
            projects_root=tmp_path,
            technical_only=True,
            extraction_workers=1,
            analysis_workers=4,
            save_snapshot=False,
        )
    )

    assert summary["status"] == "technical_only"
    assert calls == [4]
