import json

import pytest

from site_audit.report_lifecycle import (
    BUILDING_MARKER,
    COMPLETE_MARKER,
    PREVIOUS_INDEX,
    begin_report_build,
    complete_report_build,
)


def test_begin_report_build_moves_stale_index_and_clears_complete_marker(tmp_path) -> None:
    (tmp_path / "index.html").write_text("old", encoding="utf-8")
    (tmp_path / COMPLETE_MARKER).write_text("{}", encoding="utf-8")

    marker = begin_report_build(tmp_path, metadata={"domain": "example.com"})

    assert marker == tmp_path / BUILDING_MARKER
    assert not (tmp_path / "index.html").exists()
    assert (tmp_path / PREVIOUS_INDEX).read_text(encoding="utf-8") == "old"
    assert not (tmp_path / COMPLETE_MARKER).exists()
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["status"] == "building"
    assert payload["metadata"]["domain"] == "example.com"


def test_complete_report_build_requires_fresh_files(tmp_path) -> None:
    begin_report_build(tmp_path)

    with pytest.raises(FileNotFoundError, match="index.html"):
        complete_report_build(tmp_path)

    assert (tmp_path / BUILDING_MARKER).is_file()
    assert not (tmp_path / COMPLETE_MARKER).exists()


def test_complete_report_build_writes_complete_marker_last(tmp_path) -> None:
    begin_report_build(tmp_path)
    (tmp_path / "index.html").write_text("fresh", encoding="utf-8")
    (tmp_path / "site_metrics.json").write_text("{}", encoding="utf-8")

    marker = complete_report_build(
        tmp_path,
        metadata={"mode": "full"},
        required_files=("index.html", "site_metrics.json"),
    )

    assert marker == tmp_path / COMPLETE_MARKER
    assert not (tmp_path / BUILDING_MARKER).exists()
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["status"] == "complete"
    assert payload["metadata"]["mode"] == "full"
    assert payload["required_files"] == ["index.html", "site_metrics.json"]
