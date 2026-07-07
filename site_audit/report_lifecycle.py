"""Atomic-ish report lifecycle markers."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable

BUILDING_MARKER = ".report_building.json"
COMPLETE_MARKER = ".report_complete.json"
PREVIOUS_INDEX = "index.previous.html"


def begin_report_build(report_dir: Path, *, metadata: dict | None = None) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    complete = report_dir / COMPLETE_MARKER
    if complete.exists():
        complete.unlink()
    index = report_dir / "index.html"
    if index.exists():
        index.replace(report_dir / PREVIOUS_INDEX)
    marker = report_dir / BUILDING_MARKER
    _write_json_atomic(
        marker,
        {
            "status": "building",
            "started_at": time.time(),
            "metadata": metadata or {},
        },
    )
    return marker


def complete_report_build(
    report_dir: Path,
    *,
    metadata: dict | None = None,
    required_files: Iterable[str] = ("index.html",),
) -> Path:
    missing = [name for name in required_files if not (report_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"report is missing required files: {', '.join(missing)}")
    marker = report_dir / COMPLETE_MARKER
    _write_json_atomic(
        marker,
        {
            "status": "complete",
            "completed_at": time.time(),
            "metadata": metadata or {},
            "required_files": list(required_files),
        },
    )
    building = report_dir / BUILDING_MARKER
    if building.exists():
        building.unlink()
    return marker


def _write_json_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
