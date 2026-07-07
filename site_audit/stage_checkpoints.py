"""Versioned stage checkpoints with completion markers."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any


def fingerprint(value: Any) -> str:
    blob = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


class StageCheckpointStore:
    def __init__(self, root: Path, *, schema_version: int = 1) -> None:
        self.root = Path(root)
        self.schema_version = int(schema_version)

    def write(self, name: str, payload: Any, *, inputs: Any, metadata: dict | None = None) -> Path:
        stage_dir = self._stage_dir(name)
        stage_dir.mkdir(parents=True, exist_ok=True)
        payload_path = stage_dir / "payload.json"
        complete_path = stage_dir / "complete.json"
        if complete_path.exists():
            complete_path.unlink()
        _write_json_atomic(payload_path, payload)
        _write_json_atomic(
            complete_path,
            {
                "stage": name,
                "schema_version": self.schema_version,
                "input_fingerprint": fingerprint(inputs),
                "metadata": metadata or {},
                "completed_at": time.time(),
                "payload_file": payload_path.name,
            },
        )
        return payload_path

    def read(self, name: str, *, inputs: Any) -> Any | None:
        stage_dir = self._stage_dir(name)
        complete_path = stage_dir / "complete.json"
        payload_path = stage_dir / "payload.json"
        try:
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if int(complete.get("schema_version") or 0) != self.schema_version:
            return None
        if complete.get("input_fingerprint") != fingerprint(inputs):
            return None
        try:
            return json.loads(payload_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _stage_dir(self, name: str) -> Path:
        safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name).strip("_")
        return self.root / (safe or "stage")


def _write_json_atomic(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
