"""Small benchmark harness for cached audit stages."""

from __future__ import annotations

import hashlib
import json
import resource
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable


@dataclass
class BenchmarkResult:
    name: str
    wall_seconds: float
    user_cpu_seconds: float
    system_cpu_seconds: float
    max_rss_mb: float
    output_fingerprint: str

    def to_dict(self) -> dict:
        return asdict(self)


def benchmark_callable(name: str, fn: Callable[[], object]) -> BenchmarkResult:
    before = resource.getrusage(resource.RUSAGE_SELF)
    started = time.perf_counter()
    output = fn()
    elapsed = time.perf_counter() - started
    after = resource.getrusage(resource.RUSAGE_SELF)
    return BenchmarkResult(
        name=name,
        wall_seconds=round(elapsed, 6),
        user_cpu_seconds=round(after.ru_utime - before.ru_utime, 6),
        system_cpu_seconds=round(after.ru_stime - before.ru_stime, 6),
        max_rss_mb=round(_rss_to_mb(after.ru_maxrss), 3),
        output_fingerprint=_fingerprint(output),
    )


def write_benchmark(path: Path, result: BenchmarkResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def fingerprint_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda p: str(p)):
        digest.update(str(path).encode("utf-8"))
        if path.is_file():
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _fingerprint(value: object) -> str:
    try:
        blob = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    except TypeError:
        blob = repr(value)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _rss_to_mb(raw: int) -> float:
    # macOS reports bytes; Linux reports KiB.
    if raw > 10_000_000:
        return raw / (1024 * 1024)
    return raw / 1024
