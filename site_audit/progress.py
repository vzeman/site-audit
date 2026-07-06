"""Progress reporting helpers for long audit stages."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

LOG = logging.getLogger(__name__)


@dataclass
class ProgressSnapshot:
    name: str
    processed: int
    total: int | None
    elapsed_seconds: float
    rate_per_second: float
    percent: float | None
    eta_seconds: float | None


class ProgressLogger:
    def __init__(
        self,
        name: str,
        *,
        total: int | None = None,
        interval_seconds: float = 30.0,
        percent_step: float = 5.0,
        logger: logging.Logger | None = None,
        clock=time.perf_counter,
    ) -> None:
        self.name = name
        self.total = total if total and total > 0 else None
        self.interval_seconds = max(0.0, float(interval_seconds))
        self.percent_step = max(0.1, float(percent_step))
        self.logger = logger or LOG
        self.clock = clock
        self.started = clock()
        self.last_logged_at = self.started
        self.last_percent = 0.0

    def update(self, processed: int, *, force: bool = False) -> ProgressSnapshot | None:
        now = self.clock()
        snapshot = self.snapshot(processed, now=now)
        should_log = force or (now - self.last_logged_at) >= self.interval_seconds
        if snapshot.percent is not None and snapshot.percent - self.last_percent >= self.percent_step:
            should_log = True
        if not should_log:
            return None
        self.last_logged_at = now
        self.last_percent = snapshot.percent or self.last_percent
        self.logger.info("  progress: %s", format_progress(snapshot))
        return snapshot

    def snapshot(self, processed: int, *, now: float | None = None) -> ProgressSnapshot:
        now = self.clock() if now is None else now
        elapsed = max(0.0, now - self.started)
        rate = processed / elapsed if elapsed > 0 else 0.0
        percent = None
        eta = None
        if self.total:
            percent = min(100.0, max(0.0, processed / self.total * 100.0))
            if rate > 0 and processed < self.total:
                eta = (self.total - processed) / rate
        return ProgressSnapshot(
            name=self.name,
            processed=processed,
            total=self.total,
            elapsed_seconds=elapsed,
            rate_per_second=rate,
            percent=percent,
            eta_seconds=eta,
        )


def format_progress(snapshot: ProgressSnapshot) -> str:
    if snapshot.total:
        eta = _format_duration(snapshot.eta_seconds) if snapshot.eta_seconds is not None else "unknown"
        return (
            f"{snapshot.name} {snapshot.processed}/{snapshot.total} "
            f"({snapshot.percent:.1f}%) · {snapshot.rate_per_second:.1f}/s · ETA {eta}"
        )
    return f"{snapshot.name} {snapshot.processed} · {snapshot.rate_per_second:.1f}/s"


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    seconds = max(0, int(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {sec}s"
    return f"{sec}s"
