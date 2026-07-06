"""Adaptive worker selection for large audit stages."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Mapping

LOG = logging.getLogger(__name__)


NATIVE_THREAD_LIMITS: Mapping[str, str] = {
    "OMP_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
}


def configure_native_thread_limits() -> dict[str, str]:
    """Set conservative native-thread defaults for nested parallel stages."""
    changed: dict[str, str] = {}
    for key, value in NATIVE_THREAD_LIMITS.items():
        if key not in os.environ:
            os.environ[key] = value
            changed[key] = value
    if changed:
        LOG.info(
            "  native thread limits: %s",
            ", ".join(f"{key}={value}" for key, value in sorted(changed.items())),
        )
    return changed


@dataclass(frozen=True)
class SystemSnapshot:
    cpu_count: int
    load_1m: float | None = None
    available_memory_mb: float | None = None


@dataclass(frozen=True)
class StageProfile:
    name: str
    kind: str = "cpu"
    min_workers: int = 1
    max_workers: int | None = None
    estimated_worker_rss_mb: int | None = None
    io_cap: int | None = None


@dataclass(frozen=True)
class WorkerDecision:
    stage: str
    workers: int
    max_workers: int
    reasons: tuple[str, ...]


def current_system_snapshot() -> SystemSnapshot:
    cpu_count = max(1, os.cpu_count() or 1)
    try:
        load_1m = float(os.getloadavg()[0])
    except (AttributeError, OSError):
        load_1m = None
    return SystemSnapshot(
        cpu_count=cpu_count,
        load_1m=load_1m,
        available_memory_mb=_available_memory_mb(),
    )


def _available_memory_mb() -> float | None:
    if hasattr(os, "sysconf"):
        try:
            pages = os.sysconf("SC_AVPHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            if pages and page_size:
                return float(pages * page_size) / (1024 * 1024)
        except (OSError, ValueError):
            pass
    return None


class AdaptiveWorkerController:
    """Choose active workers from stage profile and current system pressure."""

    def __init__(
        self,
        *,
        max_workers: int | None = None,
        snapshot_provider=current_system_snapshot,
    ) -> None:
        snapshot = snapshot_provider()
        self._snapshot_provider = snapshot_provider
        self.max_workers = max(1, int(max_workers or snapshot.cpu_count or 1))

    def select(
        self,
        profile: StageProfile,
        *,
        item_count: int | None = None,
        explicit_workers: int | None = None,
    ) -> WorkerDecision:
        snapshot = self._snapshot_provider()
        cap = min(self.max_workers, max(1, int(profile.max_workers or self.max_workers)))
        reasons: list[str] = [f"cap={cap}"]
        if explicit_workers and explicit_workers > 0:
            workers = min(cap, max(1, int(explicit_workers)))
            reasons.append(f"explicit={explicit_workers}")
            return WorkerDecision(profile.name, workers, cap, tuple(reasons))

        item_cap = max(1, int(item_count or cap))
        cap = min(cap, item_cap)
        reasons.append(f"items={item_cap}")

        cpu_cap = max(1, int(snapshot.cpu_count or 1))
        if snapshot.load_1m is not None:
            idle_estimate = max(1, int(round(cpu_cap - max(0.0, snapshot.load_1m))))
            if snapshot.load_1m >= cpu_cap:
                idle_estimate = max(1, cpu_cap // 2)
            cpu_cap = min(cpu_cap, max(1, idle_estimate))
            reasons.append(f"load1={snapshot.load_1m:.2f}")
        cap = min(cap, cpu_cap)
        reasons.append(f"cpu={cpu_cap}")

        if profile.io_cap and profile.kind == "io":
            cap = min(cap, max(1, int(profile.io_cap)))
            reasons.append(f"io_cap={profile.io_cap}")
        elif profile.kind == "memory_heavy":
            cap = min(cap, max(profile.min_workers, min(4, cap)))
            reasons.append("memory_heavy_cap=4")

        if profile.estimated_worker_rss_mb and snapshot.available_memory_mb is not None:
            memory_cap = max(1, int(snapshot.available_memory_mb // profile.estimated_worker_rss_mb))
            cap = min(cap, memory_cap)
            reasons.append(
                f"mem={snapshot.available_memory_mb:.0f}mb/{profile.estimated_worker_rss_mb}mb"
            )

        workers = max(1, min(cap, max(1, int(profile.max_workers or cap))))
        workers = max(min(workers, cap), min(max(1, profile.min_workers), cap))
        return WorkerDecision(profile.name, workers, self.max_workers, tuple(reasons))

    def log_decision(self, decision: WorkerDecision) -> None:
        LOG.info(
            "  adaptive workers: %s uses %d/%d (%s)",
            decision.stage,
            decision.workers,
            decision.max_workers,
            "; ".join(decision.reasons),
        )
