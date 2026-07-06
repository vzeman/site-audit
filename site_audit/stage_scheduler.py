"""Dependency-aware stage execution helpers."""

from __future__ import annotations

import logging
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Callable

from .adaptive_workers import AdaptiveWorkerController, StageProfile

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class StageTask:
    name: str
    run: Callable[[], object]
    dependencies: tuple[str, ...] = ()
    profile: StageProfile = field(default_factory=lambda: StageProfile("stage"))


def run_stage_tasks(
    tasks: list[StageTask],
    *,
    controller: AdaptiveWorkerController,
    max_workers: int | None = None,
) -> dict[str, object]:
    """Run a small DAG of tasks and return results keyed by stage name."""
    task_by_name = {task.name: task for task in tasks}
    if len(task_by_name) != len(tasks):
        raise ValueError("stage names must be unique")
    for task in tasks:
        missing = [dep for dep in task.dependencies if dep not in task_by_name]
        if missing:
            raise ValueError(f"stage {task.name!r} has unknown dependencies: {missing}")
    _validate_acyclic(task_by_name)

    if not tasks:
        return {}

    worker_count = max_workers
    if worker_count is None:
        decision = controller.select(
            StageProfile("stage-dag", kind="mixed"),
            item_count=len(tasks),
        )
        worker_count = decision.workers
        controller.log_decision(decision)
    worker_count = max(1, int(worker_count))

    pending = set(task_by_name)
    running = {}
    completed: set[str] = set()
    results: dict[str, object] = {}

    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        while pending or running:
            ready = sorted(
                name
                for name in pending
                if all(dep in completed for dep in task_by_name[name].dependencies)
            )
            while ready and len(running) < worker_count:
                name = ready.pop(0)
                task = task_by_name[name]
                pending.remove(name)
                LOG.info("  scheduled stage: %s", name)
                running[pool.submit(_run_one, task)] = name
            if not running:
                blocked = ", ".join(sorted(pending))
                raise RuntimeError(f"no runnable stages; blocked: {blocked}")
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in done:
                name = running.pop(future)
                results[name] = future.result()
                completed.add(name)

    return results


def _run_one(task: StageTask) -> object:
    started = time.perf_counter()
    LOG.info("  stage start: %s", task.name)
    status = "ok"
    try:
        return task.run()
    except Exception:
        status = "error"
        raise
    finally:
        LOG.info(
            "  stage done: %s in %.1fs (%s)",
            task.name,
            time.perf_counter() - started,
            status,
        )


def _validate_acyclic(tasks: dict[str, StageTask]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(name: str) -> None:
        if name in visited:
            return
        if name in visiting:
            raise ValueError(f"stage dependency cycle includes {name!r}")
        visiting.add(name)
        for dep in tasks[name].dependencies:
            visit(dep)
        visiting.remove(name)
        visited.add(name)

    for name in tasks:
        visit(name)
