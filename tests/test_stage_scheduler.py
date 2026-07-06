import pytest

from site_audit.adaptive_workers import AdaptiveWorkerController, SystemSnapshot
from site_audit.stage_scheduler import StageTask, run_stage_tasks


def _controller(workers: int = 2) -> AdaptiveWorkerController:
    return AdaptiveWorkerController(
        max_workers=workers,
        snapshot_provider=lambda: SystemSnapshot(cpu_count=workers, load_1m=0.0, available_memory_mb=4096),
    )


def test_stage_scheduler_runs_dependencies_before_dependents() -> None:
    order: list[str] = []

    tasks = [
        StageTask("a", lambda: order.append("a") or "A"),
        StageTask("b", lambda: order.append("b") or "B", dependencies=("a",)),
        StageTask("c", lambda: order.append("c") or "C", dependencies=("b",)),
    ]

    results = run_stage_tasks(tasks, controller=_controller(), max_workers=2)

    assert results == {"a": "A", "b": "B", "c": "C"}
    assert order == ["a", "b", "c"]


def test_stage_scheduler_rejects_unknown_dependency() -> None:
    with pytest.raises(ValueError, match="unknown dependencies"):
        run_stage_tasks(
            [StageTask("a", lambda: None, dependencies=("missing",))],
            controller=_controller(),
        )


def test_stage_scheduler_rejects_cycles() -> None:
    with pytest.raises(ValueError, match="cycle"):
        run_stage_tasks(
            [
                StageTask("a", lambda: None, dependencies=("b",)),
                StageTask("b", lambda: None, dependencies=("a",)),
            ],
            controller=_controller(),
        )


def test_stage_scheduler_uses_adaptive_worker_count(monkeypatch) -> None:
    calls: list[int] = []

    class RecordingExecutor:
        def __init__(self, max_workers):
            calls.append(max_workers)
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            class Result:
                def result(self_inner):
                    return fn(*args, **kwargs)

            return Result()

    def fake_wait(running, return_when=None):
        return set(running), set()

    monkeypatch.setattr("site_audit.stage_scheduler.ThreadPoolExecutor", RecordingExecutor)
    monkeypatch.setattr("site_audit.stage_scheduler.wait", fake_wait)

    results = run_stage_tasks(
        [
            StageTask("a", lambda: "A"),
            StageTask("b", lambda: "B"),
            StageTask("c", lambda: "C"),
        ],
        controller=_controller(workers=2),
    )

    assert calls == [2]
    assert results == {"a": "A", "b": "B", "c": "C"}
