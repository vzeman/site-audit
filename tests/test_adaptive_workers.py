from site_audit.adaptive_workers import (
    AdaptiveWorkerController,
    StageProfile,
    SystemSnapshot,
    configure_native_thread_limits,
)


def _snapshot(cpu=8, load=None, memory=8192):
    return lambda: SystemSnapshot(cpu_count=cpu, load_1m=load, available_memory_mb=memory)


def test_adaptive_workers_use_cpu_count_when_cap_is_auto() -> None:
    controller = AdaptiveWorkerController(snapshot_provider=_snapshot(cpu=6, load=0.0))

    decision = controller.select(StageProfile("analysis"), item_count=20)

    assert decision.workers == 6
    assert decision.max_workers == 6
    assert "cpu=6" in decision.reasons


def test_adaptive_workers_respect_max_cap_and_item_count() -> None:
    controller = AdaptiveWorkerController(max_workers=10, snapshot_provider=_snapshot(cpu=16, load=0.0))

    decision = controller.select(StageProfile("analysis"), item_count=3)

    assert decision.workers == 3


def test_adaptive_workers_respect_explicit_stage_override() -> None:
    controller = AdaptiveWorkerController(max_workers=8, snapshot_provider=_snapshot(cpu=16, load=0.0))

    decision = controller.select(
        StageProfile("analysis"),
        item_count=20,
        explicit_workers=12,
    )

    assert decision.workers == 8
    assert "explicit=12" in decision.reasons


def test_adaptive_workers_back_off_under_load() -> None:
    controller = AdaptiveWorkerController(max_workers=8, snapshot_provider=_snapshot(cpu=8, load=6.2))

    decision = controller.select(StageProfile("analysis"), item_count=20)

    assert decision.workers == 2
    assert any(reason.startswith("load1=") for reason in decision.reasons)


def test_memory_heavy_stage_limits_workers() -> None:
    controller = AdaptiveWorkerController(max_workers=12, snapshot_provider=_snapshot(cpu=12, load=0.0))

    decision = controller.select(StageProfile("paragraph-clusters", kind="memory_heavy"), item_count=50)

    assert decision.workers == 4


def test_estimated_worker_memory_caps_workers() -> None:
    controller = AdaptiveWorkerController(max_workers=10, snapshot_provider=_snapshot(cpu=10, load=0.0, memory=2500))

    decision = controller.select(
        StageProfile("extraction", estimated_worker_rss_mb=800),
        item_count=100,
    )

    assert decision.workers == 3
    assert any(reason.startswith("mem=2500mb/800mb") for reason in decision.reasons)


def test_configure_native_thread_limits_sets_missing_values(monkeypatch) -> None:
    keys = [
        "OMP_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TOKENIZERS_PARALLELISM",
    ]
    for key in keys:
        monkeypatch.delenv(key, raising=False)

    changed = configure_native_thread_limits()

    assert changed["OMP_NUM_THREADS"] == "1"
    assert changed["TOKENIZERS_PARALLELISM"] == "false"


def test_configure_native_thread_limits_preserves_existing_values(monkeypatch) -> None:
    keys = [
        "VECLIB_MAXIMUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TOKENIZERS_PARALLELISM",
    ]
    for key in keys:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    changed = configure_native_thread_limits()

    assert "OMP_NUM_THREADS" not in changed
    assert changed["TOKENIZERS_PARALLELISM"] == "false"
