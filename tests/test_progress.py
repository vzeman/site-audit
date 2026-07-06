from site_audit.progress import ProgressLogger, ProgressSnapshot, format_progress


class Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


def test_progress_logger_reports_by_interval() -> None:
    clock = Clock()
    progress = ProgressLogger("anchors", total=100, interval_seconds=10, percent_step=50, clock=clock)

    assert progress.update(10) is None
    clock.value = 10
    snapshot = progress.update(20)

    assert snapshot is not None
    assert snapshot.processed == 20
    assert snapshot.percent == 20.0


def test_progress_logger_reports_by_percent_step() -> None:
    clock = Clock()
    progress = ProgressLogger("anchors", total=100, interval_seconds=999, percent_step=25, clock=clock)

    clock.value = 1
    snapshot = progress.update(25)

    assert snapshot is not None
    assert snapshot.percent == 25.0


def test_format_progress_with_eta() -> None:
    rendered = format_progress(
        ProgressSnapshot(
            name="anchors",
            processed=50,
            total=100,
            elapsed_seconds=10,
            rate_per_second=5,
            percent=50,
            eta_seconds=10,
        )
    )

    assert rendered == "anchors 50/100 (50.0%) · 5.0/s · ETA 10s"


def test_format_progress_without_total() -> None:
    rendered = format_progress(
        ProgressSnapshot(
            name="links",
            processed=50,
            total=None,
            elapsed_seconds=10,
            rate_per_second=5,
            percent=None,
            eta_seconds=None,
        )
    )

    assert rendered == "links 50 · 5.0/s"
