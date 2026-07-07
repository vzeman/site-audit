import json

from site_audit.benchmark import benchmark_callable, fingerprint_files, write_benchmark


def test_benchmark_callable_records_fingerprint() -> None:
    result = benchmark_callable("demo", lambda: {"b": 2, "a": 1})

    assert result.name == "demo"
    assert result.wall_seconds >= 0
    assert len(result.output_fingerprint) == 64


def test_write_benchmark_writes_json(tmp_path) -> None:
    result = benchmark_callable("demo", lambda: {"ok": True})

    write_benchmark(tmp_path / "bench.json", result)

    payload = json.loads((tmp_path / "bench.json").read_text(encoding="utf-8"))
    assert payload["name"] == "demo"
    assert payload["output_fingerprint"] == result.output_fingerprint


def test_fingerprint_files_is_stable(tmp_path) -> None:
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text("a", encoding="utf-8")
    b.write_text("b", encoding="utf-8")

    assert fingerprint_files([b, a]) == fingerprint_files([a, b])
