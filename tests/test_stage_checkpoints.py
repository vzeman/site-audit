from site_audit.stage_checkpoints import StageCheckpointStore, fingerprint


def test_checkpoint_reuses_matching_inputs(tmp_path) -> None:
    store = StageCheckpointStore(tmp_path, schema_version=2)

    store.write("anchor relevance", {"links": [1]}, inputs={"pages": 10}, metadata={"workers": 2})

    assert store.read("anchor relevance", inputs={"pages": 10}) == {"links": [1]}
    assert (tmp_path / "anchor_relevance" / "complete.json").is_file()


def test_checkpoint_invalidates_changed_inputs(tmp_path) -> None:
    store = StageCheckpointStore(tmp_path)

    store.write("stage", {"ok": True}, inputs={"a": 1})

    assert store.read("stage", inputs={"a": 2}) is None


def test_checkpoint_invalidates_schema_version(tmp_path) -> None:
    StageCheckpointStore(tmp_path, schema_version=1).write("stage", {"ok": True}, inputs={"a": 1})

    assert StageCheckpointStore(tmp_path, schema_version=2).read("stage", inputs={"a": 1}) is None


def test_checkpoint_ignores_partial_without_complete_marker(tmp_path) -> None:
    stage_dir = tmp_path / "stage"
    stage_dir.mkdir()
    (stage_dir / "payload.json").write_text('{"ok": true}', encoding="utf-8")

    assert StageCheckpointStore(tmp_path).read("stage", inputs={}) is None


def test_fingerprint_is_stable_for_key_order() -> None:
    assert fingerprint({"b": 2, "a": 1}) == fingerprint({"a": 1, "b": 2})
