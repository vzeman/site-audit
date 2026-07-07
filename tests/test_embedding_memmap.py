import json

import numpy as np

from site_audit.embedding_memmap import ensure_embedding_memmap, load_embedding_memmap


def test_ensure_embedding_memmap_creates_readonly_sidecar(tmp_path) -> None:
    npz = tmp_path / "embeddings.npz"
    expected = np.arange(12, dtype=np.float32).reshape(3, 4)
    np.savez_compressed(npz, urls=np.array(["a", "b", "c"]), hashes=np.array(["1", "2", "3"]), embeddings=expected)

    sidecar = ensure_embedding_memmap(npz)
    loaded = load_embedding_memmap(sidecar)

    assert sidecar.name == "embeddings.npz.embeddings.npy"
    assert isinstance(loaded, np.memmap)
    np.testing.assert_array_equal(loaded, expected)


def test_ensure_embedding_memmap_reuses_valid_sidecar(tmp_path) -> None:
    npz = tmp_path / "embeddings.npz"
    np.savez_compressed(npz, embeddings=np.ones((2, 2), dtype=np.float32))

    first = ensure_embedding_memmap(npz)
    marker = first.stat().st_mtime
    second = ensure_embedding_memmap(npz)

    assert second == first
    assert first.stat().st_mtime == marker


def test_ensure_embedding_memmap_rebuilds_when_source_changes(tmp_path) -> None:
    npz = tmp_path / "embeddings.npz"
    np.savez_compressed(npz, embeddings=np.ones((2, 2), dtype=np.float32))
    sidecar = ensure_embedding_memmap(npz)
    meta = sidecar.with_suffix(sidecar.suffix.replace(".npy", ".json"))
    old_meta = json.loads(meta.read_text(encoding="utf-8"))

    np.savez_compressed(npz, embeddings=np.zeros((2, 2), dtype=np.float32))
    rebuilt = ensure_embedding_memmap(npz)
    loaded = load_embedding_memmap(rebuilt)
    new_meta = json.loads(meta.read_text(encoding="utf-8"))

    assert old_meta["source_size"] != new_meta["source_size"] or old_meta["source_mtime"] != new_meta["source_mtime"]
    np.testing.assert_array_equal(loaded, np.zeros((2, 2), dtype=np.float32))
