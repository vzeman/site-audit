"""Memory-map sidecars for compressed embedding caches."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def ensure_embedding_memmap(npz_path: Path, *, key: str = "embeddings") -> Path:
    npz_path = Path(npz_path)
    mmap_path = _mmap_path(npz_path, key)
    meta_path = _meta_path(npz_path, key)
    if _sidecar_valid(npz_path, mmap_path, meta_path, key):
        return mmap_path
    data = np.load(npz_path, allow_pickle=False)
    array = np.asarray(data[key], dtype=np.float32)
    mmap_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(mmap_path, array)
    _write_meta(meta_path, npz_path, key, array.shape, str(array.dtype))
    return mmap_path


def load_embedding_memmap(path: Path) -> np.memmap:
    return np.load(Path(path), mmap_mode="r", allow_pickle=False)


def _mmap_path(npz_path: Path, key: str) -> Path:
    return npz_path.with_suffix(npz_path.suffix + f".{key}.npy")


def _meta_path(npz_path: Path, key: str) -> Path:
    return npz_path.with_suffix(npz_path.suffix + f".{key}.json")


def _sidecar_valid(npz_path: Path, mmap_path: Path, meta_path: Path, key: str) -> bool:
    if not npz_path.is_file() or not mmap_path.is_file() or not meta_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    stat = npz_path.stat()
    return (
        meta.get("key") == key
        and int(meta.get("source_size") or -1) == int(stat.st_size)
        and float(meta.get("source_mtime") or -1.0) == float(stat.st_mtime)
    )


def _write_meta(meta_path: Path, npz_path: Path, key: str, shape: tuple[int, ...], dtype: str) -> None:
    stat = npz_path.stat()
    tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(
            {
                "key": key,
                "source": str(npz_path),
                "source_size": stat.st_size,
                "source_mtime": stat.st_mtime,
                "shape": list(shape),
                "dtype": dtype,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    tmp.replace(meta_path)
