import numpy as np
import pytest
import torch

from site_audit.cache import EmbeddingCache, content_hash
from site_audit.embedder import (
    EmbedInput,
    Embedder,
    _cpu_thread_boost,
    _non_finite_row_indices,
    _reset_position_ids,
)


class FakeEmbeddingCache:
    def __init__(self) -> None:
        self.entries = {}
        self.save_count = 0

    def get(self, url: str, hash_: str):
        entry = self.entries.get(url)
        if entry and entry[0] == hash_:
            return entry[1]
        return None

    def put(self, url: str, hash_: str, embedding: np.ndarray) -> None:
        self.entries[url] = (hash_, embedding)

    def save(self) -> None:
        self.save_count += 1


class FakeEmbedder(Embedder):
    def __init__(self) -> None:
        super().__init__("fake-model")

    def encode(self, texts: list[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        return np.asarray([[float(len(text)), float(i)] for i, text in enumerate(texts)], dtype=np.float32)


class CpuFallbackEmbedder(Embedder):
    def __init__(self) -> None:
        super().__init__("fake-model")
        self.ensure_calls = []

    def _ensure(self, device: str | None = None) -> None:
        self._device = device
        self._model = object()
        self.ensure_calls.append(device)

    def _encode_current_model(
        self,
        texts: list[str],
        *,
        batch_size: int,
        show_progress: bool,
    ) -> np.ndarray:
        if self._device == "cpu":
            return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)
        return np.asarray([[np.nan, np.nan] for _ in texts], dtype=np.float32)


class FakeModel:
    def __init__(self, max_seq_length: int | None = 8192) -> None:
        self.max_seq_length = max_seq_length


class MaxSeqLengthEmbedder(Embedder):
    def __init__(self, max_seq_length: int | None) -> None:
        super().__init__("fake-model", max_seq_length=max_seq_length)
        self.fake_model = FakeModel()

    def _ensure(self, device: str | None = None) -> None:
        self._model = self.fake_model
        self._device = device
        if self.max_seq_length is not None:
            current_max = getattr(self._model, "max_seq_length", None)
            if current_max is None or current_max > self.max_seq_length:
                self._model.max_seq_length = self.max_seq_length


def test_encode_pages_persists_embedding_cache_in_chunks(monkeypatch) -> None:
    monkeypatch.setenv("SITE_AUDIT_EMBED_CACHE_SAVE_EVERY", "2")
    cache = FakeEmbeddingCache()
    inputs = [
        EmbedInput(url=f"https://example.com/{i}", text="x" * (i + 1))
        for i in range(5)
    ]

    embeddings = FakeEmbedder().encode_pages(inputs, cache, use_cache=True, batch_size=2)

    assert embeddings.shape == (5, 2)
    assert cache.save_count == 3
    assert set(cache.entries) == {item.url for item in inputs}


def test_encode_retries_non_finite_embeddings_on_cpu() -> None:
    embeddings = CpuFallbackEmbedder().encode(["text"], batch_size=1)

    assert np.isfinite(embeddings).all()
    assert embeddings.tolist() == [[1.0, 0.0]]


def test_embedder_caps_model_max_sequence_length() -> None:
    embedder = MaxSeqLengthEmbedder(512)

    embedder._ensure()

    assert embedder.fake_model.max_seq_length == 512


def test_reset_position_ids_preserves_existing_buffer_rank() -> None:
    class Embeddings(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer(
                "position_ids",
                torch.zeros((1, 8), dtype=torch.long),
                persistent=False,
            )

    class Config:
        max_position_embeddings = 8

    class Model:
        config = Config()

        def __init__(self) -> None:
            self.embeddings = Embeddings()

    model = Model()

    _reset_position_ids(model)

    assert model.embeddings.position_ids.shape == (1, 8)
    assert model.embeddings.position_ids.tolist() == [list(range(8))]


class ScriptedEmbedder(Embedder):
    """Embedder whose model calls follow a scripted sequence of outputs."""

    def __init__(self, script) -> None:
        super().__init__("stub-model")
        self._script = list(script)
        self.calls: list[tuple[str | None, list[str]]] = []

    def _ensure(self, device: str | None = None) -> None:
        self._device = device
        self._model = object()

    def _clear_accelerator_cache(self) -> None:
        pass

    def _encode_current_model(self, texts, *, batch_size, show_progress):
        self.calls.append((self._device, list(texts)))
        return np.asarray(self._script.pop(0)(texts), dtype=np.float32)


def _finite(texts):
    return np.ones((len(texts), 3), dtype=np.float32)


def _nan_at(bad_positions):
    def fn(texts):
        arr = np.ones((len(texts), 3), dtype=np.float32)
        for pos in bad_positions:
            arr[pos] = np.nan
        return arr

    return fn


def test_non_finite_row_indices() -> None:
    arr = np.ones((3, 2), dtype=np.float32)
    arr[1, 0] = np.inf
    assert _non_finite_row_indices(arr).tolist() == [1]
    assert _non_finite_row_indices(np.zeros((0, 0), dtype=np.float32)).tolist() == []


def test_encode_finite_passthrough_makes_one_call() -> None:
    emb = ScriptedEmbedder([_finite])

    result = emb.encode(["a", "b", "c"])

    assert result.shape == (3, 3)
    assert len(emb.calls) == 1


def test_accelerator_retry_reencodes_only_bad_rows() -> None:
    emb = ScriptedEmbedder([_nan_at([1]), _finite])

    result = emb.encode(["a", "b", "c"])

    assert np.isfinite(result).all()
    assert len(emb.calls) == 2
    # The retry stays on the accelerator (no device switch) and receives
    # only the non-finite row's text.
    assert emb.calls[1] == (None, ["b"])


def test_cpu_fallback_reencodes_only_still_bad_rows() -> None:
    emb = ScriptedEmbedder([_nan_at([0, 2]), _nan_at([0, 1]), _finite])

    result = emb.encode(["a", "b", "c"])

    assert np.isfinite(result).all()
    assert len(emb.calls) == 3
    assert emb.calls[1] == (None, ["a", "c"])
    assert emb.calls[2] == ("cpu", ["a", "c"])


def test_raises_when_rows_stay_non_finite() -> None:
    emb = ScriptedEmbedder([_nan_at([1]), _nan_at([0]), _nan_at([0])])

    with pytest.raises(RuntimeError, match="non-finite"):
        emb.encode(["a", "b"])


def test_cpu_thread_boost_sets_and_restores(monkeypatch) -> None:
    seen: list[int] = []
    monkeypatch.setattr(torch, "get_num_threads", lambda: 1)
    monkeypatch.setattr(torch, "set_num_threads", lambda n: seen.append(n))

    with _cpu_thread_boost():
        pass

    assert len(seen) == 2
    assert seen[0] >= 1
    assert seen[1] == 1


def test_embedding_cache_ignores_and_rejects_non_finite_vectors(tmp_path) -> None:
    cache = EmbeddingCache(tmp_path / "embeddings.npz")
    hash_ = content_hash("text|model")
    cache._cache["https://example.com/"] = (
        hash_,
        np.asarray([np.nan, np.nan], dtype=np.float32),
    )

    assert cache.get("https://example.com/", hash_) is None
    with pytest.raises(ValueError):
        cache.put("https://example.com/", hash_, np.asarray([np.nan], dtype=np.float32))
