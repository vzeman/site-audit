import logging
import sys
import types

import numpy as np
import pytest
import torch

from site_audit.cache import EmbeddingCache, content_hash
from site_audit.embedder import (
    EmbedInput,
    Embedder,
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

    def encode(self, texts: list[str], batch_size: int = 32, show_progress: bool = False, **kwargs) -> np.ndarray:
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
        arr = np.ones((len(texts), 2), dtype=np.float32)
        for idx, text in enumerate(texts):
            if text == "bad":
                arr[idx] = np.nan
        return arr


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
    embeddings = CpuFallbackEmbedder().encode(["ok", "bad"], batch_size=1)

    assert np.isfinite(embeddings).all()
    assert embeddings.tolist() == [[1.0, 1.0], [1.0, 0.0]]


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


def test_whole_chunk_failure_zero_fills_without_cpu_fallback() -> None:
    emb = ScriptedEmbedder([_nan_at([0, 1, 2]), _nan_at([0, 1, 2])])

    result = emb.encode(["a", "b", "c"], zero_fill_bad=True)

    assert np.isfinite(result).all()
    assert not result.any()
    assert [device for device, _ in emb.calls] == [None, None]


def test_whole_chunk_failure_raises_without_cpu_fallback() -> None:
    emb = ScriptedEmbedder([_nan_at([0, 1, 2]), _nan_at([0, 1, 2])])

    with pytest.raises(RuntimeError, match="3/3 non-finite"):
        emb.encode(["a", "b", "c"])

    assert [device for device, _ in emb.calls] == [None, None]


def test_chunk_size_one_failure_zero_fills_without_cpu_fallback() -> None:
    emb = ScriptedEmbedder([_nan_at([0]), _nan_at([0])])

    result = emb.encode(["a"], batch_size=1, zero_fill_bad=True)

    assert result.tolist() == [[0.0, 0.0, 0.0]]
    assert [device for device, _ in emb.calls] == [None, None]


def test_raises_when_rows_stay_non_finite() -> None:
    emb = ScriptedEmbedder([_nan_at([1]), _nan_at([0]), _nan_at([0])])

    with pytest.raises(RuntimeError, match="non-finite"):
        emb.encode(["a", "b"])


def test_encode_reloads_model_before_each_retry() -> None:
    emb = ScriptedEmbedder([_nan_at([0]), _nan_at([0]), _finite])
    emb.ensure_calls = []
    original_ensure = emb._ensure

    def tracking_ensure(device=None):
        emb.ensure_calls.append(device)
        original_ensure(device)

    emb._ensure = tracking_ensure

    result = emb.encode(["a", "b"])

    assert np.isfinite(result).all()
    # initial load, accelerator reload, CPU reload
    assert emb.ensure_calls == [None, None, "cpu"]


def test_zero_fill_bad_rows_instead_of_raising() -> None:
    emb = ScriptedEmbedder([_nan_at([1]), _nan_at([0]), _nan_at([0])])

    result = emb.encode(["a", "b"], zero_fill_bad=True)

    assert np.isfinite(result).all()
    assert not result[1].any()
    assert result[0].any()


class PoisonEmbedder(Embedder):
    """Embedder whose model NaNs the text \"poison\" on every device."""

    def __init__(self) -> None:
        super().__init__("fake-model")

    def _ensure(self, device=None):
        self._device = device
        self._model = object()

    def _clear_accelerator_cache(self):
        pass

    def _encode_current_model(self, texts, *, batch_size, show_progress):
        arr = np.ones((len(texts), 2), dtype=np.float32)
        for k, text in enumerate(texts):
            if text == "poison":
                arr[k] = np.nan
        return arr


def test_encode_pages_never_caches_zero_filled_rows() -> None:
    cache = FakeEmbeddingCache()
    inputs = [
        EmbedInput(url="https://example.com/ok", text="fine"),
        EmbedInput(url="https://example.com/bad", text="poison"),
    ]

    embeddings = PoisonEmbedder().encode_pages(inputs, cache, use_cache=True, batch_size=2)

    assert embeddings.shape == (2, 2)
    assert embeddings[0].any()
    assert not embeddings[1].any()
    assert "https://example.com/ok" in cache.entries
    assert "https://example.com/bad" not in cache.entries


def test_encode_and_cache_skips_zero_rows_and_logs_identifiers(caplog) -> None:
    """The shared helper works for (url, hash)-keyed caches (paragraphs)."""

    class FakeParagraphCache:
        def __init__(self) -> None:
            self.entries: dict[tuple[str, str], np.ndarray] = {}

        def put(self, url: str, hash_: str, embedding: np.ndarray) -> None:
            self.entries[(url, hash_)] = embedding

    cache = FakeParagraphCache()
    texts = ["fine", "poison", "also fine"]
    urls = ["https://example.com/a", "https://example.com/a", "https://example.com/b"]
    hashes = ["h0", "h1", "h2"]
    identifiers = [
        "https://example.com/a (paragraph 0)",
        "https://example.com/a (paragraph 1)",
        "https://example.com/b (paragraph 0)",
    ]

    with caplog.at_level(logging.WARNING, logger="site_audit.embedder"):
        embs = PoisonEmbedder().encode_and_cache(
            texts, urls, hashes, cache, batch_size=2, identifiers=identifiers,
        )

    assert embs.shape == (3, 2)
    assert embs[0].any() and embs[2].any()
    assert not embs[1].any()
    # Two paragraphs of the same URL cached independently; the poison row
    # is absent so a later run retries it.
    assert set(cache.entries) == {
        ("https://example.com/a", "h0"),
        ("https://example.com/b", "h2"),
    }
    assert "https://example.com/a (paragraph 1)" in caplog.text


def test_encode_returns_to_accelerator_after_cpu_fallback() -> None:
    emb = ScriptedEmbedder([_nan_at([0]), _nan_at([0]), _finite, _finite])

    first = emb.encode(["a", "ok"])
    second = emb.encode(["b"])

    assert np.isfinite(first).all()
    assert np.isfinite(second).all()
    # ladder for call 1: accelerator, accelerator reload, cpu fallback;
    # call 2 must not stay stranded on cpu.
    assert [device for device, _ in emb.calls] == [None, None, "cpu", None]


def test_reload_defeats_real_ensure_early_return(monkeypatch) -> None:
    """With the real ``_ensure``/``_reload``, every retry rung constructs a
    fresh model — the regression of the previous fix was retrying on a
    stale (corrupted) model after only clearing the allocator cache."""
    constructed_devices: list[str | None] = []

    class FakeSentenceTransformer:
        def __init__(self, model_name, trust_remote_code=False, device=None):
            constructed_devices.append(device)
            self.max_seq_length = 8192

        def __getitem__(self, idx):
            return types.SimpleNamespace(auto_model=types.SimpleNamespace())

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    monkeypatch.delenv("SITE_AUDIT_DEVICE", raising=False)

    class RealEnsureEmbedder(Embedder):
        def __init__(self) -> None:
            super().__init__("stub-model")
            self._script = [_nan_at([0]), _nan_at([0]), _finite]

        def _encode_current_model(self, texts, *, batch_size, show_progress):
            return np.asarray(self._script.pop(0)(texts), dtype=np.float32)

        def _clear_accelerator_cache(self) -> None:
            pass

    result = RealEnsureEmbedder().encode(["a", "b"])

    assert np.isfinite(result).all()
    # initial load, accelerator reload, cpu reload — three constructions
    # prove _reload discards the model instead of hitting _ensure's
    # early-return.
    assert constructed_devices == [None, None, "cpu"]


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
