"""Tests cho defensive parsing of env vars (Bonus regression).

Lưu ý: các test này reload module `task9_retrieval_pipeline` để
re-evaluate module-level constants. Sau khi test kết thúc cần reload lại
để state của module phản ánh env hiện tại (không bị leak sang test khác).
"""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture(autouse=True)
def _reload_pipeline_after_test(request):
    """Sau mỗi test, reload pipeline để module-level constants khớp env hiện tại."""
    yield
    try:
        import src.task9_retrieval_pipeline as pipeline
        importlib.reload(pipeline)
    except Exception:
        pass


def test_default_threshold_when_env_empty(monkeypatch):
    monkeypatch.setenv("SCORE_THRESHOLD", "")
    monkeypatch.delenv("DENSE_WEIGHT", raising=False)
    monkeypatch.delenv("BM25_WEIGHT", raising=False)
    import src.task9_retrieval_pipeline as pipeline
    importlib.reload(pipeline)
    assert pipeline.DEFAULT_SCORE_THRESHOLD == 0.50


def test_default_threshold_when_env_garbage(monkeypatch):
    monkeypatch.setenv("SCORE_THRESHOLD", "not-a-number")
    monkeypatch.delenv("DENSE_WEIGHT", raising=False)
    monkeypatch.delenv("BM25_WEIGHT", raising=False)
    import src.task9_retrieval_pipeline as pipeline
    importlib.reload(pipeline)
    assert pipeline.DEFAULT_SCORE_THRESHOLD == 0.50


def test_default_weights_when_env_empty(monkeypatch):
    monkeypatch.setenv("SCORE_THRESHOLD", "0.50")
    monkeypatch.setenv("DENSE_WEIGHT", "")
    monkeypatch.setenv("BM25_WEIGHT", "")
    import src.task9_retrieval_pipeline as pipeline
    importlib.reload(pipeline)
    assert pipeline.DEFAULT_DENSE_WEIGHT == 1.5
    assert pipeline.DEFAULT_BM25_WEIGHT == 1.0


def test_custom_threshold_parsed(monkeypatch):
    monkeypatch.setenv("SCORE_THRESHOLD", "0.42")
    monkeypatch.setenv("DENSE_WEIGHT", "2.0")
    monkeypatch.setenv("BM25_WEIGHT", "0.8")
    import src.task9_retrieval_pipeline as pipeline
    importlib.reload(pipeline)
    assert abs(pipeline.DEFAULT_SCORE_THRESHOLD - 0.42) < 1e-9
    assert abs(pipeline.DEFAULT_DENSE_WEIGHT - 2.0) < 1e-9
    assert abs(pipeline.DEFAULT_BM25_WEIGHT - 0.8) < 1e-9
