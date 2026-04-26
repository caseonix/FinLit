"""Pipeline cache: lazy build, key-based reuse, thread-safe."""
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest

from finlit.integrations.mcp import pipeline_cache


@pytest.fixture(autouse=True)
def _clear_cache():
    pipeline_cache.clear_cache()
    yield
    pipeline_cache.clear_cache()


def _fake_pipeline_factory(call_log):
    """Returns a callable that records calls and returns a sentinel object."""
    class FakePipeline:
        def __init__(self, schema, extractor, review_threshold, vision_extractor):
            call_log.append((schema.name, extractor, review_threshold, vision_extractor))
            self.schema = schema
            self.extractor = extractor

    return FakePipeline


def test_cache_builds_once_per_key():
    calls = []
    fake = _fake_pipeline_factory(calls)

    with patch("finlit.integrations.mcp.pipeline_cache.DocumentPipeline", fake):
        p1 = pipeline_cache.get_pipeline("claude", None, "cra.t4", 0.85)
        p2 = pipeline_cache.get_pipeline("claude", None, "cra.t4", 0.85)

    assert p1 is p2
    assert len(calls) == 1


def test_cache_separates_by_extractor():
    calls = []
    fake = _fake_pipeline_factory(calls)

    with patch("finlit.integrations.mcp.pipeline_cache.DocumentPipeline", fake):
        pipeline_cache.get_pipeline("claude", None, "cra.t4", 0.85)
        pipeline_cache.get_pipeline("ollama", None, "cra.t4", 0.85)

    assert len(calls) == 2


def test_cache_separates_by_schema():
    calls = []
    fake = _fake_pipeline_factory(calls)

    with patch("finlit.integrations.mcp.pipeline_cache.DocumentPipeline", fake):
        pipeline_cache.get_pipeline("claude", None, "cra.t4", 0.85)
        pipeline_cache.get_pipeline("claude", None, "cra.t5", 0.85)

    assert len(calls) == 2


def test_cache_thread_safe_under_contention():
    """Two threads requesting the same key should still build only once."""
    calls = []
    fake = _fake_pipeline_factory(calls)

    with patch("finlit.integrations.mcp.pipeline_cache.DocumentPipeline", fake):
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [
                pool.submit(pipeline_cache.get_pipeline, "claude", None, "cra.t4", 0.85)
                for _ in range(8)
            ]
            results = [f.result() for f in futures]

    assert all(r is results[0] for r in results)
    assert len(calls) == 1


def test_unknown_schema_raises_valueerror():
    with pytest.raises(ValueError, match="Unknown schema"):
        pipeline_cache.get_pipeline("claude", None, "cra.t99", 0.85)
