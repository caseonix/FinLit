"""Lazy, thread-safe (extractor, vision, schema, threshold) -> DocumentPipeline cache.

Used by the MCP server so that repeated tool calls with the same configuration
reuse one DocumentPipeline (and one underlying pydantic-ai client) instead of
rebuilding it every time.
"""
from __future__ import annotations

import threading

from finlit.extractors.vision_extractor import VisionExtractor
from finlit.integrations._schema_resolver import _resolve_schema
from finlit.pipeline import DocumentPipeline

# (extractor, vision_extractor_or_None, schema_key, review_threshold)
CacheKey = tuple[str, str | None, str, float]

_CACHE: dict[CacheKey, DocumentPipeline] = {}
_LOCK = threading.Lock()

_VISION_ALIASES = {
    "claude": "anthropic:claude-sonnet-4-6",
    "openai": "openai:gpt-4o",
    "ollama": "ollama:llama3.2-vision",
}


def get_pipeline(
    extractor: str,
    vision_extractor: str | None,
    schema_key: str,
    review_threshold: float,
) -> DocumentPipeline:
    """Return a cached DocumentPipeline for this configuration, building if needed.

    Raises:
        ValueError: if `schema_key` is not a known dotted registry key.
    """
    key: CacheKey = (extractor, vision_extractor, schema_key, review_threshold)
    with _LOCK:
        if key in _CACHE:
            return _CACHE[key]

        schema = _resolve_schema(schema_key)  # raises ValueError on unknown key

        ve = None
        if vision_extractor is not None:
            model_str = _VISION_ALIASES.get(vision_extractor, vision_extractor)
            ve = VisionExtractor(model=model_str)

        pipeline = DocumentPipeline(
            schema=schema,
            extractor=extractor,
            review_threshold=review_threshold,
            vision_extractor=ve,
        )
        _CACHE[key] = pipeline
        return pipeline


def clear_cache() -> None:
    """Test-only helper: drop all cached pipelines."""
    with _LOCK:
        _CACHE.clear()
