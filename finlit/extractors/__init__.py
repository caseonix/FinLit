"""Extractor interfaces and implementations.

Re-exports the abstract base classes so users can subclass them via the
shorter `from finlit.extractors import BaseExtractor` / `BaseVisionExtractor`
path used in the README "bring your own extractor" example.
"""
from finlit.extractors.base import BaseExtractor
from finlit.extractors.base_vision import BaseVisionExtractor

__all__ = ["BaseExtractor", "BaseVisionExtractor"]
