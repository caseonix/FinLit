"""Reusable stub extractor: returns a preconfigured ExtractionOutput.

Used in tests to avoid making real LLM calls. Per CLAUDE.md, tests must
pass with no API keys set.
"""
from __future__ import annotations

from finlit.extractors.base import BaseExtractor
from finlit.extractors.pydantic_ai_extractor import ExtractionOutput
from finlit.schema import Schema


class StubExtractor(BaseExtractor):
    """Returns a preconfigured ExtractionOutput regardless of input text."""

    def __init__(self, canned_output: ExtractionOutput) -> None:
        self.canned_output = canned_output
        self.call_count = 0

    def extract(self, text: str, schema: Schema) -> ExtractionOutput:
        self.call_count += 1
        return self.canned_output
