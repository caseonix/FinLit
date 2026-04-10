"""
Abstract base class for extractors. A concrete extractor takes parsed
document text plus a Schema and returns an ExtractionOutput with
fields, confidence, and notes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from finlit.schema import Schema


class BaseExtractor(ABC):
    """All extractor backends must implement this interface."""

    @abstractmethod
    def extract(self, text: str, schema: "Schema") -> Any:
        """Synchronous extraction. Returns an object with .fields, .confidence, .notes."""

    async def extract_async(self, text: str, schema: "Schema") -> Any:
        """Async extraction — default implementation delegates to sync."""
        return self.extract(text, schema)
