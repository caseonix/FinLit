"""
Abstract base class for vision-based extractors.

A concrete vision extractor takes a list of PNG-encoded page images plus
a Schema and returns an ExtractionOutput (fields, confidence, notes).

This ABC is deliberately separate from BaseExtractor (which takes text)
so that DocumentPipeline can type-check the two slots independently:
`extractor: BaseExtractor | str` for text, `vision_extractor:
BaseVisionExtractor | None` for vision. This prevents accidentally
passing a text extractor as the vision fallback.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from finlit.schema import Schema


class BaseVisionExtractor(ABC):
    """All vision extractor backends must implement this interface."""

    @abstractmethod
    def extract(
        self,
        images: list[bytes],
        schema: "Schema",
        text: str = "",
    ) -> Any:
        """Extract structured fields from page images.

        Parameters
        ----------
        images:
            List of PNG-encoded page images, one per page, in document
            order.
        schema:
            The FinLit Schema describing which fields to extract.
        text:
            Optional text hint. When the vision extractor is used as a
            fallback from the text path, this contains whatever text
            Docling managed to recover from the document. Most
            implementations will ignore it. Extractors that want to use
            the text path's partial output as additional context (e.g.,
            "the text extractor found 'Acme Corp' as employer; verify
            from the image") can read it here.
        """
