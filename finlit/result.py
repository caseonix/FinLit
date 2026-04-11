"""
The ExtractionResult is the single object returned by DocumentPipeline.run().
It contains typed fields, confidence scores, source references, PII flags,
and the full audit log.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExtractionResult:
    # Core output
    fields: dict[str, Any]
    confidence: dict[str, float]
    source_ref: dict[str, dict]

    # Compliance
    pii_entities: list[dict] = field(default_factory=list)
    audit_log: list[dict] = field(default_factory=list)

    # Review queue
    review_threshold: float = 0.85
    review_fields: list[dict] = field(default_factory=list)

    # Pipeline-level warnings (e.g. sparse text, ocr fallback used)
    warnings: list[dict] = field(default_factory=list)

    # Metadata
    document_path: str = ""
    schema_name: str = ""
    extractor_model: str = ""
    extraction_path: str = "text"  # "text" or "vision"

    @property
    def needs_review(self) -> bool:
        return len(self.review_fields) > 0 or len(self.warnings) > 0

    @property
    def extracted_field_count(self) -> int:
        return sum(1 for v in self.fields.values() if v is not None)

    def get(self, field_name: str, default: Any = None) -> Any:
        return self.fields.get(field_name, default)
