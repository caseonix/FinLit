"""FinLitLoader — LangChain BaseLoader wrapper around DocumentPipeline."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Literal, Union

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from finlit.extractors.base import BaseExtractor
from finlit.integrations._schema_resolver import _resolve_schema
from finlit.pipeline import DocumentPipeline
from finlit.result import ExtractionResult
from finlit.schema import Schema


_log = logging.getLogger(__name__)

PathLike = Union[str, Path]
OnError = Literal["raise", "skip", "include"]


class FinLitLoader(BaseLoader):
    """Load files through a FinLit DocumentPipeline and emit LangChain Documents.

    See design doc: docs/superpowers/specs/2026-04-22-langchain-llamaindex-readers-design.md
    """

    def __init__(
        self,
        file_path: PathLike | list[PathLike],
        *,
        schema: Schema | str | None = None,
        extractor: str | BaseExtractor = "claude",
        pipeline: DocumentPipeline | None = None,
        on_error: OnError = "raise",
        include_audit_log: bool = False,
    ) -> None:
        if isinstance(file_path, (str, Path)):
            self._paths: list[Path] = [Path(file_path)]
        else:
            self._paths = [Path(p) for p in file_path]

        if pipeline is not None:
            self._pipeline = pipeline
        elif schema is not None:
            self._pipeline = DocumentPipeline(
                schema=_resolve_schema(schema),
                extractor=extractor,
            )
        else:
            raise ValueError(
                "FinLitLoader requires either schema=... or pipeline=..."
            )

        if on_error not in ("raise", "skip", "include"):
            raise ValueError(
                f"on_error must be 'raise', 'skip', or 'include', got {on_error!r}"
            )
        self._on_error = on_error
        self._include_audit_log = include_audit_log

        self.last_results: list[ExtractionResult | None] = []

    def lazy_load(self) -> Iterator[Document]:
        self.last_results = []
        for path in self._paths:
            try:
                parsed = self._pipeline._parser.parse(path)
                result = self._pipeline.run(path)
            except Exception as exc:
                if self._on_error == "raise":
                    raise
                if self._on_error == "skip":
                    _log.warning(
                        "FinLit extraction failed for %s: %s", path, exc
                    )
                    continue
                # on_error == "include"
                _log.warning(
                    "FinLit extraction failed for %s (emitted as error Document): %s",
                    path,
                    exc,
                )
                self.last_results.append(None)
                yield Document(
                    page_content="",
                    metadata={
                        "source": str(path),
                        "finlit_error": repr(exc),
                        "finlit_error_type": type(exc).__name__,
                    },
                )
                continue
            self.last_results.append(result)
            yield _build_document(
                path, parsed.full_text, result, self._include_audit_log
            )


def _build_document(
    path: Path,
    full_text: str,
    result: ExtractionResult,
    include_audit_log: bool,
) -> Document:
    metadata: dict = {
        "source": str(path),
        "finlit_schema": result.schema_name,
        "finlit_model": result.extractor_model,
        "finlit_extraction_path": result.extraction_path,
        "finlit_needs_review": result.needs_review,
        "finlit_extracted_field_count": result.extracted_field_count,
        "finlit_fields": dict(result.fields),
        "finlit_confidence": dict(result.confidence),
        "finlit_source_ref": dict(result.source_ref),
        "finlit_warnings": list(result.warnings),
        "finlit_review_fields": list(result.review_fields),
        "finlit_pii_entities": list(result.pii_entities),
    }
    if include_audit_log:
        metadata["finlit_audit_log"] = list(result.audit_log)
    return Document(page_content=full_text, metadata=metadata)
