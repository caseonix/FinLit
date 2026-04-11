"""
DocumentPipeline orchestrates the full extraction flow for one document.
BatchPipeline runs DocumentPipeline over many files in parallel.
"""
from __future__ import annotations

import csv
import json
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from finlit.audit.audit_log import AuditLog
from finlit.audit.pii import CanadianPIIDetector
from finlit.extractors.base import BaseExtractor
from finlit.extractors.base_vision import BaseVisionExtractor
from finlit.extractors.pydantic_ai_extractor import PydanticAIExtractor
from finlit.parsers.docling_parser import DoclingParser
from finlit.parsers.image_renderer import render_pages
from finlit.result import ExtractionResult
from finlit.schema import Schema
from finlit.validators.field_validator import FieldValidator


_EXTRACTOR_ALIASES = {
    "claude": "anthropic:claude-sonnet-4-6",
    "openai": "openai:gpt-4o",
    "ollama": "ollama:llama3.2",
}

# Minimum number of stripped characters before we consider a parsed document
# "sparse" (likely a scanned image with no text layer).
SPARSE_TEXT_THRESHOLD = 100


class DocumentPipeline:
    """Full extraction pipeline for a single document."""

    def __init__(
        self,
        schema: Schema,
        extractor: str | BaseExtractor = "claude",
        model: str | None = None,
        audit: bool = True,
        pii_redact: bool = False,
        review_threshold: float = 0.85,
        ocr_fallback: bool = True,
        sparse_text_threshold: int = SPARSE_TEXT_THRESHOLD,
        vision_extractor: BaseVisionExtractor | None = None,
        vision_fallback_when: Callable[[Any], bool] | None = None,
    ):
        self.schema = schema
        self.audit_enabled = audit
        self.pii_redact = pii_redact
        self.review_threshold = review_threshold
        self.ocr_fallback = ocr_fallback
        self.sparse_text_threshold = sparse_text_threshold

        if isinstance(extractor, BaseExtractor):
            self._extractor: BaseExtractor = extractor
            self._model_name = "custom"
        else:
            model_str = model or _EXTRACTOR_ALIASES.get(extractor, extractor)
            self._extractor = PydanticAIExtractor(model=model_str)
            self._model_name = model_str

        self._parser = DoclingParser()
        self._ocr_parser: DoclingParser | None = None  # lazy-initialized
        self._pii_detector = CanadianPIIDetector()
        self._validator = FieldValidator()
        self.vision_extractor = vision_extractor
        self.vision_fallback_when = vision_fallback_when

    def _get_ocr_parser(self) -> DoclingParser:
        if self._ocr_parser is None:
            self._ocr_parser = DoclingParser(ocr=True)
        return self._ocr_parser

    def run(self, path: str | Path) -> ExtractionResult:
        run_id = str(uuid.uuid4())
        audit = AuditLog(run_id=run_id)
        warnings: list[dict] = []

        path = Path(path)

        # Step 1: parse
        audit.log("document_load_start", file=str(path))
        parsed = self._parser.parse(path)
        audit.log(
            "document_loaded",
            file=parsed.metadata["filename"],
            sha256=parsed.metadata["sha256"],
            num_pages=parsed.metadata.get("num_pages"),
        )

        # Step 1b: OCR fallback on sparse text
        stripped_len = len(parsed.full_text.strip())
        if stripped_len < self.sparse_text_threshold and self.ocr_fallback:
            audit.log(
                "ocr_fallback_triggered",
                reason="sparse_text",
                initial_chars=stripped_len,
            )
            parsed = self._get_ocr_parser().parse(path)
            audit.log(
                "document_loaded_ocr",
                file=parsed.metadata["filename"],
                sha256=parsed.metadata["sha256"],
                num_pages=parsed.metadata.get("num_pages"),
                chars=len(parsed.full_text.strip()),
            )

        # Step 1c: sparse warning if still unreadable
        final_stripped_len = len(parsed.full_text.strip())
        if final_stripped_len < self.sparse_text_threshold:
            warnings.append(
                {
                    "code": "sparse_document",
                    "message": (
                        f"Parsed text is only {final_stripped_len} chars; "
                        "the document may be a scanned image with no text "
                        "layer that OCR could not recover."
                    ),
                    "chars": final_stripped_len,
                }
            )
            audit.log(
                "sparse_document_warning",
                chars=final_stripped_len,
                threshold=self.sparse_text_threshold,
            )

        # Step 2: PII scan
        pii_entities: list[dict] = []
        if self.audit_enabled:
            pii_entities = self._pii_detector.analyze(parsed.full_text)
            if pii_entities:
                audit.log(
                    "pii_detected",
                    count=len(pii_entities),
                    entities=[e["entity_type"] for e in pii_entities],
                )

        # Step 3: LLM text extraction
        audit.log(
            "extraction_start", schema=self.schema.name, model=self._model_name
        )
        extraction = self._extractor.extract(parsed.full_text, self.schema)
        audit.log("extraction_complete", fields_returned=len(extraction.fields))

        # Step 4: validate
        validated_fields, validation_errors = self._validator.validate(
            extraction.fields, self.schema
        )
        if validation_errors:
            audit.log("validation_errors", errors=validation_errors)

        # Step 4b: required fields missing warning
        required_field_names = {
            f.name for f in self.schema.fields if f.required
        }
        missing_required = sorted(
            fname
            for fname in required_field_names
            if validated_fields.get(fname) is None
        )
        if missing_required:
            warnings.append(
                {
                    "code": "required_fields_missing",
                    "message": (
                        f"{len(missing_required)} required field(s) missing "
                        f"after extraction: {', '.join(missing_required)}"
                    ),
                    "missing_fields": missing_required,
                }
            )
            audit.log(
                "required_fields_missing_warning",
                count=len(missing_required),
                fields=missing_required,
            )

        # Step 5: review queue
        review_fields = [
            {
                "field": fname,
                "confidence": extraction.confidence.get(fname, 0.0),
                "raw": validated_fields.get(fname),
            }
            for fname in self.schema.field_names()
            if extraction.confidence.get(fname, 0.0) < self.review_threshold
            and validated_fields.get(fname) is not None
        ]
        if review_fields:
            audit.log(
                "review_flagged",
                count=len(review_fields),
                fields=[r["field"] for r in review_fields],
            )

        # Step 6: source refs (placeholder until Docling bbox wiring is added)
        source_ref = {
            fname: {
                "doc": parsed.metadata["filename"],
                "page": None,
                "bbox": None,
            }
            for fname in self.schema.field_names()
        }

        # Build the provisional text-path result
        text_result = ExtractionResult(
            fields=validated_fields,
            confidence=extraction.confidence,
            source_ref=source_ref,
            pii_entities=pii_entities,
            audit_log=audit.to_dict(),  # snapshot, rebuilt after finalize
            review_fields=review_fields,
            warnings=list(warnings),
            review_threshold=self.review_threshold,
            document_path=str(path),
            schema_name=self.schema.name,
            extractor_model=self._model_name,
            extraction_path="text",
        )

        # Step 7: vision fallback decision
        if self.vision_extractor is not None:
            vision_result = self._maybe_run_vision_fallback(
                path=path,
                parsed_text=parsed.full_text,
                text_result=text_result,
                audit=audit,
                warnings=warnings,
                source_ref=source_ref,
                pii_entities=pii_entities,
            )
            if vision_result is not None:
                return vision_result

        audit.log(
            "pipeline_complete",
            fields_total=len(self.schema.fields),
            fields_extracted=sum(1 for v in validated_fields.values() if v is not None),
            needs_review=len(review_fields) > 0 or len(warnings) > 0,
            extraction_path="text",
        )
        audit.finalize()

        # Refresh audit log on the text result (finalize may freeze it)
        text_result.audit_log = audit.to_dict()
        text_result.warnings = list(warnings)
        return text_result


    def _maybe_run_vision_fallback(
        self,
        *,
        path: Path,
        parsed_text: str,
        text_result: ExtractionResult,
        audit: AuditLog,
        warnings: list[dict],
        source_ref: dict,
        pii_entities: list[dict],
    ) -> ExtractionResult | None:
        """Evaluate the fallback callback and, if True, run vision extraction.

        Returns a new ExtractionResult on successful vision run, or None
        to signal "no fallback happened, return the text result".

        On any failure (callback exception, render failure, extraction
        failure), this method appends a vision_fallback_failed warning
        to `warnings` (mutating the text_result's warnings list in place
        via the shared reference), logs the appropriate audit event, and
        returns None so the caller returns the text result.
        """
        assert self.vision_extractor is not None

        callback = self.vision_fallback_when or (lambda r: r.needs_review)

        # Evaluate callback
        try:
            should_fire = callback(text_result)
        except Exception as e:
            audit.log(
                "vision_fallback_callback_error",
                error_type=type(e).__name__,
                error_message=str(e),
            )
            warnings.append(
                {
                    "code": "vision_fallback_failed",
                    "reason": "callback",
                    "message": f"vision_fallback_when callback raised: {e}",
                }
            )
            return None

        if not should_fire:
            return None

        audit.log(
            "vision_fallback_triggered",
            provisional_needs_review=text_result.needs_review,
            provisional_warning_codes=[w["code"] for w in text_result.warnings],
        )

        # Render pages
        try:
            dpi = getattr(self.vision_extractor, "dpi", 200)
            audit.log("vision_render_start", dpi=dpi, path=str(path))
            images = render_pages(path, dpi=dpi)
            audit.log(
                "vision_render_complete",
                page_count=len(images),
                total_bytes=sum(len(i) for i in images),
            )
        except Exception as e:
            audit.log(
                "vision_render_failed",
                error_type=type(e).__name__,
                error_message=str(e),
            )
            warnings.append(
                {
                    "code": "vision_fallback_failed",
                    "reason": "render",
                    "message": f"render_pages failed: {e}",
                }
            )
            return None

        # Run vision extractor
        try:
            audit.log(
                "vision_extraction_start",
                model=getattr(self.vision_extractor, "model", "custom"),
                page_count=len(images),
            )
            vision_output = self.vision_extractor.extract(
                images, self.schema, text=parsed_text
            )
            audit.log(
                "vision_extraction_complete",
                fields_returned=len(vision_output.fields),
            )
        except Exception as e:
            audit.log(
                "vision_extraction_failed",
                error_type=type(e).__name__,
                error_message=str(e),
            )
            warnings.append(
                {
                    "code": "vision_fallback_failed",
                    "reason": "extraction",
                    "message": f"vision extractor raised: {e}",
                }
            )
            return None

        # Re-validate vision output through the same validator
        v_validated, v_errors = self._validator.validate(
            vision_output.fields, self.schema
        )
        if v_errors:
            audit.log("vision_validation_errors", errors=v_errors)

        # Build vision-path warnings (re-check required fields for the new result)
        vision_warnings: list[dict] = []
        required_field_names = {
            f.name for f in self.schema.fields if f.required
        }
        v_missing = sorted(
            fname
            for fname in required_field_names
            if v_validated.get(fname) is None
        )
        if v_missing:
            vision_warnings.append(
                {
                    "code": "required_fields_missing",
                    "message": (
                        f"{len(v_missing)} required field(s) missing "
                        f"after vision extraction: {', '.join(v_missing)}"
                    ),
                    "missing_fields": v_missing,
                }
            )
            audit.log(
                "required_fields_missing_warning",
                count=len(v_missing),
                fields=v_missing,
                path="vision",
            )

        v_review_fields = [
            {
                "field": fname,
                "confidence": vision_output.confidence.get(fname, 0.0),
                "raw": v_validated.get(fname),
            }
            for fname in self.schema.field_names()
            if vision_output.confidence.get(fname, 0.0) < self.review_threshold
            and v_validated.get(fname) is not None
        ]
        if v_review_fields:
            audit.log(
                "review_flagged",
                count=len(v_review_fields),
                fields=[r["field"] for r in v_review_fields],
                path="vision",
            )

        audit.log(
            "pipeline_complete",
            fields_total=len(self.schema.fields),
            fields_extracted=sum(1 for v in v_validated.values() if v is not None),
            needs_review=len(v_review_fields) > 0 or len(vision_warnings) > 0,
            extraction_path="vision",
        )
        audit.finalize()

        return ExtractionResult(
            fields=v_validated,
            confidence=vision_output.confidence,
            source_ref=source_ref,
            pii_entities=pii_entities,
            audit_log=audit.to_dict(),
            review_fields=v_review_fields,
            warnings=vision_warnings,
            review_threshold=self.review_threshold,
            document_path=str(path),
            schema_name=self.schema.name,
            extractor_model=getattr(
                self.vision_extractor, "model", "custom-vision"
            ),
            extraction_path="vision",
        )


class BatchPipeline:
    """Runs DocumentPipeline over many files with a ThreadPoolExecutor."""

    def __init__(
        self,
        schema: Schema,
        extractor: str | BaseExtractor = "claude",
        workers: int = 4,
        **pipeline_kwargs: Any,
    ):
        self.schema = schema
        self.extractor = extractor
        self.workers = workers
        self.pipeline_kwargs = pipeline_kwargs
        self._paths: list[Path] = []

    def add(self, path: str | Path) -> None:
        self._paths.append(Path(path))

    def run(self) -> "BatchResult":
        results: list[ExtractionResult] = []
        errors: list[dict] = []

        pipeline = DocumentPipeline(
            schema=self.schema,
            extractor=self.extractor,
            **self.pipeline_kwargs,
        )

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {pool.submit(pipeline.run, p): p for p in self._paths}
            for future in as_completed(futures):
                p = futures[future]
                try:
                    results.append(future.result())
                except Exception as e:  # noqa: BLE001
                    errors.append({"path": str(p), "error": str(e)})

        return BatchResult(results=results, errors=errors)


@dataclass
class BatchResult:
    results: list[ExtractionResult] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def review_count(self) -> int:
        return sum(1 for r in self.results if r.needs_review)

    def export_csv(self, path: str) -> None:
        if not self.results:
            return
        fieldnames = ["document"] + list(self.results[0].fields.keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in self.results:
                row: dict[str, Any] = {"document": result.document_path}
                row.update(result.fields)
                writer.writerow(row)

    def export_jsonl(self, path: str) -> None:
        with open(path, "w") as f:
            for result in self.results:
                f.write(
                    json.dumps(
                        {
                            "document": result.document_path,
                            "fields": result.fields,
                            "confidence": result.confidence,
                            "needs_review": result.needs_review,
                        }
                    )
                    + "\n"
                )
