"""Tests for finlit.pipeline.DocumentPipeline and BatchPipeline."""
from __future__ import annotations


import pytest

from finlit import schemas
from finlit.parsers.docling_parser import ParsedDocument
from finlit.pipeline import BatchPipeline, BatchResult, DocumentPipeline
from finlit.result import ExtractionResult
from tests.conftest import StubExtractor


def _patch_parser(monkeypatch, parsed: ParsedDocument) -> None:
    def _fake_parse(self, path):  # noqa: ARG001
        return parsed

    monkeypatch.setattr(
        "finlit.parsers.docling_parser.DoclingParser.parse", _fake_parse
    )


def test_file_not_found_raises(monkeypatch, high_confidence_t4_extractor):
    # Parser is NOT patched here — we want real FileNotFoundError from DoclingParser
    pipeline = DocumentPipeline(
        schema=schemas.CRA_T4,
        extractor=high_confidence_t4_extractor,
    )
    with pytest.raises(FileNotFoundError):
        pipeline.run("/tmp/__nonexistent_finlit_test__.pdf")


def test_happy_path_all_confidence_above_threshold(
    monkeypatch, synthetic_parsed_document, high_confidence_t4_extractor, tmp_path
):
    _patch_parser(monkeypatch, synthetic_parsed_document)
    fake = tmp_path / "t4.pdf"
    fake.write_bytes(b"not a real pdf")

    pipeline = DocumentPipeline(
        schema=schemas.CRA_T4,
        extractor=high_confidence_t4_extractor,
        review_threshold=0.85,
    )
    result = pipeline.run(fake)

    assert isinstance(result, ExtractionResult)
    assert result.fields["employer_name"] == "Acme Corp"
    assert result.fields["box_14_employment_income"] == 87500.0
    assert isinstance(result.fields["box_14_employment_income"], float)
    assert result.needs_review is False
    assert result.review_fields == []
    events = {e["event"] for e in result.audit_log}
    assert "document_loaded" in events
    assert "extraction_complete" in events
    assert "pipeline_complete" in events


def test_low_confidence_field_goes_to_review(
    monkeypatch, synthetic_parsed_document, tmp_path
):
    _patch_parser(monkeypatch, synthetic_parsed_document)
    fake = tmp_path / "t4.pdf"
    fake.write_bytes(b"not a real pdf")

    low = StubExtractor(
        fields={
            "employer_name": "Acme Corp",
            "employee_sin": "123-456-789",
            "tax_year": 2024,
            "box_14_employment_income": 87500.00,
            "box_22_income_tax_deducted": 15200.00,
        },
        confidence={
            "employer_name": 0.99,
            "employee_sin": 0.99,
            "tax_year": 0.99,
            "box_14_employment_income": 0.50,  # below threshold
            "box_22_income_tax_deducted": 0.99,
        },
    )
    pipeline = DocumentPipeline(
        schema=schemas.CRA_T4, extractor=low, review_threshold=0.85
    )
    result = pipeline.run(fake)

    assert result.needs_review is True
    flagged = {r["field"] for r in result.review_fields}
    assert "box_14_employment_income" in flagged


def test_ocr_fallback_triggers_on_sparse_text(
    monkeypatch, synthetic_parsed_document, high_confidence_t4_extractor, tmp_path
):
    """If the initial parse returns sparse text, the pipeline should
    re-parse with OCR enabled and log ocr_fallback_triggered."""
    fake = tmp_path / "scanned_t4.pdf"
    fake.write_bytes(b"x")

    sparse_parsed = ParsedDocument(
        full_text="<!-- image -->",
        tables=[],
        metadata={
            "source": str(fake),
            "sha256": "cafe" * 16,
            "filename": "scanned_t4.pdf",
            "num_pages": 1,
        },
        source_path=str(fake),
    )

    call_log = {"default": 0, "ocr": 0}

    def _fake_parse(self, path):  # noqa: ARG001
        if self.ocr:
            call_log["ocr"] += 1
            return synthetic_parsed_document
        call_log["default"] += 1
        return sparse_parsed

    monkeypatch.setattr(
        "finlit.parsers.docling_parser.DoclingParser.parse", _fake_parse
    )

    pipeline = DocumentPipeline(
        schema=schemas.CRA_T4,
        extractor=high_confidence_t4_extractor,
        ocr_fallback=True,
    )
    result = pipeline.run(fake)

    assert call_log["default"] == 1
    assert call_log["ocr"] == 1
    events = [e["event"] for e in result.audit_log]
    assert "ocr_fallback_triggered" in events
    # Post-fallback we got real text, so no sparse warning should remain
    warning_codes = {w["code"] for w in result.warnings}
    assert "sparse_document" not in warning_codes
    # Fields should be populated from the OCR-parsed synthetic doc
    assert result.fields["employer_name"] == "Acme Corp"


def test_sparse_document_warning_when_ocr_also_fails(
    monkeypatch, high_confidence_t4_extractor, tmp_path
):
    """If text is still sparse after OCR fallback, a sparse_document warning
    is added and needs_review becomes True."""
    fake = tmp_path / "unreadable.pdf"
    fake.write_bytes(b"x")

    sparse_parsed = ParsedDocument(
        full_text="<!-- image -->",
        tables=[],
        metadata={
            "source": str(fake),
            "sha256": "dead" * 16,
            "filename": "unreadable.pdf",
            "num_pages": 1,
        },
        source_path=str(fake),
    )

    def _fake_parse(self, path):  # noqa: ARG001
        return sparse_parsed  # always sparse, even with ocr=True

    monkeypatch.setattr(
        "finlit.parsers.docling_parser.DoclingParser.parse", _fake_parse
    )

    pipeline = DocumentPipeline(
        schema=schemas.CRA_T4,
        extractor=high_confidence_t4_extractor,
        ocr_fallback=True,
    )
    result = pipeline.run(fake)

    warning_codes = {w["code"] for w in result.warnings}
    assert "sparse_document" in warning_codes
    assert result.needs_review is True
    events = [e["event"] for e in result.audit_log]
    assert "sparse_document_warning" in events


def test_required_fields_missing_forces_review(
    monkeypatch, synthetic_parsed_document, tmp_path
):
    """If the extractor returns nothing for required fields (as happens on a
    blank CRA form), the validator flags required_fields_missing and the
    pipeline must set needs_review=True even though the text wasn't sparse."""
    _patch_parser(monkeypatch, synthetic_parsed_document)
    fake = tmp_path / "blank.pdf"
    fake.write_bytes(b"x")

    # Stub returns confident-but-None for every required field
    empty = StubExtractor(
        fields={
            "employer_name": None,
            "employee_sin": None,
            "tax_year": None,
            "box_14_employment_income": None,
            "box_22_income_tax_deducted": None,
        },
        confidence={
            "employer_name": 0.99,
            "employee_sin": 0.99,
            "tax_year": 0.99,
            "box_14_employment_income": 0.99,
            "box_22_income_tax_deducted": 0.99,
        },
    )
    pipeline = DocumentPipeline(schema=schemas.CRA_T4, extractor=empty)
    result = pipeline.run(fake)

    warning_codes = {w["code"] for w in result.warnings}
    assert "required_fields_missing" in warning_codes
    assert result.needs_review is True
    events = [e["event"] for e in result.audit_log]
    assert "required_fields_missing_warning" in events


def test_ocr_fallback_disabled_skips_retry(
    monkeypatch, high_confidence_t4_extractor, tmp_path
):
    """ocr_fallback=False should NOT retry with OCR; sparse text should
    still produce a warning though."""
    fake = tmp_path / "scan.pdf"
    fake.write_bytes(b"x")

    sparse_parsed = ParsedDocument(
        full_text="<!-- image -->",
        tables=[],
        metadata={"source": str(fake), "sha256": "0" * 64,
                  "filename": "scan.pdf", "num_pages": 1},
        source_path=str(fake),
    )

    call_log = {"ocr": 0}

    def _fake_parse(self, path):  # noqa: ARG001
        if self.ocr:
            call_log["ocr"] += 1
        return sparse_parsed

    monkeypatch.setattr(
        "finlit.parsers.docling_parser.DoclingParser.parse", _fake_parse
    )

    pipeline = DocumentPipeline(
        schema=schemas.CRA_T4,
        extractor=high_confidence_t4_extractor,
        ocr_fallback=False,
    )
    result = pipeline.run(fake)

    assert call_log["ocr"] == 0
    warning_codes = {w["code"] for w in result.warnings}
    assert "sparse_document" in warning_codes


def test_batch_result_export_csv(
    monkeypatch, synthetic_parsed_document, high_confidence_t4_extractor, tmp_path
):
    _patch_parser(monkeypatch, synthetic_parsed_document)
    fake1 = tmp_path / "t4_a.pdf"
    fake2 = tmp_path / "t4_b.pdf"
    fake1.write_bytes(b"x")
    fake2.write_bytes(b"x")

    batch = BatchPipeline(
        schema=schemas.CRA_T4,
        extractor=high_confidence_t4_extractor,
        workers=2,
    )
    batch.add(fake1)
    batch.add(fake2)
    results: BatchResult = batch.run()

    assert results.total == 2
    out = tmp_path / "out.csv"
    results.export_csv(str(out))
    header_line = out.read_text().splitlines()[0]
    assert "document" in header_line
    assert "employer_name" in header_line
    assert "box_14_employment_income" in header_line


# ---------------- Vision fallback integration tests (v0.3) ----------------

from tests.conftest import StubVisionExtractor


def test_vision_fallback_fires_when_callback_returns_true(
    monkeypatch, synthetic_parsed_document, tmp_path
):
    """Text extractor returns all-None (needs_review=True). Default callback
    fires vision. Final result is the vision result."""
    _patch_parser(monkeypatch, synthetic_parsed_document)

    # Stub render_pages so we don't rasterize anything
    monkeypatch.setattr(
        "finlit.pipeline.render_pages",
        lambda path, dpi=200: [b"fakepng1"],
    )

    fake = tmp_path / "blank_t4.pdf"
    fake.write_bytes(b"x")

    empty_text = StubExtractor(
        fields={name: None for name in schemas.CRA_T4.field_names()},
        confidence={name: 0.99 for name in schemas.CRA_T4.field_names()},
    )
    vision = StubVisionExtractor(
        fields={
            "employer_name": "Acme Corp",
            "employee_sin": "123-456-789",
            "tax_year": 2024,
            "box_14_employment_income": 87500.0,
            "box_22_income_tax_deducted": 15200.0,
        },
        confidence={
            "employer_name": 0.98,
            "employee_sin": 0.99,
            "tax_year": 0.99,
            "box_14_employment_income": 0.97,
            "box_22_income_tax_deducted": 0.96,
        },
    )

    pipeline = DocumentPipeline(
        schema=schemas.CRA_T4,
        extractor=empty_text,
        vision_extractor=vision,
    )
    result = pipeline.run(fake)

    assert vision.call_count == 1
    assert result.extraction_path == "vision"
    assert result.fields["employer_name"] == "Acme Corp"
    assert result.fields["box_14_employment_income"] == 87500.0
    events = [e["event"] for e in result.audit_log]
    assert "vision_fallback_triggered" in events
    assert "vision_extraction_complete" in events


def test_vision_fallback_skipped_when_callback_returns_false(
    monkeypatch, synthetic_parsed_document, high_confidence_t4_extractor, tmp_path
):
    """Text path produces a clean result (needs_review=False). Default
    callback returns False. Vision must NOT be called."""
    _patch_parser(monkeypatch, synthetic_parsed_document)
    fake = tmp_path / "clean.pdf"
    fake.write_bytes(b"x")

    vision = StubVisionExtractor(fields={}, confidence={})
    pipeline = DocumentPipeline(
        schema=schemas.CRA_T4,
        extractor=high_confidence_t4_extractor,
        vision_extractor=vision,
    )
    result = pipeline.run(fake)

    assert vision.call_count == 0
    assert result.extraction_path == "text"
    assert result.needs_review is False


def test_vision_fallback_skipped_when_vision_extractor_not_provided(
    monkeypatch, synthetic_parsed_document, tmp_path
):
    """When vision_extractor is None, no vision audit events appear even
    if the text result has needs_review=True."""
    _patch_parser(monkeypatch, synthetic_parsed_document)
    fake = tmp_path / "blank.pdf"
    fake.write_bytes(b"x")

    empty = StubExtractor(
        fields={name: None for name in schemas.CRA_T4.field_names()},
        confidence={name: 0.99 for name in schemas.CRA_T4.field_names()},
    )
    pipeline = DocumentPipeline(schema=schemas.CRA_T4, extractor=empty)
    result = pipeline.run(fake)

    assert result.extraction_path == "text"
    events = [e["event"] for e in result.audit_log]
    assert "vision_fallback_triggered" not in events


def test_vision_fallback_custom_callback_overrides_default(
    monkeypatch, synthetic_parsed_document, high_confidence_t4_extractor, tmp_path
):
    """A custom callback can force vision to run even on a clean text result."""
    _patch_parser(monkeypatch, synthetic_parsed_document)
    monkeypatch.setattr(
        "finlit.pipeline.render_pages",
        lambda path, dpi=200: [b"fakepng"],
    )
    fake = tmp_path / "doc.pdf"
    fake.write_bytes(b"x")

    vision = StubVisionExtractor(
        fields={"employer_name": "Vision Wins Co", "employee_sin": None,
                "tax_year": 2024, "box_14_employment_income": 99999.0,
                "box_22_income_tax_deducted": 5000.0},
        confidence={"employer_name": 0.9, "employee_sin": 0.0,
                    "tax_year": 0.9, "box_14_employment_income": 0.9,
                    "box_22_income_tax_deducted": 0.9},
    )

    pipeline = DocumentPipeline(
        schema=schemas.CRA_T4,
        extractor=high_confidence_t4_extractor,
        vision_extractor=vision,
        vision_fallback_when=lambda r: True,  # always
    )
    result = pipeline.run(fake)

    assert vision.call_count == 1
    assert result.extraction_path == "vision"
    assert result.fields["employer_name"] == "Vision Wins Co"


def test_vision_render_failure_falls_back_to_text_result(
    monkeypatch, synthetic_parsed_document, high_confidence_t4_extractor, tmp_path
):
    """If render_pages raises, we return the text result with a vision_fallback_failed
    warning and an audit event."""
    _patch_parser(monkeypatch, synthetic_parsed_document)

    def _boom(path, dpi=200):
        raise RuntimeError("corrupted pdf")

    monkeypatch.setattr("finlit.pipeline.render_pages", _boom)
    fake = tmp_path / "bad.pdf"
    fake.write_bytes(b"x")

    vision = StubVisionExtractor(fields={}, confidence={})
    pipeline = DocumentPipeline(
        schema=schemas.CRA_T4,
        extractor=high_confidence_t4_extractor,
        vision_extractor=vision,
        vision_fallback_when=lambda r: True,  # force vision
    )
    result = pipeline.run(fake)

    assert vision.call_count == 0  # never got past render
    assert result.extraction_path == "text"
    warning_codes = {w["code"] for w in result.warnings}
    assert "vision_fallback_failed" in warning_codes
    matching = [w for w in result.warnings if w["code"] == "vision_fallback_failed"]
    assert matching[0]["reason"] == "render"
    events = [e["event"] for e in result.audit_log]
    assert "vision_render_failed" in events


def test_vision_extraction_failure_falls_back_to_text_result(
    monkeypatch, synthetic_parsed_document, high_confidence_t4_extractor, tmp_path
):
    """If the vision extractor raises, we return the text result with a
    vision_fallback_failed warning and an audit event."""
    _patch_parser(monkeypatch, synthetic_parsed_document)
    monkeypatch.setattr(
        "finlit.pipeline.render_pages",
        lambda path, dpi=200: [b"fakepng"],
    )
    fake = tmp_path / "doc.pdf"
    fake.write_bytes(b"x")

    vision = StubVisionExtractor(
        fields={}, confidence={}, raises=RuntimeError("api down"),
    )
    pipeline = DocumentPipeline(
        schema=schemas.CRA_T4,
        extractor=high_confidence_t4_extractor,
        vision_extractor=vision,
        vision_fallback_when=lambda r: True,
    )
    result = pipeline.run(fake)

    assert vision.call_count == 1
    assert result.extraction_path == "text"
    warning_codes = {w["code"] for w in result.warnings}
    assert "vision_fallback_failed" in warning_codes
    matching = [w for w in result.warnings if w["code"] == "vision_fallback_failed"]
    assert matching[0]["reason"] == "extraction"
    events = [e["event"] for e in result.audit_log]
    assert "vision_extraction_failed" in events


def test_vision_callback_exception_falls_back_to_text_result(
    monkeypatch, synthetic_parsed_document, high_confidence_t4_extractor, tmp_path
):
    """A callback that raises must not crash the pipeline."""
    _patch_parser(monkeypatch, synthetic_parsed_document)
    fake = tmp_path / "doc.pdf"
    fake.write_bytes(b"x")

    def _bad_callback(result):
        raise ValueError("buggy callback")

    vision = StubVisionExtractor(fields={}, confidence={})
    pipeline = DocumentPipeline(
        schema=schemas.CRA_T4,
        extractor=high_confidence_t4_extractor,
        vision_extractor=vision,
        vision_fallback_when=_bad_callback,
    )
    result = pipeline.run(fake)

    assert vision.call_count == 0
    assert result.extraction_path == "text"
    warning_codes = {w["code"] for w in result.warnings}
    assert "vision_fallback_failed" in warning_codes
    matching = [w for w in result.warnings if w["code"] == "vision_fallback_failed"]
    assert matching[0]["reason"] == "callback"


def test_vision_result_replaces_text_result_fully(
    monkeypatch, synthetic_parsed_document, tmp_path
):
    """When vision runs successfully, its fields replace the text result
    completely — no merging."""
    _patch_parser(monkeypatch, synthetic_parsed_document)
    monkeypatch.setattr(
        "finlit.pipeline.render_pages",
        lambda path, dpi=200: [b"fakepng"],
    )
    fake = tmp_path / "doc.pdf"
    fake.write_bytes(b"x")

    text = StubExtractor(
        fields={"employer_name": "WRONG", "employee_sin": "111-111-111",
                "tax_year": 2023, "box_14_employment_income": 1.0,
                "box_22_income_tax_deducted": 2.0},
        confidence={"employer_name": 0.5, "employee_sin": 0.5,
                    "tax_year": 0.5, "box_14_employment_income": 0.5,
                    "box_22_income_tax_deducted": 0.5},
    )
    vision = StubVisionExtractor(
        fields={"employer_name": "CORRECT", "employee_sin": "999-999-999",
                "tax_year": 2024, "box_14_employment_income": 87500.0,
                "box_22_income_tax_deducted": 15200.0},
        confidence={"employer_name": 0.99, "employee_sin": 0.99,
                    "tax_year": 0.99, "box_14_employment_income": 0.99,
                    "box_22_income_tax_deducted": 0.99},
    )

    pipeline = DocumentPipeline(
        schema=schemas.CRA_T4,
        extractor=text,
        vision_extractor=vision,
        # Text result has review_fields (confidence < 0.85) so default fires
    )
    result = pipeline.run(fake)

    assert result.extraction_path == "vision"
    assert result.fields["employer_name"] == "CORRECT"
    assert result.fields["employee_sin"] == "999-999-999"
    assert result.fields["box_14_employment_income"] == 87500.0


def test_vision_audit_trail_complete(
    monkeypatch, synthetic_parsed_document, tmp_path
):
    """A successful vision run must log the full audit event sequence in order."""
    _patch_parser(monkeypatch, synthetic_parsed_document)
    monkeypatch.setattr(
        "finlit.pipeline.render_pages",
        lambda path, dpi=200: [b"fakepng"],
    )
    fake = tmp_path / "doc.pdf"
    fake.write_bytes(b"x")

    empty = StubExtractor(
        fields={name: None for name in schemas.CRA_T4.field_names()},
        confidence={name: 0.99 for name in schemas.CRA_T4.field_names()},
    )
    vision = StubVisionExtractor(
        fields={"employer_name": "Acme", "employee_sin": "123-456-789",
                "tax_year": 2024, "box_14_employment_income": 1.0,
                "box_22_income_tax_deducted": 1.0},
        confidence={"employer_name": 0.99, "employee_sin": 0.99,
                    "tax_year": 0.99, "box_14_employment_income": 0.99,
                    "box_22_income_tax_deducted": 0.99},
    )

    pipeline = DocumentPipeline(
        schema=schemas.CRA_T4, extractor=empty, vision_extractor=vision
    )
    result = pipeline.run(fake)

    events = [e["event"] for e in result.audit_log]
    required_in_order = [
        "vision_fallback_triggered",
        "vision_render_start",
        "vision_render_complete",
        "vision_extraction_start",
        "vision_extraction_complete",
    ]
    # All five must appear in this exact relative order. Walk the event
    # stream once with a single cursor so duplicates or out-of-order
    # occurrences cannot be masked by events.index() returning the first hit.
    it = iter(events)
    for expected in required_in_order:
        assert any(e == expected for e in it), (
            f"missing or out-of-order audit event {expected!r} in {events}"
        )
