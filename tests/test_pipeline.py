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
