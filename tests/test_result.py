"""Tests for finlit.result.ExtractionResult."""
from finlit.result import ExtractionResult


def test_needs_review_false_when_no_review_fields():
    r = ExtractionResult(fields={"a": 1}, confidence={"a": 0.95}, source_ref={})
    assert r.needs_review is False
    assert r.extracted_field_count == 1


def test_needs_review_true_when_review_fields_present():
    r = ExtractionResult(
        fields={"a": 1},
        confidence={"a": 0.5},
        source_ref={},
        review_fields=[{"field": "a", "confidence": 0.5, "raw": 1}],
    )
    assert r.needs_review is True


def test_extracted_field_count_ignores_none_values():
    r = ExtractionResult(
        fields={"a": 1, "b": None, "c": "hello"},
        confidence={},
        source_ref={},
    )
    assert r.extracted_field_count == 2


def test_get_returns_field_value_or_default():
    r = ExtractionResult(fields={"a": 42}, confidence={}, source_ref={})
    assert r.get("a") == 42
    assert r.get("missing") is None
    assert r.get("missing", "fallback") == "fallback"


def test_warnings_default_empty():
    r = ExtractionResult(fields={"a": 1}, confidence={}, source_ref={})
    assert r.warnings == []


def test_needs_review_true_when_warning_present():
    r = ExtractionResult(
        fields={"a": 1},
        confidence={"a": 0.99},
        source_ref={},
        warnings=[{"code": "sparse_document", "message": "Parsed text < 100 chars"}],
    )
    assert r.needs_review is True
