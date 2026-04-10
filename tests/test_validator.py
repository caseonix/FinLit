"""Tests for finlit.validators.field_validator.FieldValidator."""
from finlit.schema import Schema, Field
from finlit.validators.field_validator import FieldValidator


def _schema_with(*fields: Field) -> Schema:
    return Schema(name="test", fields=list(fields))


def test_coerces_string_to_float():
    schema = _schema_with(Field(name="amount", dtype=float, required=True))
    v = FieldValidator()
    out, errors = v.validate({"amount": "87500.00"}, schema)
    assert out["amount"] == 87500.0
    assert errors == []


def test_required_missing_field_reports_error():
    schema = _schema_with(Field(name="amount", dtype=float, required=True))
    v = FieldValidator()
    out, errors = v.validate({}, schema)
    assert out["amount"] is None
    assert any("Required field missing" in e for e in errors)


def test_optional_missing_field_no_error():
    schema = _schema_with(Field(name="amount", dtype=float, required=False))
    out, errors = FieldValidator().validate({}, schema)
    assert out["amount"] is None
    assert errors == []


def test_regex_failure_reports_error():
    schema = _schema_with(
        Field(name="sin", dtype=str, regex=r"^\d{3}-\d{3}-\d{3}$")
    )
    out, errors = FieldValidator().validate({"sin": "not-a-sin"}, schema)
    assert out["sin"] == "not-a-sin"
    assert any("Regex validation failed on sin" in e for e in errors)


def test_regex_success_no_error():
    schema = _schema_with(
        Field(name="sin", dtype=str, regex=r"^\d{3}-\d{3}-\d{3}$")
    )
    out, errors = FieldValidator().validate({"sin": "123-456-789"}, schema)
    assert out["sin"] == "123-456-789"
    assert errors == []


def test_dtype_coercion_failure_keeps_raw_and_reports_error():
    schema = _schema_with(Field(name="amount", dtype=float))
    out, errors = FieldValidator().validate({"amount": "not a number"}, schema)
    assert out["amount"] == "not a number"
    assert any("Type error on amount" in e for e in errors)
