"""Response trimming + field-level PII redaction for MCP responses."""
from finlit.integrations.mcp.responses import (
    apply_pii_redaction,
    build_extraction_response,
)
from finlit.result import ExtractionResult
from finlit.schema import Field, Schema


def _make_schema():
    return Schema(
        name="t4",
        fields=[
            Field(name="sin", pii=True),
            Field(name="employee_name", pii=True),
            Field(name="employer_name"),
            Field(name="box_14_employment_income", dtype=float),
        ],
    )


def _make_result():
    return ExtractionResult(
        fields={
            "sin": "123-456-789",
            "employee_name": "Test User",
            "employer_name": "Acme Corp",
            "box_14_employment_income": 87500.0,
        },
        confidence={"sin": 0.99, "employee_name": 0.95, "employer_name": 0.99, "box_14_employment_income": 0.98},
        source_ref={"sin": {"page": 1, "bbox": [0, 0, 10, 10]}},
        pii_entities=[{"entity_type": "CA_SIN", "score": 0.99, "start": 0, "end": 11, "text": "123-456-789"}],
        audit_log=[{"event": "extraction_complete"}],
        review_fields=[],
        warnings=[],
        document_path="/tmp/t4.pdf",
        schema_name="t4",
        extraction_path="text",
    )


# --- apply_pii_redaction --------------------------------------------------

def test_redact_replaces_pii_fields_only():
    schema = _make_schema()
    fields = {"sin": "123-456-789", "employer_name": "Acme Corp"}

    out = apply_pii_redaction(fields, schema, redact=True)

    assert out["sin"] == "[REDACTED]"
    assert out["employer_name"] == "Acme Corp"


def test_redact_does_not_mutate_input():
    schema = _make_schema()
    fields = {"sin": "123-456-789"}

    apply_pii_redaction(fields, schema, redact=True)

    assert fields["sin"] == "123-456-789"


def test_redact_off_returns_copy_with_raw_values():
    schema = _make_schema()
    fields = {"sin": "123-456-789"}

    out = apply_pii_redaction(fields, schema, redact=False)

    assert out == {"sin": "123-456-789"}
    assert out is not fields  # still a copy


def test_redact_skips_none_values():
    schema = _make_schema()
    fields = {"sin": None, "employer_name": "Acme"}

    out = apply_pii_redaction(fields, schema, redact=True)

    assert out["sin"] is None  # don't redact missing values
    assert out["employer_name"] == "Acme"


# --- build_extraction_response -------------------------------------------

def test_response_default_shape():
    """Default: no audit_log, source_ref, or pii_entities keys."""
    response = build_extraction_response(
        result=_make_result(),
        schema=_make_schema(),
        schema_key="cra.t4",
        document_path="/tmp/t4.pdf",
        redact=True,
        include_audit_log=False,
        include_source_ref=False,
        include_pii_entities=False,
    )

    assert set(response.keys()) == {
        "fields", "confidence", "needs_review", "review_fields",
        "extraction_path", "extracted_field_count", "schema", "document",
        "warnings",
    }
    assert response["fields"]["sin"] == "[REDACTED]"
    assert response["fields"]["employer_name"] == "Acme Corp"
    assert response["schema"] == "cra.t4"
    assert response["document"] == "/tmp/t4.pdf"


def test_response_redact_off_returns_raw_values():
    response = build_extraction_response(
        result=_make_result(), schema=_make_schema(), schema_key="cra.t4",
        document_path="/tmp/t4.pdf", redact=False,
        include_audit_log=False, include_source_ref=False, include_pii_entities=False,
    )

    assert response["fields"]["sin"] == "123-456-789"


def test_response_include_audit_log():
    response = build_extraction_response(
        result=_make_result(), schema=_make_schema(), schema_key="cra.t4",
        document_path="/tmp/t4.pdf", redact=True,
        include_audit_log=True, include_source_ref=False, include_pii_entities=False,
    )

    assert "audit_log" in response
    assert response["audit_log"] == [{"event": "extraction_complete"}]


def test_response_include_source_ref():
    response = build_extraction_response(
        result=_make_result(), schema=_make_schema(), schema_key="cra.t4",
        document_path="/tmp/t4.pdf", redact=True,
        include_audit_log=False, include_source_ref=True, include_pii_entities=False,
    )

    assert "source_ref" in response
    assert response["source_ref"]["sin"] == {"page": 1, "bbox": [0, 0, 10, 10]}


def test_response_include_pii_entities():
    response = build_extraction_response(
        result=_make_result(), schema=_make_schema(), schema_key="cra.t4",
        document_path="/tmp/t4.pdf", redact=True,
        include_audit_log=False, include_source_ref=False, include_pii_entities=True,
    )

    assert "pii_entities" in response
    assert response["pii_entities"][0]["entity_type"] == "CA_SIN"
