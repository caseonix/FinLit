"""Response trimming and field-level PII redaction for MCP tool responses.

The redaction here is the *MCP layer's* policy, appropriate to the
chat-transcript trust model. It does not change the underlying library
behavior - the original ExtractionResult is never mutated.
"""
from __future__ import annotations

from typing import Any

from finlit.result import ExtractionResult
from finlit.schema import Schema

_REDACTED = "[REDACTED]"


def apply_pii_redaction(
    fields: dict[str, Any], schema: Schema, redact: bool
) -> dict[str, Any]:
    """Return a new dict with PII-flagged field values replaced by [REDACTED].

    `redact=False` returns a shallow copy with raw values. The input dict
    is never mutated.
    """
    if not redact:
        return dict(fields)

    pii_names = {f.name for f in schema.fields if f.pii}
    out: dict[str, Any] = {}
    for name, value in fields.items():
        if name in pii_names and value is not None:
            out[name] = _REDACTED
        else:
            out[name] = value
    return out


def build_extraction_response(
    result: ExtractionResult,
    schema: Schema,
    schema_key: str,
    document_path: str,
    *,
    redact: bool,
    include_audit_log: bool,
    include_source_ref: bool,
    include_pii_entities: bool,
) -> dict[str, Any]:
    """Build the dict returned by the extract_document MCP tool."""
    response: dict[str, Any] = {
        "fields": apply_pii_redaction(result.fields, schema, redact=redact),
        "confidence": dict(result.confidence),
        "needs_review": result.needs_review,
        "review_fields": list(result.review_fields),
        "extraction_path": result.extraction_path,
        "extracted_field_count": result.extracted_field_count,
        "schema": schema_key,
        "document": document_path,
        "warnings": list(result.warnings),
    }
    if include_audit_log:
        response["audit_log"] = list(result.audit_log)
    if include_source_ref:
        response["source_ref"] = dict(result.source_ref)
    if include_pii_entities:
        response["pii_entities"] = list(result.pii_entities)
    return response
