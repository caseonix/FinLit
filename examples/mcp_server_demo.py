"""Demo: drive the FinLit MCP server in-process with a fake pipeline.

Runs without API keys. Shows the four tools end-to-end against a tiny
synthetic T4 stub. Bypasses parsing and LLM calls so contributors can
try the demo without setup.

Run: python examples/mcp_server_demo.py
"""
from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

from finlit.integrations.mcp.server import build_app
from finlit.result import ExtractionResult
from finlit.schema import Field, Schema


def _payload(call_result: Any) -> Any:
    """Extract structured payload from FastMCP `call_tool` return.

    FastMCP returns either `tuple[blocks, {"result": ...}]` (for typed list
    returns) or `list[ContentBlock]` (for plain dict returns). Handle both.
    """
    if isinstance(call_result, tuple) and len(call_result) == 2:
        _, structured = call_result
        if isinstance(structured, dict) and "result" in structured:
            return structured["result"]
        return structured
    if isinstance(call_result, list) and call_result and hasattr(call_result[0], "text"):
        return json.loads(call_result[0].text)
    return call_result


def _t4_schema() -> Schema:
    return Schema(
        name="t4",
        document_type="T4 Statement of Remuneration Paid",
        fields=[
            Field(name="sin", pii=True, required=True),
            Field(name="employee_name", pii=True, required=True),
            Field(name="employer_name", required=True),
            Field(name="box_14_employment_income", dtype=float),
        ],
    )


def _canned_result(path: Path) -> ExtractionResult:
    return ExtractionResult(
        fields={
            "sin": "123-456-789",
            "employee_name": "Demo User",
            "employer_name": "Acme Corp",
            "box_14_employment_income": 87500.00,
        },
        confidence={
            "sin": 0.99, "employee_name": 0.97,
            "employer_name": 0.99, "box_14_employment_income": 0.98,
        },
        source_ref={},
        pii_entities=[],
        audit_log=[{"event": "extraction_complete", "fields_returned": 4}],
        review_fields=[],
        warnings=[],
        document_path=str(path),
        schema_name="t4",
        extraction_path="text",
    )


class _FakePipeline:
    def __init__(self, schema: Schema):
        self.schema = schema

    def run(self, path):
        return _canned_result(Path(path))


def _section(title: str) -> None:
    print(f"\n--- {title} ---")


async def main() -> None:
    fake = _FakePipeline(_t4_schema())

    def fake_get(extractor, vision_extractor, schema_key, review_threshold):
        return fake

    app = build_app(extractor="claude", vision_extractor=None,
                    review_threshold=0.85, pii_mode="redact")

    with tempfile.TemporaryDirectory() as td:
        doc = Path(td) / "demo_t4.pdf"
        doc.write_bytes(b"%PDF-1.4 demo")

        with patch("finlit.integrations.mcp.server.get_pipeline", fake_get):
            _section("list_schemas")
            schemas = _payload(await app.call_tool("list_schemas", {}))
            print(json.dumps(schemas[:2], indent=2))
            print(f"... ({len(schemas)} schemas total)")

            _section("extract_document (default: PII redacted)")
            r = _payload(await app.call_tool("extract_document", {
                "path": str(doc), "schema": "cra.t4",
            }))
            print(json.dumps(r["fields"], indent=2))

            _section("extract_document (redact_pii=False)")
            r = _payload(await app.call_tool("extract_document", {
                "path": str(doc), "schema": "cra.t4", "redact_pii": False,
            }))
            print(json.dumps(r["fields"], indent=2))

            _section("detect_pii")
            r = _payload(await app.call_tool("detect_pii", {
                "text": "John lives at M5V 3A8, SIN 123-456-789",
                "return_redacted": True,
            }))
            print(json.dumps(r, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
