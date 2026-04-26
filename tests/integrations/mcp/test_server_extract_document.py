"""extract_document tool: happy path, redact default, redact override, includes, errors."""
from pathlib import Path
from unittest.mock import patch

import pytest

from finlit.integrations.mcp import pipeline_cache
from finlit.integrations.mcp.server import build_app
from finlit.result import ExtractionResult
from finlit.schema import Field, Schema
from tests.integrations.mcp.conftest import call_payload


def _t4_schema() -> Schema:
    return Schema(
        name="t4",
        document_type="T4 Statement of Remuneration Paid",
        fields=[
            Field(name="sin", pii=True, required=True),
            Field(name="employer_name"),
        ],
    )


def _canned_result() -> ExtractionResult:
    return ExtractionResult(
        fields={"sin": "123-456-789", "employer_name": "Acme"},
        confidence={"sin": 0.99, "employer_name": 0.98},
        source_ref={"sin": {"page": 1, "bbox": [0, 0, 10, 10]}},
        pii_entities=[
            {"entity_type": "CA_SIN", "score": 0.99, "start": 0, "end": 11, "text": "123-456-789"}
        ],
        audit_log=[{"event": "extraction_complete"}],
        review_fields=[],
        warnings=[],
        document_path="",  # set by FakePipeline.run
        schema_name="t4",
        extraction_path="text",
    )


class _FakePipeline:
    """Fake DocumentPipeline that bypasses parsing/extraction and returns a canned result."""

    def __init__(self, schema: Schema, canned: ExtractionResult):
        self.schema = schema
        self._canned = canned

    def run(self, path):
        # Mutate the canned result's document_path each call so tests see a real path.
        self._canned.document_path = str(path)
        return self._canned


@pytest.fixture(autouse=True)
def _clear_cache():
    pipeline_cache.clear_cache()
    yield
    pipeline_cache.clear_cache()


@pytest.fixture
def patched_get_pipeline():
    """Replace pipeline_cache.get_pipeline so server tools use _FakePipeline."""
    schema = _t4_schema()
    canned = _canned_result()
    fake = _FakePipeline(schema, canned)

    def fake_get(extractor, vision_extractor, schema_key, review_threshold):
        return fake

    with patch.object(pipeline_cache, "get_pipeline", fake_get), \
         patch("finlit.integrations.mcp.server.get_pipeline", fake_get):
        yield fake


@pytest.fixture
def doc_file(tmp_path) -> Path:
    p = tmp_path / "t4.pdf"
    p.write_bytes(b"%PDF-1.4 fake")
    return p


@pytest.mark.asyncio
async def test_extract_document_redacts_pii_by_default(patched_get_pipeline, doc_file):
    app = build_app(extractor="claude", vision_extractor=None,
                    review_threshold=0.85, pii_mode="redact")

    raw = await app.call_tool("extract_document", {
        "path": str(doc_file),
        "schema": "cra.t4",
    })
    payload = call_payload(raw)

    assert payload["fields"]["sin"] == "[REDACTED]"
    assert payload["fields"]["employer_name"] == "Acme"
    assert payload["schema"] == "cra.t4"
    assert payload["document"] == str(doc_file.resolve())


@pytest.mark.asyncio
async def test_extract_document_redact_override_returns_raw(patched_get_pipeline, doc_file):
    app = build_app(extractor="claude", vision_extractor=None,
                    review_threshold=0.85, pii_mode="redact")

    raw = await app.call_tool("extract_document", {
        "path": str(doc_file), "schema": "cra.t4", "redact_pii": False,
    })
    assert call_payload(raw)["fields"]["sin"] == "123-456-789"


@pytest.mark.asyncio
async def test_extract_document_server_pii_mode_raw(patched_get_pipeline, doc_file):
    app = build_app(extractor="claude", vision_extractor=None,
                    review_threshold=0.85, pii_mode="raw")

    raw = await app.call_tool("extract_document", {
        "path": str(doc_file), "schema": "cra.t4",
    })
    # Server default is raw; per-call did not override; expect raw.
    assert call_payload(raw)["fields"]["sin"] == "123-456-789"


@pytest.mark.asyncio
async def test_extract_document_include_audit_log(patched_get_pipeline, doc_file):
    app = build_app(extractor="claude", vision_extractor=None,
                    review_threshold=0.85, pii_mode="redact")

    raw = await app.call_tool("extract_document", {
        "path": str(doc_file), "schema": "cra.t4", "include_audit_log": True,
    })
    payload = call_payload(raw)
    assert "audit_log" in payload
    assert payload["audit_log"] == [{"event": "extraction_complete"}]


@pytest.mark.asyncio
async def test_extract_document_unknown_schema(doc_file):
    app = build_app(extractor="claude", vision_extractor=None,
                    review_threshold=0.85, pii_mode="redact")

    with pytest.raises(Exception) as excinfo:
        await app.call_tool("extract_document", {"path": str(doc_file), "schema": "cra.t99"})
    assert "Unknown schema" in str(excinfo.value)


@pytest.mark.asyncio
async def test_extract_document_missing_file():
    app = build_app(extractor="claude", vision_extractor=None,
                    review_threshold=0.85, pii_mode="redact")

    with pytest.raises(Exception) as excinfo:
        await app.call_tool("extract_document", {
            "path": "/no/such/file.pdf", "schema": "cra.t4",
        })
    assert "does not exist" in str(excinfo.value)
