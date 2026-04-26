"""batch_extract tool: happy path + on_error raise/skip/include + index alignment."""
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


def _make_canned() -> ExtractionResult:
    return ExtractionResult(
        fields={"sin": "123-456-789", "employer_name": "Acme"},
        confidence={"sin": 0.99, "employer_name": 0.98},
        source_ref={},
        pii_entities=[],
        audit_log=[],
        review_fields=[],
        warnings=[],
        document_path="",
        schema_name="t4",
        extraction_path="text",
    )


class _FlakyFakePipeline:
    """Fake pipeline whose .run(path) raises if path basename matches a fail-set."""

    def __init__(self, schema: Schema, fail_basenames: set[str] | None = None):
        self.schema = schema
        self._fail = fail_basenames or set()

    def run(self, path):
        p = Path(path)
        if p.name in self._fail:
            raise RuntimeError(f"synthetic failure for {p.name}")
        canned = _make_canned()
        canned.document_path = str(p)
        return canned


@pytest.fixture(autouse=True)
def _clear_cache():
    pipeline_cache.clear_cache()
    yield
    pipeline_cache.clear_cache()


def _patch_with(fake):
    def fake_get(extractor, vision_extractor, schema_key, review_threshold):
        return fake
    return patch("finlit.integrations.mcp.server.get_pipeline", fake_get)


def _make_doc(tmp_path: Path, name: str) -> Path:
    p = tmp_path / name
    p.write_bytes(b"%PDF-1.4 fake")
    return p


@pytest.mark.asyncio
async def test_batch_extract_happy_path(tmp_path):
    fake = _FlakyFakePipeline(_t4_schema())
    paths = [str(_make_doc(tmp_path, f"t4_{i}.pdf")) for i in range(3)]

    app = build_app(extractor="claude", vision_extractor=None,
                    review_threshold=0.85, pii_mode="redact")

    with _patch_with(fake):
        raw = await app.call_tool("batch_extract", {
            "paths": paths, "schema": "cra.t4",
        })
    payload = call_payload(raw)

    assert len(payload["results"]) == 3
    assert all(r is not None for r in payload["results"])
    assert payload["errors"] == []


@pytest.mark.asyncio
async def test_batch_extract_on_error_raise(tmp_path):
    fake = _FlakyFakePipeline(_t4_schema(), fail_basenames={"t4_1.pdf"})
    paths = [str(_make_doc(tmp_path, f"t4_{i}.pdf")) for i in range(3)]

    app = build_app(extractor="claude", vision_extractor=None,
                    review_threshold=0.85, pii_mode="redact")

    with _patch_with(fake):
        with pytest.raises(Exception) as excinfo:
            await app.call_tool("batch_extract", {
                "paths": paths, "schema": "cra.t4", "on_error": "raise",
            })
    assert "synthetic failure" in str(excinfo.value) or "t4_1.pdf" in str(excinfo.value)


@pytest.mark.asyncio
async def test_batch_extract_on_error_skip(tmp_path):
    fake = _FlakyFakePipeline(_t4_schema(), fail_basenames={"t4_1.pdf"})
    paths = [str(_make_doc(tmp_path, f"t4_{i}.pdf")) for i in range(3)]

    app = build_app(extractor="claude", vision_extractor=None,
                    review_threshold=0.85, pii_mode="redact")

    with _patch_with(fake):
        raw = await app.call_tool("batch_extract", {
            "paths": paths, "schema": "cra.t4", "on_error": "skip",
        })
    payload = call_payload(raw)

    # 1 failure out of 3; skip drops the failed slot
    assert len(payload["results"]) == 2
    assert all(r is not None for r in payload["results"])
    assert len(payload["errors"]) == 1


@pytest.mark.asyncio
async def test_batch_extract_on_error_include_aligns_indices(tmp_path):
    fake = _FlakyFakePipeline(_t4_schema(), fail_basenames={"t4_1.pdf"})
    paths = [str(_make_doc(tmp_path, f"t4_{i}.pdf")) for i in range(3)]

    app = build_app(extractor="claude", vision_extractor=None,
                    review_threshold=0.85, pii_mode="redact")

    with _patch_with(fake):
        raw = await app.call_tool("batch_extract", {
            "paths": paths, "schema": "cra.t4", "on_error": "include",
        })
    payload = call_payload(raw)

    assert len(payload["results"]) == 3
    assert payload["results"][0] is not None
    assert payload["results"][1] is None  # the failed one
    assert payload["results"][2] is not None
    assert len(payload["errors"]) == 1
    assert payload["errors"][0]["path"] == paths[1]
