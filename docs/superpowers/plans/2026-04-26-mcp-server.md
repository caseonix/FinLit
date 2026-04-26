# MCP Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship an MCP server that exposes FinLit as four stdio tools (`list_schemas`, `extract_document`, `batch_extract`, `detect_pii`) so any MCP-compatible host can extract Canadian financial documents through tool calls without writing Python glue.

**Architecture:** A new package `finlit/integrations/mcp/` that mirrors the existing `finlit/integrations/langchain/` layout. The server is a thin FastMCP app over `DocumentPipeline` with two cross-cutting layers: a `(extractor, vision, schema_key, threshold)` pipeline cache, and a response-trimming + field-level PII-redaction module. Two launch paths share one implementation: `finlit mcp serve` (typer) and `python -m finlit.integrations.mcp` (Claude Desktop config-friendly).

**Tech Stack:** `mcp>=1.0` (Anthropic OSS Python SDK, FastMCP-style decorators), `pydantic` v2 (already a dep via pydantic-ai), `typer` + `rich` (existing CLI). Tests use the SDK's in-memory client transport plus a stub `BaseExtractor` — no network, no API keys.

**Spec:** `docs/superpowers/specs/2026-04-26-mcp-server-design.md`

---

## Pre-flight

- Verify you are on a feature branch off `main` (e.g. `feat/mcp-server`). The `feat/langchain-reader` work in this repo introduced `finlit/integrations/_schema_resolver.py` and `finlit/integrations/langchain/`; this plan assumes both are present on whatever base branch you start from. Confirm with `ls finlit/integrations/_schema_resolver.py finlit/integrations/langchain/loader.py`.
- Verify Python 3.11+ and an existing editable install: `pip install -e ".[dev]"`.
- Run the existing test suite once to confirm a green starting state: `pytest tests/ -v`.

---

## Task 1: Package skeleton + `mcp` extra + install guard

**Files:**
- Create: `finlit/integrations/mcp/__init__.py`
- Create: `tests/integrations/__init__.py` (if missing)
- Create: `tests/integrations/mcp/__init__.py`
- Create: `tests/integrations/mcp/test_install_guard.py`
- Modify: `pyproject.toml` (add `mcp` optional extra)

- [ ] **Step 1: Add the `mcp` optional extra to `pyproject.toml`**

Open `pyproject.toml`. Find the `[project.optional-dependencies]` table (it already contains `dev`, `langchain`, etc.). Add:

```toml
mcp = ["mcp>=1.0"]
```

If a `[project.optional-dependencies]` table does not exist, search for `langchain = [` to find where extras live and add `mcp = ["mcp>=1.0"]` adjacent to it.

- [ ] **Step 2: Reinstall with the new extra**

Run: `pip install -e ".[dev,mcp]"`
Expected: installation succeeds; `python -c "import mcp; print(mcp.__version__)"` prints a version `>= 1.0`.

- [ ] **Step 3: Create the `__init__.py` install guard**

Mirror the langchain pattern in `finlit/integrations/langchain/__init__.py`. Create `finlit/integrations/mcp/__init__.py`:

```python
"""MCP integration for FinLit. Install with: pip install finlit[mcp]."""
try:
    from finlit.integrations.mcp.server import serve
except ImportError as exc:  # pragma: no cover - exercised via sys.modules patching
    raise ImportError(
        "finlit[mcp] extras not installed. "
        "Run: pip install finlit[mcp]"
    ) from exc

__all__ = ["serve"]
```

Note: `server.py` does not exist yet; the import will fail. We will create it in Task 5. The install guard is correct as-is.

- [ ] **Step 4: Create the test package directories**

Run:
```bash
mkdir -p tests/integrations/mcp
touch tests/integrations/__init__.py tests/integrations/mcp/__init__.py
```

- [ ] **Step 5: Write the failing install-guard test**

Create `tests/integrations/mcp/test_install_guard.py`:

```python
"""Verify the MCP integration raises a helpful ImportError when mcp is missing."""
import sys
from unittest.mock import patch

import pytest


def test_missing_mcp_extra_raises_helpful_importerror():
    # Pretend the `mcp` package is not installed.
    blocked = {name: None for name in list(sys.modules) if name == "mcp" or name.startswith("mcp.")}
    blocked["mcp"] = None

    # Force re-import of finlit.integrations.mcp from scratch.
    for mod in list(sys.modules):
        if mod.startswith("finlit.integrations.mcp"):
            del sys.modules[mod]

    with patch.dict(sys.modules, blocked):
        with pytest.raises(ImportError, match=r"finlit\[mcp\] extras not installed"):
            import finlit.integrations.mcp  # noqa: F401
```

- [ ] **Step 6: Run test (will fail at server-import line, not at the guard)**

Run: `pytest tests/integrations/mcp/test_install_guard.py -v`
Expected: FAIL — but the failure mode is fine for now (server.py does not yet exist; the import error message is right). We will revisit this test in Task 5 once `server.py` exists.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml finlit/integrations/mcp/__init__.py tests/integrations/__init__.py tests/integrations/mcp/__init__.py tests/integrations/mcp/test_install_guard.py
git commit -m "feat(mcp): add finlit[mcp] extra and integration package skeleton"
```

---

## Task 2: Stub extractor fixture

**Files:**
- Create: `tests/fixtures/__init__.py` (if missing)
- Create: `tests/fixtures/stub_extractor.py`
- Create: `tests/fixtures/test_stub_extractor.py`

- [ ] **Step 1: Verify `tests/fixtures/` exists**

Run: `ls tests/fixtures/`
Expected: directory exists. If not, `mkdir -p tests/fixtures && touch tests/fixtures/__init__.py`.

- [ ] **Step 2: Write failing test for stub extractor**

Create `tests/fixtures/test_stub_extractor.py`:

```python
"""Stub extractor returns canned output regardless of input."""
from finlit.extractors.pydantic_ai_extractor import ExtractionOutput
from finlit.schema import Field, Schema
from tests.fixtures.stub_extractor import StubExtractor


def test_stub_returns_canned_output():
    canned = ExtractionOutput(
        fields={"sin": "123-456-789", "employee_name": "Test User"},
        confidence={"sin": 0.99, "employee_name": 0.95},
        notes="",
    )
    extractor = StubExtractor(canned)
    schema = Schema(name="x", fields=[Field(name="sin", pii=True), Field(name="employee_name")])

    out = extractor.extract("ignored text", schema)

    assert out.fields == {"sin": "123-456-789", "employee_name": "Test User"}
    assert out.confidence == {"sin": 0.99, "employee_name": 0.95}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/fixtures/test_stub_extractor.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tests.fixtures.stub_extractor'`.

- [ ] **Step 4: Implement the stub**

Create `tests/fixtures/stub_extractor.py`:

```python
"""Reusable stub extractor: returns a preconfigured ExtractionOutput.

Used in tests to avoid making real LLM calls. Per CLAUDE.md, tests must
pass with no API keys set.
"""
from __future__ import annotations

from finlit.extractors.base import BaseExtractor
from finlit.extractors.pydantic_ai_extractor import ExtractionOutput
from finlit.schema import Schema


class StubExtractor(BaseExtractor):
    """Returns a preconfigured ExtractionOutput regardless of input text."""

    def __init__(self, canned_output: ExtractionOutput) -> None:
        self.canned_output = canned_output
        self.call_count = 0

    def extract(self, text: str, schema: Schema) -> ExtractionOutput:
        self.call_count += 1
        return self.canned_output
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/fixtures/test_stub_extractor.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tests/fixtures/stub_extractor.py tests/fixtures/test_stub_extractor.py
git commit -m "test(mcp): add reusable StubExtractor fixture"
```

---

## Task 3: Pipeline cache module

**Files:**
- Create: `finlit/integrations/mcp/pipeline_cache.py`
- Create: `tests/integrations/mcp/test_pipeline_cache.py`

- [ ] **Step 1: Write failing tests for the cache**

Create `tests/integrations/mcp/test_pipeline_cache.py`:

```python
"""Pipeline cache: lazy build, key-based reuse, thread-safe."""
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest

from finlit.integrations.mcp import pipeline_cache


@pytest.fixture(autouse=True)
def _clear_cache():
    pipeline_cache.clear_cache()
    yield
    pipeline_cache.clear_cache()


def _fake_pipeline_factory(call_log):
    """Returns a callable that records calls and returns a sentinel object."""
    class FakePipeline:
        def __init__(self, schema, extractor, review_threshold, vision_extractor):
            call_log.append((schema.name, extractor, review_threshold, vision_extractor))
            self.schema = schema
            self.extractor = extractor

    return FakePipeline


def test_cache_builds_once_per_key():
    calls = []
    fake = _fake_pipeline_factory(calls)

    with patch("finlit.integrations.mcp.pipeline_cache.DocumentPipeline", fake):
        p1 = pipeline_cache.get_pipeline("claude", None, "cra.t4", 0.85)
        p2 = pipeline_cache.get_pipeline("claude", None, "cra.t4", 0.85)

    assert p1 is p2
    assert len(calls) == 1


def test_cache_separates_by_extractor():
    calls = []
    fake = _fake_pipeline_factory(calls)

    with patch("finlit.integrations.mcp.pipeline_cache.DocumentPipeline", fake):
        pipeline_cache.get_pipeline("claude", None, "cra.t4", 0.85)
        pipeline_cache.get_pipeline("ollama", None, "cra.t4", 0.85)

    assert len(calls) == 2


def test_cache_separates_by_schema():
    calls = []
    fake = _fake_pipeline_factory(calls)

    with patch("finlit.integrations.mcp.pipeline_cache.DocumentPipeline", fake):
        pipeline_cache.get_pipeline("claude", None, "cra.t4", 0.85)
        pipeline_cache.get_pipeline("claude", None, "cra.t5", 0.85)

    assert len(calls) == 2


def test_cache_thread_safe_under_contention():
    """Two threads requesting the same key should still build only once."""
    calls = []
    fake = _fake_pipeline_factory(calls)

    with patch("finlit.integrations.mcp.pipeline_cache.DocumentPipeline", fake):
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [
                pool.submit(pipeline_cache.get_pipeline, "claude", None, "cra.t4", 0.85)
                for _ in range(8)
            ]
            results = [f.result() for f in futures]

    assert all(r is results[0] for r in results)
    assert len(calls) == 1


def test_unknown_schema_raises_valueerror():
    with pytest.raises(ValueError, match="Unknown schema"):
        pipeline_cache.get_pipeline("claude", None, "cra.t99", 0.85)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/integrations/mcp/test_pipeline_cache.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'finlit.integrations.mcp.pipeline_cache'`.

- [ ] **Step 3: Implement the pipeline cache**

Create `finlit/integrations/mcp/pipeline_cache.py`:

```python
"""Lazy, thread-safe (extractor, vision, schema, threshold) -> DocumentPipeline cache.

Used by the MCP server so that repeated tool calls with the same configuration
reuse one DocumentPipeline (and one underlying pydantic-ai client) instead of
rebuilding it every time.
"""
from __future__ import annotations

import threading

from finlit.extractors.vision_extractor import VisionExtractor
from finlit.integrations._schema_resolver import _resolve_schema
from finlit.pipeline import DocumentPipeline

# (extractor, vision_extractor_or_None, schema_key, review_threshold)
CacheKey = tuple[str, str | None, str, float]

_CACHE: dict[CacheKey, DocumentPipeline] = {}
_LOCK = threading.Lock()

_VISION_ALIASES = {
    "claude": "anthropic:claude-sonnet-4-6",
    "openai": "openai:gpt-4o",
    "ollama": "ollama:llama3.2-vision",
}


def get_pipeline(
    extractor: str,
    vision_extractor: str | None,
    schema_key: str,
    review_threshold: float,
) -> DocumentPipeline:
    """Return a cached DocumentPipeline for this configuration, building if needed.

    Raises:
        ValueError: if `schema_key` is not a known dotted registry key.
    """
    key: CacheKey = (extractor, vision_extractor, schema_key, review_threshold)
    with _LOCK:
        if key in _CACHE:
            return _CACHE[key]

        schema = _resolve_schema(schema_key)  # raises ValueError on unknown key

        ve = None
        if vision_extractor is not None:
            model_str = _VISION_ALIASES.get(vision_extractor, vision_extractor)
            ve = VisionExtractor(model=model_str)

        pipeline = DocumentPipeline(
            schema=schema,
            extractor=extractor,
            review_threshold=review_threshold,
            vision_extractor=ve,
        )
        _CACHE[key] = pipeline
        return pipeline


def clear_cache() -> None:
    """Test-only helper: drop all cached pipelines."""
    with _LOCK:
        _CACHE.clear()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/integrations/mcp/test_pipeline_cache.py -v`
Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add finlit/integrations/mcp/pipeline_cache.py tests/integrations/mcp/test_pipeline_cache.py
git commit -m "feat(mcp): add (extractor, vision, schema, threshold) pipeline cache"
```

---

## Task 4: Response builder + PII redaction

**Files:**
- Create: `finlit/integrations/mcp/responses.py`
- Create: `tests/integrations/mcp/test_responses.py`

- [ ] **Step 1: Write failing tests for response builder**

Create `tests/integrations/mcp/test_responses.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/integrations/mcp/test_responses.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'finlit.integrations.mcp.responses'`.

- [ ] **Step 3: Implement the response builder**

Create `finlit/integrations/mcp/responses.py`:

```python
"""Response trimming and field-level PII redaction for MCP tool responses.

The redaction here is the *MCP layer's* policy, appropriate to the
chat-transcript trust model. It does not change the underlying library
behavior — the original ExtractionResult is never mutated.
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/integrations/mcp/test_responses.py -v`
Expected: 9 PASS.

- [ ] **Step 5: Commit**

```bash
git add finlit/integrations/mcp/responses.py tests/integrations/mcp/test_responses.py
git commit -m "feat(mcp): response trimming + field-level PII redaction"
```

---

## Task 5: Server skeleton + `list_schemas` tool

**Files:**
- Create: `finlit/integrations/mcp/server.py`
- Modify: `tests/integrations/mcp/test_install_guard.py` (now passes once server.py exists)
- Create: `tests/integrations/mcp/test_server_list_schemas.py`

> **MCP SDK note:** This plan targets `mcp>=1.0` which provides `mcp.server.fastmcp.FastMCP` for declarative tool registration and an in-memory client transport for tests. If the precise import paths or test-helper names differ in the version you installed, consult the SDK's README and adapt — the FastMCP decorator pattern itself is stable.

- [ ] **Step 1: Write failing test for `list_schemas`**

Create `tests/integrations/mcp/test_server_list_schemas.py`:

```python
"""list_schemas tool returns one entry per built-in registry schema."""
import pytest

from finlit.integrations.mcp.server import build_app


@pytest.mark.asyncio
async def test_list_schemas_returns_all_builtins():
    app = build_app(extractor="claude", vision_extractor=None,
                    review_threshold=0.85, pii_mode="redact")

    # Call the underlying tool function directly. (We test the MCP transport
    # in test_server.py once more tools exist.)
    schemas_list = await app.call_tool("list_schemas", {})

    # FastMCP returns a list of content blocks; the structured payload is in
    # the first block's structured_content (SDK >= 1.x). For tools returning
    # plain Python objects, we assert on the structured value.
    keys = {entry["key"] for entry in schemas_list.structured_content}
    assert keys == {"cra.t4", "cra.t5", "cra.t4a", "cra.nr4", "banking.bank_statement"}


@pytest.mark.asyncio
async def test_list_schemas_entry_shape():
    app = build_app(extractor="claude", vision_extractor=None,
                    review_threshold=0.85, pii_mode="redact")

    schemas_list = await app.call_tool("list_schemas", {})
    t4 = next(e for e in schemas_list.structured_content if e["key"] == "cra.t4")

    assert t4["name"]                          # non-empty document_type string
    assert t4["version"] == "1.0"
    assert t4["field_count"] > 0
    assert isinstance(t4["required_fields"], list)
    assert isinstance(t4["description"], str)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/integrations/mcp/test_server_list_schemas.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'finlit.integrations.mcp.server'`.

- [ ] **Step 3: Implement `server.py` with `build_app` + `list_schemas`**

Create `finlit/integrations/mcp/server.py`:

```python
"""FinLit MCP server — FastMCP app + tool registrations + serve() entry point.

The module exposes:

  - build_app(...)  — build a FastMCP app with the given server-startup config.
                       Pure construction; no I/O. Used by tests.
  - serve(...)      — build_app + run stdio. The CLI and __main__ launchers
                       both call this.
"""
from __future__ import annotations

from typing import Literal

from mcp.server.fastmcp import FastMCP

from finlit.integrations._schema_resolver import _DOTTED_TO_ATTR, _resolve_schema

PIIMode = Literal["redact", "raw"]


def build_app(
    *,
    extractor: str,
    vision_extractor: str | None,
    review_threshold: float,
    pii_mode: PIIMode,
) -> FastMCP:
    """Construct a FastMCP app with the given server-startup configuration."""
    app = FastMCP("finlit")

    # Server-startup config is captured in the closures below.
    server_default_redact = pii_mode == "redact"

    @app.tool()
    def list_schemas() -> list[dict]:
        """List all built-in FinLit schemas with field counts and required fields."""
        out = []
        for dotted_key in sorted(_DOTTED_TO_ATTR):
            schema = _resolve_schema(dotted_key)
            out.append({
                "key": dotted_key,
                "name": schema.document_type or schema.name,
                "version": schema.version,
                "field_count": len(schema.fields),
                "required_fields": [f.name for f in schema.fields if f.required],
                "description": schema.description,
            })
        return out

    # Stash config on the app for downstream tools added in later tasks.
    app._finlit_extractor = extractor                # type: ignore[attr-defined]
    app._finlit_vision = vision_extractor            # type: ignore[attr-defined]
    app._finlit_threshold = review_threshold         # type: ignore[attr-defined]
    app._finlit_default_redact = server_default_redact  # type: ignore[attr-defined]

    return app


def serve(
    *,
    extractor: str = "claude",
    vision_extractor: str | None = None,
    review_threshold: float = 0.85,
    pii_mode: PIIMode = "redact",
) -> None:
    """Build the app and run it over stdio. Blocks until the host disconnects."""
    app = build_app(
        extractor=extractor,
        vision_extractor=vision_extractor,
        review_threshold=review_threshold,
        pii_mode=pii_mode,
    )
    app.run()  # FastMCP defaults to stdio transport.
```

- [ ] **Step 4: Run the list_schemas tests**

Run: `pytest tests/integrations/mcp/test_server_list_schemas.py -v`
Expected: 2 PASS.

If tests fail because `app.call_tool` returns a different shape than `structured_content`, adjust the assertion to use whatever the installed SDK exposes (e.g. `result.content[0].text` parsed as JSON, or `result.data`). The tool implementation is correct; only the test's accessor may need to be updated.

- [ ] **Step 5: Re-run the install-guard test from Task 1**

Run: `pytest tests/integrations/mcp/test_install_guard.py -v`
Expected: PASS now that `server.py` exists.

- [ ] **Step 6: Commit**

```bash
git add finlit/integrations/mcp/server.py tests/integrations/mcp/test_server_list_schemas.py
git commit -m "feat(mcp): add server skeleton with list_schemas tool"
```

---

## Task 6: `extract_document` tool

**Files:**
- Modify: `finlit/integrations/mcp/server.py`
- Create: `tests/integrations/mcp/test_server_extract_document.py`

- [ ] **Step 1: Write failing tests for `extract_document`**

Create `tests/integrations/mcp/test_server_extract_document.py`:

```python
"""extract_document tool: happy path, redact default, redact override, includes, errors."""
from pathlib import Path
from unittest.mock import patch

import pytest

from finlit.extractors.pydantic_ai_extractor import ExtractionOutput
from finlit.integrations.mcp import pipeline_cache
from finlit.integrations.mcp.server import build_app
from finlit.result import ExtractionResult
from tests.fixtures.stub_extractor import StubExtractor


@pytest.fixture(autouse=True)
def _clear_cache():
    pipeline_cache.clear_cache()
    yield
    pipeline_cache.clear_cache()


@pytest.fixture
def stub_pipeline_factory(tmp_path):
    """Patch DocumentPipeline so it uses a stub extractor regardless of extractor= arg."""
    canned = ExtractionOutput(
        fields={"sin": "123-456-789", "employer_name": "Acme"},
        confidence={"sin": 0.99, "employer_name": 0.98},
        notes="",
    )
    stub = StubExtractor(canned)

    # Patch the pipeline_cache module to substitute the extractor.
    real_pipeline_ctor = pipeline_cache.DocumentPipeline

    def fake_ctor(*, schema, extractor, review_threshold, vision_extractor):
        return real_pipeline_ctor(
            schema=schema, extractor=stub, review_threshold=review_threshold,
            vision_extractor=vision_extractor,
        )

    with patch("finlit.integrations.mcp.pipeline_cache.DocumentPipeline", fake_ctor):
        yield stub


def _make_t4_text_file(tmp_path: Path) -> Path:
    """Create a tiny text file the docling parser can ingest as a document."""
    p = tmp_path / "t4.txt"
    p.write_text("T4 Statement of Remuneration Paid\nEmployee SIN: 123-456-789\n")
    return p


@pytest.mark.asyncio
async def test_extract_document_redacts_pii_by_default(stub_pipeline_factory, tmp_path):
    app = build_app(extractor="claude", vision_extractor=None,
                    review_threshold=0.85, pii_mode="redact")
    doc = _make_t4_text_file(tmp_path)

    result = await app.call_tool("extract_document", {
        "path": str(doc),
        "schema": "cra.t4",
    })

    payload = result.structured_content
    assert payload["fields"]["sin"] == "[REDACTED]"
    assert payload["fields"]["employer_name"] == "Acme"
    assert payload["schema"] == "cra.t4"
    assert payload["document"] == str(doc)


@pytest.mark.asyncio
async def test_extract_document_redact_override_returns_raw(stub_pipeline_factory, tmp_path):
    app = build_app(extractor="claude", vision_extractor=None,
                    review_threshold=0.85, pii_mode="redact")
    doc = _make_t4_text_file(tmp_path)

    result = await app.call_tool("extract_document", {
        "path": str(doc), "schema": "cra.t4", "redact_pii": False,
    })

    assert result.structured_content["fields"]["sin"] == "123-456-789"


@pytest.mark.asyncio
async def test_extract_document_server_pii_mode_raw(stub_pipeline_factory, tmp_path):
    app = build_app(extractor="claude", vision_extractor=None,
                    review_threshold=0.85, pii_mode="raw")
    doc = _make_t4_text_file(tmp_path)

    result = await app.call_tool("extract_document", {
        "path": str(doc), "schema": "cra.t4",
    })

    # Server default is raw; per-call did not override; expect raw.
    assert result.structured_content["fields"]["sin"] == "123-456-789"


@pytest.mark.asyncio
async def test_extract_document_include_audit_log(stub_pipeline_factory, tmp_path):
    app = build_app(extractor="claude", vision_extractor=None,
                    review_threshold=0.85, pii_mode="redact")
    doc = _make_t4_text_file(tmp_path)

    result = await app.call_tool("extract_document", {
        "path": str(doc), "schema": "cra.t4", "include_audit_log": True,
    })

    assert "audit_log" in result.structured_content
    assert isinstance(result.structured_content["audit_log"], list)


@pytest.mark.asyncio
async def test_extract_document_unknown_schema(tmp_path):
    app = build_app(extractor="claude", vision_extractor=None,
                    review_threshold=0.85, pii_mode="redact")
    doc = _make_t4_text_file(tmp_path)

    with pytest.raises(Exception) as excinfo:
        await app.call_tool("extract_document", {"path": str(doc), "schema": "cra.t99"})
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/integrations/mcp/test_server_extract_document.py -v`
Expected: FAIL — `extract_document` tool does not yet exist.

- [ ] **Step 3: Add the `extract_document` tool to `server.py`**

In `finlit/integrations/mcp/server.py`, inside `build_app`, add the tool **after** the `list_schemas` registration. Add these imports at the top of the file alongside the existing ones:

```python
import asyncio
from pathlib import Path

from finlit.integrations.mcp.pipeline_cache import get_pipeline
from finlit.integrations.mcp.responses import build_extraction_response
```

Add the tool inside `build_app`:

```python
    @app.tool()
    async def extract_document(
        path: str,
        schema: str,
        extractor_override: str | None = None,
        vision_extractor_override: str | None = None,
        redact_pii: bool | None = None,
        include_audit_log: bool = False,
        include_source_ref: bool = False,
        include_pii_entities: bool = False,
    ) -> dict:
        """Extract structured fields from a single Canadian financial document."""
        doc_path = Path(path)
        if not doc_path.exists():
            raise ValueError(f"path does not exist: {path}")

        chosen_extractor = extractor_override or extractor
        chosen_vision = vision_extractor_override if vision_extractor_override is not None else vision_extractor
        effective_redact = redact_pii if redact_pii is not None else server_default_redact

        try:
            pipeline = get_pipeline(
                chosen_extractor, chosen_vision, schema, review_threshold,
            )
        except ValueError:
            raise  # Unknown schema or auth issues — surface as-is.

        # Run the sync pipeline in a thread so the event loop stays responsive.
        result = await asyncio.to_thread(pipeline.run, doc_path)

        return build_extraction_response(
            result=result,
            schema=pipeline.schema,
            schema_key=schema,
            document_path=str(doc_path.resolve()),
            redact=effective_redact,
            include_audit_log=include_audit_log,
            include_source_ref=include_source_ref,
            include_pii_entities=include_pii_entities,
        )
```

> **Note on parameter naming:** the tool exposes `extractor_override` / `vision_extractor_override` rather than `extractor` / `vision_extractor` to avoid colliding with the closure-captured server defaults. The agent-facing JSON Schema documents these as the per-call override fields.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/integrations/mcp/test_server_extract_document.py -v`
Expected: 6 PASS.

If tests fail on the `result.structured_content` accessor, adjust as in Task 5 Step 4. If the FastMCP `call_tool` raises a wrapped exception that hides the inner `ValueError`, change the test assertions to use `match=` against the wrapped message or unwrap via the SDK's exception attribute.

- [ ] **Step 5: Commit**

```bash
git add finlit/integrations/mcp/server.py tests/integrations/mcp/test_server_extract_document.py
git commit -m "feat(mcp): add extract_document tool with PII and include flags"
```

---

## Task 7: `batch_extract` tool

**Files:**
- Modify: `finlit/integrations/mcp/server.py`
- Create: `tests/integrations/mcp/test_server_batch_extract.py`

> **Implementation note (refines spec §6.4):** `BatchPipeline` uses `as_completed` and does not preserve input order, which breaks the index alignment requirement of `on_error="include"`. We therefore use `concurrent.futures.ThreadPoolExecutor` directly inside the tool and submit one future per path, indexed by position.

- [ ] **Step 1: Write failing tests for `batch_extract`**

Create `tests/integrations/mcp/test_server_batch_extract.py`:

```python
"""batch_extract tool: happy path + on_error raise/skip/include + index alignment."""
from pathlib import Path
from unittest.mock import patch

import pytest

from finlit.extractors.pydantic_ai_extractor import ExtractionOutput
from finlit.integrations.mcp import pipeline_cache
from finlit.integrations.mcp.server import build_app
from finlit.schema import Schema
from tests.fixtures.stub_extractor import StubExtractor


@pytest.fixture(autouse=True)
def _clear_cache():
    pipeline_cache.clear_cache()
    yield
    pipeline_cache.clear_cache()


def _make_doc(tmp_path: Path, name: str) -> Path:
    p = tmp_path / name
    p.write_text("T4 Statement of Remuneration Paid\nEmployer: Acme\n")
    return p


class _FlakyStub(StubExtractor):
    """Raises on the second call; succeeds otherwise."""

    def __init__(self, canned, fail_indices: set[int]):
        super().__init__(canned)
        self.fail_indices = fail_indices

    def extract(self, text, schema):
        idx = self.call_count
        self.call_count += 1
        if idx in self.fail_indices:
            raise RuntimeError(f"synthetic failure on call {idx}")
        return self.canned_output


@pytest.fixture
def patch_pipeline_with(monkeypatch):
    """Helper: patch DocumentPipeline ctor inside pipeline_cache to inject a stub."""
    def _patch(stub):
        real_pipeline_ctor = pipeline_cache.DocumentPipeline

        def fake_ctor(*, schema, extractor, review_threshold, vision_extractor):
            return real_pipeline_ctor(
                schema=schema, extractor=stub, review_threshold=review_threshold,
                vision_extractor=vision_extractor,
            )
        monkeypatch.setattr(
            "finlit.integrations.mcp.pipeline_cache.DocumentPipeline", fake_ctor,
        )
        return stub
    return _patch


@pytest.mark.asyncio
async def test_batch_extract_happy_path(patch_pipeline_with, tmp_path):
    canned = ExtractionOutput(
        fields={"sin": "123-456-789", "employer_name": "Acme"},
        confidence={"sin": 0.99, "employer_name": 0.99},
        notes="",
    )
    patch_pipeline_with(StubExtractor(canned))

    app = build_app(extractor="claude", vision_extractor=None,
                    review_threshold=0.85, pii_mode="redact")
    paths = [str(_make_doc(tmp_path, f"t4_{i}.txt")) for i in range(3)]

    result = await app.call_tool("batch_extract", {
        "paths": paths, "schema": "cra.t4",
    })

    payload = result.structured_content
    assert len(payload["results"]) == 3
    assert all(r is not None for r in payload["results"])
    assert payload["errors"] == []


@pytest.mark.asyncio
async def test_batch_extract_on_error_raise(patch_pipeline_with, tmp_path):
    canned = ExtractionOutput(fields={"sin": "x"}, confidence={"sin": 0.9}, notes="")
    patch_pipeline_with(_FlakyStub(canned, fail_indices={1}))

    app = build_app(extractor="claude", vision_extractor=None,
                    review_threshold=0.85, pii_mode="redact")
    paths = [str(_make_doc(tmp_path, f"t4_{i}.txt")) for i in range(3)]

    with pytest.raises(Exception) as excinfo:
        await app.call_tool("batch_extract", {
            "paths": paths, "schema": "cra.t4", "on_error": "raise",
        })
    assert "synthetic failure" in str(excinfo.value) or "extraction failed" in str(excinfo.value)


@pytest.mark.asyncio
async def test_batch_extract_on_error_skip(patch_pipeline_with, tmp_path):
    canned = ExtractionOutput(fields={"sin": "x"}, confidence={"sin": 0.9}, notes="")
    patch_pipeline_with(_FlakyStub(canned, fail_indices={1}))

    app = build_app(extractor="claude", vision_extractor=None,
                    review_threshold=0.85, pii_mode="redact")
    paths = [str(_make_doc(tmp_path, f"t4_{i}.txt")) for i in range(3)]

    result = await app.call_tool("batch_extract", {
        "paths": paths, "schema": "cra.t4", "on_error": "skip",
    })
    payload = result.structured_content

    # 1 failure out of 3; skip drops the failed slot
    assert len(payload["results"]) == 2
    assert all(r is not None for r in payload["results"])
    assert len(payload["errors"]) == 1


@pytest.mark.asyncio
async def test_batch_extract_on_error_include_aligns_indices(patch_pipeline_with, tmp_path):
    canned = ExtractionOutput(fields={"sin": "x"}, confidence={"sin": 0.9}, notes="")
    patch_pipeline_with(_FlakyStub(canned, fail_indices={1}))

    app = build_app(extractor="claude", vision_extractor=None,
                    review_threshold=0.85, pii_mode="redact")
    paths = [str(_make_doc(tmp_path, f"t4_{i}.txt")) for i in range(3)]

    result = await app.call_tool("batch_extract", {
        "paths": paths, "schema": "cra.t4", "on_error": "include",
    })
    payload = result.structured_content

    assert len(payload["results"]) == 3
    assert payload["results"][0] is not None
    assert payload["results"][1] is None  # the failed one
    assert payload["results"][2] is not None
    assert len(payload["errors"]) == 1
    assert payload["errors"][0]["path"] == paths[1]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/integrations/mcp/test_server_batch_extract.py -v`
Expected: FAIL — `batch_extract` tool does not yet exist.

- [ ] **Step 3: Add `batch_extract` to `server.py`**

Add this import to the top of `finlit/integrations/mcp/server.py`:

```python
from concurrent.futures import ThreadPoolExecutor
```

Inside `build_app`, **after** the `extract_document` tool, add:

```python
    @app.tool()
    async def batch_extract(
        paths: list[str],
        schema: str,
        extractor_override: str | None = None,
        vision_extractor_override: str | None = None,
        redact_pii: bool | None = None,
        on_error: Literal["raise", "skip", "include"] = "raise",
        max_workers: int | None = None,
        include_audit_log: bool = False,
        include_source_ref: bool = False,
        include_pii_entities: bool = False,
    ) -> dict:
        """Extract from many documents in parallel; returns aligned results + errors."""
        if on_error not in ("raise", "skip", "include"):
            raise ValueError(
                f"on_error must be 'raise', 'skip', or 'include', got {on_error!r}"
            )

        doc_paths = [Path(p) for p in paths]
        for i, p in enumerate(doc_paths):
            if not p.exists():
                raise ValueError(f"paths[{i}] does not exist: {p}")

        chosen_extractor = extractor_override or extractor
        chosen_vision = vision_extractor_override if vision_extractor_override is not None else vision_extractor
        effective_redact = redact_pii if redact_pii is not None else server_default_redact

        pipeline = get_pipeline(chosen_extractor, chosen_vision, schema, review_threshold)
        workers = max_workers if max_workers is not None else 4

        def _run(path: Path):
            return pipeline.run(path)

        results: list[dict | None] = [None] * len(doc_paths)
        errors: list[dict] = []

        def _do_batch():
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_run, p): (i, p) for i, p in enumerate(doc_paths)}
                for fut in futures:
                    i, p = futures[fut]
                    try:
                        result = fut.result()
                    except Exception as e:
                        if on_error == "raise":
                            raise
                        errors.append({"path": str(p), "error": str(e), "stage": "extract"})
                        # leave results[i] as None for both skip and include
                        continue
                    results[i] = build_extraction_response(
                        result=result, schema=pipeline.schema, schema_key=schema,
                        document_path=str(p.resolve()), redact=effective_redact,
                        include_audit_log=include_audit_log,
                        include_source_ref=include_source_ref,
                        include_pii_entities=include_pii_entities,
                    )

        await asyncio.to_thread(_do_batch)

        if on_error == "skip":
            results = [r for r in results if r is not None]

        return {"results": results, "errors": errors}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/integrations/mcp/test_server_batch_extract.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add finlit/integrations/mcp/server.py tests/integrations/mcp/test_server_batch_extract.py
git commit -m "feat(mcp): add batch_extract tool with on_error modes and index alignment"
```

---

## Task 8: `detect_pii` tool

**Files:**
- Modify: `finlit/integrations/mcp/server.py`
- Create: `tests/integrations/mcp/test_server_detect_pii.py`

- [ ] **Step 1: Write failing tests for `detect_pii`**

Create `tests/integrations/mcp/test_server_detect_pii.py`:

```python
"""detect_pii tool: standalone Presidio + Canadian recognizers, no LLM."""
import pytest

from finlit.integrations.mcp.server import build_app


@pytest.mark.asyncio
async def test_detect_pii_finds_sin_and_postal():
    app = build_app(extractor="claude", vision_extractor=None,
                    review_threshold=0.85, pii_mode="redact")

    text = "John Doe lives at M5V 3A8 with SIN 123-456-789."
    result = await app.call_tool("detect_pii", {"text": text})
    payload = result.structured_content

    types = {e["entity_type"] for e in payload["entities"]}
    assert "CA_SIN" in types
    assert "CA_POSTAL_CODE" in types


@pytest.mark.asyncio
async def test_detect_pii_returns_redacted_when_requested():
    app = build_app(extractor="claude", vision_extractor=None,
                    review_threshold=0.85, pii_mode="redact")

    result = await app.call_tool("detect_pii", {
        "text": "SIN 123-456-789", "return_redacted": True,
    })
    payload = result.structured_content

    assert "redacted_text" in payload
    assert "123-456-789" not in payload["redacted_text"]
    assert "***-***-***" in payload["redacted_text"]


@pytest.mark.asyncio
async def test_detect_pii_omits_redacted_by_default():
    app = build_app(extractor="claude", vision_extractor=None,
                    review_threshold=0.85, pii_mode="redact")

    result = await app.call_tool("detect_pii", {"text": "SIN 123-456-789"})
    payload = result.structured_content

    assert "redacted_text" not in payload or payload["redacted_text"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/integrations/mcp/test_server_detect_pii.py -v`
Expected: FAIL — `detect_pii` tool does not yet exist.

- [ ] **Step 3: Add `detect_pii` to `server.py`**

Add this import to the top of `finlit/integrations/mcp/server.py`:

```python
from finlit.audit.pii import CanadianPIIDetector
```

Inside `build_app`, **after** the `batch_extract` tool, add:

```python
    # Built once per server, reused across tool calls. Presidio is heavy.
    pii_detector = CanadianPIIDetector()

    @app.tool()
    def detect_pii(text: str, return_redacted: bool = False) -> dict:
        """Detect Canadian + standard PII in arbitrary text. No LLM, no pipeline."""
        if return_redacted:
            redacted = pii_detector.redact(text)
            return {
                "entities": redacted.detected_entities,
                "redacted_text": redacted.redacted_text,
            }
        entities = pii_detector.analyze(text)
        return {"entities": entities, "redacted_text": None}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/integrations/mcp/test_server_detect_pii.py -v`
Expected: 3 PASS. (Presidio model load takes a few seconds on first run.)

- [ ] **Step 5: Commit**

```bash
git add finlit/integrations/mcp/server.py tests/integrations/mcp/test_server_detect_pii.py
git commit -m "feat(mcp): add detect_pii standalone tool"
```

---

## Task 9: Launchers — `__main__.py` and CLI subcommand

**Files:**
- Create: `finlit/integrations/mcp/__main__.py`
- Modify: `finlit/cli/main.py`
- Create: `tests/integrations/mcp/test_launchers.py`

- [ ] **Step 1: Write failing test for the launchers**

Create `tests/integrations/mcp/test_launchers.py`:

```python
"""Both launch paths build the same app and call serve()."""
from unittest.mock import patch

from typer.testing import CliRunner

from finlit.cli.main import app as cli_app


def test_python_m_entrypoint_calls_serve():
    """python -m finlit.integrations.mcp invokes serve() with default args."""
    with patch("finlit.integrations.mcp.server.serve") as mock_serve:
        from finlit.integrations.mcp import __main__  # noqa: F401
        # __main__ does its work at import time; assert serve was called.
        mock_serve.assert_called_once()


def test_finlit_mcp_serve_cli_calls_serve():
    runner = CliRunner()
    with patch("finlit.integrations.mcp.server.serve") as mock_serve:
        result = runner.invoke(cli_app, [
            "mcp", "serve", "--extractor", "ollama", "--pii-mode", "raw",
        ])
    assert result.exit_code == 0, result.output
    mock_serve.assert_called_once()
    kwargs = mock_serve.call_args.kwargs
    assert kwargs["extractor"] == "ollama"
    assert kwargs["pii_mode"] == "raw"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/integrations/mcp/test_launchers.py -v`
Expected: FAIL — neither `__main__.py` nor the `mcp` CLI subcommand exists yet.

- [ ] **Step 3: Create `__main__.py`**

Create `finlit/integrations/mcp/__main__.py`:

```python
"""Entry point for `python -m finlit.integrations.mcp`.

Reads server-startup config from environment variables (matching the CLI
flag names), then calls serve(). Used by Claude Desktop / Cursor / any
mcpServers config that prefers `python -m` over a console script.
"""
from __future__ import annotations

import os

from finlit.integrations.mcp.server import serve

_PII_MODES = {"redact", "raw"}


def _get_pii_mode() -> str:
    mode = os.environ.get("FINLIT_PII_MODE", "redact")
    if mode not in _PII_MODES:
        raise SystemExit(f"FINLIT_PII_MODE must be one of {_PII_MODES}, got {mode!r}")
    return mode


serve(
    extractor=os.environ.get("FINLIT_EXTRACTOR", "claude"),
    vision_extractor=os.environ.get("FINLIT_VISION_EXTRACTOR") or None,
    review_threshold=float(os.environ.get("FINLIT_REVIEW_THRESHOLD", "0.85")),
    pii_mode=_get_pii_mode(),  # type: ignore[arg-type]
)
```

- [ ] **Step 4: Add the `mcp` typer sub-app to the CLI**

Modify `finlit/cli/main.py`. After the existing `app = typer.Typer(...)` line, register the `mcp` sub-app:

```python
mcp_app = typer.Typer(name="mcp", help="MCP server commands")
app.add_typer(mcp_app, name="mcp")


@mcp_app.command("serve")
def mcp_serve(
    extractor: str = typer.Option("claude", help="Default text extractor for the server"),
    vision_extractor: str = typer.Option(None, "--vision-extractor", help="Optional default vision extractor"),
    review_threshold: float = typer.Option(0.85, help="Confidence threshold below which fields are flagged"),
    pii_mode: str = typer.Option("redact", help="PII default: 'redact' (recommended) or 'raw'"),
):
    """Run the FinLit MCP server over stdio."""
    if pii_mode not in ("redact", "raw"):
        console.print(f"[red]--pii-mode must be 'redact' or 'raw', got {pii_mode!r}[/red]")
        raise typer.Exit(1)
    from finlit.integrations.mcp.server import serve
    serve(
        extractor=extractor,
        vision_extractor=vision_extractor,
        review_threshold=review_threshold,
        pii_mode=pii_mode,  # type: ignore[arg-type]
    )
```

Place this block above the `if __name__ == "__main__":` guard at the bottom of the file.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/integrations/mcp/test_launchers.py -v`
Expected: 2 PASS.

- [ ] **Step 6: Verify the CLI help renders the new subcommand**

Run: `finlit mcp serve --help`
Expected: typer prints the help table for `serve` with the four options.

- [ ] **Step 7: Commit**

```bash
git add finlit/integrations/mcp/__main__.py finlit/cli/main.py tests/integrations/mcp/test_launchers.py
git commit -m "feat(mcp): add python -m and finlit mcp serve launchers"
```

---

## Task 10: Example script

**Files:**
- Create: `examples/mcp_server_demo.py`

- [ ] **Step 1: Write the example**

Create `examples/mcp_server_demo.py`:

```python
"""Demo: drive the FinLit MCP server in-process with a stub extractor.

Runs without API keys. Shows the four tools end-to-end against a tiny
synthetic T4 text file.

Run: python examples/mcp_server_demo.py
"""
from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from finlit.extractors.pydantic_ai_extractor import ExtractionOutput
from finlit.integrations.mcp import pipeline_cache
from finlit.integrations.mcp.server import build_app

# ---- Wire a stub extractor so we don't need any API keys ----------------
from tests.fixtures.stub_extractor import StubExtractor  # demo-only import

CANNED = ExtractionOutput(
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
    notes="",
)


async def main():
    pipeline_cache.clear_cache()
    app = build_app(extractor="claude", vision_extractor=None,
                    review_threshold=0.85, pii_mode="redact")

    with tempfile.TemporaryDirectory() as td:
        doc = Path(td) / "demo_t4.txt"
        doc.write_text("T4 Statement of Remuneration Paid for Demo User\n")

        real_ctor = pipeline_cache.DocumentPipeline

        def stub_ctor(*, schema, extractor, review_threshold, vision_extractor):
            return real_ctor(
                schema=schema, extractor=StubExtractor(CANNED),
                review_threshold=review_threshold, vision_extractor=vision_extractor,
            )

        with patch("finlit.integrations.mcp.pipeline_cache.DocumentPipeline", stub_ctor):
            print("--- list_schemas ---")
            r = await app.call_tool("list_schemas", {})
            print(json.dumps(r.structured_content, indent=2)[:500] + "...")

            print("\n--- extract_document (default: PII redacted) ---")
            r = await app.call_tool("extract_document", {
                "path": str(doc), "schema": "cra.t4",
            })
            print(json.dumps(r.structured_content["fields"], indent=2))

            print("\n--- extract_document (redact_pii=False) ---")
            r = await app.call_tool("extract_document", {
                "path": str(doc), "schema": "cra.t4", "redact_pii": False,
            })
            print(json.dumps(r.structured_content["fields"], indent=2))

            print("\n--- detect_pii ---")
            r = await app.call_tool("detect_pii", {
                "text": "John lives at M5V 3A8, SIN 123-456-789",
                "return_redacted": True,
            })
            print(json.dumps(r.structured_content, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Run the example**

Run: `python examples/mcp_server_demo.py`
Expected: prints all four tool outputs. PII is redacted in the first `extract_document` call and raw in the second.

- [ ] **Step 3: Commit**

```bash
git add examples/mcp_server_demo.py
git commit -m "docs(examples): add MCP server end-to-end demo"
```

---

## Task 11: Documentation — README, roadmap, CLAUDE.md

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add the MCP server section to the README**

Open `README.md`. Find the existing `### LangChain integration` section (roughly line 246). Add a new section **immediately after it**, before the `## CLI` heading:

````markdown
### MCP server

Expose FinLit as a Model Context Protocol server so any MCP-compatible host
(Claude Desktop, Claude Code, Cursor, custom agents) can extract documents
through tool calls — no Python glue.

Install the extra:

```bash
pip install finlit[mcp]
```

Run the server (two equivalent ways):

```bash
# Human-facing
finlit mcp serve --extractor claude

# Claude Desktop mcpServers config
python -m finlit.integrations.mcp
```

Claude Desktop config example:

```json
{
  "mcpServers": {
    "finlit": {
      "command": "python",
      "args": ["-m", "finlit.integrations.mcp"],
      "env": {
        "ANTHROPIC_API_KEY": "...",
        "FINLIT_EXTRACTOR": "claude",
        "FINLIT_PII_MODE": "redact"
      }
    }
  }
}
```

Tools exposed:

- `list_schemas()` — discover the built-in CRA / banking schemas
- `extract_document(path, schema, ...)` — extract one document
- `batch_extract(paths, schema, ...)` — extract many in parallel
- `detect_pii(text, ...)` — standalone Presidio + Canadian recognizers

PII fields (per schema annotation) are redacted in tool responses by
default — appropriate to the chat-transcript trust model. Pass
`redact_pii=false` per call, or start with `--pii-mode raw`, to opt out.
````

- [ ] **Step 2: Flip the roadmap line**

In `README.md`, find the roadmap section (around line 479). Change:

```markdown
- [ ] MCP tool definitions for agentic workflows
```

to:

```markdown
- [x] MCP tool definitions for agentic workflows
```

- [ ] **Step 3: Update CLAUDE.md OSS Stack table**

Open `CLAUDE.md`. Find the table starting `| Layer | Library | Notes |`. Add this row at the bottom of the table, before the next heading:

```markdown
| MCP server | `mcp` (Anthropic OSS) | Optional extra `finlit[mcp]`. Stdio transport only |
```

- [ ] **Step 4: Add design rule to CLAUDE.md**

In `CLAUDE.md`, find the `### Do not host anything` rule under "Key Design Rules". After that subsection, add a new subsection:

```markdown
### MCP server is a thin presentation layer
The `finlit/integrations/mcp/` package wraps `DocumentPipeline` for MCP
hosts. It contains no business logic — only protocol mapping, response
trimming, and a layer-specific PII redaction policy that does NOT change
underlying library behavior. `DocumentPipeline.run()` still returns
un-redacted `ExtractionResult.fields`; the MCP server applies an
explicit, MCP-only redaction step appropriate to the chat-transcript
trust model.
```

- [ ] **Step 5: Verify documentation builds (no broken links)**

Run: `grep -n "MCP server\|mcp serve\|finlit\[mcp\]" README.md CLAUDE.md`
Expected: matches in both files; no other docs reference MCP yet so no other files need updating.

- [ ] **Step 6: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "docs(mcp): document MCP server, flip roadmap, update CLAUDE.md"
```

---

## Task 12: Final verification

**Files:** none modified — verification only.

- [ ] **Step 1: Run the full test suite**

Run: `pytest tests/ -v`
Expected: all tests pass, including the new `tests/integrations/mcp/` and `tests/fixtures/test_stub_extractor.py`. No real LLM API calls; no API keys required.

- [ ] **Step 2: Run lint**

Run: `ruff check finlit/ tests/`
Expected: no findings. If any, fix in place and re-run.

- [ ] **Step 3: Run type checker**

Run: `mypy finlit/`
Expected: no findings. If any, fix in place and re-run.

- [ ] **Step 4: Manual smoke test of the server**

In one terminal:
```bash
finlit mcp serve --extractor claude --pii-mode redact
```

In another terminal, drive it with the official MCP CLI inspector or any MCP-aware client. The `mcp` SDK ships an `mcp` CLI command; confirm `mcp` is on `PATH`, then:

```bash
mcp dev finlit  # or equivalent inspector command from the installed SDK
```

If the SDK's inspector tool name differs, run `mcp --help` to find the right invocation. Verify all four tools are listed.

- [ ] **Step 5: Verify the example runs without API keys**

Run: `python examples/mcp_server_demo.py`
Expected: prints output for all four tools. PII redaction visible (first call masked, second call raw).

- [ ] **Step 6: Verify the CLI integrates cleanly**

Run: `finlit --help`
Expected: top-level help shows the existing `extract`, `schema-list` commands plus a new `mcp` group.

Run: `finlit mcp --help`
Expected: shows the `serve` subcommand.

- [ ] **Step 7: Final commit (if any cleanup)**

If lint or mypy required fixes, commit them:

```bash
git add -A
git commit -m "chore(mcp): fix lint/mypy findings from final verification"
```

- [ ] **Step 8: Confirm the branch is ready for PR**

Run:
```bash
git log --oneline main..HEAD
```

Expected: a clean series of feature commits, each scoped to one task. Open a PR with the spec link in the body and a note pointing to the manual smoke test in Step 4.

---

## Notes for the implementer

- **MCP SDK version drift.** The plan assumes `mcp>=1.0` with the FastMCP decorator pattern (`@app.tool()`) and `app.call_tool(name, args)` returning a result with `.structured_content`. If the installed SDK exposes different accessor names, adjust the test assertions; the tool implementations themselves are correct.
- **Pre-existing `_schema_resolver`.** This module already exists from the langchain integration. Do not modify it — both integrations share it.
- **No real LLM calls in tests.** Per CLAUDE.md, every test must pass with no API keys set. The plan uses `StubExtractor` + `patch` on `DocumentPipeline`'s constructor for this. If you find yourself reaching for a real provider in a test, stop and revisit.
- **Field-level redaction is `[REDACTED]` always.** `Field` carries no entity-type attribute, so the spec's "entity_hint" idea collapses to the fallback. If a richer hint is wanted later, it's a `Field` schema change, not an MCP change.
- **`BatchPipeline` was deliberately NOT used in `batch_extract`.** It uses `as_completed` and loses input order, which breaks `on_error="include"` index alignment. The MCP tool implements its own indexed `ThreadPoolExecutor` instead.
