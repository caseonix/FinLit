# MCP Server for FinLit — Design

**Status:** Draft
**Date:** 2026-04-26
**Owner:** Caseonix / FinLit maintainers
**Roadmap item:** `MCP tool definitions for agentic workflows` (README.md:494)

---

## 1. Goal

Expose FinLit's extraction pipeline as a Model Context Protocol (MCP) server so any MCP-compatible host — Claude Desktop, Claude Code, Cursor, custom agents — can extract structured data from Canadian financial documents through tool calls, without writing Python glue code. The server is a thin presentation layer over `DocumentPipeline`; it adds no business logic, only protocol mapping, response trimming, and an MCP-only PII-redaction policy appropriate to the chat-transcript trust model.

## 2. Non-goals

- New extractors, parsers, or schemas. This feature is purely a presentation layer.
- HTTP/SSE transport. Stdio only in v1; matches what every MCP host on the market consumes today.
- A hosted/server-side deployment. FinLit remains a library (CLAUDE.md rule); the MCP server runs locally as the user.
- MCP **resources** or **prompts**. Tools only in v1. Resources are appealing for schema browsing but only some clients render them and they double the protocol surface to test. Defer to a future spec.
- Custom-schema authoring tools (`validate_custom_schema(yaml)`). Almost no agent reliably hand-writes a valid schema; YAGNI.
- Inline base64 document content. Path-based only. Agents handle paths fine and binary in transcripts is a transcript-bloat anti-pattern.
- Allowed-roots / chroot-style path restriction. Stdio inherits the host's trust boundary; an explicit allow-list is theatre in v1.
- Modifying the top-level `finlit/__init__.py` public API. The MCP server lives entirely under `finlit/integrations/mcp/`.

## 3. Scope

Single PR shipping in v0.4.0. Backwards compatible — existing users see no change. Layered exactly like the existing `finlit/integrations/langchain/` package, including the install-guard ImportError pattern.

## 4. Architecture

### 4.1 Module layout

```
finlit/integrations/
├── __init__.py                 # unchanged namespace package
├── _schema_resolver.py         # already exists; reused as-is
└── mcp/
    ├── __init__.py             # extras-install guard ("pip install finlit[mcp]")
    ├── server.py               # FastMCP app, tool registration, serve() entry point
    ├── pipeline_cache.py       # (extractor, vision, schema_key, threshold) → DocumentPipeline
    ├── responses.py            # ExtractionResult → trimmed dict; PII redaction
    └── __main__.py             # `python -m finlit.integrations.mcp` launcher
```

### 4.2 Public surface

Two launch paths:

```bash
# Human-facing
finlit mcp serve --extractor claude --vision-extractor claude

# Claude Desktop / mcpServers config
python -m finlit.integrations.mcp
```

The `finlit mcp` command is a thin typer sub-app (registered in `finlit/cli/main.py`) that delegates to `finlit.integrations.mcp.server:serve()`. Both paths accept the same flags.

The top-level `finlit/__init__.py` is **not** modified. The MCP server is fully isolated under `finlit.integrations.mcp` and discoverable via the documented launch commands.

### 4.3 Dependency

New optional extra in `pyproject.toml`:

```toml
[project.optional-dependencies]
mcp = ["mcp>=1.0"]
```

Install: `pip install finlit[mcp]`. The `finlit/integrations/mcp/__init__.py` raises a clear ImportError pointing at this command if `mcp` is missing — same pattern as `finlit/integrations/langchain/__init__.py`.

## 5. Tools

Four tools, registered with FastMCP-style decorators. Pydantic models on inputs and outputs so the SDK auto-generates JSON Schema for the agent. The MCP server is registered with name `"finlit"` — clients show this as the namespace, so tool names use plain snake_case without a `finlit_` prefix.

### 5.1 `list_schemas`

```python
def list_schemas() -> list[SchemaInfo]
```

Read-only. Returns one entry per built-in schema (today: `cra.t4`, `cra.t5`, `cra.t4a`, `cra.nr4`, `banking.statement`).

```python
class SchemaInfo(BaseModel):
    key: str                    # "cra.t4"
    name: str                   # "T4 Statement of Remuneration Paid"
    version: str                # "1.0"
    field_count: int
    required_fields: list[str]
    description: str
```

The agent uses this to discover available schemas before calling `extract_document`.

### 5.2 `extract_document`

```python
def extract_document(
    path: str,
    schema: str,
    *,
    extractor: str | None = None,
    vision_extractor: str | None = None,
    redact_pii: bool | None = None,
    include_audit_log: bool = False,
    include_source_ref: bool = False,
    include_pii_entities: bool = False,
) -> ExtractionResponse
```

The main tool. `path` is any absolute or relative path the server process can read; `schema` is a registry key from `list_schemas`. `extractor` / `vision_extractor` / `redact_pii` are optional per-call overrides; `None` means use the server-startup default. The PII default at server start is `redact` (per §6.1) unless `--pii-mode raw` is set.

Default response shape:

```python
class ExtractionResponse(BaseModel):
    fields: dict[str, Any]              # PII-redacted by default per Q5
    confidence: dict[str, float]
    needs_review: bool
    review_fields: list[dict]
    extraction_path: Literal["text", "vision"]
    extracted_field_count: int
    schema: str                         # the resolved schema key
    document: str                       # the absolute path
    # Conditional keys (omitted unless requested):
    audit_log: list[dict] | None        # include_audit_log=True
    source_ref: dict[str, dict] | None  # include_source_ref=True
    pii_entities: list[dict] | None     # include_pii_entities=True
```

### 5.3 `batch_extract`

```python
def batch_extract(
    paths: list[str],
    schema: str,
    *,
    extractor: str | None = None,
    vision_extractor: str | None = None,
    redact_pii: bool | None = None,
    on_error: Literal["raise", "skip", "include"] = "raise",
    max_workers: int | None = None,    # None means BatchPipeline's own default
    include_audit_log: bool = False,
    include_source_ref: bool = False,
    include_pii_entities: bool = False,
) -> BatchResponse
```

Wraps `BatchPipeline`. `on_error` follows the convention already established in the langchain integration spec:

- `raise` (default): first failure aborts the batch; the tool returns a JSON-RPC error.
- `skip`: failed paths dropped from `results`; `errors` array populated.
- `include`: failed paths get `null` in `results` at the same index as the input; `errors` array populated. Index alignment lets the agent correlate.

```python
class BatchResponse(BaseModel):
    results: list[ExtractionResponse | None]   # null only when on_error="include"
    errors: list[BatchError]                   # always present; empty on full success

class BatchError(BaseModel):
    path: str
    error: str
    stage: Literal["parse", "extract", "validate"]
```

### 5.4 `detect_pii`

```python
def detect_pii(text: str, *, return_redacted: bool = False) -> PIIResponse
```

Standalone Presidio + Canadian recognizers pass over arbitrary text. No pipeline, no LLM. Useful as a pre-flight check before sending text to a non-Canadian LLM, or as a standalone PIPEDA-aware utility.

```python
class PIIResponse(BaseModel):
    entities: list[PIIEntity]
    redacted_text: str | None              # only when return_redacted=True

class PIIEntity(BaseModel):
    type: str                              # "CA_SIN", "CA_POSTAL_CODE", "PERSON", ...
    start: int
    end: int
    score: float
```

## 6. Server lifecycle

### 6.1 Startup flags

```bash
finlit mcp serve \
  --extractor claude \
  --vision-extractor claude \
  --review-threshold 0.85 \
  --pii-mode redact            # redact | raw  (default: redact)
```

All flags optional. Defaults match the rest of the codebase. Env vars (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, etc.) are read at pipeline-build time, not at server start — so a server can start without keys and only fail when a tool call needs the missing provider.

`--pii-mode` sets the server-wide default that applies whenever a tool call passes `redact_pii=None` (the default). A per-call `redact_pii=True` or `redact_pii=False` always overrides. The default at server start is `redact`.

### 6.2 Transport

Stdio only in v1. The MCP SDK's `Server.run_stdio()` (or FastMCP equivalent) is the entrypoint. HTTP/SSE deferred until a real use case appears.

### 6.3 Pipeline cache

The per-call extractor override (§5.2) means we cannot pre-build one pipeline at startup. The cache lives in `pipeline_cache.py`:

```python
_CACHE: dict[CacheKey, DocumentPipeline] = {}
_LOCK = threading.Lock()

CacheKey = tuple[str, str | None, str, float]  # (extractor, vision, schema_key, threshold)

def get_pipeline(
    extractor: str,
    vision_extractor: str | None,
    schema_key: str,
    review_threshold: float,
) -> DocumentPipeline:
    ...
```

First call for any tuple builds the pipeline and caches it under the lock; subsequent calls reuse. No LRU, no TTL — server lifetimes are short and the number of distinct tuples is small in practice (typically 1–3). Lock guards the dict during build to prevent two concurrent calls from racing on the same key.

### 6.4 Concurrency

The MCP SDK is async; `DocumentPipeline.run()` is sync. Each tool handler awaits its work via `asyncio.to_thread(pipeline.run, path)`. `batch_extract` does **not** add its own threadpool — `BatchPipeline` already parallelizes internally via `ThreadPoolExecutor`; the MCP wrapper just `await`s the whole call via `to_thread`.

## 7. PII redaction layer

Lives in `responses.py` as `_apply_pii_redaction(result, schema, mode)`. The function:

1. Walks `result.fields`.
2. For each field where `schema.fields[name].pii is True`, replaces the value in a *copy* with `f"[{entity_hint}]"` where `entity_hint` is the field's PII entity type (e.g. `"CA_SIN"`) or `"REDACTED"` if not specified.
3. Returns a new dict; never mutates the underlying `ExtractionResult`.

Critical invariant: the `audit_log` and `source_ref` (when included) reference the *un-redacted* values. Redaction operates only on the response payload sent back to the agent. This preserves the audit log's forensic integrity — the log was produced before the MCP layer existed and continues to record what the pipeline actually saw.

This MCP-layer policy does **not** change the underlying library behavior. `DocumentPipeline.run()` still returns un-redacted `ExtractionResult.fields`, consistent with the CLAUDE.md rule that PII flagging is annotation, not auto-redaction. The MCP server applies a separate, explicit redaction step appropriate to the chat-transcript trust model.

## 8. Error handling

Three failure surfaces:

### 8.1 Tool-input validation

Bad path, unknown schema, malformed args: tool handler raises `ValueError`; the MCP SDK converts to JSON-RPC `code: -32602` (invalid params). Examples:

- `path does not exist: /foo/bar.pdf`
- `unknown schema 'cra.t99' — call list_schemas() to see available keys`
- `extractor must be one of {claude, openai, ollama} or a pydantic-ai model string`

### 8.2 Pipeline errors

Parser crash, LLM timeout, validation failure: caught in the tool handler; returned as JSON-RPC `code: -32000` (server error) with structured `data`:

```python
{
  "code": -32000,
  "message": "extraction failed: <short reason>",
  "data": {
    "stage": "parse" | "extract" | "validate",
    "path": "...",
    "exception_type": "DoclingParseError"
  }
}
```

Stack traces are written to the server's stderr (which the MCP host typically captures into a log file) but never sent over the wire. Keeps transcripts clean and avoids leaking internal paths.

### 8.3 `batch_extract` per-document errors

Per §5.3 — `raise` / `skip` / `include`, mirroring the langchain convention.

### 8.4 Missing API keys

The pipeline-cache builder catches `pydantic_ai`'s auth error and re-raises as `ValueError("ANTHROPIC_API_KEY is not set; either export it or pass --extractor ollama")`. The agent gets actionable guidance, not a stack trace.

### 8.5 Server-startup errors

Bad CLI flag, unable to bind stdio: exit non-zero with a stderr message. The MCP host shows the user the disconnect.

## 9. Testing

Two layers, both run with no API keys set (per CLAUDE.md "never make real LLM API calls in tests").

### 9.1 Unit tests

`tests/integrations/mcp/test_responses.py`, `test_pipeline_cache.py`. Pure functions; no MCP runtime needed.

- `_apply_pii_redaction`: PII fields redacted, non-PII untouched, original `ExtractionResult` not mutated, redaction hint matches the field's PII type.
- `_trim_response`: each combination of `include_*` flags produces the expected key set.
- `pipeline_cache.get_pipeline`: cache hit/miss by key tuple, lock contention under `ThreadPoolExecutor` (two threads requesting the same key build only once).

### 9.2 In-process MCP integration tests

`tests/integrations/mcp/test_server.py`. The official `mcp` SDK ships an in-memory client that connects to a server via a memory transport — no subprocess, no stdio. Spin up the FinLit server with a stub extractor, call tools through the client, assert on responses.

Coverage:

- `list_schemas` returns the expected 5 entries with the right keys, names, field counts.
- `extract_document` happy path with stub → expected trimmed response shape.
- `extract_document` with `redact_pii=True` (default) → SIN field is `"[CA_SIN]"`.
- `extract_document` with `redact_pii=False` → raw value.
- `extract_document` with each `include_*` flag → corresponding key present.
- `batch_extract` with each `on_error` mode (one stub raises, two succeed); index alignment for `include`.
- `detect_pii` on a string with a SIN and a postal code → both detected; `return_redacted=True` returns redacted text.
- Error mapping: bad schema key → JSON-RPC -32602 with helpful message; missing API key → -32602 with the actionable hint from §8.4.
- Pipeline cache: same args twice → builder called once.

### 9.3 Test fixtures

New `tests/fixtures/stub_extractor.py`:

```python
class StubExtractor(BaseExtractor):
    """Returns a preconfigured ExtractionOutput regardless of input."""
    def __init__(self, canned_output: ExtractionOutput): ...
    def extract(self, text: str, schema: Schema) -> ExtractionOutput: ...
```

Reusable beyond MCP tests.

### 9.4 CI

`tests/integrations/mcp/` runs in the existing `pytest tests/` job. Optional `mcp` extra is installed in CI's dev install (`pip install -e ".[dev,mcp]"`).

### 9.5 Out of scope for tests

- Real Claude Desktop / Cursor / Claude Code consumption: manual smoke test in the PR description.
- HTTP/SSE transport: not shipped in v1.
- Real LLM extraction: covered by existing `tests/test_pipeline.py`; the MCP layer is a thin wrapper.

## 10. Documentation and packaging

### 10.1 README

New "MCP server" section placed after the existing "LangChain integration" section. Includes:

- Canonical Claude Desktop `mcpServers` config snippet (using `python -m finlit.integrations.mcp`).
- The `finlit mcp serve` form for humans.
- One-paragraph "what tools you get" with the four tool names.
- A note that PII is redacted by default and how to opt out.

Roadmap line flipped: `- [x] MCP tool definitions for agentic workflows`.

### 10.2 `pyproject.toml`

- New optional extra: `mcp = ["mcp>=1.0"]`.
- CLI entry point unchanged — `mcp` subcommand registered inside the existing `finlit` typer app.

### 10.3 CLAUDE.md

- Add a row to the "OSS Stack" table for `mcp` (Anthropic OSS, MIT).
- Add a paragraph under "Key Design Rules" stating the MCP server is a thin presentation layer over `DocumentPipeline` with no business logic, and that MCP-layer PII redaction does not change underlying library behavior.

### 10.4 Examples

`examples/mcp_server_demo.py`: a 30-line script that uses the in-memory MCP client to call each tool against a sample T4 with a stub extractor — same in-spirit pattern as `examples/langchain_rag.py`. Runs without API keys so contributors can try it immediately.

## 11. Release

Ships in v0.4.0. Single PR, single commit series. Backwards compatible — no changes to existing public API, no changes to existing extractor/parser/schema behavior.

## 12. Open questions

None at design-approval time. Implementation may surface SDK-specific details (FastMCP decorator ergonomics, exact error-mapping API) that are tactical and resolved during the writing-plans step.
