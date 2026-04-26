# CLAUDE.md — FinLit Project Memory

This file is read automatically by Claude Code at the start of every session.
Do not delete or rename it.

---

## What This Project Is

**FinLit** is an open-source Python library for extracting structured,
compliance-ready data from Canadian financial documents (T4, T5, T4A, NR4,
SEDAR filings, bank statements). It is a **pip-installable developer library**
with no UI, no hosted server, and no backend API.

Maintained by **Caseonix** (Waterloo, Ontario, Canada).
GitHub: `https://github.com/caseonix/finlit`

---

## OSS Stack — What We Use and Why

| Layer | Library | Notes |
|---|---|---|
| Document parsing | `docling` | IBM OSS, MIT license. Do NOT replace with PyMuPDF or pdfplumber |
| LLM orchestration | `pydantic-ai` | Model-agnostic. Do NOT use LangChain, LlamaIndex, or direct Anthropic SDK |
| Data validation | `pydantic v2` | Bundled with pydantic-ai |
| PII detection | `presidio-analyzer` + `presidio-anonymizer` | Microsoft OSS. Do NOT write custom regex-only PII detection |
| Schema definitions | `PyYAML` | YAML files in `finlit/schemas/` |
| CLI | `typer` + `rich` | Do NOT use argparse or click |
| Tests | `pytest` + `pytest-asyncio` | |
| Package | `pyproject.toml` PEP 517 | Do NOT use setup.py |
| MCP server | `mcp` (Anthropic OSS) | Optional extra `finlit[mcp]`. Stdio transport only |

**LLM model strings (pydantic-ai convention):**
- Default: `anthropic:claude-sonnet-4-6`
- OpenAI: `openai:gpt-4o`
- Local: `ollama:llama3.2`

---

## Repository Layout

```
finlit/                          ← pip package root
├── __init__.py                  ← public API: DocumentPipeline, BatchPipeline,
│                                  Schema, Field, ExtractionResult, schemas
├── pipeline.py                  ← DocumentPipeline + BatchPipeline
├── schema.py                    ← Schema + Field dataclasses
├── result.py                    ← ExtractionResult dataclass
│
├── schemas/                     ← built-in YAML schema registry
│   ├── __init__.py              ← exposes CRA_T4, CRA_T5, CRA_T4A, CRA_NR4, BANK_STATEMENT
│   ├── cra/
│   │   ├── t4.yaml
│   │   ├── t5.yaml
│   │   ├── t4a.yaml
│   │   └── nr4.yaml
│   └── banking/
│       └── bank_statement.yaml
│
├── parsers/
│   └── docling_parser.py        ← DoclingParser → ParsedDocument
│
├── extractors/
│   ├── base.py                  ← BaseExtractor ABC
│   └── pydantic_ai_extractor.py ← PydanticAIExtractor (Claude / OpenAI / Ollama)
│
├── validators/
│   └── field_validator.py       ← dtype coercion, regex, required field checks
│
├── audit/
│   ├── audit_log.py             ← AuditLog (append-only, finalizable)
│   └── pii.py                   ← CanadianPIIDetector (Presidio + custom recognizers)
│
└── cli/
    └── main.py                  ← typer CLI: `finlit extract`, `finlit schema list`

tests/
├── conftest.py
├── test_schema.py
├── test_pipeline.py
├── test_pii.py
├── test_validator.py
└── fixtures/
    └── sample_t4.txt            ← synthetic T4 text, no real PII

examples/
├── extract_t4.py
├── extract_batch.py
└── custom_schema.py
```

---

## Key Design Rules — Always Follow These

### Public API surface is minimal
Only these names are importable from the top-level `finlit` package:
```python
from finlit import DocumentPipeline, BatchPipeline, Schema, Field, ExtractionResult, schemas
```
Nothing else should leak into `__init__.py`.

### Never make real LLM API calls in tests
Always mock the extractor. Use `unittest.mock.patch` or a stub `BaseExtractor`
subclass that returns synthetic `ExtractionOutput` with hardcoded fields and
confidence scores. Tests must pass with no API keys set.

### Monetary values are always float, never string
The validator coerces `"87,500.00"` → `87500.0`. The LLM prompt instructs
numeric output. Never store dollar amounts as strings in `ExtractionResult.fields`.

### PII flag is an annotation, not automatic redaction
`Field(pii=True)` marks a field as sensitive for downstream consumers.
It does NOT automatically redact the value in `ExtractionResult.fields`.
Redaction is a separate, explicit step via `CanadianPIIDetector.redact()`.

### Audit log is append-only
`AuditLog.log()` raises `RuntimeError` after `finalize()` is called.
`finalize()` is called at the end of `DocumentPipeline.run()`. Never mutate
the log after that point.

### Schemas are YAML-first
All built-in schemas live in `finlit/schemas/**/*.yaml`.
The `finlit/schemas/__init__.py` loads them at import time via `Schema.from_yaml()`.
Never hardcode schema field lists in Python — always load from YAML.

### Do not host anything
FinLit is a library. It has no FastAPI app, no Flask server, no Docker entrypoint,
no cloud deployment. If asked to add a server layer, decline and suggest the
caller use `docling-serve` or build their own thin wrapper.

### MCP server is a thin presentation layer
The `finlit/integrations/mcp/` package wraps `DocumentPipeline` for MCP
hosts. It contains no business logic — only protocol mapping, response
trimming, and a layer-specific PII redaction policy that does NOT change
underlying library behavior. `DocumentPipeline.run()` still returns
un-redacted `ExtractionResult.fields`; the MCP server applies an
explicit, MCP-only redaction step appropriate to the chat-transcript
trust model.

---

## Common Commands

```bash
# Install in editable mode with dev deps
pip install -e ".[dev]"

# spaCy model required for Presidio (run once after install)
python -m spacy download en_core_web_lg

# Run all tests
pytest tests/ -v

# Run tests without network (safe in CI)
pytest tests/ -v --ignore=tests/integration/

# Lint
ruff check finlit/ tests/

# Type check
mypy finlit/

# CLI — list built-in schemas
finlit schema list

# CLI — extract a document (requires ANTHROPIC_API_KEY)
finlit extract my_t4.pdf --schema cra.t4 --extractor claude

# CLI — extract locally with Ollama (no API key needed)
finlit extract my_t4.pdf --schema cra.t4 --extractor ollama

# Build package
python -m build

# Publish to PyPI (maintainers only)
twine upload dist/*
```

---

## Canadian PII Recognizers (Custom — Do Not Remove)

These are registered in `CanadianPIIDetector.__init__()` and extend Presidio's
defaults. Do not remove them — they are the primary compliance differentiator:

| Entity type | Pattern | Example |
|---|---|---|
| `CA_SIN` | `\d{3}-\d{3}-\d{3}` | `123-456-789` |
| `CA_POSTAL_CODE` | `[A-Z]\d[A-Z]\s?\d[A-Z]\d` | `M5V 3A8` |
| `CA_CRA_BN` | `\d{9}\s?(?:RT\|RP\|RC\|RZ)\d{4}` | `123456789RT0001` |

---

## Built-in Schema Registry

| Python name | YAML file | Document type |
|---|---|---|
| `schemas.CRA_T4` | `cra/t4.yaml` | T4 Statement of Remuneration Paid |
| `schemas.CRA_T5` | `cra/t5.yaml` | T5 Statement of Investment Income |
| `schemas.CRA_T4A` | `cra/t4a.yaml` | T4A Pension, Retirement, Annuity |
| `schemas.CRA_NR4` | `cra/nr4.yaml` | NR4 Non-Resident Income |
| `schemas.BANK_STATEMENT` | `banking/bank_statement.yaml` | Generic Canadian bank statement |

To add a new schema: create the YAML in the appropriate subfolder, then add
a `_load()` line in `finlit/schemas/__init__.py`. No other changes needed.

---

## ExtractionResult Shape

```python
result.fields          # dict[str, Any]   — typed, validated field values
result.confidence      # dict[str, float] — per-field confidence 0.0–1.0
result.source_ref      # dict[str, dict]  — {page, bbox, doc} per field
result.pii_entities    # list[dict]       — Presidio detections on raw text
result.audit_log       # list[dict]       — immutable structured event log
result.review_fields   # list[dict]       — fields below review_threshold
result.needs_review    # bool             — True if review_fields is non-empty
result.extracted_field_count  # int       — count of non-None fields
result.extraction_path # str              — "text" or "vision" (which path produced the result)
```

---

## Adding a New Extractor Backend

1. Subclass `BaseExtractor` in `finlit/extractors/`
2. Implement `extract(text, schema) -> ExtractionOutput`
3. Optionally implement `extract_async(text, schema) -> ExtractionOutput`
4. Register a shorthand alias in `_EXTRACTOR_ALIASES` in `pipeline.py`
5. Add a test in `tests/test_pipeline.py`

---

## Relationship to LocalMind Sovereign

FinLit is the **extraction engine**. LocalMind Sovereign (also by Caseonix) is
a **document intelligence platform** that uses FinLit internally.

- LocalMind's Ollama integration → maps to `extractor="ollama"` in FinLit
- LocalMind's PIPEDA/OSFI positioning → same compliance story, shared messaging
- Do NOT couple FinLit code to LocalMind — FinLit must remain fully standalone

---

## What NOT to Do

- Do NOT add a web server, REST API, or any hosted service to this repo
- Do NOT replace `docling` with another PDF parser
- Do NOT replace `pydantic-ai` with direct Anthropic/OpenAI SDK calls
- Do NOT replace `presidio` with hand-rolled regex PII detection
- Do NOT use `setup.py` — use `pyproject.toml` only
- Do NOT store real SINs, CRA numbers, or any real PII in test fixtures
- Do NOT make real LLM API calls in tests
- Do NOT add non-Canadian document schemas to the built-in registry (open a
  separate repo for that)
- Do NOT change the public API surface without updating this file and the README
