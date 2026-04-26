# FinLit 🍁

**Extract structured data from Canadian financial documents — T4s, T5s, SEDAR filings, bank statements — with a compliance audit trail built in.**

[![PyPI version](https://img.shields.io/pypi/v/finlit.svg)](https://pypi.org/project/finlit/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Built on Docling](https://img.shields.io/badge/parsing-Docling-orange.svg)](https://github.com/docling-project/docling)

```bash
pip install finlit
python -m spacy download en_core_web_lg   # one-time, required by Presidio
export ANTHROPIC_API_KEY=sk-ant-...
```

```python
from finlit import DocumentPipeline, schemas

result = DocumentPipeline(schema=schemas.CRA_T4, extractor="claude").run("t4_2024.pdf")

print(result.fields["box_14_employment_income"])     # → 87500.0
print(result.confidence["box_14_employment_income"]) # → 0.97
print(result.needs_review)                           # → False
```

**Who this is for:**

- Canadian **fintechs** processing user-uploaded T-slips into structured data
- **Banks and credit unions** running SEDAR filing and statement pipelines
- **Accounting and tax software** pre-filling CRA forms from client documents
- Any team that needs **on-premises** extraction with a **PIPEDA/OSFI-friendly audit trail**

Not a developer? See [docs/use-cases.md](docs/use-cases.md) for business context, compliance framing, and "build vs. buy" math.

---

## Contents

- [Why FinLit](#why-finlit)
- [Setup](#setup)
- [Usage](#usage)
  - [Extract a T4](#extract-a-t4)
  - [Batch processing](#batch-processing)
  - [Vision fallback for scans and forms](#vision-fallback-for-scans-and-forms)
  - [Fully local with Ollama](#fully-local-with-ollama)
  - [Custom schemas](#custom-schemas)
  - [Error handling](#error-handling)
- [CLI](#cli)
- [API reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [Built-in schemas](#built-in-schemas)
- [Adding a schema](#adding-a-schema)
- [Compared to alternatives](#compared-to-alternatives)
- [Roadmap](#roadmap)
- [Contributing](#contributing)

---

## Why FinLit

General-purpose extraction tools parse PDFs fine but don't know what a T4 box is, what fields CRA requires, or what a Canadian SIN looks like. FinLit is the Canadian-document layer — pre-built, open-source, on-premises — that you'd otherwise write yourself: versioned CRA schemas, per-field confidence, source traceability, PIPEDA PII detection, and an immutable audit log.

It wraps [Docling](https://github.com/docling-project/docling) (IBM's parser) and [pydantic-ai](https://github.com/pydantic/pydantic-ai) (model-agnostic LLM orchestration). Runs entirely inside your infrastructure — with `extractor="ollama"` even LLM calls stay on-prem, suitable for OSFI-regulated and air-gapped deployments.

---

## Setup

**Install:**

```bash
pip install finlit
python -m spacy download en_core_web_lg
```

The spaCy model is required by Presidio for PII detection. Skipping it will raise an `OSError` on first pipeline run.

**Pick an extractor backend** — set *one* of:

| Backend | Env / setup | Extractor string |
|---|---|---|
| Anthropic Claude (default) | `export ANTHROPIC_API_KEY=...` | `"claude"` |
| OpenAI | `export OPENAI_API_KEY=...` | `"openai"` |
| Local Ollama | [Install Ollama](https://ollama.ai) · `ollama pull llama3.2` | `"ollama"` |

Docling pulls its layout models from HuggingFace on first run (~500MB, cached afterwards).

---

## Usage

### Extract a T4

```python
from finlit import DocumentPipeline, schemas

pipeline = DocumentPipeline(
    schema=schemas.CRA_T4,
    extractor="claude",       # or "openai" or "ollama"
    audit=True,
    review_threshold=0.85,
)

result = pipeline.run("john_doe_t4_2024.pdf")

# Typed, validated fields
print(result.fields["box_14_employment_income"])      # → 87500.0
print(result.fields["province_of_employment"])        # → "ON"

# Per-field confidence — box_52 came back at 71%, below the 0.85 threshold
print(result.confidence["box_52_pension_adjustment"]) # → 0.71
print(result.needs_review)                            # → True
print(result.review_fields)
# [{"field": "box_52_pension_adjustment", "confidence": 0.71, "raw": "..."}]

# Trace any value back to its location in the source PDF
print(result.source_ref["box_14_employment_income"])
# {"page": 1, "bbox": [120, 340, 280, 360], "doc": "john_doe_t4_2024.pdf"}

# Audit log — append-only, finalized at end of run
for ev in result.audit_log:
    print(ev["event"], ev.get("ts"))
# document_loaded ...
# pii_detected ...
# extraction_complete ...
# review_flagged ...
# pipeline_complete ...
```

### Batch processing

```python
from finlit import BatchPipeline, schemas
from glob import glob

batch = BatchPipeline(schema=schemas.CRA_T4, extractor="claude", workers=8)

for path in glob("uploads/*.pdf"):
    batch.add(path)

results = batch.run()
results.export_csv("extracted/t4s_2024.csv")

print(f"Processed:    {results.total}")
print(f"Needs review: {results.review_count}")
```

### Vision fallback for scans and forms

Text extraction fails in two cases: image-only PDFs with no text layer, and form-heavy documents (tax slips, invoices) where 2D column alignment carries meaning. For both, enable the vision fallback — it sends rendered page images to any multimodal LLM.

```python
from finlit import DocumentPipeline, VisionExtractor, schemas

pipeline = DocumentPipeline(
    schema=schemas.CRA_T5,
    extractor="claude",                     # text path (cheap, fast)
    vision_extractor=VisionExtractor(),     # vision fallback (accurate)
)
result = pipeline.run("t5_scanned.pdf")
print(result.extraction_path)               # → "text" or "vision"
```

By default the vision extractor runs only when the text result has `needs_review=True`. Pass a custom callback for finer control:

```python
pipeline = DocumentPipeline(
    schema=schemas.CRA_T5,
    extractor="claude",
    vision_extractor=VisionExtractor(model="openai:gpt-4o"),
    vision_fallback_when=lambda r: any(c < 0.80 for c in r.confidence.values()),
)
```

Vision results **replace** the text result entirely — `result.fields` is whatever the vision extractor returned, and `result.extraction_path == "vision"`. If vision fails for any reason (render error, API failure, LLM error) the pipeline keeps the text result and logs a `vision_fallback_failed` warning.

### Fully local with Ollama

No API keys, no external network — suitable for air-gapped and OSFI-regulated deployments. Text and vision can each be local independently.

```python
pipeline = DocumentPipeline(
    schema=schemas.CRA_T5,
    extractor="ollama:llama3.2",
    vision_extractor=VisionExtractor(model="ollama:qwen2.5vl:7b"),
)
```

Vision models verified against CRA slips:

| Model | Size | Ollama tag | Notes |
|---|---|---|---|
| Qwen2.5-VL | 7B | `ollama:qwen2.5vl:7b` | Strongest on form/document tasks |
| Llama 3.2 Vision | 11B | `ollama:llama3.2-vision` | General-purpose, Meta |
| MiniCPM-V | 8B | `ollama:minicpm-v` | Fast, OpenBMB |

Any pydantic-ai–compatible multimodal model works; these are the ones explicitly tested.

### Custom schemas

```python
from finlit import DocumentPipeline, Schema, Field

loan_schema = Schema(
    name="internal_loan_application",
    fields=[
        Field("applicant_name",  dtype=str,   required=True),
        Field("gross_income",    dtype=float, required=True),
        Field("sin_number",      dtype=str,   pii=True),
        Field("loan_amount",     dtype=float, required=True),
    ]
)

result = DocumentPipeline(schema=loan_schema, extractor="claude").run("loan_app.pdf")
```

### Error handling

FinLit does not raise on low-confidence fields — those go into `result.review_fields`. It *does* attach structured `warnings` for document-level problems (sparse OCR, missing required fields, vision fallback failure).

```python
result = pipeline.run("t4.pdf")

if result.needs_review:
    for flagged in result.review_fields:
        queue_for_human_review(flagged)

for warning in result.warnings:
    if warning["code"] == "sparse_document":
        # PDF had very little extractable text — likely a scan
        ...
    elif warning["code"] == "vision_fallback_failed":
        # Vision path was tried and failed; we kept the text result
        log.warn(warning["reason"])
```

Common warning codes:

| Code | Meaning |
|---|---|
| `sparse_document` | Extracted text is very short; likely an image-only PDF |
| `missing_required_fields` | One or more `required=True` fields came back empty |
| `vision_fallback_failed` | Vision path was attempted and failed; text result retained |
| `pii_detected` | Presidio found PII entities in the source text |

### LangChain integration

FinLit ships a LangChain `BaseLoader` so you can drop extracted Canadian
financial documents straight into RAG pipelines, retrievers, and agents.

Install the extra:

```bash
pip install finlit[langchain]
```

Load one file:

```python
from finlit.integrations.langchain import FinLitLoader

docs = FinLitLoader("t4.pdf", schema="cra.t4").load()
doc = docs[0]
print(doc.metadata["finlit_fields"]["employer_name"])      # "Acme Corp"
print(doc.metadata["finlit_needs_review"])                  # False
```

Batch load with compliance-friendly error surfacing:

```python
loader = FinLitLoader(
    ["t4_001.pdf", "t4_002.pdf", "t4_003.pdf"],
    schema="cra.t4",
    on_error="include",  # failures become Documents with finlit_error
)
docs = loader.load()
# Filter out failures before embedding — empty page_content breaks most embedders.
good = [d for d in docs if not d.metadata.get("finlit_error")]
```

Access the underlying `ExtractionResult` objects via `loader.last_results`
(same order as the input paths, with `None` for skipped/included failures).

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

---

## CLI

```bash
finlit extract t4_2024.pdf --schema cra.t4 --extractor claude
finlit extract t4_2024.pdf --schema cra.t4 --output json
finlit extract t5_scan.pdf --schema cra.t5 --extractor claude \
    --vision-extractor claude
finlit schema list
```

**Flags:**

| Flag | Default | Description |
|---|---|---|
| `--schema` | *required* | Schema name (`cra.t4`, `cra.t5`, …) |
| `--extractor` | `claude` | Text extractor: `claude`, `openai`, `ollama`, or a pydantic-ai model string |
| `--vision-extractor` | *none* | Enable vision fallback. Accepts `claude`/`openai`/`ollama` or a full model string like `ollama:qwen2.5vl:7b` |
| `--output` | `table` | Output format: `table`, `json`, `csv` |
| `--review-threshold` | `0.85` | Confidence below which a field is flagged for review |

---

## API reference

### `DocumentPipeline`

```python
DocumentPipeline(
    schema: Schema,
    extractor: str | BaseExtractor = "claude",
    model: str | None = None,
    vision_extractor: BaseVisionExtractor | None = None,
    vision_fallback_when: Callable[[ExtractionResult], bool] | None = None,
    audit: bool = True,
    review_threshold: float = 0.85,
)
```

`run(path: str | Path) -> ExtractionResult` — parse, extract, validate, audit. Never raises on low confidence; inspect `.needs_review` and `.warnings` instead.

### `ExtractionResult`

```python
result.fields                 # dict[str, Any]   — typed, validated values
result.confidence             # dict[str, float] — 0.0–1.0 per field
result.source_ref             # dict[str, dict]  — {page, bbox, doc} per field
result.pii_entities           # list[dict]       — Presidio detections
result.audit_log              # list[dict]       — immutable event log
result.review_fields          # list[dict]       — fields below threshold
result.needs_review           # bool
result.warnings               # list[dict]       — document-level warnings
result.extracted_field_count  # int
result.extraction_path        # "text" | "vision"
```

### `Schema` and `Field`

```python
Schema(name: str, fields: list[Field], version: str = "1")
Field(
    name: str,
    dtype: type,              # str, int, float, bool, date
    required: bool = False,
    pii: bool = False,        # annotation only — not auto-redacted
    regex: str | None = None,
    description: str = "",
)
```

### Extractor strings

`extractor=` accepts the shorthands `"claude"`, `"openai"`, `"ollama"`, any full pydantic-ai model string (`"anthropic:claude-sonnet-4-6"`, `"ollama:llama3.2"`), or your own `BaseExtractor` instance. `vision_extractor=` takes a `VisionExtractor(model=...)` or any `BaseVisionExtractor` subclass.

```python
from finlit.extractors import BaseExtractor
from finlit import BaseVisionExtractor

class MyTextExtractor(BaseExtractor):
    def extract(self, text, schema): ...

class MyVisionExtractor(BaseVisionExtractor):
    def extract(self, images, schema, text=""): ...

DocumentPipeline(
    schema=schemas.CRA_T4,
    extractor=MyTextExtractor(),
    vision_extractor=MyVisionExtractor(),
)
```

---

## Troubleshooting

**`OSError: [E050] Can't find model 'en_core_web_lg'`**
Presidio needs the spaCy model. Run `python -m spacy download en_core_web_lg` once after install.

**`anthropic.AuthenticationError` / `openai.AuthenticationError`**
`ANTHROPIC_API_KEY` / `OPENAI_API_KEY` is missing or invalid. Check `echo $ANTHROPIC_API_KEY`. These are only read when extraction actually runs — imports and tests never require them.

**`httpx.ConnectError` when using `extractor="ollama"`**
Ollama isn't running or the model isn't pulled. Run `ollama serve` and `ollama pull llama3.2` (or whichever model you passed).

**`warnings` contains `sparse_document`**
The PDF had very little extractable text — almost certainly a scan. Enable vision fallback: `vision_extractor=VisionExtractor()`.

**`warnings` contains `vision_fallback_failed`**
The vision path was attempted and raised. Check the `reason` field — common causes are `render_failed` (pypdfium2 can't rasterize the PDF), `api_error` (network/auth issue with the vision model), or `extraction_failed` (LLM returned an unparseable response). The pipeline keeps the text result when this happens.

**Box values come back in the wrong fields on tax slips**
Form-heavy documents rely on 2D layout that text extraction flattens. Enable `vision_extractor=VisionExtractor()` — the vision model reads the image directly and preserves column alignment.

**First run is slow / downloads lots of data**
Docling pulls ~500MB of layout models from HuggingFace on first use. They are cached locally after that.

---

## Built-in schemas

| Schema | Document | Source |
|---|---|---|
| `schemas.CRA_T4` | T4 Statement of Remuneration Paid | CRA XML spec |
| `schemas.CRA_T5` | T5 Statement of Investment Income | CRA XML spec |
| `schemas.CRA_T4A` | T4A Pension, Retirement, Annuity | CRA XML spec |
| `schemas.CRA_NR4` | NR4 Non-Resident Income | CRA XML spec |
| `schemas.BANK_STATEMENT` | Generic Canadian bank statement | Community |

Each schema is a versioned YAML file inside the package, updated annually when CRA publishes new XML specifications.

---

## Adding a schema

Every schema is a YAML file. To add a new Canadian document type, create the file and register it with one line.

```yaml
# finlit/schemas/cra/t2202.yaml
name: cra_t2202
version: "2024"
document_type: "CRA T2202 Tuition and Enrolment Certificate"
description: >
  Issued by post-secondary institutions to report eligible tuition
  and months of enrolment.

fields:
  - name: institution_name
    dtype: str
    required: true
    description: "Name of the post-secondary institution"

  - name: student_sin
    dtype: str
    required: true
    pii: true
    regex: '^\d{3}-\d{3}-\d{3}$'
    description: "Student's Social Insurance Number"

  - name: eligible_tuition_fees
    dtype: float
    required: true
    description: "Box 1: Total eligible tuition fees paid"

  - name: full_time_months
    dtype: int
    required: false
    description: "Number of months enrolled full-time"
```

```python
# finlit/schemas/__init__.py — add one line
CRA_T2202 = _load("cra/t2202.yaml")
```

Schema contributions are the most useful PRs this project gets. If you know the document, the YAML is the easy part.

---

## Compared to alternatives

| | FinLit | LlamaParse | Docling alone | Textract |
|---|---|---|---|---|
| Canadian document schemas | ✅ | ✗ | ✗ | ✗ |
| Runs on-premises | ✅ | ✗ SaaS only | ✅ | ✗ AWS only |
| Confidence per field | ✅ | Partial | ✗ | Partial |
| Source traceability | ✅ | Partial | ✗ | Partial |
| PIPEDA PII detection | ✅ | ✗ | ✗ | ✗ |
| Audit log | ✅ | ✗ | ✗ | ✗ |
| Custom schemas | ✅ | ✗ | ✗ | ✗ |
| Vision fallback for scans | ✅ | Partial | ✗ | ✅ |
| Open-source | ✅ | ✗ | ✅ | ✗ |

---

## Roadmap

- [x] Core extraction pipeline (Docling + pydantic-ai)
- [x] CRA schema registry (T4, T5, T4A, NR4)
- [x] Source traceability and audit log
- [x] PIPEDA PII detection — SIN, CRA BNs, postal codes
- [x] CLI
- [x] OCR auto-fallback for image-only PDFs (v0.2)
- [x] Document-level warnings for sparse and missing-required-field results (v0.2)
- [x] Vision extraction fallback — Claude, OpenAI, Gemini, or local OSS via Ollama (v0.3)
- [ ] SEDAR filing schemas (MD&A, AIF, financial statements)
- [ ] Bank statement schemas (RBC, TD, Scotiabank, BMO, CIBC)
- [ ] Accuracy benchmarks per schema
- [x] LangChain reader integration
- [ ] LlamaIndex reader integration
- [x] MCP tool definitions for agentic workflows
- [ ] French CRA form support

---

## Contributing

Open issues and PRs are welcome. If you work in a regulated Canadian industry and need a document type that is not yet here, open an issue with the document name and the fields you need.

See [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup.

---

## License

Apache 2.0. See [LICENSE](LICENSE).

---

Built by [Caseonix](https://caseonix.ca) · Waterloo, Ontario 🍁

*FinLit is the extraction engine inside [LocalMind Sovereign](https://localmind.caseonix.ca), Caseonix's document intelligence platform for Canadian regulated industries.*
