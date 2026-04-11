# FinLit 🍁

**Extract structured data from Canadian financial documents — T4s, T5s, SEDAR filings, bank statements — with a compliance audit trail built in.**

```bash
pip install finlit
```

```python
from finlit import DocumentPipeline, schemas

result = DocumentPipeline(schema=schemas.CRA_T4, extractor="claude").run("t4_2024.pdf")

print(result.fields["box_14_employment_income"])     # → 87500.0
print(result.confidence["box_14_employment_income"]) # → 0.97
print(result.needs_review)                           # → False
```

---

[![PyPI version](https://img.shields.io/pypi/v/finlit.svg)](https://pypi.org/project/finlit/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Built on Docling](https://img.shields.io/badge/parsing-Docling-orange.svg)](https://github.com/docling-project/docling)

---

## Why this exists

You are building a Canadian fintech app. Users upload their T4s. You need the numbers out — reliably, with a traceable record of where each value came from, and without sending SINs to a US cloud API.

General-purpose extraction tools handle the PDF parsing fine. They do not know what a T4 box is, what fields CRA requires, or what a Canadian SIN looks like. So you write that layer yourself. Every team building on Canadian documents writes it themselves, slightly differently, with no audit trail.

FinLit is that layer, pre-built and open-source.

---

## What it adds on top of Docling

FinLit wraps [Docling](https://github.com/docling-project/docling) (IBM's document parser) and [pydantic-ai](https://github.com/pydantic/pydantic-ai) (model-agnostic LLM orchestration) with the things regulated-industry developers actually need:

- **Canadian document schema registry** — T4, T5, T4A, NR4, SEDAR MD&A, bank statements. Each schema is a versioned YAML file built from CRA XML specifications.
- **Field-level confidence scores** — every extracted value gets a 0–1 confidence rating. Fields below your threshold go into a review queue rather than silently passing through.
- **Source traceability** — every field links back to the page and bounding box it came from.
- **PIPEDA PII detection** — [Microsoft Presidio](https://github.com/microsoft/presidio) scans for SINs, CRA business numbers, and postal codes before extraction runs. Detections go into the audit log.
- **Immutable audit log** — every run produces a structured, append-only event log: document loaded, PII found, fields extracted, review flags raised.
- **LLM-agnostic** — Claude, OpenAI, or a local Ollama model. Same API, one argument to swap.

It runs entirely inside your own infrastructure. No documents, extracted values, or audit logs leave your environment. With `extractor="ollama"` even the LLM calls stay on-premises — no API keys, no data residency questions, suitable for OSFI-regulated and air-gapped deployments.

---

## Quickstart

```bash
pip install finlit

# Presidio needs a spaCy model — one-time setup
python -m spacy download en_core_web_lg
```

### Extract a T4

```python
from finlit import DocumentPipeline, schemas

pipeline = DocumentPipeline(
    schema=schemas.CRA_T4,
    extractor="claude",      # or "openai" or "ollama"
    audit=True,
    review_threshold=0.85,
)

result = pipeline.run("john_doe_t4_2024.pdf")

# Typed, validated fields
print(result.fields["box_14_employment_income"])      # → 87500.0
print(result.fields["box_22_income_tax_deducted"])    # → 21340.0
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
print(result.audit_log)
# [
#   {"event": "document_loaded",     "sha256": "abc...", "ts": "..."},
#   {"event": "pii_detected",        "count": 1, "entities": ["CA_SIN"], "ts": "..."},
#   {"event": "extraction_complete", "fields_returned": 13, "ts": "..."},
#   {"event": "review_flagged",      "count": 1, "fields": ["box_52_pension_adjustment"], "ts": "..."},
#   {"event": "pipeline_complete",   "fields_extracted": 13, "ts": "..."}
# ]
```

### Run fully local with Ollama

```python
pipeline = DocumentPipeline(
    schema=schemas.CRA_T4,
    extractor="ollama",
    model="llama3.2",
)
result = pipeline.run("t4.pdf")
# No API keys. No external calls.
```

### Use vision extraction for scanned PDFs and form layouts

Text extraction fails in two cases: image-only PDFs with no text layer, and form-heavy documents (tax slips, invoices) where 2D column alignment carries meaning. For both, FinLit v0.3 ships an opt-in vision fallback that sends rendered page images to any multimodal LLM.

```python
from finlit import DocumentPipeline, VisionExtractor, schemas

pipeline = DocumentPipeline(
    schema=schemas.CRA_T5,
    extractor="claude",                                    # text path (cheap, fast)
    vision_extractor=VisionExtractor(),                    # vision fallback (accurate)
)
result = pipeline.run("t5_scanned.pdf")
print(result.extraction_path)   # → "text" or "vision"
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

### Running fully locally with open-source vision models

Vision extraction is model-agnostic. Any multimodal model pydantic-ai supports works — including fully-local open-source models via Ollama. No API keys, no external network, suitable for air-gapped deployments.

```python
pipeline = DocumentPipeline(
    schema=schemas.CRA_T5,
    extractor="ollama:llama3.2",
    vision_extractor=VisionExtractor(model="ollama:qwen2.5vl:7b"),
)
```

Tested open-source vision models:

| Model | Size | Ollama tag | Notes |
|---|---|---|---|
| Qwen2.5-VL | 7B | `ollama:qwen2.5vl:7b` | Strongest on form/document tasks |
| Llama 3.2 Vision | 11B | `ollama:llama3.2-vision` | General-purpose, Meta |
| MiniCPM-V | 8B | `ollama:minicpm-v` | Fast, OpenBMB |

Any pydantic-ai–compatible multimodal model will work — these are the ones that have been verified against CRA slips.

### Custom schema for your own documents

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

---

## CLI

```bash
finlit extract t4_2024.pdf --schema cra.t4 --extractor claude
finlit extract t4_2024.pdf --schema cra.t4 --output json
finlit schema list
```

```
┌──────────────────────────────────────────────────────────────┐
│ Extraction: t4_2024.pdf                                       │
├─────────────────────────────┬──────────┬────────────┬────────┤
│ Field                       │ Value    │ Confidence │ Review │
├─────────────────────────────┼──────────┼────────────┼────────┤
│ employer_name               │ Acme Corp│ 99%        │        │
│ tax_year                    │ 2024     │ 99%        │        │
│ box_14_employment_income    │ 87500.0  │ 97%        │        │
│ box_16_cpp_contributions    │ 3754.45  │ 95%        │        │
│ box_18_ei_premiums          │ 1049.12  │ 95%        │        │
│ box_22_income_tax_deducted  │ 21340.0  │ 96%        │        │
│ box_52_pension_adjustment   │ 4200.0   │ 71%        │ ⚠      │
└─────────────────────────────┴──────────┴────────────┴────────┘

⚠ 1 field(s) flagged for review
```

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

## LLM backends

```python
# Anthropic Claude (default)
DocumentPipeline(schema=schemas.CRA_T4, extractor="claude")

# OpenAI
DocumentPipeline(schema=schemas.CRA_T4, extractor="openai", model="gpt-4o")

# Fully local — no external calls
DocumentPipeline(schema=schemas.CRA_T4, extractor="ollama", model="llama3.2")

# Vision fallback (any multimodal model)
from finlit import VisionExtractor
DocumentPipeline(
    schema=schemas.CRA_T4,
    extractor="claude",
    vision_extractor=VisionExtractor(model="ollama:qwen2.5vl:7b"),
)

# Your own
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

## How it fits in your stack

FinLit has no server, dashboard, or hosted tier. It runs wherever your Python runs.

```
# Fintech T4 upload
User uploads PDF → your backend → FinLit → structured JSON → your database

# Bank batch pipeline
S3 bucket of SEDAR filings → BatchPipeline (8 workers) → CSV → data warehouse

# Accounting software, on-prem
Client portal upload → FinLit + Ollama → validated fields → CRA form pre-fill
```

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
| Human review queue | ✅ | ✗ | ✗ | ✗ |
| Open-source | ✅ | ✗ | ✅ | ✗ |

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

## Requirements

- Python 3.10+
- `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or a running [Ollama](https://ollama.ai) instance
- `python -m spacy download en_core_web_lg` (one-time, required by Presidio)

Docling pulls its layout models from HuggingFace on first run (~500MB). After that they are cached locally.

---

## Contributing

Open issues and PRs are welcome. If you work in a regulated Canadian industry and need a document type that is not yet here, open an issue with the document name and the fields you need.

See [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup.

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
- [ ] LangChain and LlamaIndex reader integrations
- [ ] MCP tool definitions for agentic workflows
- [ ] French CRA form support

---

## License

Apache 2.0. See [LICENSE](LICENSE).

---

Built by [Caseonix](https://caseonix.com) · Waterloo, Ontario 🍁

*FinLit is the extraction engine inside [LocalMind Sovereign](https://caseonix.com/localmind), Caseonix's document intelligence platform for Canadian regulated industries.*
