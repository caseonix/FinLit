# FinLit Quickstart

Extract structured data from a Canadian T4 in five minutes. For the full reference, see [README.md](../README.md).

## 1. Install

```bash
pip install finlit
python -m spacy download en_core_web_lg   # one-time, required by Presidio
```

The spaCy model is ~500MB; skipping it raises `OSError` on first run.

## 2. Pick an extractor backend

Set **one** of the following before running anything:

| Backend                  | Setup                                                                   | Extractor string |
|--------------------------|-------------------------------------------------------------------------|------------------|
| Anthropic Claude         | `export ANTHROPIC_API_KEY=sk-ant-...`                                   | `"claude"`       |
| OpenAI                   | `export OPENAI_API_KEY=sk-...`                                          | `"openai"`       |
| Ollama (local, no key)   | [Install Ollama](https://ollama.ai) and `ollama pull llama3.2`          | `"ollama"`       |

## 3. Extract your first T4

```python
from finlit import DocumentPipeline, schemas

pipeline = DocumentPipeline(schema=schemas.CRA_T4, extractor="claude")
result = pipeline.run("john_doe_t4_2024.pdf")

print(result.fields["box_14_employment_income"])   # → 87500.0
print(result.confidence["box_14_employment_income"]) # → 0.97
print(result.needs_review)                           # → False
```

What you get back on `result`:

- `fields` — typed, validated values keyed by schema field name
- `confidence` — per-field 0.0–1.0 score
- `source_ref` — page + bounding box per field, for traceability
- `audit_log` — append-only event trail (parse, PII scan, extraction, validation)
- `needs_review` / `review_fields` — fields below the confidence threshold

## 4. Use it from the CLI

```bash
finlit schema-list                                  # see available schemas
finlit extract t4_2024.pdf --schema cra.t4          # extract one document
finlit extract scan.pdf  --schema cra.t5 \
    --vision-extractor claude                       # text + vision fallback
```

## 5. Where to go next

- **Batch processing, error handling, custom schemas** → [README — Usage](../README.md#usage)
- **LangChain integration** for RAG over Canadian docs → [README — LangChain integration](../README.md#langchain-integration)
- **MCP server** for Claude Desktop / Cursor / agentic workflows → [README — MCP server](../README.md#mcp-server)
- **Vision fallback** for scanned PDFs and form-heavy documents → [README — Vision fallback](../README.md#vision-fallback-for-scans-and-forms)
- **Fully local with Ollama** for air-gapped or PIPEDA-sensitive deployments → [README — Fully local with Ollama](../README.md#fully-local-with-ollama)
