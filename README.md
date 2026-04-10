# FinLit

FinLit is an open-source Python library for extracting structured, compliance-ready data from Canadian financial documents — T4, T5, T4A, NR4, SEDAR filings, and bank statements. Built on Docling, pydantic-ai, and Microsoft Presidio. Maintained by Caseonix.

## Install

```bash
pip install finlit
python -m spacy download en_core_web_lg
```

## Quickstart

```python
from finlit import DocumentPipeline, schemas

pipeline = DocumentPipeline(schema=schemas.CRA_T4, extractor="claude")
result = pipeline.run("t4_2024.pdf")
print(result.fields)
print(f"Needs review: {result.needs_review}")
```

See `FINLIT_BUILD_PROMPT.md` for full component docs.
