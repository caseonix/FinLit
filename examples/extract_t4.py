"""
Example: Extract a T4 slip using Claude (cloud) or Ollama (local).

Usage:
  ANTHROPIC_API_KEY=... python examples/extract_t4.py my_t4.pdf
  python examples/extract_t4.py my_t4.pdf --local   # uses Ollama
"""
import sys

from finlit import DocumentPipeline, schemas

local = "--local" in sys.argv
path = next(a for a in sys.argv[1:] if not a.startswith("--"))

pipeline = DocumentPipeline(
    schema=schemas.CRA_T4,
    extractor="ollama" if local else "claude",
    audit=True,
    review_threshold=0.85,
)

result = pipeline.run(path)

print("\n=== Extracted Fields ===")
review_set = {r["field"] for r in result.review_fields}
for field_name, value in result.fields.items():
    conf = result.confidence.get(field_name, 0.0)
    flag = " review" if field_name in review_set else ""
    print(f"  {field_name:45s} {str(value):20s} ({conf:.0%}){flag}")

print("\n=== Summary ===")
print(f"  Fields extracted: {result.extracted_field_count}/{len(result.fields)}")
print(f"  Needs review:     {result.needs_review}")
print(f"  PII detected:     {len(result.pii_entities)} entities")
