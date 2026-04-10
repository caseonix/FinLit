"""
Example: Batch extract T4 slips from a directory.

Usage:
  python examples/extract_batch.py ./t4_folder
"""
import sys
from pathlib import Path

from finlit import BatchPipeline, schemas

folder = Path(sys.argv[1])
pdfs = sorted(folder.glob("*.pdf"))

batch = BatchPipeline(schema=schemas.CRA_T4, extractor="claude", workers=4)
for p in pdfs:
    batch.add(p)

result = batch.run()
print(f"Processed {result.total} documents, {result.review_count} need review")

result.export_csv("t4_output.csv")
result.export_jsonl("t4_output.jsonl")
print("Wrote t4_output.csv and t4_output.jsonl")
