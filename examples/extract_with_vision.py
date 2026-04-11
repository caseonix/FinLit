"""
Minimal example: extract a T5 with Claude vision fallback.

Requires ANTHROPIC_API_KEY in the environment. The text path runs first;
if its result has needs_review=True, the pipeline re-runs the extraction
through Claude Sonnet 4.6 multimodal using rendered page images.

Run:
    ANTHROPIC_API_KEY=sk-... python examples/extract_with_vision.py path/to/slip.pdf
"""
from __future__ import annotations

import sys

from finlit import DocumentPipeline, VisionExtractor, schemas


def main(path: str) -> None:
    pipeline = DocumentPipeline(
        schema=schemas.CRA_T5,
        extractor="claude",
        vision_extractor=VisionExtractor(model="anthropic:claude-sonnet-4-6"),
    )
    result = pipeline.run(path)

    print(f"Path taken     : {result.extraction_path}")
    print(f"Needs review   : {result.needs_review}")
    print(f"Warnings       : {[w['code'] for w in result.warnings]}")
    print()
    print("Extracted fields:")
    for name, value in result.fields.items():
        conf = result.confidence.get(name, 0.0)
        marker = " (None)" if value is None else ""
        print(f"  {name:55s} = {str(value):25s}  conf={conf:.2f}{marker}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: extract_with_vision.py <document.pdf>")
        sys.exit(1)
    main(sys.argv[1])
