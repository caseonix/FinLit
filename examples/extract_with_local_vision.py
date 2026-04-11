"""
Fully-local example: extract a T5 with Ollama + Qwen2.5-VL, no API keys.

Prerequisites:
    1. Install Ollama:   https://ollama.ai
    2. Pull the models:  ollama pull llama3.2
                         ollama pull qwen2.5vl:7b
    3. Start the Ollama server (if not already running): ollama serve

Run:
    python examples/extract_with_local_vision.py path/to/slip.pdf

Zero API keys. Zero external network. Pure open-source.
"""
from __future__ import annotations

import sys

from finlit import DocumentPipeline, VisionExtractor, schemas


def main(path: str) -> None:
    pipeline = DocumentPipeline(
        schema=schemas.CRA_T5,
        extractor="ollama:llama3.2",
        vision_extractor=VisionExtractor(model="ollama:qwen2.5vl:7b"),
    )
    result = pipeline.run(path)

    print(f"Path taken     : {result.extraction_path}")
    print(f"Needs review   : {result.needs_review}")
    print(f"Warnings       : {[w['code'] for w in result.warnings]}")
    print()
    print("Extracted fields:")
    for name, value in result.fields.items():
        conf = result.confidence.get(name, 0.0)
        print(f"  {name:55s} = {str(value):25s}  conf={conf:.2f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: extract_with_local_vision.py <document.pdf>")
        sys.exit(1)
    main(sys.argv[1])
