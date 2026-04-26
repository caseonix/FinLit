"""Shared fixtures for finlit.integrations tests.

These tests run DocumentPipeline end-to-end against a stub extractor
and a monkeypatched Docling parser, so no network or LLM calls happen.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from finlit import schemas
from finlit.pipeline import DocumentPipeline

from tests.conftest import StubExtractor  # noqa: F401  (re-exported for tests)


@pytest.fixture
def patch_docling_parser(monkeypatch, synthetic_parsed_document):
    """Replace DoclingParser.parse with a function that returns a canned
    ParsedDocument. Tests that want the default t4 text can just request
    this fixture; tests that need a different ParsedDocument can shadow
    the fixture locally."""
    def _fake_parse(self, path):  # noqa: ARG001
        return synthetic_parsed_document

    monkeypatch.setattr(
        "finlit.parsers.docling_parser.DoclingParser.parse", _fake_parse
    )
    return synthetic_parsed_document


@pytest.fixture
def t4_pipeline(high_confidence_t4_extractor) -> DocumentPipeline:
    """A DocumentPipeline pre-wired with a deterministic T4 stub extractor.

    Tests that need a pipeline but don't care about constructor branches
    in the loader should take this fixture + patch_docling_parser together."""
    return DocumentPipeline(
        schema=schemas.CRA_T4,
        extractor=high_confidence_t4_extractor,
    )


@pytest.fixture
def fake_t4_pdf(tmp_path) -> Path:
    """A temp file path that DoclingParser will never actually read (the
    parser is monkeypatched) but whose existence satisfies os.stat calls."""
    p = tmp_path / "t4.pdf"
    p.write_bytes(b"not a real pdf")
    return p
