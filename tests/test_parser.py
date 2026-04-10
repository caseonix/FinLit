"""Tests for finlit.parsers.docling_parser.DoclingParser."""
from __future__ import annotations

import pytest

from finlit.parsers.docling_parser import DoclingParser


def test_default_parser_has_ocr_disabled():
    parser = DoclingParser()
    assert parser.ocr is False


def test_parser_stores_ocr_flag_when_enabled():
    parser = DoclingParser(ocr=True)
    assert parser.ocr is True


def test_ocr_parser_configures_pdf_pipeline_with_do_ocr():
    """When ocr=True, the DocumentConverter is built with do_ocr=True on
    the PDF pipeline options. Verify the pipeline_options are wired through
    without actually running the converter."""
    from docling.datamodel.base_models import InputFormat

    parser = DoclingParser(ocr=True)
    # Internal format_options should contain a PDF entry
    fmt_options = parser._converter.format_to_options
    assert InputFormat.PDF in fmt_options
    pdf_opt = fmt_options[InputFormat.PDF]
    assert pdf_opt.pipeline_options is not None
    assert pdf_opt.pipeline_options.do_ocr is True


def test_parse_missing_file_raises():
    parser = DoclingParser()
    with pytest.raises(FileNotFoundError):
        parser.parse("/tmp/__finlit_nonexistent_parser_test__.pdf")
