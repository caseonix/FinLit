"""Tests for finlit.parsers.image_renderer.render_pages()."""
from __future__ import annotations

from pathlib import Path

import pypdfium2 as pdfium
import pytest
from PIL import Image

from finlit.parsers.image_renderer import render_pages


PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _write_blank_pdf(path: Path, num_pages: int = 1) -> None:
    """Generate a tiny blank PDF at `path` using pypdfium2."""
    pdf = pdfium.PdfDocument.new()
    try:
        for _ in range(num_pages):
            pdf.new_page(612, 792)  # US-letter points
        pdf.save(str(path))
    finally:
        pdf.close()


def test_render_pdf_returns_png_bytes(tmp_path: Path):
    pdf_path = tmp_path / "blank.pdf"
    _write_blank_pdf(pdf_path, num_pages=1)

    images = render_pages(pdf_path)

    assert isinstance(images, list)
    assert len(images) == 1
    assert images[0].startswith(PNG_MAGIC)


def test_render_respects_dpi(tmp_path: Path):
    """Higher DPI must produce an image with larger pixel dimensions."""
    import io as _io

    pdf_path = tmp_path / "blank.pdf"
    _write_blank_pdf(pdf_path, num_pages=1)

    low = render_pages(pdf_path, dpi=72)
    high = render_pages(pdf_path, dpi=200)

    low_w, low_h = Image.open(_io.BytesIO(low[0])).size
    high_w, high_h = Image.open(_io.BytesIO(high[0])).size
    assert high_w > low_w
    assert high_h > low_h


def test_render_multipage_pdf(tmp_path: Path):
    pdf_path = tmp_path / "three_pages.pdf"
    _write_blank_pdf(pdf_path, num_pages=3)

    images = render_pages(pdf_path)

    assert len(images) == 3
    for img in images:
        assert img.startswith(PNG_MAGIC)


def test_render_png_input_passthrough(tmp_path: Path):
    """A .png input is returned as a single-element list of the raw bytes
    with no re-encoding."""
    png_path = tmp_path / "pic.png"
    Image.new("RGB", (100, 100), color="white").save(png_path, "PNG")
    original_bytes = png_path.read_bytes()

    images = render_pages(png_path)

    assert len(images) == 1
    assert images[0] == original_bytes


def test_render_jpg_input_passthrough(tmp_path: Path):
    jpg_path = tmp_path / "pic.jpg"
    Image.new("RGB", (100, 100), color="white").save(jpg_path, "JPEG")
    original_bytes = jpg_path.read_bytes()

    images = render_pages(jpg_path)

    assert len(images) == 1
    assert images[0] == original_bytes


def test_render_file_not_found_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        render_pages(tmp_path / "does_not_exist.pdf")


def test_render_unsupported_format_raises(tmp_path: Path):
    txt = tmp_path / "notes.txt"
    txt.write_text("hello")
    with pytest.raises(ValueError, match="unsupported"):
        render_pages(txt)
