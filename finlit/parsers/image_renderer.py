"""
Render PDFs and image files to PNG bytes for vision-based extraction.

Standalone utility — no dependency on Docling, the pipeline, or any
extractor. Can be used directly by a consumer if they want to pre-render
documents themselves.

PDFs are rasterized page-by-page via pypdfium2 (Google's PDFium engine,
already in the dependency tree via Docling). Image files (.png, .jpg,
.jpeg) are returned as a single-element list of the raw file bytes with
no re-encoding — vision models accept them directly.
"""
from __future__ import annotations

import io
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image

_PDF_SUFFIXES = {".pdf"}
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def render_pages(path: str | Path, dpi: int = 200) -> list[bytes]:
    """Render a document to a list of PNG-encoded page images.

    Parameters
    ----------
    path:
        Path to a PDF or image file. Must exist.
    dpi:
        Resolution used when rasterizing PDF pages. Default 200 — a
        balance between OCR legibility on small box numbers and image
        token cost. Ignored for image inputs (no re-encoding).

    Returns
    -------
    list[bytes]
        One entry per page, each a PNG byte string.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the file extension is not .pdf, .png, .jpg, or .jpeg.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"image_renderer: file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in _IMAGE_SUFFIXES:
        return [path.read_bytes()]

    if suffix in _PDF_SUFFIXES:
        return _render_pdf(path, dpi)

    raise ValueError(
        f"image_renderer: unsupported format {suffix!r}. "
        f"Supported: .pdf, .png, .jpg, .jpeg"
    )


def _render_pdf(path: Path, dpi: int) -> list[bytes]:
    """Render every page of a PDF to PNG bytes at the given DPI.

    Native ``FPDF_PAGE`` handles are closed eagerly after each page is
    encoded so memory usage stays flat regardless of page count.
    """
    scale = dpi / 72.0  # pypdfium2 scale is relative to 72 DPI
    out: list[bytes] = []
    pdf = pdfium.PdfDocument(str(path))
    try:
        for i in range(len(pdf)):
            page = pdf.get_page(i)
            try:
                bitmap = page.render(scale=scale)
                pil_image: Image.Image = bitmap.to_pil()
                # Force a copy out of pypdfium2's shared buffer so the
                # PIL image remains valid after the bitmap is freed.
                pil_image.load()
                buf = io.BytesIO()
                pil_image.save(buf, format="PNG")
                out.append(buf.getvalue())
            finally:
                page.close()
    finally:
        pdf.close()
    return out
