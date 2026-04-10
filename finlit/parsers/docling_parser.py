"""
Wraps Docling's DocumentConverter to extract text and layout
from PDF, DOCX, and image files.

Returns a ParsedDocument with:
  - full_text: str
  - tables: list of dict rows
  - metadata: filename, sha256, num_pages
  - source_path
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from docling.document_converter import DocumentConverter


@dataclass
class ParsedDocument:
    full_text: str
    tables: list[dict]
    metadata: dict
    source_path: str


class DoclingParser:
    """Local-only document parser built on Docling.

    Parameters
    ----------
    ocr:
        When True, enables Docling's OCR pipeline for PDF inputs. Required
        for scanned / image-only PDFs that have no embedded text layer.
        Default is False (faster, works for native text PDFs).
    """

    def __init__(self, ocr: bool = False) -> None:
        self.ocr = ocr
        if ocr:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import PdfFormatOption

            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = True
            self._converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options
                    )
                }
            )
        else:
            self._converter = DocumentConverter()

    def parse(self, path: str | Path) -> ParsedDocument:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

        result = self._converter.convert(str(path))
        doc = result.document

        full_text = doc.export_to_markdown()

        tables: list[dict] = []
        for table in getattr(doc, "tables", []) or []:
            try:
                df = table.export_to_dataframe()
                tables.append(df.to_dict(orient="records"))
            except Exception:
                # Table export is best-effort; unreadable tables are skipped.
                pass

        metadata = {
            "source": str(path),
            "sha256": sha256,
            "filename": path.name,
            "num_pages": len(doc.pages) if hasattr(doc, "pages") else None,
        }

        return ParsedDocument(
            full_text=full_text,
            tables=tables,
            metadata=metadata,
            source_path=str(path),
        )
