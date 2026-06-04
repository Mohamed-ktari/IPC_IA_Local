# pdf_parser.py
# Extracts text from PDF files using Docling.
#
# Docling handles automatically:
#   - Digital PDFs (text extraction)
#   - Scanned PDFs (OCR with layout awareness)
#   - Multi-column layouts (common in technical reports)
#   - Tables (converts them to readable text)
#   - Mixed French/English documents
#
# Replaces the previous pdfplumber + pytesseract implementation.
# The interface (parse() method and return format) is identical —
# nothing outside this file needs to change.

from pathlib import Path
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    AcceleratorOptions,
    AcceleratorDevice,
)
from docling.document_converter import PdfFormatOption


# Singleton converter — expensive to initialize, reuse across calls
_converter: DocumentConverter | None = None

###################### For GPU use case ################################
# def _get_converter() -> DocumentConverter:
#     # Lazy initialization — only created on first parse call
#     # OCR is enabled by default in Docling for scanned pages
#     global _converter
#     if _converter is None:
#         pipeline_options = PdfPipelineOptions()
#         pipeline_options.do_ocr = True              # enable OCR for scanned pages
#         pipeline_options.do_table_structure = True  # extract tables as structured text
#
#         _converter = DocumentConverter(
#             format_options={
#                 InputFormat.PDF: PdfFormatOption(
#                     pipeline_options=pipeline_options
#                 )
#             }
#         )
#     return _converter


###################### For CPU use case ################################
def _get_converter() -> DocumentConverter:
    global _converter
    if _converter is None:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True

        # Force CPU — GPU is reserved for Ollama (the LLM)
        # On the production server revisit this once you measure VRAM usage
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=4,
            device=AcceleratorDevice.CPU,
        )

        _converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )
    return _converter


class PDFParser:

    def parse(self, file_path: str | Path) -> dict:
        # Main entry point — identical interface to previous implementation.
        # Returns a dict with full text and metadata.
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF not found: {file_path}")


        converter = _get_converter()

        # Docling does all the heavy lifting here —
        # OCR, layout detection, table extraction, language handling
        result = converter.convert(str(file_path))
        doc = result.document

        # Export to markdown — preserves structure (headings, tables, lists)
        # much cleaner than plain text for LLM consumption
        full_text = doc.export_to_markdown()
        full_text = self._clean_text(full_text)

        # Per-page breakdown for metadata
        pages_text = self._extract_pages(doc, full_text)

        return {
            "file_name": file_path.name,
            "total_pages": len(doc.pages),
            "scanned_pages": self._count_scanned_pages(result),
            "digital_pages": len(doc.pages) - self._count_scanned_pages(result),
            "full_text": full_text,
            "pages": pages_text,
            "char_count": len(full_text),
            "word_count": len(full_text.split()),
        }

    def _extract_pages(self, doc, full_text: str) -> list[dict]:
        # Docling processes the whole document at once,
        # so we approximate page breakdown from the full text
        # This is used for metadata only — agents always receive full_text
        pages = []
        for i, page in enumerate(doc.pages.values(), start=1):
            pages.append({
                "page": i,
                "text": f"[see full_text — page {i} of {len(doc.pages)}]",
            })
        return pages

    def _count_scanned_pages(self, result) -> int:
        # Count pages where Docling used OCR
        try:
            scanned = 0
            for page in result.document.pages.values():
                if hasattr(page, 'predictions') and page.predictions:
                    scanned += 1
            return scanned
        except Exception:
            return 0

    def _clean_text(self, text: str) -> str:
        import re
        text = re.sub(r'<!--\s*image\s*-->', '', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

# Convenience function — used everywhere outside this file
def parse_pdf(file_path: str | Path) -> dict:
    return PDFParser().parse(file_path)