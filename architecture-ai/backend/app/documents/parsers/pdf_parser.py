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
# Per-page text is now extracted properly using Docling 2.96.1's
# export_to_markdown(page_no=N) API, enabling page-range slicing
# in DocumentCleaner when a human-provided config specifies
# start_page / end_page for the materials section.

from pathlib import Path
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    AcceleratorOptions,
    AcceleratorDevice,
)
from docling.document_converter import PdfFormatOption

# Marker injected between pages — recognised by DocumentCleaner
# for page-range slicing. Chosen to be unambiguous in markdown.
PAGE_BREAK_MARKER = "<!-- page {page_no} -->"

# Singleton converter — expensive to initialize, reuse across calls
_converter: DocumentConverter | None = None


###################### For GPU use case ################################
def _get_converter() -> DocumentConverter:
    global _converter
    if _converter is None:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True

        _converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )
    return _converter


###################### For CPU use case ################################
# def _get_converter() -> DocumentConverter:
#     global _converter
#     if _converter is None:
#         pipeline_options = PdfPipelineOptions()
#         pipeline_options.do_ocr = True
#         pipeline_options.do_table_structure = True
#         pipeline_options.accelerator_options = AcceleratorOptions(
#             num_threads=4,
#             device=AcceleratorDevice.CPU,
#         )
#         _converter = DocumentConverter(
#             format_options={
#                 InputFormat.PDF: PdfFormatOption(
#                     pipeline_options=pipeline_options
#                 )
#             }
#         )
#     return _converter


class PDFParser:

    def parse(self, file_path: str | Path) -> dict:
        """
        Main entry point.
        Returns a dict with:
          - full_text   : complete markdown with <!-- page N --> markers
          - pages       : list of {page, text} — one entry per PDF page
          - total_pages, scanned_pages, digital_pages, char_count, word_count
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF not found: {file_path}")

        converter = _get_converter()
        result    = converter.convert(str(file_path))
        doc       = result.document

        total_pages = doc.num_pages()

        # ── Per-page extraction ───────────────────────────────────────────
        # Export each page individually using Docling 2.96.1's page_no arg.
        # This gives us accurate page boundaries without relying on markers
        # that Docling doesn't emit by default.
        pages_text: list[dict] = []
        page_chunks: list[str] = []

        for page_no in range(1, total_pages + 1):
            page_md = doc.export_to_markdown(page_no=page_no)
            page_md = self._clean_text(page_md)

            pages_text.append({
                "page": page_no,
                "text": page_md,
            })
            # Prefix each chunk with its page marker so DocumentCleaner
            # can split by page number when a config specifies a page range
            page_chunks.append(
                f"{PAGE_BREAK_MARKER.format(page_no=page_no)}\n{page_md}"
            )

        # ── Full text assembly ────────────────────────────────────────────
        # Join all pages with a blank line — markers are embedded above
        full_text = "\n\n".join(page_chunks)

        return {
            "file_name":      file_path.name,
            "total_pages":    total_pages,
            "scanned_pages":  self._count_scanned_pages(result),
            "digital_pages":  total_pages - self._count_scanned_pages(result),
            "full_text":      full_text,
            "pages":          pages_text,
            "char_count":     len(full_text),
            "word_count":     len(full_text.split()),
        }

    def _count_scanned_pages(self, result) -> int:
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