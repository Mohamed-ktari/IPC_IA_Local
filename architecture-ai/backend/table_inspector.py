# table_inspector.py — diagnostic script, not production code yet
"""
Run this against a real PDF to see exactly what Docling's structured
table objects look like — column headers, cell spans, page numbers.
This informs how we build the "is this table clean enough to map
deterministically" classifier next.
"""

from pathlib import Path
from docling.document_converter import DocumentConverter


def inspect_tables(pdf_path: str, page_start: int | None = None, page_end: int | None = None):
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    doc = result.document

    print(f"Total tables found: {len(doc.tables)}\n")

    for i, table in enumerate(doc.tables):
        page_no = table.prov[0].page_no if table.prov else None

        if page_start is not None and page_end is not None:
            if page_no is None or not (page_start <= page_no <= page_end):
                continue

        print(f"{'=' * 70}")
        print(f"TABLE {i} — page {page_no}")
        print(f"  num_rows={table.data.num_rows}  num_cols={table.data.num_cols}")
        print(f"{'=' * 70}")

        # Dump every cell with its span info
        for cell in table.data.table_cells:
            span_r = cell.end_row_offset_idx - cell.start_row_offset_idx
            span_c = cell.end_col_offset_idx - cell.start_col_offset_idx
            spanning = " [SPANNING]" if (span_r > 1 or span_c > 1) else ""
            header = " [HEADER]" if cell.column_header else ""
            text_preview = (cell.text or "").replace("\n", " ")[:60]
            print(
                f"  r{cell.start_row_offset_idx}-{cell.end_row_offset_idx} "
                f"c{cell.start_col_offset_idx}-{cell.end_col_offset_idx}"
                f"{header}{spanning} : {text_preview!r}"
            )
        print()


if __name__ == "__main__":
    import sys
    pdf_path = sys.argv[1]
    p_start = int(sys.argv[2]) if len(sys.argv) > 2 else None
    p_end   = int(sys.argv[3]) if len(sys.argv) > 3 else None
    inspect_tables(pdf_path, p_start, p_end)