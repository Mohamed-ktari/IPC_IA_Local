from pathlib import Path
import json
import re

from docling.document_converter import DocumentConverter
from docling_core.types.doc import DocItemLabel


INPUT_DIR = Path("/home/mohamed_ktari/IPC_IA_Local/architecture-ai/data/memoires")
OUTPUT_FILE = "toc_headers.json"

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx"}

TOC_MARKERS = [
    "sommaire",
    "table des matières",
    "table de matieres",
    "table de matière",
    "contents",
]

# Example matches:
# 1 Introduction...................3
# 1.1 Context.......................5
# Annexes..........................42
TOC_ENTRY = re.compile(
    r"^(?:\d+(?:\.\d+)*\s+)?(.+?)\s*\.{2,}\s*\d+\s*$"
)


def extract_title(document):
    """Return the first TITLE found."""
    for item, _ in document.iterate_items():
        if getattr(item, "label", None) == DocItemLabel.TITLE:
            return item.text.strip()

    return None


def extract_headers_from_toc(document):
    """Extract section names from the Table of Contents."""

    lines = []

    for item, _ in document.iterate_items():
        if hasattr(item, "text"):
            text = item.text.strip()
            if text:
                lines.append(text)

    collecting = False
    headers = []

    for line in lines:

        lower = line.lower()

        # Start collecting after TOC marker
        if not collecting:
            if any(marker in lower for marker in TOC_MARKERS):
                collecting = True
            continue

        # Ignore empty lines
        if not line:
            continue

        # Stop if we start reading normal paragraphs
        if len(headers) > 3 and len(line.split()) > 20:
            break

        match = TOC_ENTRY.match(line)

        if match:
            title = match.group(1).strip()

            if title not in headers:
                headers.append(title)

    return headers


def process_document(converter, file_path):

    result = converter.convert(str(file_path))
    document = result.document

    return {
        "document": file_path.name,
        "title": extract_title(document),
        "headers": extract_headers_from_toc(document),
    }


def main():

    converter = DocumentConverter()

    files = sorted(
        f for f in INPUT_DIR.iterdir()
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    # Test only first 5 documents
    files = files[:5]

    results = []

    for file in files:
        print(f"Processing {file.name}")

        try:
            results.append(process_document(converter, file))

        except Exception as e:
            results.append({
                "document": file.name,
                "error": str(e)
            })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()