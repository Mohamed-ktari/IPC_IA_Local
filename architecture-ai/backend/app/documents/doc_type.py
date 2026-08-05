# doc_type.py
# Enum for the kinds of documents that flow through ingestion.
# Mirrors how base.py defines Role as a str-Enum — same reasoning: type
# safety at the call site, while still serializing as a plain string in
# JSON (metadata.json, Chroma metadata, API responses) without needing
# a custom encoder.

from enum import Enum


class DocType(str, Enum):
    memoire = "memoire"
    programme = "programme"
    rc = "rc"
    unspecified = "unspecified"