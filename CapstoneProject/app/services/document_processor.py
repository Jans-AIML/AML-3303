"""Document text extraction and chunking service.

Supports PDF, TXT, CSV, and DOCX file types.
"""

import csv
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings


def extract_text(file_path: str, file_type: str) -> str:
    """Extract raw text from a document file."""
    if file_type == "pdf":
        return _extract_pdf(file_path)
    if file_type == "txt":
        return _extract_txt(file_path)
    if file_type == "csv":
        return _extract_csv(file_path)
    if file_type == "docx":
        return _extract_docx(file_path)
    if file_type == "json":
        return _extract_elsevier_json(file_path)
    raise ValueError(f"Unsupported file type: {file_type}")


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)
    # Filter out chunks that are too short to be meaningful
    return [c for c in chunks if len(c.split()) >= 20]


# ── Private helpers ───────────────────────────────────────────────────────────


def _clean_pdf_text(text: str) -> str:
    """Remove NIH watermarks, repeated headers, and figure captions from PDF text."""
    text = re.sub(r'(NIH-PA\s*\n?Author\s*\n?Manuscript\s*\n?){1,}', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'NIH Public Access\s*\n?Author Manuscript\s*\n?', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'Q Rev Biophys\.?\s*(Author manuscript)?[^\n]*\n?', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'Author manuscript;\s*available in PMC[^\n]*\n?', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'Published in final edited form[^\n]*\n?', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\n(Figure|Fig\.|Scheme|Table)\s+\d+[^\n]{0,120}\n', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'\n[A-Za-z\s]+et al\.\s+Page\s+\d+\s*\n', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()


def _extract_pdf(path: str) -> str:
    import pdfplumber

    pages: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    raw = "\n\n".join(pages)
    return _clean_pdf_text(raw)


def _extract_txt(path: str) -> str:
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def _extract_csv(path: str) -> str:
    """Convert CSV rows into prose-like chunks (header: value pairs)."""
    lines: list[str] = []
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entry = " | ".join(f"{k}: {v}" for k, v in row.items() if v)
            lines.append(entry)
    return "\n".join(lines)


def _extract_docx(path: str) -> str:
    from docx import Document as DocxDocument

    doc = DocxDocument(path)
    return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())


def extract_elsevier_metadata(path: str) -> dict:
    """Return a flat dict of article-level metadata for ChromaDB storage.

    All values are str or int so ChromaDB accepts them without coercion.
    The ``authors`` field is formatted as "Last1 I., Last2 I." (max 3, then "et al.").
    """
    import json

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    meta = data.get("metadata", {})
    authors_raw = meta.get("authors", [])
    formatted = [
        f"{a.get('last', '')} {a.get('initial', '')}".strip()
        for a in authors_raw[:3]
        if a.get("last")
    ]
    if len(authors_raw) > 3:
        formatted.append("et al.")

    result: dict = {}
    if meta.get("title"):
        result["title"] = str(meta["title"])
    if meta.get("pub_year") is not None:
        result["pub_year"] = int(meta["pub_year"])
    if formatted:
        result["authors"] = ", ".join(formatted)
    if meta.get("doi"):
        result["doi"] = str(meta["doi"])
    areas = meta.get("subjareas", [])
    if areas:
        result["subjareas"] = ", ".join(areas)
    return result


def _extract_elsevier_json(path: str) -> str:
    """Extract readable text from an Elsevier Open Access JSON article."""
    import json

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    meta = data.get("metadata", {})
    parts: list[str] = []

    # ── Metadata header ───────────────────────────────────────
    parts.append(f"Title: {meta.get('title', 'Untitled')}")

    authors = meta.get("authors", [])
    if authors:
        names = [
            f"{a.get('first', '')} {a.get('last', '')}".strip()
            for a in authors[:5]
        ]
        parts.append(f"Authors: {', '.join(names)}")

    if meta.get("pub_year"):
        parts.append(f"Year: {meta['pub_year']}")
    if meta.get("doi"):
        parts.append(f"DOI: {meta['doi']}")

    keywords = meta.get("keywords", [])
    if keywords:
        parts.append(f"Keywords: {', '.join(keywords)}")

    subjareas = meta.get("subjareas", [])
    if subjareas:
        parts.append(f"Subject Areas: {', '.join(subjareas)}")

    parts.append("")

    # ── Abstract ──────────────────────────────────────────────
    abstract = data.get("abstract", "")
    if abstract:
        parts.append("ABSTRACT")
        parts.append(abstract)
        parts.append("")

    # ── Author highlights ─────────────────────────────────────
    highlights = data.get("author_highlights", [])
    if highlights:
        parts.append("HIGHLIGHTS")
        for h in highlights:
            sentence = h.get("sentence", "")
            if sentence:
                parts.append(f"• {sentence}")
        parts.append("")

    # ── Body text grouped by section ──────────────────────────
    current_section: str | None = None
    for entry in data.get("body_text", []):
        section = entry.get("title", "")
        if section and section != current_section:
            current_section = section
            parts.append(f"\n{section.upper()}")
        sentence = entry.get("sentence", "")
        if sentence:
            parts.append(sentence)

    return "\n".join(parts)
