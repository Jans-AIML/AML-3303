"""Document text extraction and chunking service.

Supports PDF, TXT, CSV, and DOCX file types.
"""

import csv
import io

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
    raise ValueError(f"Unsupported file type: {file_type}")


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


# ── Private helpers ───────────────────────────────────────────────────────────

def _extract_pdf(path: str) -> str:
    import pdfplumber

    pages: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n\n".join(pages)


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
