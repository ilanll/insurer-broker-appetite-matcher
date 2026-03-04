"""PDF text extraction utilities."""

from __future__ import annotations

import io
from typing import Optional

from PyPDF2 import PdfReader


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extract all text from a PDF file.

    Args:
        pdf_bytes: Raw bytes of the PDF file.

    Returns:
        Concatenated text from all pages.
    """
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages_text: list[str] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages_text.append(f"--- Page {i + 1} ---\n{text}")
    return "\n\n".join(pages_text)


def chunk_text(text: str, chunk_size: int = 4000, overlap: int = 200) -> list[str]:
    """
    Split text into overlapping chunks for LLM processing.

    Args:
        text: Full text to chunk.
        chunk_size: Target characters per chunk.
        overlap: Overlap between consecutive chunks.

    Returns:
        List of text chunks.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        # Try to break at a paragraph or sentence boundary
        if end < len(text):
            # Look for a paragraph break
            newline_pos = text.rfind("\n\n", start + chunk_size // 2, end)
            if newline_pos != -1:
                end = newline_pos + 2
            else:
                # Fall back to sentence boundary
                period_pos = text.rfind(". ", start + chunk_size // 2, end)
                if period_pos != -1:
                    end = period_pos + 2
        chunks.append(text[start:end].strip())
        start = end - overlap
    return chunks
