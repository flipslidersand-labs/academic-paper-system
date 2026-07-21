"""PDF text extraction and file hashing.

Extracts text content from PDF files with page tracking,
and provides file integrity verification via SHA-256 hashing.
"""

import hashlib
import pdfplumber


def extract_text(pdf_path: str) -> list[dict]:
    """Extract text from PDF with page numbers.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of dicts with "page" (int) and "text" (str) keys.
        Pages with no text are skipped.
    """
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                pages.append({"page": page_num, "text": text})
    return pages


def hash_file(path: str) -> str:
    """Calculate SHA-256 hash of a file for duplicate detection.

    Args:
        path: Path to the file.

    Returns:
        Hexadecimal SHA-256 hash string.
    """
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()
