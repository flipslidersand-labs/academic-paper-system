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
            # extract_words() guarantees a space between every token, which avoids
            # the word-merging artefact that extract_text() produces on 2-column PDFs
            # (e.g. 'KVcachesprecomputedoffline' instead of 'KV caches precomputed offline').
            words = page.extract_words(x_tolerance=3, y_tolerance=3)
            text = " ".join(w["text"] for w in words)
            if text.strip():
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
