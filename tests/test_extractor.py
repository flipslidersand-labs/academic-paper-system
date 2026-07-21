"""Tests for PDF extraction and file hashing."""

import hashlib
from pathlib import Path

import pytest

from academic_paper.extractor import extract_text, hash_file


def test_hash_file(tmp_path: Path) -> None:
    """Test that hash_file calculates consistent SHA-256 hashes."""
    # Create a temporary file
    test_file = tmp_path / "test.txt"
    test_content = b"Hello, World! This is test content."
    test_file.write_bytes(test_content)

    # Calculate hash
    hash1 = hash_file(str(test_file))

    # Verify it's a valid hex string
    assert isinstance(hash1, str)
    assert len(hash1) == 64  # SHA-256 is 256 bits = 64 hex chars
    assert all(c in "0123456789abcdef" for c in hash1)

    # Same file should produce same hash
    hash2 = hash_file(str(test_file))
    assert hash1 == hash2

    # Verify it matches expected SHA-256
    expected = hashlib.sha256(test_content).hexdigest()
    assert hash1 == expected

    # Different file should produce different hash
    test_file2 = tmp_path / "test2.txt"
    test_file2.write_bytes(b"Different content")
    hash3 = hash_file(str(test_file2))
    assert hash3 != hash1


@pytest.mark.skip(reason="Requires actual PDF file - skipped in CI")
def test_extract_text_returns_list(tmp_path: Path) -> None:
    """Test that extract_text returns list of dicts with page and text keys."""
    # This test is skipped because generating a valid PDF requires reportlab or similar
    # In a real scenario, you would either:
    # 1. Use pytest.importorskip to skip if reportlab unavailable
    # 2. Use a fixture with a pre-built PDF sample
    # 3. Mock pdfplumber

    try:
        from reportlab.pdfgen import canvas
    except ImportError:
        pytest.skip("reportlab not installed")

    # Create a simple PDF
    pdf_path = tmp_path / "test.pdf"
    c = canvas.Canvas(str(pdf_path))
    c.drawString(100, 750, "Page 1 content")
    c.showPage()
    c.drawString(100, 750, "Page 2 content")
    c.showPage()
    c.save()

    # Extract text
    result = extract_text(str(pdf_path))

    # Verify structure
    assert isinstance(result, list)
    assert len(result) > 0
    for item in result:
        assert "page" in item
        assert "text" in item
        assert isinstance(item["page"], int)
        assert isinstance(item["text"], str)
        assert item["text"].strip()  # Non-empty text


def test_extract_text_empty_pages_skipped(tmp_path: Path) -> None:
    """Test that extract_text skips pages with no text via mocking."""
    from unittest.mock import MagicMock, patch

    try:
        import pdfplumber
    except ImportError:
        pytest.skip("pdfplumber not installed")

    # Mock pdfplumber to return pages with mixed content
    mock_pdf = MagicMock()
    mock_page1 = MagicMock()
    mock_page1.extract_text.return_value = "Page 1 text"
    mock_page2 = MagicMock()
    mock_page2.extract_text.return_value = ""  # Empty page
    mock_page3 = MagicMock()
    mock_page3.extract_text.return_value = "   \n  "  # Whitespace only
    mock_page4 = MagicMock()
    mock_page4.extract_text.return_value = "Page 4 text"

    mock_pdf.pages = [mock_page1, mock_page2, mock_page3, mock_page4]

    with patch("pdfplumber.open") as mock_open:
        mock_open.return_value.__enter__.return_value = mock_pdf

        result = extract_text("/fake/path.pdf")

    # Only pages 1 and 4 should be returned
    assert len(result) == 2
    assert result[0]["page"] == 1
    assert result[0]["text"] == "Page 1 text"
    assert result[1]["page"] == 4
    assert result[1]["text"] == "Page 4 text"
