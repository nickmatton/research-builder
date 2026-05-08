"""Tests for PDF paper extraction."""

from pathlib import Path

import pytest

from research_builder.llm.paper import extract_full_text, extract_pages, get_page_count

FIXTURE_PDF = Path(__file__).parent / "fixtures" / "test_paper.pdf"


class TestExtractPages:
    def test_single_page(self):
        text = extract_pages(FIXTURE_PDF, 1)
        assert "Test Paper" in text
        assert "Abstract" in text

    def test_page_range(self):
        text = extract_pages(FIXTURE_PDF, 2, 3)
        assert "Methods" in text
        assert "Results" in text

    def test_last_page(self):
        text = extract_pages(FIXTURE_PDF, 3)
        assert "95.2%" in text

    def test_page_markers(self):
        text = extract_pages(FIXTURE_PDF, 1, 2)
        assert "--- Page 1 ---" in text
        assert "--- Page 2 ---" in text

    def test_out_of_bounds(self):
        with pytest.raises(ValueError, match="out of bounds"):
            extract_pages(FIXTURE_PDF, 0, 3)

    def test_out_of_bounds_high(self):
        with pytest.raises(ValueError, match="out of bounds"):
            extract_pages(FIXTURE_PDF, 1, 99)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            extract_pages("/nonexistent.pdf", 1)


class TestGetPageCount:
    def test_count(self):
        assert get_page_count(FIXTURE_PDF) == 3


class TestExtractFullText:
    def test_all_pages(self):
        text = extract_full_text(FIXTURE_PDF)
        assert "Abstract" in text
        assert "Methods" in text
        assert "Results" in text
        assert "--- Page 1 ---" in text
        assert "--- Page 3 ---" in text

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            extract_full_text("/nonexistent.pdf")
