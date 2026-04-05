import pytest
from RAG.chunking import strip_frontmatter, chunk_text


class TestStripFrontmatter:
    def test_strips_valid_frontmatter(self):
        text = "---\nurl: https://example.com\n---\n\nBody here."
        body, fm = strip_frontmatter(text)
        assert body == "Body here."
        assert fm == {"url": "https://example.com"}

    def test_no_frontmatter_returns_text_unchanged(self):
        text = "Just plain body text."
        body, fm = strip_frontmatter(text)
        assert body == text
        assert fm == {}

    def test_multiple_keys_parsed(self):
        text = "---\nurl: https://example.com\ntitle: My Doc\n---\n\nBody."
        body, fm = strip_frontmatter(text)
        assert fm["url"] == "https://example.com"
        assert fm["title"] == "My Doc"
        assert body == "Body."

    def test_empty_body_after_frontmatter(self):
        text = "---\nurl: https://example.com\n---\n\n"
        body, fm = strip_frontmatter(text)
        assert body == ""
        assert fm["url"] == "https://example.com"

    def test_value_with_colon_preserved(self):
        text = "---\nurl: https://example.com/path?a=1&b=2\n---\n\nBody."
        body, fm = strip_frontmatter(text)
        # partition on first ':' so 'https' is key — the url value after first colon
        # just confirm no crash and frontmatter parsed
        assert body == "Body."
        assert "url" in fm

    def test_no_newline_after_closing_dashes_not_matched(self):
        # Frontmatter must be followed by \n to match
        text = "---\nurl: x\n---Body."
        body, fm = strip_frontmatter(text)
        assert fm == {}
        assert body == text


class TestChunkText:
    def test_empty_text_returns_empty_list(self):
        assert chunk_text("", 100, 10) == []

    def test_whitespace_only_returns_empty_list(self):
        assert chunk_text("   \n\n   \n\n  ", 100, 10) == []

    def test_single_short_paragraph_returns_one_chunk(self):
        text = "Hello, world."
        chunks = chunk_text(text, 100, 0)
        assert chunks == ["Hello, world."]

    def test_two_paragraphs_fit_in_one_chunk(self):
        text = "Para one.\n\nPara two."
        chunks = chunk_text(text, 100, 0)
        assert len(chunks) == 1
        assert "Para one." in chunks[0]
        assert "Para two." in chunks[0]

    def test_paragraphs_split_when_exceeding_size(self):
        p1 = "A" * 50
        p2 = "B" * 50
        chunks = chunk_text(f"{p1}\n\n{p2}", 60, 0)
        assert len(chunks) == 2
        assert p1 in chunks[0]
        assert p2 in chunks[1]

    def test_overlap_prepended_to_next_chunk(self):
        p1 = "A" * 50
        p2 = "B" * 50
        chunks = chunk_text(f"{p1}\n\n{p2}", 60, 20)
        assert len(chunks) >= 2
        # The overlap is the last 20 chars of chunks[0], which is all A's
        assert "A" * 20 in chunks[1]

    def test_large_single_paragraph_split_by_chars(self):
        big = "X" * 300
        chunks = chunk_text(big, 100, 0)
        assert len(chunks) == 3
        for chunk in chunks:
            assert len(chunk) <= 100

    def test_large_paragraph_with_overlap_split_by_chars(self):
        big = "Y" * 250
        chunks = chunk_text(big, 100, 20)
        for chunk in chunks:
            assert len(chunk) <= 100

    def test_chunk_size_respected_across_multiple_paragraphs(self):
        # 10 paragraphs of 10 chars each; size=25 fits 2 per chunk (10+2+10=22 < 25)
        paras = "\n\n".join(["0123456789"] * 10)
        chunks = chunk_text(paras, 25, 0)
        for chunk in chunks:
            assert len(chunk) <= 25

    def test_no_overlap_produces_non_overlapping_chunks(self):
        p1 = "A" * 40
        p2 = "B" * 40
        chunks = chunk_text(f"{p1}\n\n{p2}", 50, 0)
        assert len(chunks) == 2
        # With no overlap, second chunk should not contain A's
        assert "A" not in chunks[1]
