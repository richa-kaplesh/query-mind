import fitz
import pytest
from core.chunker import TextChunker
from core.extractors.pdf_extractor import PDFExtractor
from core.models import ExtractedPage, PageMetadata


def test_empty_text_gives_no_chunks():
    assert TextChunker().chunk_text("") == []
    assert TextChunker().chunk_text("\n   \n") == []


def test_short_text_is_one_chunk_and_keeps_metadata():
    chunks = TextChunker().chunk_text("hello\nworld", {"source": "a.pdf", "page": 3})
    assert len(chunks) == 1
    assert chunks[0]["text"] == "hello\nworld"
    assert chunks[0]["metadata"]["source"] == "a.pdf"
    assert chunks[0]["metadata"]["page"] == 3
    assert chunks[0]["metadata"]["chunk_index"] == 0


def test_long_text_is_split_and_no_chunk_exceeds_the_limit():
    lines = [f"line number {i:02d} .........." for i in range(20)]  # 26 chars each
    chunker = TextChunker(chunk_size=80, chunk_overlap=0)
    chunks = chunker.chunk_text("\n".join(lines))

    assert len(chunks) > 1
    assert all(len(c["text"]) <= 80 for c in chunks)
    assert [c["metadata"]["chunk_index"] for c in chunks] == list(range(len(chunks)))
    joined = "\n".join(c["text"] for c in chunks)
    assert all(line in joined for line in lines)      # nothing lost


def test_overlap_repeats_the_end_of_the_previous_chunk():
    lines = [f"line number {i:02d} .........." for i in range(10)]
    chunker = TextChunker(chunk_size=80, chunk_overlap=30)
    chunks = chunker.chunk_text("\n".join(lines))

    assert len(chunks) > 1
    last_line_of_first = chunks[0]["text"].split("\n")[-1]
    assert last_line_of_first in chunks[1]["text"].split("\n")


def test_headings_split_sections_and_marker_is_removed():
    text = "[[HEADING:Intro]]Hello\nworld\n[[HEADING:Methods]]We did things"
    chunks = TextChunker().chunk_text(text)

    assert [c["metadata"]["heading"] for c in chunks] == ["Intro", "Methods"]
    assert chunks[0]["text"] == "Hello\nworld"
    assert "[[HEADING" not in "".join(c["text"] for c in chunks)


def test_chunk_pages_carries_each_pages_metadata():
    pages = [
        ExtractedPage(text="first page text",
                      metadata=PageMetadata(source="a.pdf", file_type="pdf", page=0)),
        ExtractedPage(text="second page text",
                      metadata=PageMetadata(source="a.pdf", file_type="pdf", page=1)),
    ]
    chunks = TextChunker().chunk_pages(pages)
    assert [c["metadata"]["page"] for c in chunks] == [0, 1]


@pytest.mark.xfail(strict=True, reason=(
    "Known issue: pdf_extractor writes the literal text 'None' as the heading "
    "marker when a page has no heading, so chunks get heading == 'None' (a string)."))
def test_page_without_heading_gives_heading_none(tmp_path):
    pdf_path = tmp_path / "plain.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "just plain body text")
    doc.save(str(pdf_path))
    doc.close()

    pages = PDFExtractor().extract(str(pdf_path)).pages
    chunks = TextChunker().chunk_pages(pages)
    assert chunks[0]["metadata"]["heading"] is None