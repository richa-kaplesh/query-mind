import fitz
import pytest
from core.exceptions import PDFCorruptError, PDFPasswordProtectedError
from core.extractors.pdf_extractor import PDFExtractor

def test_missing_file_raises():
    extractor = PDFExtractor()
    with pytest.raises(FileNotFoundError):
        extractor.extract("this_file_does_not_exist.pdf")
    
def test_folder_raises(tmp_path):
    extractor = PDFExtractor()
    with pytest.raises(IsADirectoryError):
        extractor.extract(tmp_path)

def test_module_loads_when_tesseract_path_is_set(monkeypatch):
    import importlib
    import config
    import core.extractors.pdf_extractor as pdf_extractor

    monkeypatch.setattr(config.settings, "tesseract_path", "/fake/path/to/tesseract")
    importlib.reload(pdf_extractor)   # re-runs the top-of-file code

def test_corrupt_file_raises(tmp_path):
    bad_pdf = tmp_path / "bad.pdf"
    bad_pdf.write_text("this is not a real pdf")
    with pytest.raises(PDFCorruptError):
        PDFExtractor().extract(str(bad_pdf))


def test_password_protected_pdf_raises(tmp_path):
    locked = tmp_path / "locked.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "secret contents")
    doc.save(str(locked), encryption=fitz.PDF_ENCRYPT_AES_256,
             user_pw="pw123", owner_pw="pw123")
    doc.close()
    with pytest.raises(PDFPasswordProtectedError):
        PDFExtractor().extract(str(locked))

def test_normal_pdf_returns_text_and_metadata(tmp_path):
    pdf_path = tmp_path / "normal.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "Hello QueryMind page one")
    doc.new_page().insert_text((72, 72), "Second page here")
    doc.save(str(pdf_path))
    doc.close()

    result = PDFExtractor().extract(str(pdf_path))

    assert len(result.pages) == 2
    first, second = result.pages
    assert "Hello QueryMind page one" in first.text
    assert "Second page here" in second.text
    assert first.metadata.page == 0
    assert second.metadata.page == 1
    assert first.metadata.total_pages == 2
    assert first.metadata.source == "normal.pdf"
    assert first.metadata.file_type == "pdf"
    assert first.metadata.ocr_used is False
    assert first.metadata.warnings == []


# def test_blank_page_gives_warning_not_crash(tmp_path):
#     blank = tmp_path / "blank.pdf"
#     doc = fitz.open()
#     doc.new_page()          # a page with nothing on it
#     doc.save(str(blank))
#     doc.close()

#     result = PDFExtractor().extract(str(blank))

#     assert len(result.pages) == 1
#     assert result.pages[0].text == ""
#     assert len(result.pages[0].metadata.warnings) >= 1