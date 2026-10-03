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