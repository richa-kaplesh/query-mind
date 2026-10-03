import pytest
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
