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
        