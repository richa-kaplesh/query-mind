import pytest
import pytesseract
from PIL import Image
from config import settings
from core.ocr.factory import get_ocr_engine
from core.ocr.tesseract_ocr import TesseractOCREngine


def white_image():
    return Image.new("RGB", (20, 20), "white")


def test_factory_returns_tesseract_by_default():
    assert isinstance(get_ocr_engine(), TesseractOCREngine)


def test_factory_rejects_unknown_provider(monkeypatch):
    monkeypatch.setattr(settings, "ocr_provider", "magic-cloud-ocr")
    with pytest.raises(NotImplementedError, match="magic-cloud-ocr"):
        get_ocr_engine()


def test_recognised_text_is_stripped_and_has_no_warning(monkeypatch):
    monkeypatch.setattr(pytesseract, "image_to_string", lambda *a, **k: "  hello  \n")
    result = TesseractOCREngine().extract_text(white_image())
    assert result.text == "hello" and result.warning is None


def test_no_text_found_gives_a_warning(monkeypatch):
    monkeypatch.setattr(pytesseract, "image_to_string", lambda *a, **k: "   \n")
    result = TesseractOCREngine().extract_text(white_image())
    assert result.text == "" and "no text" in result.warning


def test_timeout_gives_a_warning(monkeypatch):
    def slow(*a, **k):
        raise RuntimeError("Tesseract process timeout")
    monkeypatch.setattr(pytesseract, "image_to_string", slow)
    result = TesseractOCREngine(timeout_seconds=7).extract_text(white_image())
    assert result.text == "" and "timed out after 7s" in result.warning


def test_missing_tesseract_binary_gives_a_warning(monkeypatch):
    def missing(*a, **k):
        raise pytesseract.TesseractNotFoundError()
    monkeypatch.setattr(pytesseract, "image_to_string", missing)
    result = TesseractOCREngine().extract_text(white_image())
    assert result.text == "" and "not found" in result.warning