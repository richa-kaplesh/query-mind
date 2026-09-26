from core.ocr.base_ocr import BaseOCREngine
from core.ocr.tesseract_ocr import TesseractOCREngine
from config import settings


def get_ocr_engine() -> BaseOCREngine:
    if settings.ocr_provider == "tesseract":
        return TesseractOCREngine(timeout_seconds=settings.ocr_timeout_seconds)
    raise NotImplementedError(
        f"OCR provider '{settings.ocr_provider}' is not implemented. "
        "Add a new BaseOCREngine subclass and wire it in here."
    )