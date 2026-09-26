from PIL import Image, ImageOps
import pytesseract

from core.ocr.base_ocr import BaseOCREngine, OCRResult


class TesseractOCREngine(BaseOCREngine):
    """Self-hosted OCR via pytesseract. Preprocessing is intentionally simple
    (grayscale + autocontrast + binarize) — no deskew yet, that needs OpenCV
    which isn't a project dependency. Add opencv-python-headless + a deskew
    step here if client scans turn out skewed/rotated in practice."""

    def __init__(self, timeout_seconds: int = 20, binarize_threshold: int = 150):
        self.timeout_seconds = timeout_seconds
        self.binarize_threshold = binarize_threshold

    def _preprocess(self, image: Image.Image) -> Image.Image:
        gray = ImageOps.grayscale(image)
        contrasted = ImageOps.autocontrast(gray)
        # simple binarization: anything darker than threshold -> black, else white
        return contrasted.point(lambda p: 0 if p < self.binarize_threshold else 255)

    def extract_text(self, image: Image.Image) -> OCRResult:
        processed = self._preprocess(image)
        try:
            text = pytesseract.image_to_string(processed, timeout=self.timeout_seconds)
        except RuntimeError:
            # pytesseract raises RuntimeError on timeout (wraps its own timeout logic)
            return OCRResult(text="", warning=f"OCR timed out after {self.timeout_seconds}s")
        except pytesseract.TesseractNotFoundError:
            return OCRResult(text="", warning="Tesseract binary not found on this system")

        text = text.strip()
        if not text:
            return OCRResult(text="", warning="OCR ran but recognized no text")
        return OCRResult(text=text)