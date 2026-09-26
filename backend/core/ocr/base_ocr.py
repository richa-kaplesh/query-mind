from abc import ABC, abstractmethod
from dataclasses import dataclass
from PIL import Image


@dataclass
class OCRResult:
    text: str
    warning: str | None = None  # e.g. "OCR timed out", "no text recognized"


class BaseOCREngine(ABC):
    """Swappable OCR interface. PDFExtractor depends on this, never on a
    specific OCR library — swapping Tesseract for a hosted API later means
    writing one new class here, no changes to PDFExtractor."""

    @abstractmethod
    def extract_text(self, image: Image.Image) -> OCRResult:
        pass