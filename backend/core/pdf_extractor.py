# backend/core/pdf_extractor.py

import fitz
from pathlib import Path

class PDFExtractor:
    def extract(self, pdf_path: str) -> list[dict]:
        doc = fitz.open(pdf_path)
        pages = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            text = self._clean(text)

            if not text:
                continue

            pages.append({
                "text": text,
                "metadata": {
                    "source": Path(pdf_path).name,
                    "page": page_num + 1,
                    "total_pages": len(doc)
                }
            })

        doc.close()
        return pages

    def _clean(self, text: str) -> str:
        lines = text.split("\n")
        lines = [line.strip() for line in lines]
        lines = [line for line in lines if line]
        return " ".join(lines)