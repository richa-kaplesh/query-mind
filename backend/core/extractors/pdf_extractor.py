import fitz  # PyMuPDF
import pytesseract
from collections import Counter
from pathlib import Path
from PIL import Image

from core.models import ExtractedPage, PageMetadata, PDFExtractionResult
from core.extractors.base_extractor import BaseExtractor
from core.exceptions import PDFPasswordProtectedError, PDFCorruptError
from core.ocr.base_ocr import BaseOCREngine
from core.ocr.factory import get_ocr_engine
from config import settings

if settings.tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = settings.tesseract_path


class PDFExtractor(BaseExtractor):
    """
    Extracts structured text + tables from a PDF using PyMuPDF (fitz), with an
    OCR fallback (via an injected BaseOCREngine) for pages with no text layer.

    Per-page pipeline:
      1. Compute doc-wide body font size (most common span size, via Counter).
      2. Detect tables → collect their bboxes.
      3. Get text blocks (type==0) → drop any that intersect a table bbox.
      4. Merge tables + filtered text blocks, sort by y0 for reading order.
      5. Walk merged elements:
           - table  → DataFrame → Markdown table → append to page text
           - text block → classify each span:
               size > body_size+0.5  → heading: update current_heading (not appended to body)
               size < body_size-0.5  → footnote/caption: append as "[Footnote/Caption]: ..."
               else                  → body text, append normally
      6. current_heading persists across pages until a new heading is found.
      7. If a page has no extractable text layer, rasterize it and run OCR
         instead of skipping it. OCR'd pages carry the last known heading
         forward (no font metadata exists post-OCR, so per-line heading
         detection can't apply) and are flagged via metadata.ocr_used.
      8. A page-level OCR failure (timeout, no text recognized) produces a
         warning and empty text for that page — it never fails the whole
         document.
    """

    def __init__(self, ocr_engine: BaseOCREngine | None = None):
        self.ocr_engine = ocr_engine or get_ocr_engine()

    def extract(self, file_path: str) -> PDFExtractionResult:
        self.validate_file(file_path)

        filename = Path(file_path).name

        try:
            doc = fitz.open(file_path)
        except fitz.FileDataError as exc:
            raise PDFCorruptError(f"'{filename}' could not be opened — file appears corrupt.") from exc

        if doc.needs_pass:
            doc.close()
            raise PDFPasswordProtectedError(f"'{filename}' is password-protected.")

        total_pages = len(doc)
        body_size = self._get_body_size(doc)

        pages: list[ExtractedPage] = []
        current_heading: str | None = None

        for page_index in range(total_pages):
            page = doc[page_index]
            extracted, current_heading = self._process_page(
                page, page_index, filename, total_pages, body_size, current_heading
            )
            pages.append(extracted)

        doc.close()
        return PDFExtractionResult(pages=pages)

    def _get_body_size(self, doc) -> float:
        sizes: list[float] = []
        for page in doc:
            for block in page.get_text("dict")["blocks"]:
                if block["type"] == 0:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            sizes.append(round(span["size"], 1))
        if not sizes:
            return 12.0
        return Counter(sizes).most_common(1)[0][0]

    def _process_page(
        self, page, page_index: int, filename: str, total_pages: int,
        body_size: float, current_heading: str | None,
    ) -> tuple[ExtractedPage, str | None]:
        warnings: list[str] = []
        ocr_used = False

        table_finder = page.find_tables(strategy="lines_strict")
        tables = table_finder.tables if table_finder.tables else []
        table_rects = [fitz.Rect(t.bbox) for t in tables]

        raw_blocks = [b for b in page.get_text("dict")["blocks"] if b["type"] == 0]
        filtered_blocks = self._exclude_table_overlaps(raw_blocks, table_rects)
        ordered = self._build_ordered_elements(tables, filtered_blocks)

        page_text = ""
        for _y0, el_type, content in ordered:
            if el_type == "table":
                page_text += self._render_table(content)
            else:
                block_text, new_heading = self._classify_text_block(content, body_size, current_heading)
                if new_heading:
                    current_heading = new_heading
                if block_text.strip():
                    page_text += block_text

        if not page_text.strip():
            page_text, ocr_warning = self._ocr_fallback(page, page_index)
            ocr_used = bool(page_text.strip())
            if ocr_warning:
                warnings.append(ocr_warning)
            if not page_text.strip() and not ocr_warning:
                warnings.append(
                    f"Page {page_index} has no extractable text and OCR found nothing."
                )

        metadata = PageMetadata(
            source=filename,
            file_type="pdf",
            page=page_index,
            total_pages=total_pages,
            heading=current_heading,
            warnings=warnings,
            ocr_used=ocr_used,
        )
        return ExtractedPage(text=page_text.strip(), metadata=metadata), current_heading

    def _ocr_fallback(self, page, page_index: int) -> tuple[str, str | None]:
        """Rasterize the page and run it through the injected OCR engine.
        Never raises — a failure here becomes a warning, not a crash,
        so one bad page can't take down the whole document."""
        try:
            pix = page.get_pixmap(dpi=settings.ocr_dpi, colorspace=fitz.csRGB, alpha=False)
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        except Exception as exc:
            return "", f"Page {page_index}: could not rasterize for OCR ({exc})"

        result = self.ocr_engine.extract_text(image)
        return result.text, result.warning

    def _exclude_table_overlaps(self, text_blocks: list, table_rects: list[fitz.Rect]) -> list:
        filtered = []
        for block in text_blocks:
            block_rect = fitz.Rect(block["bbox"])
            if not any(block_rect.intersects(t_rect) for t_rect in table_rects):
                filtered.append(block)
        return filtered

    def _build_ordered_elements(self, tables: list, text_blocks: list) -> list:
        elements = []
        for t in tables:
            elements.append((t.bbox[1], "table", t))
        for b in text_blocks:
            elements.append((b["bbox"][1], "text", b))
        elements.sort(key=lambda x: x[0])
        return elements

    def _render_table(self, table) -> str:
        try:
            df = table.to_pandas()
            return "\n" + df.to_markdown(index=False) + "\n\n"
        except Exception as exc:
            return f"\n[Table extraction error: {exc}]\n\n"

    def _classify_text_block(self, block: dict, body_size: float, current_heading: str | None) -> tuple[str, str | None]:
        body_text = ""
        heading: str | None = current_heading
        for line in block["lines"]:
            line_text = ""
            for span in line["spans"]:
                text = span["text"].strip()
                if not text:
                    continue
                span_size = round(span["size"], 1)
                if span_size > body_size + 0.5:
                    heading = text
                elif span_size < body_size - 0.5:
                    line_text += f"[Footnote/Caption]: {text} "
                else:
                    line_text += text + " "
            if line_text.strip():
                body_text += f"[[HEADING:{heading}]]{line_text}\n"
        return body_text, heading