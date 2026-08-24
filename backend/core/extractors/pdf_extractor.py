import fitz  # PyMuPDF
from collections import Counter
from pathlib import Path

from core.models import ExtractedPage, PageMetadata
from core.extractors.base_extractor import BaseExtractor


class PDFExtractor(BaseExtractor):
    """
    Extracts structured text + tables from a PDF using PyMuPDF (fitz).

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
      7. One ExtractedPage per page; if page has no extractable text, emit a
         warning instead of silently skipping the page.
    """

    # ── Public entry point ────────────────────────────────────────────────────

    def extract(self, file_path: str) -> list[ExtractedPage]:
        self.validate_file(file_path)

        filename    = Path(file_path).name
        doc         = fitz.open(file_path)
        total_pages = len(doc)
        body_size   = self._get_body_size(doc)

        pages: list[ExtractedPage] = []
        current_heading: str | None = None  # persists across pages

        for page_index in range(total_pages):
            page = doc[page_index]
            extracted, current_heading = self._process_page(
                page, page_index, filename, total_pages, body_size, current_heading
            )
            pages.append(extracted)

        doc.close()
        return pages

    # ── Step 1: doc-wide body font size ──────────────────────────────────────

    def _get_body_size(self, doc) -> float:
        """
        Returns the most common span font size across the whole document.
        Sizes are rounded to 1 decimal place to avoid float noise grouping
        visually identical sizes into separate buckets.
        """
        sizes: list[float] = []
        for page in doc:
            for block in page.get_text("dict")["blocks"]:
                if block["type"] == 0:          # text block
                    for line in block["lines"]:
                        for span in line["spans"]:
                            sizes.append(round(span["size"], 1))

        if not sizes:
            return 12.0  # safe fallback

        return Counter(sizes).most_common(1)[0][0]

    # ── Step 2-8: per-page processing ─────────────────────────────────────────

    def _process_page(
        self,
        page,
        page_index: int,
        filename: str,
        total_pages: int,
        body_size: float,
        current_heading: str | None,
    ) -> tuple[ExtractedPage, str | None]:
        """
        Returns (ExtractedPage, updated_current_heading).
        Always returns an ExtractedPage — empty/scanned pages get a warning.
        """
        warnings: list[str] = []

        # Step 2: detect tables and collect their bboxes
        table_finder = page.find_tables()
        tables       = table_finder.tables if table_finder.tables else []
        table_rects  = [fitz.Rect(t.bbox) for t in tables]

        # Step 3: get text blocks, drop those that overlap any table bbox
        raw_blocks      = [b for b in page.get_text("dict")["blocks"] if b["type"] == 0]
        filtered_blocks = self._exclude_table_overlaps(raw_blocks, table_rects)

        # Step 4: merge and sort by y0 for reading order
        ordered = self._build_ordered_elements(tables, filtered_blocks)

        # Step 5-6: walk elements, build page text
        page_text = ""

        for _y0, el_type, content in ordered:
            if el_type == "table":
                page_text += self._render_table(content)
            else:  # el_type == "text"
                block_text, new_heading = self._classify_text_block(content, body_size)
                if new_heading:
                    current_heading = new_heading   # step 6: carry across pages
                if block_text.strip():
                    page_text += block_text

        # Step 7: warn on empty/scanned pages
        if not page_text.strip():
            warnings.append(
                f"Page {page_index} has no extractable text — "
                "may be scanned/image-only and require OCR."
            )

        metadata = PageMetadata(
            source      = filename,           # filename, not full path
            file_type   = "pdf",
            page        = page_index,         # 0-based page index
            total_pages = total_pages,
            heading     = current_heading,    # step 6: last known heading
            warnings    = warnings,
        )

        return ExtractedPage(text=page_text.strip(), metadata=metadata), current_heading

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _exclude_table_overlaps(
        self, text_blocks: list, table_rects: list[fitz.Rect]
    ) -> list:
        """Drop any text block whose bbox intersects a table bbox."""
        filtered = []
        for block in text_blocks:
            block_rect = fitz.Rect(block["bbox"])
            if not any(block_rect.intersects(t_rect) for t_rect in table_rects):
                filtered.append(block)
        return filtered

    def _build_ordered_elements(self, tables: list, text_blocks: list) -> list:
        """
        Build a unified list of (y0, el_type, content) tuples
        and sort ascending by y0 for top-to-bottom reading order.
        """
        elements = []
        for t in tables:
            elements.append((t.bbox[1], "table", t))
        for b in text_blocks:
            elements.append((b["bbox"][1], "text", b))
        elements.sort(key=lambda x: x[0])
        return elements

    def _render_table(self, table) -> str:
        """Convert a fitz table to a Markdown table string."""
        try:
            df = table.to_pandas()
            return "\n" + df.to_markdown(index=False) + "\n\n"
        except Exception as exc:
            return f"\n[Table extraction error: {exc}]\n\n"

    def _classify_text_block(
        self, block: dict, body_size: float
    ) -> tuple[str, str | None]:
        """
        Walk every span in the block and classify by font size:
          - size > body_size + 0.5  → heading  (captured, NOT added to body text)
          - size < body_size - 0.5  → footnote / caption (prefixed and added inline)
          - else                    → body text

        Returns (body_text_accumulated, last_heading_found_or_None).
        """
        body_text = ""
        heading: str | None = None

        for line in block["lines"]:
            for span in line["spans"]:
                text = span["text"].strip()
                if not text:
                    continue

                span_size = round(span["size"], 1)

                if span_size > body_size + 0.5:
                    # Heading: update running heading, do NOT append to body
                    heading = text

                elif span_size < body_size - 0.5:
                    # Footnote / caption: tag and append inline
                    body_text += f"[Footnote/Caption]: {text} "

                else:
                    # Normal body text
                    body_text += text + " "

            body_text += "\n"   # preserve line breaks within the block

        return body_text, heading