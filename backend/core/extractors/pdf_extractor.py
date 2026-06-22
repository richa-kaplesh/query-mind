import fitz
import pytesseract
from pathlib import Path
from PIL import Image
import io
from core.models import ExtractedPage, PageMetadata
from core.extractors.base_extractor import BaseExtractor

class PDFExtractor(BaseExtractor):
    
    def extract(self, file_path: str) -> list[ExtractedPage]:
        self.validate_file(file_path)
        doc = fitz.open(file_path)
        pages = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            result = self._process_page(page, page_num, file_path, len(doc))
            if result:
                pages.append(result)
        
        doc.close()
        return pages
    
    def _process_page(self, page, page_num:int, file_path: str, total_pages: int)-> ExtractedPage | None:
        text = page.get_text("text")
        images = page.get_images()
        has_text = bool(text.strip())
        has_images = len(images)>0
        warnings = []

        if has_text and not has_images:
            return self._build_page(text, page_num, file_path, total_pages, warnings)
        
        if has_text and has_images:
            warnings.append("This page contains images or charts which were skipped")
            for i , img in enumerate(images):
                bbox = page.get_image_rects(img[0])
                annotation = f"\n[IMAGE_{i+1}: page {page_num+1}]\n"
                text += annotation
            return self.build_page(text, page_num, file_path, total_pages , warnings)
        
        if not has_text and has_images:
            ocr_text = self._run_ocr(page)
            if ocr_text:
                return self._build_page(ocr_text, page_num, file_path, total_pages, warnings)
            warnings.append("This page contains unsupported content like photos or charts")
            return None
        return None 
    
    def _run_ocr(self, page) -> str:
        pix = page.get_pixmap()
        img_bytes = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_bytes))
        text = pytesseract.image_to_string(img)
        return text.strip()
    
    def _build_page(self , text:str, page_num: int , file_path: str, total_pages: int , warnings : list[str]) -> ExtractedPage:
        metadata = PageMetadata(
            page_number = page_num+1,
            total_pages = total_pages,
            file_path = file_path,
            has_warnings=len(warnings)> 0
        )
        return ExtractedPage(
            text=text,
            metadata = metadata,
            warnings=warnings
        )