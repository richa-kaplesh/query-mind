from models import PageMetadata , ExtractedPage
from base_extractor import BaseExtractor
from typing import List
from PIL import Image
import io
import pytesseract
from config import settings
import fitz

class PDFExtractor(BaseExtractor):

        def __init__(self):
                 pytesseract.pytesseract.tesseract_cmd = settings.tesseract_path

        def extract(self, file_path:str) ->List[ExtractedPage]:
            results =[]
            self.validate_file(file_path)
            doc = fitz.open(file_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_result = self._process_page(page, file_path, page_num,len(doc))
                if page_result is not None:
                      results.append(page_result)
            doc.close()
            return results

        def _build_page(self, text:str, page_num:int, file_path:str, total_pages:int , warnings:List[str]):
            metadata = PageMetadata(source = file_path, file_type = "pdf", page= page_num+1,total_pages = total_pages, warnings = warnings)
            Page = ExtractedPage(text = text, metadata= metadata)
            return Page

        def _run_ocr(self, page):
            pix = page.get_pixmap()
            img_bytes = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_bytes))
            text = pytesseract.image_to_string(img)
            return text.strip()

        def _process_page(self, page, file_path:str,page_num:int, total_page:int)-> ExtractedPage| None:
            text = page.get_text("text")
            images = page.get_images()
            has_text = bool(text.strip())
            has_images = bool(len(images)>0)
            warnings = []

            if has_text and not has_images:
                return self._build_page(text, page_num, file_path, total_page,[])
                 
            elif has_text and has_images:
                for i, img in enumerate(images):
                    text +=f"\n[IMAGE_{i+1}:page {page_num+1}]\n"
                warnings.append("This page contains images or charts which were skipped")
                return self._build_page(text, page_num, file_path, total_page, warnings)

            elif not has_text and has_images:
                 ocr_text = self._run_ocr(page) 
                 if ocr_text:
                      return self._build_page(ocr_text, page_num, file_path, total_page,[])
                 else:
                      return None

            elif not has_text and not has_images:
                 return None
                
                


