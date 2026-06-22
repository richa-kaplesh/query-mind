from pydantic import BaseModel
from typing import Optional, List

class PageMetadata(BaseModel):
    source: str
    file_type: str
    page: Optional[int] = None
    total_pages: Optional[int] = None
    heading: Optional[str] = None
    warnings: List[str] = []

class ExtractedPage(BaseModel):
    text: str
    metadata: PageMetadata