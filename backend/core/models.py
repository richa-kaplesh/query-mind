from pydantic import BaseModel
from typing import Optional

class PageMetadata(BaseModel):
    source: str
    file_type: str
    page: Optional[int] = None
    total_pages: Optional[int] = None
    heading: Optional[str] = None

class ExtractedPage(BaseModel):
    text: str
    metadata: PageMetadata