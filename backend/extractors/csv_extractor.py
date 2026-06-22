import pandas as pd
from pathlib import Path
from backend.core.models import ExtractedPage, PageMetadata
from backend.core.extractors.base_extractor import BaseExtractor


class CSVExtractor(BaseExtractor):

    def extract(self, file_path: str) -> list[ExtractedPage]:
        self.validate_file(file_path)
        df = self._load(file_path)
        pages = []

        for idx, row in df.iterrows():
            text = self._row_to_text(row)
            page = self._build_page(text, file_path)
            pages.append(page)

        return pages

    def _load(self, file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise ValueError(f"Failed to load CSV: {e}")

    def _row_to_text(self, row: pd.Series) -> str:
        parts = [f"{col}: {val}" for col, val in row.items()]
        return " | ".join(parts)

    def _build_page(self, text: str, file_path: str) -> ExtractedPage:
        metadata = PageMetadata(
            source=file_path,
            file_type="csv"
        )
        return ExtractedPage(
            text=text,
            metadata=metadata
        )