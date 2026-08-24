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

class ColumnSchema(BaseModel):
    name: str
    dtype: str
    null_count: int
    samples: List[str]
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    unique_count: Optional[int] = None
    unique_values: Optional[List[str]] = None

class CSVSchema(BaseModel):
    source: str
    file_type: str = "csv"
    columns: List[ColumnSchema]
    row_count: int
    warnings: List[str] = []

    def to_prompt_string(self) -> str:
        """
        Serialise the schema into a compact, human-readable string suitable
        for injection into the LLM system prompt.  Mirrors the format that
        the old _format_schema_to_text() produced so the prompt stays stable.
        """
        lines = [
            f"File: {self.source} | Rows: {self.row_count} | Cols: {len(self.columns)}"
        ]
        for col in self.columns:
            line = (
                f"  Col: {col.name} | dtype: {col.dtype} "
                f"| Nulls: {col.null_count} | Samples: {col.samples}"
            )
            if col.min is not None:
                line += f" | Min: {col.min} | Max: {col.max} | Mean: {col.mean}"
            if col.unique_count is not None:
                line += f" | Unique: {col.unique_count}"
            if col.unique_values is not None:
                line += f" | Values: {col.unique_values}"
            lines.append(line)
        if self.warnings:
            lines.append("Warnings: " + "; ".join(self.warnings))
        return "\n".join(lines)