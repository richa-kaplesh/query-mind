from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, List

class QueryRequest(BaseModel):
    question: str
    conversation_history: list = []
    
class PageMetadata(BaseModel):
    source: str
    file_type: str
    page: Optional[int] = None
    total_pages: Optional[int] = None
    heading: Optional[str] = None
    warnings: List[str] = []
    ocr_used: bool = False   # ← new

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
        for injection into the LLM system prompt.
        """
        THRESHOLD = 20

        def _format_col(col) -> str:
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
            return line

        lines = [
            f"File: {self.source} | Rows: {self.row_count} | Cols: {len(self.columns)}"
        ]

        if len(self.columns) > THRESHOLD:
            for col in self.columns[:THRESHOLD]:
                lines.append(_format_col(col))

            # Group remaining columns by dtype and list them compactly
            groups: dict[str, list] = {}
            for col in self.columns[THRESHOLD:]:
                groups.setdefault(col.dtype, []).append(col)
            for dtype, cols in groups.items():
                col_names = ", ".join(c.name for c in cols)
                lines.append(f"  +{len(cols)} more {dtype} columns not detailed above: {col_names}")

            lines.append(
                "Note: some columns are only listed by name above, not in full detail. "
                "If asked about a column not shown in detail, do not assume it doesn't exist — "
                "use the pandas_sandbox tool to check df.columns or inspect it directly."
            )
        else:
            for col in self.columns:
                lines.append(_format_col(col))

        if self.warnings:
            lines.append("Warnings: " + "; ".join(self.warnings))

        return "\n".join(lines)


class ExtractionResult(BaseModel):
    pass


class PDFExtractionResult(ExtractionResult):
    pages: List[ExtractedPage]


class CSVExtractionResult(ExtractionResult):
    model_config = ConfigDict(populate_by_name=True, protected_namespaces=())
    csv_schema: CSVSchema = Field(..., alias="schema")

    @property
    def schema(self) -> CSVSchema:
        return self.csv_schema

