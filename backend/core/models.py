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
        THRESHOLD = 20
        print(f"[DEBUG] to_prompt_string called, {len(self.columns)} columns, THRESHOLD={THRESHOLD}")
        """
        Serialise the schema into a compact, human-readable string suitable
        for injection into the LLM system prompt.  Mirrors the format that
        the old _format_schema_to_text() produced so the prompt stays stable.
        """
        lines = [
            f"File: {self.source} | Rows: {self.row_count} | Cols: {len(self.columns)}"
        ]
       
        if len(self.columns)>THRESHOLD:
                detailed_cols = self.columns[:THRESHOLD]
                remaining_cols = self.columns[THRESHOLD:]
                for col in detailed_cols:
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
                groups: dict[str, list] = {}
                for col in remaining_cols:
                    if col.dtype not in groups:
                        groups[col.dtype] = []
                    groups[col.dtype].append(col)
                print(groups)
                if self.warnings:
                    lines.append("Warnings: " + "; ".join(self.warnings))

                result = "\n".join(lines)
                print(f"[DEBUG] final schema string length: {len(result)} chars")
                return result

                
                     
                     
        else:
            detailed_cols = self.columns
            remaining_cols = []
            for col in detailed_cols:
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

