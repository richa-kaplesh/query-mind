import logging
import pandas as pd
from pathlib import Path

from core.models import CSVSchema, ColumnSchema, CSVExtractionResult
from core.extractors.base_extractor import BaseExtractor
from core.utils.data_cleaning import strip_stray_quotes
from core.utils.memory_usage import rss_mb, fmt
from core.utils.csv_loader import load_csv
log = logging.getLogger("csv_extractor")


class CSVExtractor(BaseExtractor):
    """
    Extracts a structured CSVSchema from a CSV file.

    Public API
    ----------
    extract_schema(file_path)  → CSVSchema
    extract(file_path)         → CSVExtractionResult
    """

    # ── BaseExtractor contract ────────────────────────────────────────────────

    def extract(self, file_path: str) -> CSVExtractionResult:
        """
        Satisfies BaseExtractor.extract().
        Internally calls extract_schema() and returns a CSVExtractionResult wrapping the CSVSchema.
        """
        schema = self.extract_schema(file_path)
        return CSVExtractionResult(schema=schema)

    # ── Primary method ────────────────────────────────────────────────────────

    def extract_schema(self, file_path: str) -> CSVSchema:
        """
        Load the CSV and return a fully populated CSVSchema instance.
        Warnings are appended for:
          - files with 0 rows
          - columns that are entirely null
        """
        self.validate_file(file_path)
        df       = load_csv(file_path)
        log.info(f"[MEM] CSV parsed for schema in main process: main={fmt(rss_mb())}")
        filename = Path(file_path).name
        warnings: list[str] = []

        if len(df) == 0:
            warnings.append("File has 0 rows — dataset is empty.")

        columns = self._build_columns(df, warnings)

        return CSVSchema(
            source   = filename,
            columns  = columns,
            row_count= len(df),
            warnings = warnings,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    
   
    
        
    
    def _build_columns(
        self, df: pd.DataFrame, warnings: list[str]
    ) -> list[ColumnSchema]:
        columns: list[ColumnSchema] = []

        for col_name in df.columns:
            series     = df[col_name]
            null_count = int(series.isna().sum())

            # Warn if the entire column is null
            if null_count == len(df) and len(df) > 0:
                warnings.append(f"Column '{col_name}' is entirely null.")

            # First 3 non-null values → always stored as strings for the LLM
            samples: list[str] = [
                str(v) for v in series.dropna().head(3).tolist()
            ]

            col = ColumnSchema(
                name       = col_name,
                dtype      = str(series.dtype),
                null_count = null_count,
                samples    = samples,
            )

            if pd.api.types.is_numeric_dtype(series):
                col.min  = float(series.min())
                col.max  = float(series.max())
                col.mean = float(series.mean())
            else:
                unique_count = int(series.nunique())
                col.unique_count = unique_count
                if unique_count < 10:
                    col.unique_values = [str(v) for v in series.unique().tolist()]

            columns.append(col)

        return columns