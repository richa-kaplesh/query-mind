import pandas as pd
from pathlib import Path

from core.models import CSVSchema, ColumnSchema, ExtractedPage, PageMetadata
from core.extractors.base_extractor import BaseExtractor


class CSVExtractor(BaseExtractor):
    """
    Extracts a structured CSVSchema from a CSV file.

    Public API
    ----------
    extract_schema(file_path)  → CSVSchema          ← primary method
    extract(file_path)         → list[ExtractedPage] ← keeps BaseExtractor contract;
                                                       delegates to extract_schema and
                                                       serialises via to_prompt_string()
    """

    # ── BaseExtractor contract ────────────────────────────────────────────────

    def extract(self, file_path: str) -> list[ExtractedPage]:
        """
        Satisfies BaseExtractor.extract().
        Internally calls extract_schema(), then converts the CSVSchema to a
        prompt-ready string and wraps it in a single ExtractedPage so that
        the existing routes/_extract_schema_sync() call still works unchanged.
        """
        schema = self.extract_schema(file_path)
        metadata = PageMetadata(
            source=schema.source,
            file_type="csv",
        )
        return [ExtractedPage(text=schema.to_prompt_string(), metadata=metadata)]

    # ── Primary method ────────────────────────────────────────────────────────

    def extract_schema(self, file_path: str) -> CSVSchema:
        """
        Load the CSV and return a fully populated CSVSchema instance.
        Warnings are appended for:
          - files with 0 rows
          - columns that are entirely null
        """
        self.validate_file(file_path)
        df       = self._load(file_path)
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

    # Encodings tried in order; first UnicodeDecodeError moves to the next.
    _ENCODINGS = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]

    def _load(self, file_path: str) -> pd.DataFrame:
        """
        Robustly load a CSV regardless of encoding or delimiter.

        Phase 1 — delimiter detection:
          Read the first 8 KB of the file (trying each encoding) and pass it
          to csv.Sniffer to detect the real separator (comma, semicolon, tab,
          pipe, etc.).  Falls back to ',' if sniffing raises an error.

        Phase 2 — full load:
          Try pd.read_csv with the detected separator, iterating encodings.
          on_bad_lines='warn' skips isolated malformed rows instead of aborting.
        """
        import csv as _csv

        sep = self._sniff_separator(file_path)

        last_error: Exception = RuntimeError("No encodings tried")
        for enc in self._ENCODINGS:
            try:
                return pd.read_csv(
                    file_path,
                    sep=sep,
                    encoding=enc,
                    low_memory=False,
                    on_bad_lines="warn",   # skip malformed rows, don't crash
                )
            except UnicodeDecodeError as e:
                last_error = e
                continue
            except Exception as e:
                raise ValueError(f"Failed to load CSV ({enc}): {e}") from e

        raise ValueError(
            f"Failed to load CSV — tried encodings {self._ENCODINGS}: {last_error}"
        )

    def _sniff_separator(self, file_path: str) -> str:
        """
        Use csv.Sniffer on the first 8 KB to detect the actual delimiter.
        Returns ',' as a safe fallback if sniffing fails for any reason.
        """
        import csv as _csv

        for enc in self._ENCODINGS:
            try:
                with open(file_path, "r", encoding=enc, errors="replace") as fh:
                    sample = fh.read(8192)
                dialect = _csv.Sniffer().sniff(sample, delimiters=",;\t|")
                return dialect.delimiter
            except UnicodeDecodeError:
                continue
            except _csv.Error:
                break   # sniffer failed on content — fall through to default

        return ","  # safe default

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