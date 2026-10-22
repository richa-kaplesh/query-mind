import pandas as pd

from core.models import CSVSchema, ColumnSchema


def _build_columns(df: pd.DataFrame, warnings: list[str]) -> list[ColumnSchema]:
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


def build_schema(df: pd.DataFrame, source: str) -> CSVSchema:
    """Build a CSVSchema from an already-loaded DataFrame."""
    warnings: list[str] = []

    if len(df) == 0:
        warnings.append("File has 0 rows — dataset is empty.")

    columns = _build_columns(df, warnings)

    return CSVSchema(
        source   = source,
        columns  = columns,
        row_count= len(df),
        warnings = warnings,
    )