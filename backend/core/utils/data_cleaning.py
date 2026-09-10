import pandas as pd


def strip_stray_quotes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Some source CSVs embed literal quote characters inside string cell
    values (e.g. "'1'" instead of "1"), not just as CSV-level field
    quoting. This strips leading/trailing ' and " characters from every
    object-dtype (string) column, and converts the column to numeric if
    every stripped value parses cleanly (e.g. quoted numeric labels like
    class IDs).

    Shared between CSVExtractor (schema building) and PandasSandboxTool
    (sandboxed query execution) so both see identical, cleaned data —
    they load the CSV independently (the sandbox runs in a separate
    process and can't receive a live DataFrame across the process
    boundary), so this logic must be applied in both places consistently.
    """
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip("'\"")
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().all():
            df[col] = converted
    return df