import os
import pandas as pd

from core.utils.data_cleaning import strip_stray_quotes

# One parsed DataFrame per file, reloaded only if the file changes on disk.
_CACHE: dict[str, tuple[float, pd.DataFrame]] = {}

ACTIONS = ["shape", "columns", "dtypes", "describe", "value_counts", "null_count", "unique_count", "head"]


def _load(file_path: str) -> pd.DataFrame:
    mtime = os.path.getmtime(file_path)
    cached = _CACHE.get(file_path)
    if cached and cached[0] == mtime:
        return cached[1]
    df = strip_stray_quotes(pd.read_csv(file_path, low_memory=False))
    _CACHE[file_path] = (mtime, df)
    return df


class DataInspectTool:
    """Read-only, fixed-menu lookups on the dataset (no code execution).

    The reviewer uses this instead of re-running pandas code, so its check is
    independent of how the generator computed the answer: same data, different route.
    Every failure is returned as text so the reviewer can read it and adapt."""

    name = "inspect_data"

    def __init__(self, file_path: str, max_rows: int = 10):
        self.file_path = file_path
        self.max_rows = max_rows

    def run(self, action: str, column: str | None = None) -> str:
        try:
            df = _load(self.file_path)
        except Exception as e:
            return f"Error: could not load the dataset ({type(e).__name__}: {e})"

        if action not in ACTIONS:
            return f"Error: unknown action '{action}'. Valid actions: {', '.join(ACTIONS)}."

        if action == "shape":
            return f"rows={df.shape[0]}, columns={df.shape[1]}"
        if action == "columns":
            cols = list(df.columns)
            return f"{len(cols)} columns: " + ", ".join(map(str, cols[:200])) + (" ..." if len(cols) > 200 else "")
        if action == "dtypes":
            counts = df.dtypes.astype(str).value_counts().to_dict()
            return "dtype counts: " + ", ".join(f"{k}={v}" for k, v in counts.items())
        if action == "head":
            return df.head(self.max_rows).to_string(max_cols=12)
        if action == "null_count" and column is None:
            nulls = df.isna().sum()
            nulls = nulls[nulls > 0]
            return "no missing values" if nulls.empty else "missing values per column:\n" + nulls.to_string()

        if column is None:
            return f"Error: action '{action}' needs a 'column'."
        if column not in df.columns:
            close = [c for c in df.columns if str(c).lower() == str(column).lower()]
            hint = f" Did you mean '{close[0]}'?" if close else ""
            return f"Error: column '{column}' does not exist.{hint}"

        series = df[column]
        if action == "describe":
            return series.describe().to_string()
        if action == "value_counts":
            return series.value_counts(dropna=False).head(self.max_rows).to_string()
        if action == "null_count":
            return f"{int(series.isna().sum())} missing of {len(series)}"
        if action == "unique_count":
            return f"{int(series.nunique(dropna=True))} unique values"
        return "Error: unsupported request."

    def run_json_args(self, args: dict) -> str:
        if not isinstance(args, dict):
            return "Error: arguments must be a JSON object."
        return self.run(str(args.get("action", "")), args.get("column"))


INSPECT_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "inspect_data",
        "description": (
            "Read-only lookup on the dataset. Use it to independently check facts in a draft answer "
            "(row/column counts, dtypes, a column's min/max/mean, value counts, missing values)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ACTIONS, "description": "Which lookup to run."},
                "column": {"type": "string", "description": "Column name, required for all actions except shape, columns, dtypes, head."},
            },
            "required": ["action"],
        },
    },
}