import csv
import pandas as pd

from core.utils.data_cleaning import strip_stray_quotes

# Encodings tried in order; first UnicodeDecodeError moves to the next.
ENCODINGS = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]


def _sniff_separator(file_path: str) -> str:
    """Detect the delimiter from the first 8 KB. Falls back to ','."""
    for enc in ENCODINGS:
        try:
            with open(file_path, "r", encoding=enc, errors="replace") as fh:
                sample = fh.read(8192)
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            return dialect.delimiter
        except UnicodeDecodeError:
            continue
        except csv.Error:
            break   # sniffer failed on content, fall through to default
    return ","


def load_csv(file_path: str) -> pd.DataFrame:
    """Load a CSV regardless of encoding or delimiter."""
    sep = _sniff_separator(file_path)

    last_error: Exception = RuntimeError("No encodings tried")
    for enc in ENCODINGS:
        try:
            df = pd.read_csv(
                file_path,
                sep=sep,
                encoding=enc,
                low_memory=False,
                on_bad_lines="warn",   # skip malformed rows, don't crash
            )
            return strip_stray_quotes(df)
        except UnicodeDecodeError as e:
            last_error = e
            continue
        except Exception as e:
            raise ValueError(f"Failed to load CSV ({enc}): {e}") from e

    raise ValueError(
        f"Failed to load CSV, tried encodings {ENCODINGS}: {last_error}"
    )