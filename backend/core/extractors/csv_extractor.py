import logging
from pathlib import Path

from core.models import CSVSchema, CSVExtractionResult
from core.extractors.base_extractor import BaseExtractor
from core.utils.csv_loader import load_csv
from core.utils.csv_schema import build_schema

log = logging.getLogger("csv_extractor")


class CSVExtractor(BaseExtractor):
    """
    Extracts a structured CSVSchema from a CSV file.

    Note: the live upload path no longer uses this class. The pandas worker
    computes the schema from the DataFrame it already loaded (see
    PandasWorkerManager). This class is kept for tests and offline/eval use.
    """

    def extract(self, file_path: str) -> CSVExtractionResult:
        schema = self.extract_schema(file_path)
        return CSVExtractionResult(schema=schema)

    def extract_schema(self, file_path: str) -> CSVSchema:
        self.validate_file(file_path)
        df = load_csv(file_path)
        return build_schema(df, Path(file_path).name)