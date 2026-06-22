from backend.core.tools.base_tool import BaseTool
from backend.core.analysis.csv_stats import run_full_statistics
import pandas as pd

class CSVStatsTool(BaseTool):

    def __init__(self, file_path: str):
        self.file_path = file_path

    @property
    def name(self) -> str:
        return "get_csv_stats"

    @property
    def description(self) -> str:
        return "Run statistical analysis on a CSV file to answer analytical or ML related questions"

    def run(self, query: str) -> dict:
        df = pd.read_csv(self.file_path)
        report = run_full_statistics(df, problem_statement=query)
        return {"stats_report": report}