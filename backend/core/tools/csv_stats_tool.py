from core.tools.base_tool import BaseTool
from core.analysis.csv_stats import run_full_statistics
import pandas as pd

class CSVStatsTool(BaseTool):

    def __init__(self, file_path:str):
        self.file_path = file_path

    @property
    def name(self) ->str:
        return "get_csv_stats"

    @property
    def description(self) ->str:
        return (
            "Run broad statistical analysis, machine learning insights, or correlation analysis on a CSV file. "
            "Use this tool for exploratory data science reports, understanding distributions, or answering high-level analytical questions. "
            "Do NOT use this tool for specific row filtering or exact mathematical calculations."
        )

    def run(self, query:str) ->dict:
        df = pd.read_csv(self.file_path, low_memory=False)
        report = run_full_statistics(df, problem_statement=query)

        return {"stats_report":report}







