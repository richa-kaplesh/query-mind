from core.tools.base_tool import BaseTool
import textwrap
from core.tools.sandbox_security import CodeValidator, CodeValidationError
from core.tools.pandas_worker_manager import PandasWorkerManager, worker_manager


class PandasSandboxTool(BaseTool):
    """Adapts the persistent worker to the Generator's Tool interface.
    Depends on CodeValidator/PandasWorkerManager as injected abstractions
    (defaulting to shared singletons) — swappable without touching this class."""

    def __init__(self, file_path: str, timeout_seconds: int = 15,
                 manager: PandasWorkerManager = worker_manager,
                 validator: CodeValidator = None):
        self.file_path = file_path
        self.timeout_seconds = timeout_seconds
        self._manager = manager
        self._validator = validator or CodeValidator()

    @property
    def name(self) -> str:
        return "pandas_sandbox"

    @property
    def description(self) -> str:
        return (
            "Execute dynamic Python/Pandas code on DataFrame 'df' to answer exact mathematical, "
            "filtering, aggregation, or analytical queries. Assign your final answer to variable 'result'. "
            "Also use this tool to check whether a column exists (e.g. df.columns) or inspect its values, "
            "even if that column isn't shown in full detail in the schema above — the schema only details "
            "some columns individually; others are only listed by name."
        )

    def clean_code(self, raw_code: str) -> str:
        clean_code = raw_code.strip()
        if clean_code.startswith("```python"):
            clean_code = clean_code.removeprefix("```python")
        elif clean_code.startswith("```Python"):
            clean_code = clean_code.removeprefix("```Python")
        elif clean_code.startswith("```"):
            clean_code = clean_code.removeprefix("```")
        if clean_code.endswith("```"):
            clean_code = clean_code.removesuffix("```")
        clean_code = textwrap.dedent(clean_code)
        return clean_code.strip()

    def execute_query(self, llm_code: str) -> str:
        executable_code = self.clean_code(llm_code)
        try:
            self._validator.validate(executable_code)
        except CodeValidationError as e:
            return f"Rejected: {e}"
        self._manager.load_file(self.file_path)  # self-healing — correct no matter how this tool got constructed
        return self._manager.run_query(executable_code, timeout_seconds=self.timeout_seconds)

    def run(self, code: str) -> dict:
        return {"result": self.execute_query(code)}