import pytest
from core.tools.pandas_sandbox_tool import PandasSandboxTool
from core.tools.pandas_worker_manager import PandasWorkerManager


class FakeManager:
    """Stands in for the real worker process, and records how it was used."""
    def __init__(self, answer="42"):
        self.answer = answer
        self.loaded = []
        self.queries = []

    def load_file(self, path):
        self.loaded.append(path)

    def run_query(self, code, timeout_seconds=15):
        self.queries.append(code)
        return self.answer


def test_tool_name():
    assert PandasSandboxTool("x.csv", manager=FakeManager()).name == "pandas_sandbox"


def test_clean_code_strips_markdown_fences():
    tool = PandasSandboxTool("x.csv", manager=FakeManager())
    assert tool.clean_code("```python\nresult = 1\n```") == "result = 1"
    assert tool.clean_code("```\nresult = 1\n```") == "result = 1"
    assert tool.clean_code("   result = 1   ") == "result = 1"


def test_dangerous_code_is_rejected_and_never_reaches_the_worker():
    manager = FakeManager()
    tool = PandasSandboxTool("x.csv", manager=manager)

    out = tool.execute_query("```python\nimport os\n```")

    assert out.startswith("Rejected:")
    assert manager.loaded == [] and manager.queries == []


def test_valid_code_loads_the_file_then_runs_the_cleaned_code():
    manager = FakeManager(answer="60")
    tool = PandasSandboxTool("data.csv", manager=manager)

    assert tool.run("```python\nresult = df['age'].sum()\n```") == {"result": "60"}
    assert manager.loaded == ["data.csv"]
    assert manager.queries == ["result = df['age'].sum()"]


# ---- integration: a REAL worker process, tiny CSV ------------------------

@pytest.fixture
def real_tool(tmp_path):
    csv_path = tmp_path / "ages.csv"
    csv_path.write_text("age,city\n10,Delhi\n20,Mumbai\n30,Delhi\n")
    manager = PandasWorkerManager()
    yield PandasSandboxTool(str(csv_path), timeout_seconds=20, manager=manager)
    if manager.process is not None:
        manager.process.kill()
        manager.process.join()


def test_real_worker_computes_the_right_answer(real_tool):
    assert real_tool.execute_query("result = df['age'].sum()") == "60"
    assert real_tool.execute_query("result = df[df['city'] == 'Delhi']['age'].sum()") == "40"


def test_real_worker_reports_runtime_errors_as_text(real_tool):
    out = real_tool.execute_query("result = df['no_such_column'].sum()")
    assert out.startswith("Error executing Pandas code")