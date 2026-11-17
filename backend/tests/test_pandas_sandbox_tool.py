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
        return "ok", None

    def run_query(self, code, timeout_seconds=15):
        self.queries.append(code)
        return self.answer

    def run_on_file(self, path, code, timeout_seconds=15):
        self.load_file(path)
        return self.run_query(code, timeout_seconds=timeout_seconds)


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

    assert tool.run("```python\nresult = df['age'].sum()\n```") == "60"
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


@pytest.mark.parametrize("cache_size", ["1", "3"])
def test_real_worker_two_files_each_query_sees_its_own_data(tmp_path, monkeypatch, cache_size):
    monkeypatch.setenv("MAX_CACHED_DATASETS", cache_size)
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    a.write_text("age\n10\n20\n")
    b.write_text("age\n100\n200\n")
    manager = PandasWorkerManager()
    tool_a = PandasSandboxTool(str(a), timeout_seconds=20, manager=manager)
    tool_b = PandasSandboxTool(str(b), timeout_seconds=20, manager=manager)
    try:
        for _ in range(2):
            assert tool_a.execute_query("result = df['age'].sum()") == "30"
            assert tool_b.execute_query("result = df['age'].sum()") == "300"
    finally:
        if manager.process is not None:
            manager.process.kill()
            manager.process.join()