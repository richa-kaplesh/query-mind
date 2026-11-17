from concurrent.futures import ThreadPoolExecutor

import pytest

from core.tools.pandas_sandbox_tool import PandasSandboxTool
from core.tools.pandas_worker_manager import PandasWorkerManager

USERS = 10
ROUNDS = 5


def expected_sum(user: int) -> int:
    return user * 1000 + 3


def make_csv(tmp_path, user: int) -> str:
    path = tmp_path / f"user{user:02d}.csv"
    path.write_text(f"value,user_tag\n{user * 1000},u{user:02d}\n1,u{user:02d}\n2,u{user:02d}\n")
    return str(path)


def ask_many(tool: PandasSandboxTool, rounds: int) -> list[str]:
    return [tool.execute_query("result = int(df['value'].sum())") for _ in range(rounds)]


@pytest.fixture
def manager(monkeypatch):
    monkeypatch.setenv("MAX_CACHED_DATASETS", "2")
    m = PandasWorkerManager()
    yield m
    if m.process is not None:
        m.process.kill()
        m.process.join()


def test_ten_users_in_parallel_each_only_see_their_own_data(tmp_path, manager):
    tools = {u: PandasSandboxTool(make_csv(tmp_path, u), timeout_seconds=30, manager=manager) for u in range(1, USERS + 1)}

    with ThreadPoolExecutor(max_workers=USERS) as pool:
        futures = {u: pool.submit(ask_many, tool, ROUNDS) for u, tool in tools.items()}
        answers = {u: f.result(timeout=300) for u, f in futures.items()}

    wrong = {u: a for u, a in answers.items() if a != [str(expected_sum(u))] * ROUNDS}
    assert wrong == {}
