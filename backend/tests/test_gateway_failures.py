"""Gateway failure handling: stop on 429 / 5xx, no extra requests, no silent 'verified'.

Run from backend/:  python -m pytest tests/test_gateway_failures.py
"""
import asyncio

import httpx
import pytest

from core import gateway_client, generator as generator_mod, reviewer as reviewer_mod
from core.exceptions import GatewayError
from core.generator import Generator, UNVERIFIED_NOTE
from core.reviewer import Reviewer, RAGReviewer, ReviewResult


def run(coro):
    return asyncio.run(coro)


TOOL_CALL = {"id": "c1", "function": {"name": "t", "arguments": '{"x": 1}'}}
WANTS_TOOL = {"content": None, "tool_calls": [TOOL_CALL], "model_used": "m"}


@pytest.fixture
def script(monkeypatch):
    """Replace gateway_client.complete with a scripted sequence of results / exceptions."""
    calls = []

    def install(*steps):
        steps = list(steps)

        async def fake_complete(**kwargs):
            calls.append(kwargs)
            step = steps.pop(0)
            if isinstance(step, Exception):
                raise step
            return step

        monkeypatch.setattr(gateway_client, "complete", fake_complete)
        return calls

    return install


@pytest.fixture
def gen():
    g = Generator.__new__(Generator)  # skip __init__ (needs a Groq key)
    g.tools = []
    g._run_tool_safely = lambda name, raw: ("input", "tool result")
    return g


def loop(g):
    return g._run_loop(
        [{"role": "user", "content": "q"}], [{"type": "function"}], "conv", "user",
        None, None, [],
    )


# ── _run_loop ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("status", [429, 500, 503, None])
def test_fatal_gateway_error_after_tool_work_makes_no_further_request(gen, script, status):
    calls = script(WANTS_TOOL, GatewayError(status, "boom"))
    with pytest.raises(GatewayError) as exc:
        run(loop(gen))
    assert exc.value.status == status
    assert len(calls) == 2  # tool-call turn + the failure; NO forced-final-answer request


def test_fatal_error_on_first_call_raises(gen, script):
    calls = script(GatewayError(500, "boom"))
    with pytest.raises(GatewayError):
        run(loop(gen))
    assert len(calls) == 1


def test_non_fatal_4xx_still_uses_forced_answer_recovery(gen, script):
    calls = script(WANTS_TOOL, GatewayError(400, "malformed"), {"content": "forced answer"})
    answer, reason = run(loop(gen))
    assert (answer, reason) == ("forced answer", "llm_error")
    assert len(calls) == 3


def test_fatal_error_on_forced_answer_surfaces(gen, script):
    calls = script(WANTS_TOOL, GatewayError(400, "malformed"), GatewayError(500, "down"))
    with pytest.raises(GatewayError) as exc:
        run(loop(gen))
    assert exc.value.status == 500
    assert len(calls) == 3


# ── review / revise ──────────────────────────────────────────────────────────

def review_args(g, review_fn, answer="draft"):
    return dict(
        reviewer=object(), review_fn=review_fn, messages=[{"role": "user", "content": "q"}],
        tool_schema=[], conversation_id="c", user_id="u", tracer=None, token_tracker=None,
        tool_log=[("t", "i", "r")], answer=answer, stop_reason="final_answer",
        system_prompt="s", purpose_prefix="csv", record=lambda *a, **k: None,
    )


def test_revision_hitting_gateway_failure_keeps_previous_draft(gen):
    reviews = []

    async def review_fn(ans, tlog):
        reviews.append(ans)
        return ReviewResult(passed=False, feedback="bad", iterations=1, tool_calls=0)

    loops = []

    async def failing_loop(*a, **k):
        loops.append(1)
        raise GatewayError(500, "down")

    gen._run_loop = failing_loop
    answer, revisions = run(gen._review_and_revise(**review_args(gen, review_fn)))
    assert answer == "draft" + UNVERIFIED_NOTE
    assert revisions == 1
    assert len(reviews) == 1 and len(loops) == 1  # no second review, no second revision


def test_reviewer_failure_marks_draft_unverified_and_skips_revisions(gen):
    async def review_fn(ans, tlog):
        return ReviewResult(passed=True, feedback="", iterations=1, tool_calls=0, unverified=True)

    async def never(*a, **k):
        raise AssertionError("must not revise")

    gen._run_loop = never
    answer, revisions = run(gen._review_and_revise(**review_args(gen, review_fn)))
    assert answer == "draft" + UNVERIFIED_NOTE
    assert revisions == 0


def test_normal_pass_is_untouched(gen):
    async def review_fn(ans, tlog):
        return ReviewResult(passed=True, feedback="", iterations=1, tool_calls=0)

    answer, revisions = run(gen._review_and_revise(**review_args(gen, review_fn)))
    assert (answer, revisions) == ("draft", 0)


# ── reviewers fail "unverified", not "passed silently" ──────────────────────

def test_csv_reviewer_gateway_error_is_unverified(script):
    calls = script(GatewayError(429, "slow down"))
    r = Reviewer(inspect_tool=None, max_iterations=3)
    res = run(r.review(question="q", schema="s", draft_answer="a", tool_log=[],
                       conversation_id="c", user_id="u", tracer=None, token_tracker=None))
    assert res.unverified is True
    assert len(calls) == 1


def test_rag_reviewer_gateway_error_is_unverified(script):
    calls = script(GatewayError(500, "down"))
    r = RAGReviewer(max_iterations=2)
    res = run(r.review(question="q", draft_answer="a", tool_log=[],
                       conversation_id="c", user_id="u", tracer=None, token_tracker=None))
    assert res.unverified is True
    assert len(calls) == 1


# ── gateway_client ───────────────────────────────────────────────────────────

class _Resp:
    def __init__(self, status, body=None):
        self.status_code, self._body, self.text = status, body or {}, str(body)

    def json(self):
        return self._body


def fake_httpx(monkeypatch, outcome):
    class FakeClient:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, *a, **k):
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

    monkeypatch.setattr(gateway_client.httpx, "AsyncClient", FakeClient)


@pytest.fixture(autouse=True)
def _reset_budget():
    gateway_client._budget_var.set(None)
    yield
    gateway_client._budget_var.set(None)


def call():
    return run(gateway_client.complete(conversation_id="c", user_id="u", messages=[]))


@pytest.mark.parametrize("status,fatal", [(429, True), (500, True), (502, True), (400, False), (422, False)])
def test_http_errors_become_gateway_error(monkeypatch, status, fatal):
    fake_httpx(monkeypatch, _Resp(status))
    with pytest.raises(GatewayError) as exc:
        call()
    assert exc.value.status == status and exc.value.fatal is fatal


def test_timeout_is_unreachable_and_fatal(monkeypatch):
    fake_httpx(monkeypatch, httpx.ReadTimeout("slow"))
    with pytest.raises(GatewayError) as exc:
        call()
    assert exc.value.reason == "unreachable" and exc.value.status is None and exc.value.fatal


def test_success_returns_json(monkeypatch):
    fake_httpx(monkeypatch, _Resp(200, {"content": "hi"}))
    assert call() == {"content": "hi"}


def test_call_budget_stops_the_request_without_hitting_the_gateway(monkeypatch):
    fake_httpx(monkeypatch, _Resp(200, {"content": "hi"}))
    gateway_client.start_call_budget(2)
    call(); call()
    with pytest.raises(GatewayError) as exc:
        call()
    assert exc.value.reason == "budget" and exc.value.fatal