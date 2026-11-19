import asyncio

import pytest

from core import generator as generator_mod
from core.answer_check import extract_numbers, is_trivially_verified
from core.generator import Generator
from core.reviewer import ReviewResult


@pytest.mark.parametrize("text, expected", [
    ("row index **3764**, class **21**", {3764, 21}),
    ("max of f1", set()),
    ("total is 12,009", {12009}),
    ("(3764, 21)", {3764, 21}),
    ("mean is 3.50", {3.5}),
    ("-4.2 and 2nd", {-4.2}),
    ("Rejected: Attribute/method 'idxmax' is not on the allowed list", set()),
])
def test_extract_numbers(text, expected):
    assert extract_numbers(text) == expected


EXTREME_LOG = [
    ("pandas_sandbox", "a", "Rejected: Attribute/method 'idxmax' is not on the allowed list"),
    ("pandas_sandbox", "b", "(3764, 21)"),
]
EXTREME_Q = "What is the row index of the maximum value of f1, and what class does that row belong to?"


@pytest.mark.parametrize("answer, tool_log, question, expected", [
    ("The maximum value of **f1** occurs at row index **3764**, and that row belongs to class **21**.",
     EXTREME_LOG, EXTREME_Q, True),
    ("Index 3764, class 22", EXTREME_LOG, EXTREME_Q, False),
    ("The mean is 3.14", [("p", "c", "3.14159")], "mean?", True),
    ("There are 1000 rows", [("p", "c", "1004")], "how many?", False),
    ("There are 5 rows", [], "how many?", False),
    ("The most common tag is red", [("p", "c", "red")], "top tag?", False),
    ("12 rows have weight > 5", [("p", "c", "12")], "how many rows have weight > 5?", True),
])
def test_is_trivially_verified(answer, tool_log, question, expected):
    assert is_trivially_verified(answer, tool_log, question) is expected


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def gen():
    return Generator.__new__(Generator)


def review_args(review_fn, answer, question):
    return dict(
        reviewer=object(), review_fn=review_fn, messages=[{"role": "user", "content": "q"}],
        tool_schema=[], conversation_id="c", user_id="u", tracer=None, token_tracker=None,
        tool_log=[("pandas_sandbox", "code", "(3764, 21)")], answer=answer, stop_reason="final_answer",
        system_prompt="s", purpose_prefix="csv", record=lambda *a, **k: None, question=question,
    )


def counting_review(calls):
    async def review_fn(ans, tlog):
        calls.append(ans)
        return ReviewResult(passed=True, feedback="", iterations=1, tool_calls=0)
    return review_fn


def test_matching_numbers_skip_the_reviewer(gen):
    calls = []
    answer, revisions = run(gen._review_and_revise(**review_args(counting_review(calls), "Index 3764, class 21", "q")))
    assert (answer, revisions, calls) == ("Index 3764, class 21", 0, [])


def test_unmatched_number_still_reviews(gen):
    calls = []
    run(gen._review_and_revise(**review_args(counting_review(calls), "Index 3764, class 22", "q")))
    assert len(calls) == 1


def test_flag_off_always_reviews(gen, monkeypatch):
    monkeypatch.setattr(generator_mod.settings, "reviewer_skip_rules_enabled", False)
    calls = []
    run(gen._review_and_revise(**review_args(counting_review(calls), "Index 3764, class 21", "q")))
    assert len(calls) == 1
