import json
import logging
from dataclasses import dataclass

from core import gateway_client
from core.tools.data_inspect_tool import DataInspectTool, INSPECT_TOOL_SCHEMA
from core.token_utils import estimate_tokens
from config import settings

log = logging.getLogger("reviewer")

REVIEW_SYSTEM_PROMPT = """You are a fact-checker for a data-analysis assistant. You will be given a \
question, a draft answer, and the tool calls that produced it.

Your job is NOT to re-derive the answer the same way the draft did, and NOT to verify every number in \
it — that wastes time on a report with many columns. Use the `inspect_data` tool to independently \
spot-check the draft: pick at most 2 of its most important or highest-risk claims (prefer numbers that \
came from `pandas_sandbox` computation over numbers just restated from the schema, since restated \
schema facts are already reliable) and check those against the actual dataset. inspect_data does \
fixed, read-only lookups — it cannot run the same code the draft used, so it is a genuinely different \
way of checking the same fact.

Rules:
1. Call inspect_data at most twice, on the 1-2 claims most likely to be wrong. Do not try to check \
   everything — a fast, partial check is the goal, not a full audit.
2. Do not check style, wording, or completeness — only whether the specific facts you checked are \
   correct and actually supported by the data.
3. After your checks (or immediately, if the draft has nothing worth checking), respond with ONLY this \
   JSON object as your final message, no other text:
   {"verdict": "pass", "feedback": ""}
   or, if a checked claim is wrong, unsupported, or invented:
   {"verdict": "fail", "feedback": "<specific, actionable — what is wrong and what to recheck>"}
"""


@dataclass
class ReviewResult:
    passed: bool
    feedback: str
    iterations: int
    tool_calls: int


def _tc_parts(tool_call) -> tuple[str, str, str]:
    if isinstance(tool_call, dict):
        fn = tool_call.get("function") or {}
        return tool_call.get("id") or "", fn.get("name") or "", fn.get("arguments") or "{}"
    return getattr(tool_call, "id", "") or "", tool_call.function.name, tool_call.function.arguments or "{}"


def _parse_verdict(text: str) -> dict | None:
    """The model is asked for bare JSON, but models sometimes wrap it in prose or
    a code fence anyway — pull out the first {...} block rather than failing outright."""
    if not text:
        return None
    text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    if data.get("verdict") not in ("pass", "fail"):
        return None
    return data


class Reviewer:
    """Checks a generator's draft answer against the data independently, via a
    fixed-menu tool — never re-runs the generator's own code. Never edits the
    answer itself; only returns pass/fail + feedback for the generator to act on."""

    def __init__(self, inspect_tool: DataInspectTool, max_iterations: int = 3):
        self.inspect_tool = inspect_tool
        self.max_iterations = max_iterations

    def _run_tool(self, raw_args: str) -> str:
        try:
            args = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError as e:
            return f"Error: arguments were not valid JSON ({e})."
        try:
            return self.inspect_tool.run_json_args(args)
        except Exception as e:
            log.error(f"[REVIEWER] inspect_data raised: {e}", exc_info=True)
            return f"Error: inspect_data failed: {type(e).__name__}: {e}"

    async def review(self, question: str, schema: str, draft_answer: str, tool_log: list,
                      conversation_id: str, user_id: str = "query_mind_user",
                      tracer=None, token_tracker=None) -> ReviewResult:

        def record(step_type, data):
            if tracer:
                try:
                    tracer(step_type, data)
                except Exception:
                    pass

        tools_summary = "\n\n".join(
            f"Tool used: {name}\nInput: {tool_input}\nResult: {result}"
            for name, tool_input, result in tool_log
        ) or "(no tools were used to produce this draft)"

        messages = [
            {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Question: {question}\n\nDataset schema:\n{schema or '(none)'}\n\n"
                f"Draft answer to check:\n{draft_answer}\n\n"
                f"How the draft was produced:\n{tools_summary}"
            )},
        ]

        tool_calls_made = 0
        for iteration in range(1, self.max_iterations + 1):
            record(f"review_call_{iteration}", {"iteration": iteration})
            try:
                result = await gateway_client.complete(
                    conversation_id=f"{conversation_id}:review", user_id=user_id, messages=messages,
                    tools=[INSPECT_TOOL_SCHEMA], tool_choice="auto", is_tool_related=True,
                )
            except Exception as e:
                log.error(f"[REVIEWER] LLM call {iteration} failed: {e}")
                record("review_error", {"iteration": iteration, "error": str(e)})
                return ReviewResult(passed=True, feedback="", iterations=iteration, tool_calls=tool_calls_made)

            if token_tracker:
                token_tracker.log_call(
                    model=result.get("model_used", "gateway"),
                    prompt_tokens=estimate_tokens(str(messages)),
                    completion_tokens=estimate_tokens(result.get("content") or ""),
                    purpose=f"review_call{iteration}", estimated=True,
                )

            tool_calls = result.get("tool_calls")
            if not tool_calls:
                verdict = _parse_verdict(result.get("content") or "")
                if verdict is not None:
                    record("review_verdict", {**verdict, "iterations": iteration, "tool_calls": tool_calls_made})
                    return ReviewResult(
                        passed=verdict["verdict"] == "pass", feedback=verdict.get("feedback", ""),
                        iterations=iteration, tool_calls=tool_calls_made,
                    )
                messages.append({"role": "assistant", "content": result.get("content")})
                messages.append({"role": "user", "content": (
                    'Respond with ONLY the JSON object — nothing else. '
                    'Example: {"verdict": "pass", "feedback": ""}'
                )})
                continue

            messages.append({"role": "assistant", "content": result.get("content"), "tool_calls": tool_calls})
            for tc in tool_calls:
                call_id, tool_name, raw_args = _tc_parts(tc)
                tool_result = self._run_tool(raw_args)
                tool_calls_made += 1
                record("review_tool_call", {"input": raw_args, "result": tool_result})
                messages.append({"role": "tool", "tool_call_id": call_id, "name": tool_name, "content": tool_result})

        log.warning(f"[REVIEWER] No verdict after {self.max_iterations} iterations — passing by default")
        record("review_verdict", {"verdict": "pass (default, no verdict reached)", "iterations": self.max_iterations})
        return ReviewResult(passed=True, feedback="", iterations=self.max_iterations, tool_calls=tool_calls_made)

RAG_REVIEW_SYSTEM_PROMPT = """You are a faithfulness checker for a document Q&A assistant. You will be \
given a question, the passages that were retrieved from the document, and a draft answer built from \
those passages.

Check ONLY this: is every factual claim in the draft answer actually supported by the passages below — \
not from outside knowledge, and not invented. A citation that points to the wrong passage, or a claim \
that isn't backed by any passage at all, should fail.

Do not check style, completeness, or whether this is the best possible answer — only whether what it \
states is actually grounded in the passages given.

Respond with ONLY this JSON object as your final message, no other text:
{"verdict": "pass", "feedback": ""}
or, if a claim is unsupported or a citation is wrong:
{"verdict": "fail", "feedback": "<specific — which claim is unsupported, and why>"}
"""


class RAGReviewer:
    """Checks a RAG draft answer for groundedness against the passages that were
    actually retrieved — no tool calls needed, since everything to check is already
    in tool_log. Catches invented claims and prior-knowledge leakage, like the
    invented 'Location' line found during OCR testing. Never edits the answer
    itself; only returns pass/fail + feedback for the generator to act on."""

    def __init__(self, max_iterations: int = 2):
        # max_iterations here only bounds "ask again for valid JSON" retries,
        # not tool-calling rounds — there's no tool, so 2 is just one real
        # attempt plus one reformat nudge.
        self.max_iterations = max_iterations

    async def review(self, question: str, draft_answer: str, tool_log: list,
                      conversation_id: str, user_id: str = "query_mind_user",
                      tracer=None, token_tracker=None) -> ReviewResult:

        def record(step_type, data):
            if tracer:
                try:
                    tracer(step_type, data)
                except Exception:
                    pass

        passages_text = "\n\n".join(result for _, _, result in tool_log) or "(no passages were retrieved)"
        messages = [
            {"role": "system", "content": RAG_REVIEW_SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Question: {question}\n\nRetrieved passages:\n{passages_text}\n\n"
                f"Draft answer to check:\n{draft_answer}"
            )},
        ]

        for iteration in range(1, self.max_iterations + 1):
            record(f"review_call_{iteration}", {"iteration": iteration})
            try:
                result = await gateway_client.complete(
                    conversation_id=f"{conversation_id}:review", user_id=user_id, messages=messages,
                    is_tool_related=False,
                )
            except Exception as e:
                log.error(f"[RAG REVIEWER] LLM call {iteration} failed: {e}")
                record("review_error", {"iteration": iteration, "error": str(e)})
                return ReviewResult(passed=True, feedback="", iterations=iteration, tool_calls=0)

            if token_tracker:
                token_tracker.log_call(
                    model=result.get("model_used", "gateway"),
                    prompt_tokens=estimate_tokens(str(messages)),
                    completion_tokens=estimate_tokens(result.get("content") or ""),
                    purpose=f"rag_review_call{iteration}", estimated=True,
                )

            verdict = _parse_verdict(result.get("content") or "")
            if verdict is not None:
                record("review_verdict", {**verdict, "iterations": iteration})
                return ReviewResult(
                    passed=verdict["verdict"] == "pass", feedback=verdict.get("feedback", ""),
                    iterations=iteration, tool_calls=0,
                )

            messages.append({"role": "assistant", "content": result.get("content")})
            messages.append({"role": "user", "content": (
                'Respond with ONLY the JSON object — nothing else. '
                'Example: {"verdict": "pass", "feedback": ""}'
            )})

        log.warning(f"[RAG REVIEWER] No verdict after {self.max_iterations} iterations — passing by default")
        record("review_verdict", {"verdict": "pass (default, no verdict reached)", "iterations": self.max_iterations})
        return ReviewResult(passed=True, feedback="", iterations=self.max_iterations, tool_calls=0)