import sys
import os
import json
import re
import pandas as pd
from groq import Groq
from core.generator import Generator
from core.extractors.csv_extractor import CSVExtractor
from core.models import CSVSchema
from config import settings
from datetime import datetime 
import re as re_module  # already imported as `re` at top of file — reuse that import
import asyncio
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = r"D:\query-mind\backend\uploads\phpB0xrNj.csv"
DATASET_PATH = os.path.join(BASE_DIR, "golden_csv_dataset.json")
RESULTS_SCRATCH_PATH = os.path.join(BASE_DIR, "csv_eval_scratch.json")

SAVE_TO_HISTORY = False # flip to False while debugging

from core.generator import Generator
from core.tools.pandas_sandbox_tool import PandasSandboxTool  # check actual import path/class name

generator = Generator(tools=[])
generator.tools = [PandasSandboxTool(file_path=CSV_PATH)]
groq_client = Groq(api_key=settings.groq_api_key)


import re

_DASH_MAP = str.maketrans({"\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-"})

_SIGNAL_PATTERN = re.compile(
    r"(?:\bis\b|\bare\b|\bwas\b|\bwere\b|\bequals?\b|\bapproximately\b|\babout\b)"
    r"\s*\**\s*(-?\d+\.?\d*)",
    re.IGNORECASE,
)

def _clean(text: str) -> str:
    return text.translate(_DASH_MAP)

def _bare_numbers(text: str) -> list[float]:
    """Fallback: standalone numbers not glued to a letter prefix (e.g. skips f10, f617)."""
    numbers = []
    for m in re.finditer(r"[a-zA-Z]*-?\d+\.?\d*", text):
        token = m.group()
        if re.match(r"[a-zA-Z]*", token).group():
            continue
        numbers.append(float(token))
    return numbers

def extract_number(text: str):
    text = _clean(text)
    signal_matches = _SIGNAL_PATTERN.findall(text)
    if signal_matches:
        return float(signal_matches[-1])   # the final stated answer, not an echoed parameter
    numbers = _bare_numbers(text)          # fallback for answers with no linking verb
    return numbers[-1] if numbers else None

def extract_all_numbers(text: str) -> list[float]:
    text = _clean(text)
    signal_matches = _SIGNAL_PATTERN.findall(text)
    if len(signal_matches) >= 2:
        return [float(x) for x in signal_matches]
    return _bare_numbers(text)             # fallback for range answers without "is"/"are" phrasing

def setup_csv_pipeline(csv_path: str):
    """
    Returns (schema, df):
      schema -> CSVSchema object, passed into generate_with_tools() same as production
      df     -> raw pandas DataFrame, loaded via CSVExtractor._load() to guarantee
                identical parsing (separator sniffing, encoding fallback, bad-line
                handling) as what the production pipeline and PandasSandboxTool use.
    """
    extractor = CSVExtractor()
    result = extractor.extract(csv_path)
    schema = result.schema
    df = extractor._load(csv_path)   # same parsing path as extract_schema() uses internally
    return schema, df


def compute_golden_value(pandas_code: str, df):
    safe_builtins = {
        "len": len,
        "sum": sum,
        "min": min,
        "max": max,
        "abs": abs,
        "round": round,
        "sorted": sorted,
        "list": list,
        "set": set,
        "int": int,
        "float": float,
        "str": str,
    }
    safe_globals = {"__builtins__": safe_builtins}
    safe_locals = {"df": df}
    return eval(pandas_code, safe_globals, safe_locals)

async def _run_csv_query_async(question: str, schema: CSVSchema, tracer=None) -> dict:
    conversation_id = str(uuid.uuid4())  # each eval question is an independent query,
                                          # not a real multi-turn conversation
    tool_used = None
    answer_parts = []

    async for token in generator.generate_stream(
        query=question,
        schema=schema,
        conversation_id=conversation_id,
        tracer=tracer,
    ):
        if token.startswith("__tool__:"):
            tool_used = token.split(":", 1)[1]
        else:
            answer_parts.append(token)

    return {"answer": "".join(answer_parts), "tool_used": tool_used}


def run_csv_query(question: str, schema: CSVSchema, tracer=None) -> dict:
    return asyncio.run(_run_csv_query_async(question, schema, tracer=tracer))


def check_tool_app(should_use: bool, actual_tool_used) -> float:
    actual_used_bool = actual_tool_used is not None
    return 1.0 if actual_used_bool == should_use else 0.0




def score_correctness(golden_value, actual_answer: str, tolerance) -> float:
    if tolerance is not None:
        if isinstance(golden_value, (list, tuple)):
            # Range comparison — extract two numbers from the answer, compare each bound
            numbers = extract_all_numbers(actual_answer)
            if len(numbers) < 2:
                return 0.0
            actual_min, actual_max = numbers[0], numbers[1]
            golden_min, golden_max = float(golden_value[0]), float(golden_value[1])
            min_ok = abs(actual_min - golden_min) <= tolerance
            max_ok = abs(actual_max - golden_max) <= tolerance
            return 1.0 if (min_ok and max_ok) else 0.0
        else:
            # Single-value numeric comparison — existing logic
            actual_number = extract_number(actual_answer)
            if actual_number is None:
                return 0.0
            return 1.0 if abs(actual_number - float(golden_value)) <= tolerance else 0.0
    else:
        golden_str = str(golden_value).strip().lower()
        # \b = word boundary — matches "6" as a standalone token, not the "6" inside "268" or "f617"
        pattern = r'\b' + re.escape(golden_str) + r'\b'
        if re.search(pattern, actual_answer.strip().lower()):
            return 1.0
        return score_with_llm_judge(golden_value, actual_answer)



def score_with_llm_judge(golden_value, actual_answer: str) -> float:
    prompt = f"""You are an evaluation judge for a CSV question-answering system.
Ground Truth Value: {golden_value}
Generated Answer: {actual_answer}

Does the Generated Answer correctly convey the Ground Truth Value, even if worded
differently? Score from 0.0 to 1.0.

Return ONLY a JSON object like this, nothing else:
{{
    "answer_correctness": 0.0
}}"""

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        scores = json.loads(raw)
        return float(scores.get("answer_correctness", 0.0))
    except json.JSONDecodeError as e:
        print(f"\n❌ JSON Parse Error: {e}")
        print(f"Raw response: {raw[:200]}...")
        return 0.0


RESULTS_SCRATCH_PATH = os.path.join(BASE_DIR, "csv_eval_scratch.json")

def run_single_csv_eval(question_id: str):
    with open(DATASET_PATH, "r") as f:
        golden_data = json.load(f)

    item = next((q for q in golden_data if q["id"] == question_id), None)
    if item is None:
        print(f"No question found with id={question_id}")
        return

    schema, df = setup_csv_pipeline(CSV_PATH)

    question = item["question"]
    should_use = item["needs_computation"]
    pandas_code = item["golden_pandas_code"]
    tolerance = item.get("tolerance")

    golden_value = compute_golden_value(pandas_code, df)

    trace_data = {}
    def tracer(step_type, data):
        if step_type == "tool_input":
            trace_data["pandas_code"] = data.get("input")
        elif step_type == "tool_output":
            trace_data["raw_tool_result"] = data.get("result")

    result = run_csv_query(question, schema, tracer=tracer)
    actual_answer = result["answer"]
    actual_tool_used = result["tool_used"]

    tool_score = check_tool_app(should_use, actual_tool_used)
    correctness_score = score_correctness(golden_value, actual_answer, tolerance)

    print(f"Question: {question}")
    print(f"Golden value: {golden_value}")
    print(f"Answer: {actual_answer}")
    print(f"Tool used: {actual_tool_used}")
    print(f"Tool Appropriateness: {tool_score} | Answer Correctness: {correctness_score}")

    scratch = []
    if os.path.exists(RESULTS_SCRATCH_PATH):
        with open(RESULTS_SCRATCH_PATH, "r") as f:
            scratch = json.load(f)

    scratch = [r for r in scratch if r["id"] != question_id]
    scratch.append({
        "id": question_id,
        "question": question,
        "answer": actual_answer,
        "pandas_code": trace_data.get("pandas_code"),
        "raw_tool_result": trace_data.get("raw_tool_result"),
        "golden_value": str(golden_value),
        "tool_used": actual_tool_used,
        "scores": {
            "tool_appropriateness": tool_score,
            "answer_correctness": correctness_score,
        }
    })

    with open(RESULTS_SCRATCH_PATH, "w") as f:
        json.dump(scratch, f, indent=2)

    print(f"Saved to scratch ({len(scratch)}/{len(golden_data)} questions done)")

from datetime import datetime  # add this near your other imports if not already present

def finalize_csv_eval_run():
    with open(RESULTS_SCRATCH_PATH, "r") as f:
        results = json.load(f)

    avg_tool = sum(r["scores"]["tool_appropriateness"] for r in results) / len(results)
    avg_correctness = sum(r["scores"]["answer_correctness"] for r in results) / len(results)

    run_record = {
        "timestamp": datetime.utcnow().isoformat(),
        "type": "csv",
        "label": "csv-eval-gateway-production-pass",
        "description": (
            "First full eval run against the LIVE, gateway-routed production path "
            "(generate_stream() -> gateway_client -> LLM Gateway -> persistent pandas "
            "worker), replacing the earlier standalone generate_with_tools() script which "
            "was found to be dead code, never wired into any live route. 13/15 correct. "
            "Bugs found and fixed during this pass: "
            "(1) The gateway's semantic cache was incorrectly caching Call 2 (post-tool "
            "synthesis) responses. Call 2 is built with no tools/tool_choice by design (a "
            "separate earlier fix to stop unwanted repeat tool calls), which made it look "
            "identical to a plain cacheable request to router.py's tools is None check. "
            "Since Call 2's message content is mostly shared boilerplate instruction text, "
            "embeddings of different synthesis calls scored above the similarity threshold "
            "and one question's cached answer leaked into unrelated questions' responses -- "
            "confirmed live via a 33% cache-hit rate on requests that should never be "
            "cached. Fixed with an explicit is_tool_related flag added to GatewayRequest, "
            "set by the caller rather than inferred from message shape, checked alongside "
            "tools is None in the cache-skip condition. "
            "(2) PandasWorkerManager had no locking around its task_queue/result_queue "
            "exchange -- under concurrent access, one caller's response could be consumed "
            "by a different in-flight call, surfacing as spurious 'Worker found dead' "
            "respawns with no actual crash. Fixed with a threading.Lock wrapping the full "
            "send/receive exchange in _send(). "
            "(3) router.py ended up with two `async def route(...)` functions in the same "
            "file after adding retry-with-backoff logic above an older, un-deleted version "
            "-- Python silently used the second (older, no-retry) definition, so the new "
            "retry code never actually executed despite multiple confirmed, successful "
            "redeploys. The error message ('Both providers failed' vs 'Both providers "
            "failed after retries') never changed across several redeploys, which was the "
            "signal something was off; found by direct inspection of the pushed repo rather "
            "than continuing to trust deploy-status indicators. Fixed by deleting the "
            "duplicate, older function body. "
            "(4) Added retry-with-backoff (2 attempts, ~2s delay) per provider in route(), "
            "after observing a genuine simultaneous failure: Groq's real rate limit hit "
            "at the same time as one of Gemini's recurring 503 'high demand' errors, under "
            "sustained heavy testing load across a long session. "
            "REMAINING 2 FAILURES -- known scoring-methodology gaps, not execution bugs: "
            "schema_06 (model's prose answer is factually correct but doesn't string-match "
            "the golden value's raw pandas .dtypes series repr) and count_05 (a compound "
            "two-part question -- 'which class AND how many' -- scored against a "
            "single-value golden_value, so a correct count with no class label, or vice "
            "versa, can't score 1.0 under the current scorer). Needs either a "
            "compound-answer scorer or splitting these into separate single-value "
            "questions -- flagged for next investigation, not addressed this pass."
        ),
        "config": {
            "csv_extractor": "CSVExtractor (discriminated union, structured CSVSchema)",
            "execution_path": "generate_stream() -> gateway_client.complete() -> LLM Gateway /query -> persistent pandas worker (module-level singleton, boots once at FastAPI startup via lifespan, self-healing on file-load and liveness)",
            "sandbox_security": "AST allowlist (ALLOWED_ATTRIBUTES) + runtime SafeModuleProxy/SafeDataFrameProxy defense-in-depth + POSIX resource.setrlimit guarded for Windows dev",
            "gateway_caching": "Jina text-matching embeddings, semantic similarity threshold 0.90; disabled for any request where tools is not None OR is_tool_related is True",
            "gateway_routing": "sticky per-conversation provider (token bucket, Groq 30rpm/Gemini 15rpm), retry-with-backoff (2 attempts/provider) before fallback",
            "data_cleaning": "shared strip_stray_quotes() utility, used by both CSVExtractor._load() and the persistent worker's CSV load",
            "schema_serialization": "to_prompt_string() with THRESHOLD=20 truncation + dtype-grouped summary for remaining columns",
            "generator_model": settings.model_name,
            "scoring": "deterministic numeric/range/string match (via compute_golden_value using golden_pandas_code) with LLM-judge fallback; word-boundary regex match for the non-tolerance string path (fixed from raw substring match, which risked false positives on short numeric golden values like single-digit class labels)",
            "tolerance": "0.001 globally",
            "sample_size": f"{len(results)} of 45 golden questions (schema/count/stat/filter/groupby categories sampled)",
        },
        "averages": {"tool_appropriateness": avg_tool, "answer_correctness": avg_correctness},
        "per_question": results,
    }

    history_path = os.path.join(BASE_DIR, "eval_history.json")
    history = []
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            history = json.load(f)
    history.append(run_record)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"Finalized run with {len(results)} questions saved to {history_path}")
def recompute_last_run_scores():
    """Re-scores the most recent eval run using the corrected golden dataset,
    WITHOUT re-querying any LLM — the generated answers/tool_used values
    already collected are reused as-is. Only needs_computation changed."""
    with open(DATASET_PATH, "r") as f:
        golden_data = json.load(f)
    golden_by_id = {q["id"]: q for q in golden_data}

    history_path = os.path.join(BASE_DIR, "eval_history.json")
    with open(history_path, "r") as f:
        history = json.load(f)

    last_run = history[-1]
    corrected_results = []

    for item in last_run["per_question"]:
        golden = golden_by_id.get(item["id"])
        should_use = golden["needs_computation"] if golden else True
        new_tool_score = check_tool_app(should_use, item["tool_used"])
        item["scores"]["tool_appropriateness"] = new_tool_score
        corrected_results.append(item)

    avg_tool = sum(r["scores"]["tool_appropriateness"] for r in corrected_results) / len(corrected_results)
    avg_correctness = sum(r["scores"]["answer_correctness"] for r in corrected_results) / len(corrected_results)

    corrected_run = {
        "timestamp": datetime.utcnow().isoformat(),
        "type": "csv",
        "label": "csv-eval-gateway-production-pass-corrected",
        "description": (
            f"Score correction of '{last_run['label']}' — no new LLM calls made, same "
            "generated answers reused. golden_csv_dataset.json had needs_computation=true "
            "for schema_02, stat_mean_f10, and stat_range_f10, incorrectly penalizing the "
            "model's tool_appropriateness for correctly answering these schema-lookup "
            "questions without invoking pandas_sandbox (all three were answered correctly, "
            "answer_correctness was never affected). Corrected to needs_computation=false "
            "for these three, matching their actual nature as schema-derivable facts."
        ),
        "config": last_run["config"],
        "averages": {"tool_appropriateness": avg_tool, "answer_correctness": avg_correctness},
        "per_question": corrected_results,
    }

    history.append(corrected_run)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"Corrected scores: tool={avg_tool:.2f}, correctness={avg_correctness:.2f}")
if __name__ == "__main__":
    
    recompute_last_run_scores()
    if SAVE_TO_HISTORY:
        from datetime import datetime
        run_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": "csv",
            "label": "csv-eval-baseline-post-fixes",
            "description": (
                "First working CSV eval run, after fixing several blocking issues found "
                "along the way. (1) Generator() in the eval script was constructed with no "
                "tools registered (self.tools=[]), unlike production which mutates "
                "generator.tools per-request in the /query/stream route -- fixed by "
                "explicitly setting generator.tools = [PandasSandboxTool(file_path=CSV_PATH)] "
                "in the eval script to match production behavior. (2) CSVSchema.to_prompt_string() "
                "printed every column unconditionally, causing a 40k-token prompt on this "
                "617-column CSV and a Groq 413 'request too large' error -- fixed by adding a "
                "THRESHOLD-based branch: full detail for the first 20 columns, dtype-grouped "
                "summary lines for the remainder, cutting the schema string to ~2.7k chars. "
                "(3) One golden_dataset.json row (schema_06) referenced an undefined "
                "'feature_cols' variable in its golden_pandas_code -- fixed by rewriting it as "
                "a self-contained expression using df.columns directly. (4) generate_with_tools() "
                "Call 2 (post-tool synthesis) was replaying tool_calls history and passing "
                "tools/tool_choice='none', which could cause the model to attempt a further tool "
                "call and get rejected by Groq ('tool choice is none, but model called a tool') "
                "-- fixed by building a fresh system+user-only synthesis_messages list with no "
                "tools/tool_choice passed at all, and adding an explicit instruction telling the "
                "model not to perform further computation. (5) Hit Groq's on-demand tier TPM rate "
                "limit (8000 TPM) running the full 45-question suite back-to-back -- for this run, "
                "questions were validated one at a time via run_single_csv_eval() rather than the "
                "full batch loop, to avoid rate-limit interruptions during initial validation. "
                "(6) Also required activating the project's .venv before running -- an earlier "
                "attempt using the global Python interpreter failed with "
                "ModuleNotFoundError: tiktoken."
            ),
            "config": {
                "csv_extractor": "CSVExtractor (discriminated union, structured CSVSchema)",
                "schema_serialization": "to_prompt_string() with THRESHOLD=20 truncation + dtype-grouped summary for remaining columns",
                "generator_model": settings.model_name,
                "tool_registration": "generator.tools set explicitly in eval script to match production's per-request mutation",
                "synthesis_call": "fresh system+user messages, no tools/tool_choice passed, explicit no-further-computation instruction",
                "scoring": "deterministic numeric/string match (via compute_golden_value using golden_pandas_code) with LLM-judge fallback for categorical/fuzzy answers",
            },
            "averages": output["averages"],
            "per_question": output["per_question"],
                }
        history_path = os.path.join(BASE_DIR, "eval_history.json")
        history = []
        if os.path.exists(history_path):
            with open(history_path, "r") as f:
                history = json.load(f)
        history.append(run_record)
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"Saved run to {history_path}")
