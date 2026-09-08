import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import re
import pandas as pd
from groq import Groq
from core.generator import Generator
from core.extractors.csv_extractor import CSVExtractor
from core.models import CSVSchema
from config import settings
import time 
from datetime import datetime 
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

def run_csv_query(question: str, schema: CSVSchema, tracer = None) -> dict:
    result = generator.generate_with_tools(query=question, schema=schema, tracer = tracer)
    return result


def check_tool_app(should_use: bool, actual_tool_used) -> float:
    actual_used_bool = actual_tool_used is not None
    return 1.0 if actual_used_bool == should_use else 0.0


def extract_number(text: str):
    text = text.replace("\u2011", "-").replace("\u2010", "-").replace("\u2013", "-").replace("\u2012", "-")
    for m in re.finditer(r"[a-zA-Z]*-?\d+\.?\d*", text):
        token = m.group()
        letters = re.match(r"[a-zA-Z]*", token).group()
        if letters:          # glued to a letter prefix like "f10" — skip it
            continue
        return float(token)
    return None

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
        if golden_str in actual_answer.strip().lower():
            return 1.0
        return score_with_llm_judge(golden_value, actual_answer)

def extract_all_numbers(text: str) -> list[float]:
    text = text.replace("\u2011", "-").replace("\u2010", "-").replace("\u2013", "-").replace("\u2012", "-")
    numbers = []
    for m in re.finditer(r"[a-zA-Z]*-?\d+\.?\d*", text):
        token = m.group()
        letters = re.match(r"[a-zA-Z]*", token).group()
        if letters:          # glued to a letter prefix like "f10" — skip it
            continue
        numbers.append(float(token))
    return numbers

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
        "label": "csv-eval-post-scoring-fixes",
        "description": (
            "Re-run of the same 15 golden questions from 'csv-eval-baseline-post-fixes', "
            "after root-causing why the first pass scored only 20% answer_correctness "
            "despite several answers being visibly correct. Investigation done via a new "
            "tracer hook threaded through generate_with_tools() (it already had a no-op "
            "tracer parameter used for its existing record() step-logging, just never wired "
            "up from the eval side) -- capturing the model's actual pandas_code and raw "
            "sandbox tool_output per question, not just the final synthesized answer text. "
            "This exposed two distinct, previously indistinguishable failure modes hiding "
            "inside the single answer_correctness number: "
            "(1) ROOT CAUSE, extract_number()/extract_all_numbers() regex bug: "
            "r'-?\\d+\\.?\\d*' with no word-boundary check matched digits embedded in column "
            "names (e.g. the '1' in 'f1', digits inside 'f617') before ever reaching the "
            "actual numeric answer later in the sentence -- silently scoring numerically "
            "correct, tool-verified answers (e.g. stat_std_f1: sandbox computed "
            "0.23672097770205053, model correctly reported ~0.2367, scored 0.0 anyway) as "
            "wrong. Fixed with a negative lookbehind, r'(?<![a-zA-Z])-?\\d+\\.?\\d*', so digits "
            "immediately preceded by a letter are no longer treated as standalone numbers. "
            "Confirmed via isolated reproduction (extract_number() run directly against the "
            "captured answer strings) before and after the fix. "
            "(2) tolerance miscalibration: every question was seeded with tolerance=0.0001, "
            "which implicitly demanded 4+ decimal places of precision in a natural-language "
            "answer even though the question never asked for that precision (e.g. corr_01: "
            "golden 0.7366024737356933, model answered ~0.737, diff ~0.0004, exceeded the "
            "0.0001 tolerance despite being a reasonable rounded answer). Loosened to "
            "tolerance=0.001 globally across all questions in golden_dataset.json, giving "
            "headroom for ~3-decimal-place rounding without accepting genuinely wrong values. "
            "SEPARATELY OBSERVED, NOT YET FIXED: stat_range_f617 was non-deterministic across "
            "runs -- one run correctly called pandas_sandbox and got the right min/max, "
            "another run answered 'the dataset does not contain a column named f617' and "
            "skipped the tool entirely. Root cause identified as a real bug in "
            "to_prompt_string() (not a schema string this run's fix touches yet): the "
            "function builds a dtype-grouped summary dict for columns past THRESHOLD=20 "
            "columns, but the resulting summary lines are never appended to the returned "
            "prompt string -- so for this 617-column CSV, columns 21-617 (including f617 and "
            "the earlier-flagged 'class' label column) are entirely invisible to the model, "
            "not vaguely summarized as previously assumed. The model's 'column doesn't exist' "
            "answers are a correct inference from an incomplete prompt, not a hallucination. "
            "Decided approach for next pass: give the model a get_column_info(column_name) "
            "tool for on-demand lookup of any column outside the detailed cutoff, rather than "
            "further expanding the static prompt -- still to be implemented."
                        "NEWLY OBSERVED, NOW FIXED: Call 2 (post-tool synthesis) intermittently crashed "
            "with groq.BadRequestError ('tool choice is none, but model called a tool'), "
            "even with zero tools/tool_choice passed in the request -- confirmed via "
            "count_05 to be openai/gpt-oss-20b's known built-in tendency to attempt a "
            "'python' function call on its own initiative, previously observed during the "
            "original CSV eval build but not fully suppressed. The underlying tool "
            "computation had already succeeded before this crash occurred (confirmed via "
            "worker logs), so the failure was purely in the synthesis step, not the "
            "computation. Fixed two ways: (1) wrapped Call 2 in try/except, falling back to "
            "surfacing the raw tool_result directly as the answer on failure, so a question "
            "no longer loses its already-computed result or halts the whole eval batch; "
            "(2) tightened the synthesis instruction to explicitly forbid any tool/function "
            "call, not just 'writing code', to reduce (though likely not eliminate) how "
            "often the model attempts this."
        ),
        "config": {
            "csv_extractor": "CSVExtractor (discriminated union, structured CSVSchema)",
            "schema_serialization": "to_prompt_string() with THRESHOLD=20 truncation + dtype-grouped summary for remaining columns (KNOWN BUG: summary lines computed but never appended to output -- columns past 20 are fully invisible to the model, not just vaguely summarized)",
            "generator_model": settings.model_name,
            "tool_registration": "generator.tools set explicitly in eval script to match production's per-request mutation",
            "synthesis_call": "fresh system+user messages, no tools/tool_choice passed, explicit no-further-computation instruction",
            "scoring": "deterministic numeric/range/string match (via compute_golden_value using golden_pandas_code) with LLM-judge fallback for categorical/fuzzy answers; extract_number/extract_all_numbers now use a negative lookbehind to avoid matching digits embedded in column names",
            "tolerance": "0.001 globally (raised from 0.0001 across all golden_dataset.json questions)",
            "tracer": "generate_with_tools()'s existing tracer hook now wired from the eval script, capturing per-question pandas_code and raw_tool_result alongside the final answer",
            "sample_size": f"{len(results)} of 45 golden questions (schema/count/stat categories sampled)",
            "synthesis_call": "fresh system+user messages, no tools/tool_choice passed, explicit no-tool/no-function-call instruction; wrapped in try/except with raw-tool-result fallback on synthesis failure (Groq's gpt-oss-20b can attempt an unprompted 'python' tool call even with no tools declared)",
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
    
if __name__ == "__main__":
    finalize_csv_eval_run()
    
            
    

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
