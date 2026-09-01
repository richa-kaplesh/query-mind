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

def run_csv_query(question: str, schema: CSVSchema) -> dict:
    result = generator.generate_with_tools(query=question, schema=schema)
    return result


def check_tool_app(should_use: bool, actual_tool_used) -> float:
    actual_used_bool = actual_tool_used is not None
    return 1.0 if actual_used_bool == should_use else 0.0


def extract_number(text: str):
    """Pull the first numeric value out of a generated answer string."""
    match = re.search(r"-?\d+\.?\d*", text)
    return float(match.group()) if match else None


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
    matches = re.findall(r"-?\d+\.?\d*", text)
    return [float(m) for m in matches]

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
    result = run_csv_query(question, schema)
    actual_answer = result["answer"]
    actual_tool_used = result["tool_used"]

    tool_score = check_tool_app(should_use, actual_tool_used)
    correctness_score = score_correctness(golden_value, actual_answer, tolerance)

    print(f"Question: {question}")
    print(f"Golden value: {golden_value}")
    print(f"Answer: {actual_answer}")
    print(f"Tool used: {actual_tool_used}")
    print(f"Tool Appropriateness: {tool_score} | Answer Correctness: {correctness_score}")

    # Load existing scratch results, append this one, save back
    scratch = []
    if os.path.exists(RESULTS_SCRATCH_PATH):
        with open(RESULTS_SCRATCH_PATH, "r") as f:
            scratch = json.load(f)

    # Remove any previous entry for this same question_id (in case you rerun it)
    scratch = [r for r in scratch if r["id"] != question_id]
    scratch.append({
        "id": question_id,
        "question": question,
        "answer": actual_answer,
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
        "label": "csv-eval-baseline-post-fixes",
        "description": (
            "First working CSV eval run (15 of 45 golden questions, sampled across "
            "schema/count/stat categories), after fixing several blocking issues found "
            "along the way. (1) Generator() in the eval script had no tools registered "
            "(self.tools=[]) unlike production, which mutates generator.tools per-request "
            "in the /query/stream route -- fixed by explicitly setting "
            "generator.tools = [PandasSandboxTool(file_path=CSV_PATH)] in the eval script. "
            "(2) CSVSchema.to_prompt_string() printed every column unconditionally, causing "
            "a ~40k-token prompt on this 617-column CSV and a Groq 413 'request too large' "
            "error -- fixed by adding a THRESHOLD-based branch: full detail for the first 20 "
            "columns, dtype-grouped summary for the rest, cutting the schema string to ~2.7k "
            "chars. (3) One golden_dataset.json row (schema_06) referenced an undefined "
            "'feature_cols' variable in its golden_pandas_code -- fixed by rewriting it as a "
            "self-contained expression using df.columns directly. (4) generate_with_tools() "
            "Call 2 (post-tool synthesis) replayed tool_calls history and passed "
            "tools/tool_choice='none', which let the model attempt a further tool call and "
            "get rejected by Groq ('tool choice is none, but model called a tool') -- fixed "
            "by building a fresh system+user-only synthesis_messages list with no "
            "tools/tool_choice passed, plus an explicit instruction telling the model not to "
            "perform further computation. (5) Hit Groq's on-demand tier TPM rate limit "
            "(8000 TPM) running questions back-to-back -- worked around by validating "
            "questions one at a time via run_single_csv_eval() instead of a full batch loop "
            "for this run. (6) score_correctness() crashed on stat_range_* questions because "
            "golden_pandas_code for range questions returns a (min, max) tuple, but the "
            "function only handled single-value numeric comparison -- fixed by branching on "
            "whether golden_value is a list/tuple (range comparison via a new "
            "extract_all_numbers() helper) vs a single number. "
            "KNOWN ISSUE FOUND, NOT YET FIXED: schema_02 ('how many feature columns, "
            "excluding class/label') failed -- the model answered 618 and claimed the class "
            "column 'is not present in the schema', when it is present but falls outside the "
            "first-20-columns detail cutoff and only appears inside a vague grouped dtype "
            "summary line. to_prompt_string()'s truncation currently has no concept of "
            "'always show the label/target column explicitly' -- likely needs a heuristic "
            "(e.g. always detail the last column, or detect an explicit label/target column) "
            "in addition to the first-N-columns cutoff. Flagged for next investigation pass."
        ),
        "config": {
            "csv_extractor": "CSVExtractor (discriminated union, structured CSVSchema)",
            "schema_serialization": "to_prompt_string() with THRESHOLD=20 truncation + dtype-grouped summary for remaining columns (known gap: doesn't guarantee label/target column visibility)",
            "generator_model": settings.model_name,
            "tool_registration": "generator.tools set explicitly in eval script to match production's per-request mutation",
            "synthesis_call": "fresh system+user messages, no tools/tool_choice passed, explicit no-further-computation instruction",
            "scoring": "deterministic numeric/range/string match (via compute_golden_value using golden_pandas_code) with LLM-judge fallback for categorical/fuzzy answers",
            "sample_size": f"{len(results)} of 45 golden questions (schema/count/stat categories sampled)",
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
