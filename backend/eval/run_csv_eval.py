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

def run_csv_query(question: str, schema: CSVSchema, tracer = None) -> dict:
    result = generator.generate_with_tools(query=question, schema=schema, tracer = tracer)
    return result


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
        "label": "csv-eval-final",
        "description": (
            "Final pass on the canonical 15-question CSV eval set, after a full day of "
            "root-causing why the original run scored only 20% answer_correctness. Summary "
            "of every distinct bug found and fixed across the investigation: "
            "(1) extract_number()/extract_all_numbers() originally used a plain digit regex "
            "with no letter-boundary check, matching digits embedded in column names (e.g. "
            "the '1' in 'f1') as standalone numbers -- fixed with a token-based approach that "
            "rejects any digit sequence glued to a letter prefix, correctly handling "
            "multi-digit column suffixes (f10, f100, f617) that a simple lookbehind missed. "
            "(2) Unicode dash variants (U+2010-U+2013, from markdown auto-formatting) weren't "
            "recognized as negative signs by the ASCII-only '-' in the regex -- fixed by "
            "normalizing all dash variants to ASCII '-' before parsing. "
            "(3) Both extraction functions were rewritten a final time to anchor on linking "
            "words ('is', 'are', 'equals', 'approximately', etc.) rather than taking the "
            "first or last number in the text -- the earlier position-based heuristics failed "
            "on answers that echoed a question parameter before stating the result (e.g. "
            "'...for rows in class 20 is -0.4856' was extracting '20' instead of the actual "
            "answer). The signal-word anchor generalizes correctly across every phrasing "
            "pattern observed in this dataset's answers, with the old letter-prefix logic "
            "kept as a fallback for phrasing with no linking verb. "
            "(4) tolerance was loosened from 0.0001 to 0.001 globally across "
            "golden_dataset.json, since the tighter value implicitly demanded 4+ decimal "
            "places of precision in prose answers that were never asked for. "
            "(5) Found and fixed a deeper data bug: the 'class' column's values contain "
            "literal embedded quote characters in the source CSV (e.g. \"'1'\" not \"1\"), "
            "confirmed by inspecting the raw file directly. This caused the model's "
            "correctly-written comparison code (df['class']==1) to silently match zero rows. "
            "Fixed by stripping stray quote characters and converting to numeric where every "
            "value parses cleanly, in a new shared strip_stray_quotes() utility. "
            "(6) That fix initially only applied to CSVExtractor's dataframe, not "
            "PandasSandboxTool's -- the sandbox tool loads the CSV independently in its own "
            "worker process (required since a live DataFrame can't cross a multiprocessing "
            "process boundary), so it was still operating on unclean, quoted string data "
            "even after the extractor was fixed. Resolved by moving the cleaning logic into "
            "a shared utility function imported by both, eliminating the drift between the "
            "two independent CSV loads. "
            "(7) Several golden_dataset.json entries (count_04, filter_02, groupby_02) had "
            "golden_pandas_code written against the old quoted-string class format (e.g. "
            "df['class']=='1') and silently broke in the same way once the data was cleaned "
            "-- each question's own recorded golden_answer field was used as ground truth to "
            "confirm the correct value, then the golden code was updated to compare against "
            "clean numeric class values. "
            "(8) Call 2 (post-tool synthesis) intermittently crashed with "
            "groq.BadRequestError ('tool choice is none, but model called a tool') due to "
            "openai/gpt-oss-20b's tendency to attempt an unprompted 'python' tool call even "
            "with no tools declared -- fixed with a try/except fallback to the raw tool "
            "result, plus a tightened synthesis instruction explicitly forbidding any "
            "tool/function call, not just code. Call 1 was separately given retry logic (3 "
            "attempts, 1s delay) after a malformed tool-call-argument generation error was "
            "observed. "
            "(9) Root-caused and fixed the original 'model claims a column doesn't exist' "
            "failure family (schema_03, count_04, count_06, filter_02, groupby_02 in earlier "
            "runs): to_prompt_string() computed a dtype-grouped summary for columns beyond "
            "its THRESHOLD=20 detail cutoff but never appended it to the returned prompt "
            "string, so columns 21-617 (including the 'class' label column) were completely "
            "invisible to the model, not vaguely summarized as previously assumed. Fixed by "
            "appending the summary lines, and separately updated PandasSandboxTool's "
            "description to explicitly tell the model to check df.columns for anything not "
            "shown in full schema detail. "
            "KNOWN, NOT FIXED: count_05 ('which class has the fewest samples, and how many') "
            "is a compound-answer question -- the model answered only the count (298) and "
            "not the class label (6) that golden_value expects, but the count itself is "
            "correct. This is a question-design/scoring gap (single-value scoring applied to "
            "a two-part question), not a model or data bug -- needs either a compound-answer "
            "scorer or splitting into two separate questions. Flagged for next investigation, "
            "not addressed this pass. count_06 has a related but distinct issue: multiple "
            "classes tie at the maximum count (300), and the scorer expects one specific "
            "class rather than accepting any class in the tied set."
        ),
        "config": {
            "csv_extractor": "CSVExtractor (discriminated union, structured CSVSchema)",
            "data_cleaning": "shared strip_stray_quotes() utility (core/utils/data_cleaning.py), used by both CSVExtractor._load() and PandasSandboxTool's worker process -- strips embedded quote characters from object-dtype columns and converts to numeric where every value parses cleanly",
            "schema_serialization": "to_prompt_string() with THRESHOLD=20 truncation; columns past the cutoff are now summarized by name and dtype (previously computed but never appended -- fixed)",
            "generator_model": settings.model_name,
            "tool_registration": "generator.tools set explicitly in eval script to match production's per-request mutation",
            "pandas_sandbox_description": "explicitly instructs the model to check df.columns / inspect values for columns not shown in full schema detail",
            "synthesis_call": "fresh system+user messages, no tools/tool_choice passed, explicit no-tool/no-function-call instruction; wrapped in try/except with raw-tool-result fallback on synthesis failure",
            "call_1_retry": "3 attempts with 1s delay on failure, graceful error message after exhausting retries",
            "scoring": "deterministic numeric/range/string match (via compute_golden_value using golden_pandas_code) with LLM-judge fallback for categorical/fuzzy answers; extract_number/extract_all_numbers anchor on linking words (is/are/equals/approximately) rather than first-or-last-number position, with letter-prefix-rejection as fallback for answers without linking verbs",
            "tolerance": "0.001 globally (raised from 0.0001 across all golden_dataset.json questions)",
            "tracer": "generate_with_tools()'s tracer hook wired from the eval script, capturing per-question pandas_code and raw_tool_result alongside the final answer",
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

if __name__ == "__main__":
    
    run_single_csv_eval("count_05")
    run_single_csv_eval("count_07")
   
    
    run_single_csv_eval("count_03")
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
