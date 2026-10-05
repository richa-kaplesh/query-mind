"""Scoring helpers shared by the live eval runner.

The number-extraction and correctness logic is copied from run_csv_eval.py so live
results are scored exactly like your earlier local runs. It has no backend imports,
so it can run from anywhere.
"""
import json
import os
import re

_DASH_MAP = str.maketrans({"\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-"})

_SIGNAL_PATTERN = re.compile(
    r"(?:\bis\b|\bare\b|\bwas\b|\bwere\b|\bequals?\b|\bapproximately\b|\babout\b)"
    r"\s*\**\s*(-?\d+\.?\d*)",
    re.IGNORECASE,
)

JUDGE_MODEL = "openai/gpt-oss-120b"


def _clean(text: str) -> str:
    return text.translate(_DASH_MAP)


def _bare_numbers(text: str) -> list[float]:
    """Standalone numbers not glued to a letter prefix (skips f10, f617)."""
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
        return float(signal_matches[-1])
    numbers = _bare_numbers(text)
    return numbers[-1] if numbers else None


def extract_all_numbers(text: str) -> list[float]:
    text = _clean(text)
    signal_matches = _SIGNAL_PATTERN.findall(text)
    if len(signal_matches) >= 2:
        return [float(x) for x in signal_matches]
    return _bare_numbers(text)


def check_tool_app(should_use, actual_tool_used):
    """1.0 if tool use matched expectation, None if the question doesn't specify."""
    if should_use is None:
        return None
    return 1.0 if (actual_tool_used is not None) == should_use else 0.0


def _get_groq_client():
    from groq import Groq
    key = os.getenv("GROQ_API_KEY")
    if not key:
        from config import settings  # works when run from the backend/ folder
        key = settings.groq_api_key
    return Groq(api_key=key)


def llm_judge(golden, answer: str, question: str = "", use_judge: bool = True):
    """Returns a 0.0-1.0 score, or None when the judge is disabled / unavailable."""
    if not use_judge:
        return None
    prompt = f"""You are an evaluation judge for a question-answering system.
Question: {question}
Ground Truth: {golden}
Generated Answer: {answer}

Does the Generated Answer correctly convey the Ground Truth, even if worded
differently? Score from 0.0 to 1.0.

Return ONLY a JSON object like this, nothing else:
{{"answer_correctness": 0.0}}"""
    try:
        client = _get_groq_client()
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return float(json.loads(raw.strip()).get("answer_correctness", 0.0))
    except Exception as e:
        print(f"    [judge] failed: {e}")
        return None


def score_correctness(golden, answer: str, tolerance, question: str = "", use_judge: bool = True):
    if tolerance is not None:
        if isinstance(golden, (list, tuple)):
            numbers = extract_all_numbers(answer)
            if len(numbers) < 2:
                return 0.0
            ok_min = abs(numbers[0] - float(golden[0])) <= tolerance
            ok_max = abs(numbers[1] - float(golden[1])) <= tolerance
            return 1.0 if (ok_min and ok_max) else 0.0
        actual = extract_number(answer)
        if actual is None:
            return 0.0
        return 1.0 if abs(actual - float(golden)) <= tolerance else 0.0

    golden_str = str(golden).strip().lower()
    pattern = r"\b" + re.escape(golden_str) + r"\b"
    if re.search(pattern, answer.strip().lower()):
        return 1.0
    return llm_judge(golden, answer, question, use_judge)