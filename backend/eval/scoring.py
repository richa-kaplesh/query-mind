"""Scoring helpers shared by the live eval runner.

Numeric answers are scored deterministically (tolerance based). Text answers are
matched deterministically where possible and only fall back to an LLM judge when
no deterministic rule applies. The judge call retries, so a rate-limit blip does
not leave a question unscored.

Golden answer shapes handled:
  number + tolerance        -> numeric comparison
  [min, max] + tolerance    -> both numbers must match, in order
  number, no tolerance      -> whole-token match (thousands separators ignored)
  dict                      -> every value must appear in the answer
  any golden + "must_include" list in the dataset -> every item must appear
  other text                -> whole-token match, else LLM judge
"""
import json
import os
import re
import time

_DASH_MAP = str.maketrans({"\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-"})

_SIGNAL_PATTERN = re.compile(
    r"(?:\bis\b|\bare\b|\bwas\b|\bwere\b|\bequals?\b|\bapproximately\b|\babout\b)"
    r"\s*\**\s*(-?\d+\.?\d*)",
    re.IGNORECASE,
)

JUDGE_MODEL = "openai/gpt-oss-120b"
JUDGE_ATTEMPTS = 4
JUDGE_FIRST_DELAY_S = 2.0
# Errors that retrying cannot fix (bad install, bad code): fail fast instead.
_NON_RETRYABLE = (TypeError, ImportError, AttributeError, KeyError)

_groq_client = None


def _clean(text: str) -> str:
    return text.translate(_DASH_MAP)


def _normalize(text: str) -> str:
    """Lowercase and drop thousands separators so '7,797' matches 7797."""
    text = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text)
    return text.strip().lower()


def _has_token(text: str, token) -> bool:
    """Whole-token match: '26' matches 'class 26' but not '126' or '2.6'."""
    pattern = r"(?<![\w.])" + re.escape(str(token).strip().lower()) + r"(?![\w])"
    return re.search(pattern, text) is not None


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
    global _groq_client
    if _groq_client is None:
        from groq import Groq
        key = os.getenv("GROQ_API_KEY")
        if not key:
            from config import settings  # works when run from the backend/ folder
            key = settings.groq_api_key
        _groq_client = Groq(api_key=key)
    return _groq_client


def llm_judge(golden, answer: str, question: str = "", use_judge: bool = True):
    """0.0-1.0 score, or None if the judge is disabled or still failing after retries."""
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

    delay = JUDGE_FIRST_DELAY_S
    last_error = None
    for attempt in range(1, JUDGE_ATTEMPTS + 1):
        try:
            response = _get_groq_client().chat.completions.create(
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
            score = float(json.loads(raw.strip()).get("answer_correctness", 0.0))
            return max(0.0, min(1.0, score))
        except _NON_RETRYABLE as e:
            last_error = e
            break
        except Exception as e:  # rate limit, timeout, 5xx, bad JSON: worth retrying
            last_error = e
            if attempt < JUDGE_ATTEMPTS:
                time.sleep(delay)
                delay *= 2
    print(f"    [judge] gave up: {type(last_error).__name__}: {last_error}")
    return None


def score_correctness(golden, answer: str, tolerance, question: str = "",
                      use_judge: bool = True, must_include=None):
    # 1. numeric answers: deterministic, tolerance based
    if tolerance is not None and isinstance(golden, (list, tuple)):
        numbers = extract_all_numbers(answer)
        if len(numbers) < 2:
            return 0.0
        ok_min = abs(numbers[0] - float(golden[0])) <= tolerance
        ok_max = abs(numbers[1] - float(golden[1])) <= tolerance
        return 1.0 if (ok_min and ok_max) else 0.0
    if tolerance is not None and isinstance(golden, (int, float)):
        actual = extract_number(answer)
        if actual is None:
            return 0.0
        return 1.0 if abs(actual - float(golden)) <= tolerance else 0.0

    normalized = _normalize(answer)

    # 2. multi-part answers: every required piece must be present
    required = must_include
    if required is None and isinstance(golden, dict):
        required = list(golden.values())
    if required:
        return 1.0 if all(_has_token(normalized, item) for item in required) else 0.0

    # 3. single value: whole-token match, else the judge decides
    if _has_token(normalized, golden):
        return 1.0
    return llm_judge(golden, answer, question, use_judge)