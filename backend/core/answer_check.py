import re

NUMBER_PATTERN = re.compile(r"(?<![A-Za-z_\d.])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![A-Za-z_\d])")


def extract_numbers(text: str) -> set[float]:
    numbers = set()
    for match in NUMBER_PATTERN.findall(text):
        numbers.add(float(match.replace(",", "")))
    return numbers


def _decimals(n: float) -> int:
    if n.is_integer():
        return 0
    return len(str(n).split(".")[1])


def _matches_any(n: float, allowed: set[float]) -> bool:
    places = _decimals(n)
    for a in allowed:
        if round(a, places) == n:
            return True
    return False


def is_trivially_verified(answer: str, tool_log: list[tuple[str, str, str]], question: str) -> bool:
    if not tool_log:
        return False
    answer_numbers = extract_numbers(answer)
    if not answer_numbers:
        return False
    allowed = set()
    for name, tool_input, result in tool_log:
        allowed = allowed | extract_numbers(result)
    allowed = allowed | extract_numbers(question)
    for n in answer_numbers:
        if not _matches_any(n, allowed):
            return False
    return True
