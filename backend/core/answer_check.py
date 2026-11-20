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


WORD_PATTERN = re.compile(r"[a-z0-9]+")
NON_ANSWER_MAX_WORDS = 3


def is_non_answer(answer: str, question: str) -> bool:
    answer_words = WORD_PATTERN.findall(answer.lower())
    if not answer_words:
        return True
    if len(answer_words) > NON_ANSWER_MAX_WORDS:
        return False
    if any(ch.isdigit() for ch in answer):
        return False
    question_words = WORD_PATTERN.findall(question.lower())
    size = len(answer_words)
    for start in range(len(question_words) - size + 1):
        if question_words[start:start + size] == answer_words:
            return True
    return False


TERM_PATTERN = re.compile(r"[a-z]+")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
MIN_TERM_LENGTH = 3
MIN_FUZZY_LENGTH = 4
PREMISE_MIN_WORDS = 3
GENERIC_WORDS = frozenset("""
a an the of to in on for and or is are was were be been by with as at from what which who whom whose
how many much does do did have has had this that these those there their its it each every all any
some per across into than then if not no yes most least more less fewest largest smallest highest
lowest biggest top average mean median sum total count number numbers value values column columns row
rows dataset data file record records sample samples feature features share percentage percent ratio
difference between range minimum maximum min max std standard deviation variance distinct different
unique missing null nulls empty times time give find show list compute calculate about over under below
above versus including include actually should would could can will just only other another same also
both either where when why own type types kind belong belongs common
""".split())


def _stem(word: str) -> str:
    if len(word) > MIN_TERM_LENGTH and word.endswith("s"):
        return word[:-1]
    return word


def unknown_terms(question: str, schema_text: str) -> list[str]:
    known = {_stem(w) for w in TERM_PATTERN.findall(schema_text.lower())}
    fuzzy = [k for k in known if len(k) >= MIN_FUZZY_LENGTH]
    unknown = []
    for word in TERM_PATTERN.findall(question.lower()):
        if len(word) < MIN_TERM_LENGTH or word in GENERIC_WORDS:
            continue
        term = _stem(word)
        if term in known:
            continue
        if len(term) >= MIN_FUZZY_LENGTH and any(term in k or k in term for k in fuzzy):
            continue
        unknown.append(word)
    return unknown


def has_premise(question: str) -> bool:
    for sentence in SENTENCE_SPLIT.split(question.strip()):
        if sentence.endswith("?"):
            continue
        if len(TERM_PATTERN.findall(sentence.lower())) >= PREMISE_MIN_WORDS:
            return True
    return False


def skip_blocker(question: str, schema_text: str | None) -> str | None:
    if has_premise(question):
        return "premise"
    terms = unknown_terms(question, schema_text or "")
    if terms:
        return "unknown_terms: " + ", ".join(terms)
    return None
