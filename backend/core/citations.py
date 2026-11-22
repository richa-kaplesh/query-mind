import re

SNIPPET_CHARS = 200
MAX_UNCITED_SOURCES = 5

_RAW_SOURCE_RE = re.compile(r"\s*[\(\[]\s*Source\s*:[^\]\)]*[\]\)]", re.IGNORECASE)
_NUMBER_GROUP_RE = re.compile(r"\[(?:S|Passage\s*)?(\d+(?:\s*(?:,|;|and)\s*(?:S|Passage\s*)?\d+)*)\]", re.IGNORECASE)
_FILE_CITE_RE = re.compile(r"\s*\[([^\[\]]*?\.pdf)\s*(?:[,|]\s*(?:p\.?|page)\s*:?\s*(\d+))?[^\[\]]*\]", re.IGNORECASE)


def _snippet(text: str) -> str:
    flat = " ".join((text or "").split())
    if len(flat) <= SNIPPET_CHARS:
        return flat
    cut = flat[:SNIPPET_CHARS].rsplit(" ", 1)[0]
    return cut + "…"


class PassageRegistry:
    def __init__(self, display_name: str | None):
        self.display_name = display_name or "document"
        self.passages: list[dict] = []
        self._by_key: dict[tuple, int] = {}

    def add(self, chunk: dict) -> int:
        meta = chunk.get("metadata", {})
        key = (meta.get("page"), meta.get("heading"), (chunk.get("text") or "").strip()[:200])
        if key in self._by_key:
            idx = self._by_key[key]
            score = chunk.get("rerank_score")
            if score is not None and score > (self.passages[idx - 1]["score"] or 0):
                self.passages[idx - 1]["score"] = float(score)
            return idx
        self.passages.append({
            "filename": self.display_name,
            "page": meta.get("page"),
            "section": meta.get("heading") or None,
            "snippet": _snippet(chunk.get("text", "")),
            "score": float(chunk["rerank_score"]) if chunk.get("rerank_score") is not None else None,
        })
        idx = len(self.passages)
        self._by_key[key] = idx
        return idx

    def label(self, idx: int) -> str:
        p = self.passages[idx - 1]
        parts = [f"[{idx}] {p['filename']}"]
        if p["page"] is not None:
            parts.append(f"Page {p['page']}")
        if p["section"]:
            parts.append(f"Section: {p['section']}")
        return " | ".join(parts)


def finalize_citations(answer: str, registry: PassageRegistry | None) -> tuple[str, list[dict]]:
    passages = registry.passages if registry else []
    order: list[int] = []

    def number_for(original: int) -> int | None:
        if original < 1 or original > len(passages):
            return None
        if original not in order:
            order.append(original)
        return order.index(original) + 1

    def page_match(page: str | None) -> int | None:
        if page is None:
            return None
        for i, p in enumerate(passages, start=1):
            if str(p["page"]) == str(page):
                return i
        return None

    def replace_raw(match: re.Match) -> str:
        page = re.search(r"Page\s*:?\s*(\d+)", match.group(0), re.IGNORECASE)
        original = page_match(page.group(1)) if page else None
        n = number_for(original) if original else None
        return f" \x00{n}\x00" if n else ""

    def replace_file(match: re.Match) -> str:
        original = page_match(match.group(2))
        n = number_for(original) if original else None
        return f" \x00{n}\x00" if n else ""

    def replace_numbers(match: re.Match) -> str:
        nums = [int(x) for x in re.findall(r"\d+", match.group(1))]
        mapped = [number_for(x) for x in nums]
        mapped = [m for m in mapped if m]
        if not mapped:
            return match.group(0)
        return "".join(f"[{m}]" for m in dict.fromkeys(mapped))

    text = _RAW_SOURCE_RE.sub(replace_raw, answer or "")
    text = _FILE_CITE_RE.sub(replace_file, text)
    text = _NUMBER_GROUP_RE.sub(replace_numbers, text)
    text = re.sub(r"\x00(\d+)\x00", r"[\1]", text)
    text = re.sub(r"[ \t]+([.,;:!?])", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text).strip()

    cited = [passages[i - 1] for i in order]
    uncited = sorted(
        (p for i, p in enumerate(passages, start=1) if i not in order),
        key=lambda p: p["score"] or 0, reverse=True,
    )[:MAX_UNCITED_SOURCES]
    sources = [dict(p, n=k) for k, p in enumerate(cited + uncited, start=1)]
    return text, sources
