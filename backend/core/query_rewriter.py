class QueryWriter:
    def __init__(self):
        pass

    def _needs_rewrite(self, query: str , has_history: bool)-> bool:
        query_lower = query.lower()

        conjunction_words = ["and", "vs","versus","compare"]
        has_conjunction = any(word in query_lower.split() for word in conjunction_words)

        has_multiple_questions = query.count("?")>1

        reference_words = ["it", "that","this", "they"]
        has_reference_words = any(word in query_lower.split() for word in reference_words) and has_history

        is_short_followup = len(query.split()) <5 and has_history

        return has_conjunction or has_multiple_questions or has_reference_words or is_short_followup