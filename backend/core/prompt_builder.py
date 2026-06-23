from typing import List


class PromptBuilder:

    def build_system_prompt(self, has_chunks: bool = False) -> str:
        if has_chunks:
            return """You are a research assistant. Answer the user's question using ONLY the context provided below.
Rules:
- Only use information from the provided context.
- Always cite which source your answer came from.
- If the answer is not in the context, say "I cannot find this in the provided documents"."""

        return """You are a helpful research assistant with access to the following tools:

- get_csv_stats: Use this when the user asks analytical, statistical, or ML-related questions about a CSV dataset.
- search_web: Use this for questions about current events, specific companies, people, or anything requiring up to date information. When in doubt, search.

Never output raw function calls or XML in your response. If a tool returns no results, say so naturally and answer from your own knowledge instead.
For general knowledge questions you can answer directly without tools."""

    def build_context(self, chunks: List[dict]) -> str:
        context_parts = []
        for i, chunk in enumerate(chunks):
            source = chunk["metadata"].get("source", "unknown")
            page = chunk["metadata"].get("page", "N/A")
            text = chunk["text"]
            context_parts.append(f"[SOURCE {i+1} - {source}, page {page}]\n{text}")
        return "\n\n".join(context_parts)

    def build_rag_prompt(self, query: str, context: str) -> str:
        return f"CONTEXT:\n{context}\n\nQUESTION:\n{query}\n\nANSWER (with citations):"