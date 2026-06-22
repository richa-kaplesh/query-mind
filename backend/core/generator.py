from groq import Groq
from typing import List
from core.tools.base_tool import BaseTool
from config import settings
import json


class Generator:

    def __init__(self, tools: List[BaseTool]):
        self.client = Groq(api_key=settings.groq_api_key)
        self.model = settings.model_name
        self.temperature = settings.temperature
        self.tools = {tool.name: tool for tool in tools}

    def _build_tools_schema(self) -> list:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The query to run"
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
            for tool in self.tools.values()
            if tool.name != "search_documents"  # RAG handled explicitly
        ]

    def _build_system_prompt(self, has_chunks: bool = False) -> str:
        if has_chunks:
            return """You are a research assistant. Answer the user's question using ONLY the context provided below.
Rules:
- Only use information from the provided context.
- Always cite which source your answer came from.
- If the answer is not in the context, say "I cannot find this in the provided documents"."""
        
        return """You are a helpful research assistant with access to the following tools:

- get_csv_stats: Use this when the user asks analytical, statistical, or ML-related questions about a CSV dataset.
- search_web: Use this for questions about current events, specific companies, people, or anything requiring up to date information. When in doubt, search.

For general knowledge questions you can answer directly without tools."""

    def _build_context(self, chunks: List[dict]) -> str:
        context_parts = []
        for i, chunk in enumerate(chunks):
            source = chunk["metadata"].get("source", "unknown")
            page = chunk["metadata"].get("page", "N/A")
            text = chunk["text"]
            context_parts.append(f"[SOURCE {i+1} - {source}, page {page}]\n{text}")
        return "\n\n".join(context_parts)

    def _run_tool(self, tool_name: str, query: str) -> str:
        tool = self.tools.get(tool_name)
        if not tool:
            return f"Tool {tool_name} not found"
        result = tool.run(query)
        return json.dumps(result)

    def generate(self, query: str, chunks: List[dict] = []) -> dict:
        has_chunks = len(chunks) > 0

        if has_chunks:
            context = self._build_context(chunks)
            prompt = f"CONTEXT:\n{context}\n\nQUESTION:\n{query}\n\nANSWER (with citations):"
            messages = [
                {"role": "system", "content": self._build_system_prompt(has_chunks=True)},
                {"role": "user", "content": prompt}
            ]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature
            )
            return {
                "answer": response.choices[0].message.content,
                "tool_used": "search_documents",
                "sources": [
                    {
                        "source": c["metadata"].get("source", "unknown"),
                        "page": c["metadata"].get("page", "N/A"),
                        "text": c["text"],
                        "rerank_score": c.get("rerank_score", 0)
                    }
                    for c in chunks
                ]
            }

        # no chunks - use tools or own knowledge
        messages = [
            {"role": "system", "content": self._build_system_prompt(has_chunks=False)},
            {"role": "user", "content": query}
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=self._build_tools_schema(),
            tool_choice="auto",
            temperature=self.temperature
        )

        message = response.choices[0].message

        if message.tool_calls:
            tool_call = message.tool_calls[0]
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            tool_result = self._run_tool(tool_name, tool_args.get("query", query))

            messages.append({"role": "assistant", "content": None, "tool_calls": message.tool_calls})
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result
            })

            final_response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature
            )

            return {
                "answer": final_response.choices[0].message.content,
                "tool_used": tool_name,
                "sources": []
            }

        return {
            "answer": message.content,
            "tool_used": None,
            "sources": []
        }

    def generate_stream(self, query: str, chunks: List[dict] = []):
        has_chunks = len(chunks) > 0

        if has_chunks:
            context = self._build_context(chunks)
            prompt = f"CONTEXT:\n{context}\n\nQUESTION:\n{query}\n\nANSWER (with citations):"
            messages = [
                {"role": "system", "content": self._build_system_prompt(has_chunks=True)},
                {"role": "user", "content": prompt}
            ]
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                stream=True
            )
            for chunk in stream:
                token = chunk.choices[0].delta.content
                if token is not None:
                    yield token
            return

        # no chunks - tool calling then stream
        messages = [
            {"role": "system", "content": self._build_system_prompt(has_chunks=False)},
            {"role": "user", "content": query}
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=self._build_tools_schema(),
            tool_choice="auto",
            temperature=self.temperature
        )

        message = response.choices[0].message

        if message.tool_calls:
            tool_call = message.tool_calls[0]
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            tool_result = self._run_tool(tool_name, tool_args.get("query", query))

            messages.append({"role": "assistant", "content": None, "tool_calls": message.tool_calls})
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result
            })

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            stream=True
        )

        for chunk in stream:
            token = chunk.choices[0].delta.content
            if token is not None:
                yield token