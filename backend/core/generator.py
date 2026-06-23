from groq import Groq
from typing import List
from core.tools.base_tool import BaseTool
from core.prompt_builder import PromptBuilder
from core.tool_registry import ToolRegistry
from config import settings
import json


class Generator:

    def __init__(self, tools: List[BaseTool]):
        self.client = Groq(api_key=settings.groq_api_key)
        self.model = settings.model_name
        self.temperature = settings.temperature
        self.prompt_builder = PromptBuilder()
        self.registry = ToolRegistry(tools)

    @property
    def tools(self):
        return self.registry.tools

    def generate(self, query: str, chunks: List[dict] = []) -> dict:
        has_chunks = len(chunks) > 0

        if has_chunks:
            context = self.prompt_builder.build_context(chunks)
            prompt = self.prompt_builder.build_rag_prompt(query, context)
            messages = [
                {"role": "system", "content": self.prompt_builder.build_system_prompt(has_chunks=True)},
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

        messages = [
            {"role": "system", "content": self.prompt_builder.build_system_prompt(has_chunks=False)},
            {"role": "user", "content": query}
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=self.registry.build_schema(),
            tool_choice="auto",
            temperature=self.temperature
        )

        message = response.choices[0].message

        if message.tool_calls:
            tool_call = message.tool_calls[0]
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            tool_result = self.registry.run(tool_name, tool_args.get("query", query))

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
            context = self.prompt_builder.build_context(chunks)
            prompt = self.prompt_builder.build_rag_prompt(query, context)
            messages = [
                {"role": "system", "content": self.prompt_builder.build_system_prompt(has_chunks=True)},
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

        messages = [
            {"role": "system", "content": self.prompt_builder.build_system_prompt(has_chunks=False)},
            {"role": "user", "content": query}
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=self.registry.build_schema(),
            tool_choice="auto",
            temperature=self.temperature
        )

        message = response.choices[0].message

        if message.tool_calls:
            tool_call = message.tool_calls[0]
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            tool_result = self.registry.run(tool_name, tool_args.get("query", query))

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