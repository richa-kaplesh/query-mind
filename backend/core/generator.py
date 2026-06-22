from groq import Groq
from typing import List
from backend.core.tools.base_tool import BaseTool
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
        ]

    def _build_system_prompt(self) -> str:
        return """You are a research assistant with access to the following tools:

- search_documents: Use this when the user asks questions about uploaded documents, files, or any content they have provided.
- get_csv_stats: Use this when the user asks analytical, statistical, or ML-related questions about a CSV dataset (correlations, distributions, target column analysis).
- search_web: Use this when the user asks about current events, general knowledge, or anything not present in uploaded documents.

Rules:
- Always pick the most appropriate tool based on the query.
- If the query is about uploaded content, prefer search_documents.
- If the query is analytical about a CSV, prefer get_csv_stats.
- If the query requires outside knowledge, use search_web.
- Always cite your sources in the final answer.
- If you can answer directly without a tool, do so."""

    def _run_tool(self, tool_name: str, query: str) -> str:
        tool = self.tools.get(tool_name)
        if not tool:
            return f"Tool {tool_name} not found"
        result = tool.run(query)
        return json.dumps(result)

    def generate(self, query: str) -> dict:
        messages = [
            {
                "role": "system",
                "content": self._build_system_prompt()
            },
            {
                "role": "user",
                "content": query
            }
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
                "tool_used": tool_name
            }

        return {
            "answer": message.content,
            "tool_used": None
        }