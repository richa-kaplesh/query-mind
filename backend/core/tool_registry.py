from typing import List
from core.tools.base_tool import BaseTool
import json


class ToolRegistry:

    def __init__(self, tools: List[BaseTool]):
        self.tools = {tool.name: tool for tool in tools}

    def register(self, tool: BaseTool):
        self.tools[tool.name] = tool

    def build_schema(self) -> list:
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
            if tool.name != "search_documents"
        ]

    def run(self, tool_name: str, query: str) -> str:
        tool = self.tools.get(tool_name)
        if not tool:
            return f"Tool {tool_name} not found"
        result = tool.run(query)
        return json.dumps(result)