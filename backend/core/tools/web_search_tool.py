from core.tools.base_tool import BaseTool
from duckduckgo_search import DDGS

class WebSearchTool(BaseTool):

    def __init__(self, max_results: int = 5):
        self.max_results = max_results

    @property
    def name(self) -> str:
        return "search_web"

    @property
    def description(self) -> str:
        return "Search the web for current information not available in uploaded documents"

    def run(self, query: str) -> dict:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=self.max_results))
        
        formatted = []
        for r in results:
            formatted.append({
                "title": r.get("title", ""),
                "snippet": r.get("body", ""),
                "url": r.get("href", "")
            })
        
        return {"results": formatted}