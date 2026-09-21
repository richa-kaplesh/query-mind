import httpx
from config import settings

async def complete(conversation_id: str, user_id: str, messages: list[dict],
                    tools: list[dict] | None = None, tool_choice: str | None = None) -> dict:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(f"{settings.gateway_url}/query", json={
            "conversation_id": conversation_id,
            "user_id": user_id,
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
        })
        response.raise_for_status()
        return response.json()