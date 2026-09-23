import httpx
from config import settings

async def complete(conversation_id: str, user_id: str, messages: list[dict],
                    tools: list[dict] | None = None, tool_choice: str | None = None,
                    is_tool_related: bool = False) -> dict:
    async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(f"{settings.gateway_url}/query", json={
            "conversation_id": conversation_id,
            "user_id": user_id,
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
            "is_tool_related": is_tool_related,
            })
            if response.status_code >= 400:
              print(f"[GATEWAY ERROR] status={response.status_code} body={response.text}", flush=True)
            response.raise_for_status()
            return response.json()