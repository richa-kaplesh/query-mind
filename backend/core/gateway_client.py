import asyncio
import contextvars
import logging
import time
import httpx
from config import settings
from core.exceptions import GatewayError
from core.logging_setup import request_id_var

log = logging.getLogger("gateway_client")


class _CallBudget:
    def __init__(self, limit: int):
        self.limit = limit
        self.used = 0


# One budget per user question. The router calls start_call_budget() when a stream begins;
# every complete() call (agent loop, reviewer, revisions, forced answer) draws from it.
_budget_var: contextvars.ContextVar["_CallBudget | None"] = contextvars.ContextVar(
    "gateway_call_budget", default=None
)


_shared: tuple[asyncio.AbstractEventLoop, httpx.AsyncClient, asyncio.Semaphore] | None = None


def _resources() -> tuple[httpx.AsyncClient, asyncio.Semaphore]:
    global _shared
    loop = asyncio.get_running_loop()
    if _shared is None or _shared[0] is not loop:
        _shared = (
            loop,
            httpx.AsyncClient(timeout=60.0),
            asyncio.Semaphore(settings.gateway_max_concurrent_calls),
        )
    return _shared[1], _shared[2]


async def aclose() -> None:
    global _shared
    if _shared is not None:
        client = _shared[1]
        _shared = None
        await client.aclose()


def start_call_budget(limit: int | None = None) -> None:
    _budget_var.set(_CallBudget(limit if limit is not None else settings.gateway_max_calls_per_request))

def calls_used() -> int:
    budget = _budget_var.get()
    return budget.used if budget else 0

async def complete(conversation_id: str, user_id: str, messages: list[dict],
                   tools: list[dict] | None = None, tool_choice: str | None = None,
                   is_tool_related: bool = False) -> dict:
    """POST to the gateway. Raises GatewayError (never a bare httpx error) on any failure."""
    budget = _budget_var.get()
    if budget is not None:
        if budget.used >= budget.limit:
            raise GatewayError(
                None, f"gateway call budget of {budget.limit} requests used up for this question",
                reason="budget",
            )
        budget.used += 1

    client, limit = _resources()
    wait_start = time.perf_counter()
    try:
        async with limit:
            wait_ms = round((time.perf_counter() - wait_start) * 1000)
            if wait_ms >= 50:
                log.info("gateway slot wait", extra={"fields": {
                    "event": "gateway_wait", "wait_ms": wait_ms,
                    "max_concurrent": settings.gateway_max_concurrent_calls,
                }})
            response = await client.post(
                f"{settings.gateway_url}/query",
                headers={"X-Request-ID": request_id_var.get()},
                json={
                    "conversation_id": conversation_id,
                    "user_id": user_id,
                    "messages": messages,
                    "tools": tools,
                    "tool_choice": tool_choice,
                    "is_tool_related": is_tool_related,
                },
            )
    except httpx.HTTPError as e:  # timeout, connection refused, DNS, ...
        log.error(f"gateway unreachable: {type(e).__name__}: {e}")
        raise GatewayError(None, f"gateway unreachable ({type(e).__name__})", reason="unreachable") from e

    if response.status_code >= 400:
        log.error(f"gateway error status={response.status_code} body={response.text}")
        raise GatewayError(response.status_code, f"gateway returned {response.status_code}")
    return response.json()