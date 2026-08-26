import asyncio
import json
import logging
import os
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse

router = APIRouter(prefix='/dashboard')

# Path to eval_history.json — lives in backend/eval/ next to run_eval.py
_EVAL_HISTORY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "eval", "eval_history.json"
)


@router.get('/evals')
async def get_evals():
    """Return all eval run records, most-recent first. Returns [] if no history yet."""
    if not os.path.exists(_EVAL_HISTORY_PATH):
        return []
    try:
        with open(_EVAL_HISTORY_PATH, "r", encoding="utf-8") as f:
            history = json.load(f)
        # Sort newest first by timestamp string (ISO format sorts lexicographically)
        history.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
        return history
    except Exception as e:
        logging.getLogger("dashboard_routes").error(f"Failed to read eval history: {e}")
        raise HTTPException(status_code=500, detail="Could not read eval history")

@router.get('/traces')
async def get_traces(request: Request):
    store = request.app.state.trace_store
    return store.get_all()

@router.get('/traces/{trace_id}')
async def get_trace(request: Request, trace_id: str):
    store = request.app.state.trace_store
    trace = store.get(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")
    return trace

@router.get('/stream')
async def stream_traces(request: Request):
    store = request.app.state.trace_store

    async def event_generator():
        last_version = store.get_version()
        init_payload = {'event': 'init', 'traces': store.get_all()}
        yield f'data: {json.dumps(init_payload)}\n\n'

        last_ping_time = asyncio.get_event_loop().time()

        while True:
            if await request.is_disconnected():
                break

            current_version = store.get_version()
            if current_version != last_version:
                update_payload = {'event': 'update', 'traces': store.get_all()}
                yield f'data: {json.dumps(update_payload)}\n\n'
                last_version = current_version
            
            current_time = asyncio.get_event_loop().time()
            if current_time - last_ping_time >= 15:
                ping_payload = {'event': 'ping'}
                yield f'data: {json.dumps(ping_payload)}\n\n'
                last_ping_time = current_time

            await asyncio.sleep(0.4)

    return StreamingResponse(
        event_generator(),
        media_type='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )
