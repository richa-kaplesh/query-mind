import asyncio
import json
import logging
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse

router = APIRouter(prefix='/dashboard')

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
