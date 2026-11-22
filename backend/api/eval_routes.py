from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request

from core.eval_runs import get_run, list_runs

router = APIRouter(prefix="/api")


@router.get("/evals/runs")
async def eval_runs():
    return {"runs": list_runs()}


@router.get("/evals/runs/{run_id}")
async def eval_run(run_id: str):
    run = get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@router.get("/costs")
async def costs(request: Request):
    entries = request.app.state.token_tracker.get_all()
    calls = [
        {
            "call_type": e.get("purpose", "unknown"),
            "model": e.get("model", "unknown"),
            "prompt_tokens": int(e.get("prompt_tokens", 0)),
            "completion_tokens": int(e.get("completion_tokens", 0)),
            "timestamp": datetime.fromtimestamp(e["timestamp"], timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        for e in entries
        if "timestamp" in e
    ]
    calls.sort(key=lambda c: c["timestamp"], reverse=True)
    return {"calls": calls}
