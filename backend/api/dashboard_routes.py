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


_EVAL_DIR = os.path.dirname(_EVAL_HISTORY_PATH)
_BEHAVIOR_RUNS_PATH = os.path.join(_EVAL_DIR, "behavior_runs.json")
_BEHAVIOR_CATEGORIES = ["normal", "unanswerable", "false_premise", "out_of_context", "prompt_injection"]


def _wilson(passed, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = passed / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _behavior_run(meta):
    with open(os.path.join(_EVAL_DIR, meta["file"]), "r", encoding="utf-8") as f:
        records = list(json.load(f).values())
    ran = [r for r in records if r.get("status") == "ran"]
    averages = {}
    passed_by_cat = {}
    for cat in _BEHAVIOR_CATEGORIES:
        rows = [r for r in ran if r.get("category") == cat]
        passed_by_cat[cat] = (sum(1 for r in rows if r.get("passed")), len(rows))
        if rows:
            averages[cat] = passed_by_cat[cat][0] / len(rows)
    total_passed = sum(p for p, _ in passed_by_cat.values())
    averages["overall"] = total_passed / len(ran) if ran else 0.0
    summaries = [r["summary"] for r in ran if r.get("summary")]
    calls = sum(s.get("total_calls", 0) for s in summaries)
    models = {}
    for s in summaries:
        for c in s.get("llm_calls", []):
            models[c.get("model")] = models.get(c.get("model"), 0) + 1
    low, high = _wilson(total_passed, len(ran))
    config = {
        "questions": len(records),
        "ran": len(ran),
        "not_run": len(records) - len(ran),
        "correct": f"{total_passed}/{len(ran)}",
        "ci_95": f"{low * 100:.0f}–{high * 100:.0f}%",
        "llm_calls": calls,
        "calls_per_question": round(calls / len(summaries), 2) if summaries else 0,
        "calls_per_correct": round(calls / total_passed, 2) if total_passed else 0,
        "per_category": {cat: f"{p}/{n}" for cat, (p, n) in passed_by_cat.items()},
        "model_calls": models,
        "question_sets": ", ".join(sorted({r.get("set", "") for r in records})),
    }
    per_question = [
        {
            "question": f"[{r.get('category')}] {r.get('question')}",
            "answer": r.get("answer") or r.get("error") or "",
            "scores": {"passed": 1.0 if r.get("passed") else 0.0},
        }
        for r in records
    ]
    return {
        "timestamp": meta["timestamp"],
        "type": "behavior",
        "label": meta["label"],
        "description": meta.get("description", ""),
        "config": config,
        "averages": averages,
        "per_question": per_question,
    }


def _behavior_runs():
    if not os.path.exists(_BEHAVIOR_RUNS_PATH):
        return []
    with open(_BEHAVIOR_RUNS_PATH, "r", encoding="utf-8") as f:
        metas = json.load(f)
    runs = []
    for meta in metas:
        try:
            runs.append(_behavior_run(meta))
        except Exception as e:
            logging.getLogger("dashboard_routes").error(f"Failed to read behavior run {meta.get('file')}: {e}")
    return runs


@router.get('/evals')
async def get_evals():
    """Return all eval run records, most-recent first. Returns [] if no history yet."""
    history = []
    try:
        if os.path.exists(_EVAL_HISTORY_PATH):
            with open(_EVAL_HISTORY_PATH, "r", encoding="utf-8") as f:
                history = json.load(f)
        history.extend(_behavior_runs())
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

        last_ping_time = asyncio.get_running_loop().time()

        while True:
            if await request.is_disconnected():
                break

            current_version = store.get_version()
            if current_version != last_version:
                update_payload = {'event': 'update', 'traces': store.get_all()}
                yield f'data: {json.dumps(update_payload)}\n\n'
                last_version = current_version
            
            current_time = asyncio.get_running_loop().time()
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

@router.get("/tokens")
async def get_tokens(request: Request):
    tokens = request.app.state.token_tracker
    return tokens.get_all()


