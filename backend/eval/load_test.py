"""Load test for Query Mind: N virtual users, each with their own CSV, asking questions at the same time.

Each user:
  1. POST /session
  2. uploads a CSV built only for them (a unique tag and a unique sum)
  3. waits until /documents/status says ready
  4. asks the questions, with a random pause between them (or all users at the same second with --burst)

Every answer is checked against that user's own data, and against every OTHER user's data.
An answer that contains another user's tag or sum is a LEAK. The target is 0 leaks.

Run from the backend/ folder:
    python eval/load_test.py --base-url http://localhost:8000 --users 1
    python eval/load_test.py --base-url https://query-mind.onrender.com --users 5
    python eval/load_test.py --base-url https://query-mind.onrender.com --users 10 --burst

Every request carries X-Request-ID: load-<run_id>-u<NN>-q<N>, so in Grafana:
    {service="query-mind"} | json | req_id=~"load-<run_id>-.*"
"""
import argparse
import asyncio
import json
import os
import random
import re
import time
from datetime import datetime

import httpx

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(HERE, "load_runs")
SID_PREFIX_LEN = 8

QUESTIONS = [
    ("sum", "What is the sum of the value column? Reply with just the number."),
    ("tag", "What is the value in the user_tag column? Reply with just the value."),
    ("rows", "How many rows does the dataset have? Reply with just the number."),
]


def user_data(user: int) -> dict:
    rows = 3 + user
    values = [user * 1000] + list(range(1, rows))
    tag = f"QM{user:02d}TAG"
    lines = ["value,user_tag"] + [f"{v},{tag}" for v in values]
    return {
        "csv": ("\n".join(lines) + "\n").encode(),
        "filename": f"load_user{user:02d}.csv",
        "sum": sum(values),
        "tag": tag,
        "rows": rows,
    }


def numbers_in(text: str) -> set[int]:
    return {int(n.replace(",", "")) for n in re.findall(r"\d[\d,]*", text)}


def check(answer: str, kind: str, mine: dict, others: list[dict]) -> tuple[bool, list[str]]:
    nums = numbers_in(answer)
    correct = {
        "sum": mine["sum"] in nums,
        "tag": mine["tag"] in answer,
        "rows": mine["rows"] in nums,
    }[kind]
    leaks = [o["tag"] for o in others if o["tag"] in answer or o["sum"] in nums]
    return correct, leaks


def percentile(values: list[float], p: float) -> int | None:
    if not values:
        return None
    s = sorted(values)
    k = max(0, min(len(s) - 1, round(p / 100 * (len(s) - 1))))
    return int(s[k])


class Run:
    def __init__(self, args):
        self.args = args
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.started = time.perf_counter()
        os.makedirs(RUNS_DIR, exist_ok=True)
        self.out_path = os.path.join(RUNS_DIR, f"load_{self.run_id}.jsonl")
        self.records: list[dict] = []
        self.users = {u: user_data(u) for u in range(1, args.users + 1)}
        self.barrier = asyncio.Barrier(args.users) if args.burst else None

    def t(self) -> str:
        return f"[{time.perf_counter() - self.started:7.1f}s]"

    def save(self, rec: dict) -> None:
        self.records.append(rec)
        with open(self.out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")


async def wake_up(client: httpx.AsyncClient, base: str) -> None:
    for i in range(30):
        try:
            r = await client.get(f"{base}/", timeout=30)
            if r.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        print(f"waiting for backend to wake up ({i + 1}/30)...")
        await asyncio.sleep(5)
    raise SystemExit("backend did not wake up")


async def ask(client, base, headers, question) -> dict:
    start = time.perf_counter()
    first_token = None
    parts, error, status = [], None, None
    try:
        async with client.stream("POST", f"{base}/query/stream", json={"question": question}, headers=headers) as r:
            if r.status_code >= 400:
                body = (await r.aread()).decode(errors="replace")[:200]
                return {"error": f"http_{r.status_code}: {body}", "status": r.status_code,
                        "answer": "", "ttft_ms": None, "total_ms": int((time.perf_counter() - start) * 1000)}
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                event = json.loads(payload)
                if event.get("type") == "error":
                    error = event.get("content") or "error"
                    status = event.get("status")
                elif event.get("type") == "token":
                    if first_token is None:
                        first_token = time.perf_counter()
                    parts.append(event["content"])
    except httpx.HTTPError as e:
        error = f"{type(e).__name__}: {e}"
    answer = "".join(parts).strip()
    if error is None and answer.startswith("[Error"):
        error = answer
    return {
        "answer": answer,
        "error": error,
        "status": status,
        "ttft_ms": int((first_token - start) * 1000) if first_token else None,
        "total_ms": int((time.perf_counter() - start) * 1000),
    }


async def upload_and_wait(client, base, headers, data, run, user) -> bool:
    r = await client.post(f"{base}/upload", headers=headers,
                          files={"files": (data["filename"], data["csv"], "text/csv")})
    if r.status_code != 200:
        print(f"{run.t()} u{user:02d} upload failed: {r.status_code} {r.text[:200]}")
        return False
    for _ in range(240):
        st = (await client.get(f"{base}/documents/status", headers=headers)).json()
        status = st.get("documents", {}).get(data["filename"])
        if status == "ready":
            return True
        if status == "failed":
            print(f"{run.t()} u{user:02d} ingest failed")
            return False
        await asyncio.sleep(0.5)
    print(f"{run.t()} u{user:02d} ingest timed out")
    return False


async def virtual_user(client, run: Run, user: int) -> None:
    args = run.args
    base = args.base_url
    mine = run.users[user]
    others = [d for u, d in run.users.items() if u != user]

    await asyncio.sleep(random.uniform(0, args.stagger))
    r = await client.post(f"{base}/session")
    if r.status_code != 200:
        print(f"{run.t()} u{user:02d} could not create session: {r.status_code} {r.text[:200]}")
        if run.barrier:
            await run.barrier.abort()
        return
    sid = r.json()["session_id"]
    headers = {"X-Session-Id": sid}
    print(f"{run.t()} u{user:02d} session sid={sid[:SID_PREFIX_LEN]}")

    if not await upload_and_wait(client, base, headers, mine, run, user):
        if run.barrier:
            await run.barrier.abort()
        return
    print(f"{run.t()} u{user:02d} dataset ready")

    reupload_at = len(QUESTIONS) // 2 if args.reupload else None
    for q, (kind, question) in enumerate(QUESTIONS[: args.questions], start=1):
        if reupload_at is not None and q - 1 == reupload_at:
            print(f"{run.t()} u{user:02d} re-uploading")
            if not await upload_and_wait(client, base, headers, mine, run, user):
                return
        if run.barrier:
            try:
                await run.barrier.wait()
            except asyncio.BrokenBarrierError:
                pass
        elif q > 1:
            await asyncio.sleep(random.uniform(args.think_min, args.think_max))

        req_id = f"load-{run.run_id}-u{user:02d}-q{q}"
        result = await ask(client, base, {**headers, "X-Request-ID": req_id}, question)
        correct, leaks = check(result["answer"], kind, mine, others) if not result["error"] else (False, [])
        is_429 = result["status"] == 429 or "429" in str(result["error"] or "")
        rec = {"run_id": run.run_id, "user": user, "q": q, "kind": kind, "req_id": req_id,
               "sid": sid[:SID_PREFIX_LEN], "correct": correct, "leaks": leaks, "is_429": is_429,
               **result, "answer": result["answer"][:300]}
        run.save(rec)

        if leaks:
            label = f"LEAK {leaks}"
        elif result["error"]:
            label = "429" if is_429 else f"ERROR {str(result['error'])[:80]}"
        else:
            label = "correct" if correct else f"WRONG: {result['answer'][:60]!r}"
        print(f"{run.t()} u{user:02d} q{q} {kind:<4} {result['total_ms']:>6}ms  {label}")


def summarize(run: Run) -> dict:
    recs = run.records
    ok = [r for r in recs if not r["error"]]
    totals = [r["total_ms"] for r in ok]
    ttfts = [r["ttft_ms"] for r in ok if r["ttft_ms"] is not None]
    summary = {
        "run_id": run.run_id,
        "users": run.args.users,
        "burst": run.args.burst,
        "reupload": run.args.reupload,
        "questions_asked": len(recs),
        "answered": len(ok),
        "correct": sum(r["correct"] for r in recs),
        "wrong": sum(1 for r in ok if not r["correct"]),
        "leaks": sum(1 for r in recs if r["leaks"]),
        "rate_limited_429": sum(r["is_429"] for r in recs),
        "other_errors": sum(1 for r in recs if r["error"] and not r["is_429"]),
        "total_ms_p50": percentile(totals, 50),
        "total_ms_p95": percentile(totals, 95),
        "ttft_ms_p50": percentile(ttfts, 50),
        "ttft_ms_p95": percentile(ttfts, 95),
        "wall_clock_s": round(time.perf_counter() - run.started, 1),
    }
    return summary


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", required=True)
    p.add_argument("--users", type=int, default=1)
    p.add_argument("--questions", type=int, default=len(QUESTIONS), choices=range(1, len(QUESTIONS) + 1))
    p.add_argument("--think-min", type=float, default=5.0)
    p.add_argument("--think-max", type=float, default=20.0)
    p.add_argument("--stagger", type=float, default=3.0, help="users start at a random moment in this many seconds")
    p.add_argument("--burst", action="store_true", help="all users ask each question at the same second")
    p.add_argument("--reupload", action="store_true", help="each user re-uploads once in the middle of the run")
    args = p.parse_args()
    args.base_url = args.base_url.rstrip("/")

    run = Run(args)
    calls_estimate = args.users * args.questions * 4
    print(f"run {run.run_id}: {args.users} user(s) x {args.questions} question(s), "
          f"~{calls_estimate} gateway calls, burst={args.burst}, reupload={args.reupload}")
    print(f"Grafana: {{service=\"query-mind\"}} | json | req_id=~\"load-{run.run_id}-.*\"")

    async with httpx.AsyncClient(timeout=httpx.Timeout(180, connect=30)) as client:
        await wake_up(client, args.base_url)
        await asyncio.gather(*(virtual_user(client, run, u) for u in run.users))

    summary = summarize(run)
    with open(os.path.join(RUNS_DIR, f"load_{run.run_id}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== summary ===")
    for k, v in summary.items():
        print(f"{k:>18}: {v}")
    if summary["leaks"]:
        print("\n!!! LEAKS FOUND: an answer contained another user's data. Stop and investigate. !!!")
    print(f"\nper-question records: {run.out_path}")


if __name__ == "__main__":
    asyncio.run(main())
