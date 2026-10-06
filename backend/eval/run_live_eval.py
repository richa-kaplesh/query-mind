"""Live end-to-end eval runner.

Sends every golden question through the REAL API (/upload, /query/stream), exactly
like the frontend does, so the whole production path is exercised: middleware,
reviewer, tools, gateway. Each request carries the header
    X-Request-ID: eval-<run_id>-<question_id>
so in Grafana you can filter one eval run, or one question, with:
    {service="query-mind"} | json | req_id=~"eval-<run_id>-.*"

Run from the backend/ folder:
    python eval/run_live_eval.py --base-url https://query-mind.onrender.com --mode csv --limit 10
    python eval/run_live_eval.py --base-url http://localhost:8000 --mode all --repeat 3

--repeat N runs the whole selection N times (re-uploading each time) and reports the
range across runs, not a single number. Results are saved after EVERY question, so an
interrupted run keeps what it finished (the file says "complete": false).

The app keeps ONE active document, so the runner uploads the CSV, runs all CSV
questions, then uploads the PDF and runs the PDF questions. Don't use the app by
hand while a run is in progress.
"""
import argparse
import json
import mimetypes
import os
import statistics
import sys
import time
from datetime import datetime, timezone

import httpx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import check_tool_app, score_correctness, llm_judge  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
DEFAULT_CSV = os.path.join(BACKEND, "uploads", "phpB0xrNj.csv")
DEFAULT_PDF = os.path.join(HERE, "Leaflet - HDFC International Funds - GIFT Outbound Retail - Class B.pdf")
DEFAULT_CSV_SETS = [
    os.path.join(HERE, "golden_csv_dataset.json"),
    os.path.join(HERE, "golden_csv_generated.json"),
]
DEFAULT_PDF_SET = os.path.join(HERE, "golden_dataset.json")
RUNS_DIR = os.path.join(HERE, "live_runs")


# ── API helpers ──────────────────────────────────────────────────────────────

def upload_and_wait(client: httpx.Client, base: str, path: str, run_id: str, timeout_s: int = 400):
    name = os.path.basename(path)
    mime = mimetypes.guess_type(name)[0] or "application/octet-stream"
    print(f"Uploading {name} ...")
    with open(path, "rb") as fh:
        r = client.post(f"{base}/upload", files=[("files", (name, fh, mime))],
                        headers={"X-Request-ID": f"eval-{run_id}-upload"})
    r.raise_for_status()

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        status = client.get(f"{base}/documents").json().get(name)
        if status == "ready":
            print("  document ready")
            return
        if status == "failed":
            raise RuntimeError(f"Ingestion failed for {name} (search the server logs for INGEST)")
        time.sleep(3)
    raise TimeoutError(f"{name} not ready after {timeout_s}s")


def ask(client: httpx.Client, base: str, question: str, req_id: str) -> dict:
    """POST /query/stream, read the SSE stream, return answer + timings."""
    started = time.perf_counter()
    first_token_ms = None
    parts, tool_used = [], None
    error = None
    try:
        with client.stream("POST", f"{base}/query/stream", json={"question": question},
                           headers={"X-Request-ID": req_id}) as r:
            if r.status_code >= 400:
                error = f"http_{r.status_code}"
            else:
                for line in r.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        event = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    if event.get("type") == "tool":
                        tool_used = event.get("content")
                    elif event.get("type") == "token":
                        if first_token_ms is None:
                            first_token_ms = (time.perf_counter() - started) * 1000
                        parts.append(event.get("content", ""))
    except Exception as e:  # network / timeout
        error = f"{type(e).__name__}: {e}"

    answer = "".join(parts).strip()
    if error is None and answer.startswith("[Error"):
        error = "server_error"
    return {
        "answer": answer,
        "tool_used": tool_used,
        "error": error,
        "total_ms": round((time.perf_counter() - started) * 1000),
        "ttft_ms": round(first_token_ms) if first_token_ms is not None else None,
    }


# ── Dataset loading ──────────────────────────────────────────────────────────

def load_csv_questions(paths):
    items, seen = [], set()
    for p in paths:
        if not os.path.exists(p):
            continue
        for q in json.load(open(p, encoding="utf-8")):
            if q["id"] in seen:
                continue
            seen.add(q["id"])
            items.append({
                "id": q["id"], "question": q["question"], "category": q.get("category", "csv"),
                "golden": q["golden_answer"], "tolerance": q.get("tolerance"),
                "needs_computation": q.get("needs_computation"),
                "must_include": q.get("must_include"),
            })
    return items


def load_pdf_questions(path):
    items = []
    for i, q in enumerate(json.load(open(path, encoding="utf-8")), 1):
        items.append({
            "id": f"pdf_{i:02d}", "question": q["question"], "category": "pdf",
            "golden": q["ground_truth"], "tolerance": None, "needs_computation": None,
            "must_include": None,
        })
    return items


# ── Saving ───────────────────────────────────────────────────────────────────

def save_run(run_id, base, results, complete):
    os.makedirs(RUNS_DIR, exist_ok=True)
    path = os.path.join(RUNS_DIR, f"{run_id}.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"run_id": run_id, "base_url": base, "complete": complete,
                   "saved_at": datetime.now(timezone.utc).isoformat(), "results": results}, f, indent=2)
    os.replace(tmp, path)  # never leaves a half-written file behind
    return path


# ── Runner ───────────────────────────────────────────────────────────────────

def run_set(client, base, items, kind, run_id, delay, use_judge, checkpoint):
    results = []
    for n, item in enumerate(items, 1):
        req_id = f"eval-{run_id}-{item['id']}"
        print(f"[{kind} {n}/{len(items)}] {item['id']}: {item['question'][:70]}")
        out = ask(client, base, item["question"], req_id)

        if out["error"]:
            correctness, tool_score = 0.0, None
        elif kind == "pdf":
            correctness = llm_judge(item["golden"], out["answer"], item["question"], use_judge)
            tool_score = None
        else:
            correctness = score_correctness(item["golden"], out["answer"], item["tolerance"],
                                            item["question"], use_judge, item["must_include"])
            tool_score = check_tool_app(item["needs_computation"], out["tool_used"])

        flag = "ERROR " + out["error"] if out["error"] else f"correct={correctness}"
        print(f"    {flag} | {out['total_ms']} ms | tool={out['tool_used']}")
        results.append({
            "id": item["id"], "kind": kind, "category": item["category"], "req_id": req_id,
            "question": item["question"], "golden": item["golden"], "answer": out["answer"],
            "tool_used": out["tool_used"], "error": out["error"],
            "total_ms": out["total_ms"], "ttft_ms": out["ttft_ms"],
            "scores": {"answer_correctness": correctness, "tool_appropriateness": tool_score},
        })
        checkpoint(results)
        time.sleep(delay)
    return results


def run_once(client, base, args, run_id, use_judge):
    finished = []  # results of sets that are already complete (csv, then pdf)

    def checkpoint(current_set_results):
        save_run(run_id, base, finished + current_set_results, False)

    if args.mode in ("csv", "all"):
        items = load_csv_questions(args.csv_sets)
        if args.only:
            items = [i for i in items if i["id"] in args.only]
        items = items[: args.limit] if args.limit else items
        if items:
            upload_and_wait(client, base, args.csv_path, run_id)
            finished += run_set(client, base, items, "csv", run_id, args.delay, use_judge, checkpoint)

    if args.mode in ("pdf", "all"):
        items = load_pdf_questions(args.pdf_set)
        if args.only:
            items = [i for i in items if i["id"] in args.only]
        items = items[: args.limit] if args.limit else items
        if items:
            upload_and_wait(client, base, args.pdf_path, run_id)
            finished += run_set(client, base, items, "pdf", run_id, args.delay, use_judge, checkpoint)

    path = save_run(run_id, base, finished, True)
    return finished, path


# ── Metrics & reporting ──────────────────────────────────────────────────────

def pct(values, p):
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, round(p / 100 * (len(values) - 1)))]


def metrics(results):
    """Per-kind metrics for one run."""
    out = {}
    for kind in sorted({r["kind"] for r in results}):
        rs = [r for r in results if r["kind"] == kind]
        scored = [r["scores"]["answer_correctness"] for r in rs if r["scores"]["answer_correctness"] is not None]
        tools = [r["scores"]["tool_appropriateness"] for r in rs if r["scores"]["tool_appropriateness"] is not None]
        lat = [r["total_ms"] for r in rs if not r["error"]]
        cats = {}
        for r in rs:
            s = r["scores"]["answer_correctness"]
            if s is not None:
                cats.setdefault(r["category"], []).append(s)
        out[kind] = {
            "n": len(rs),
            "errors": sum(1 for r in rs if r["error"]),
            "unscored": [r["id"] for r in rs if r["scores"]["answer_correctness"] is None],
            "scored": len(scored),
            "correctness": statistics.mean(scored) if scored else None,
            "tool": statistics.mean(tools) if tools else None,
            "p50": pct(lat, 50), "p95": pct(lat, 95), "max": max(lat) if lat else None,
            "categories": {c: (statistics.mean(v), len(v)) for c, v in cats.items()},
        }
    return out


def fmt_range(values, fmt="{:.3f}"):
    values = [v for v in values if v is not None]
    if not values:
        return "n/a"
    if len(values) == 1:
        return fmt.format(values[0])
    return f"{fmt.format(min(values))} - {fmt.format(max(values))}  (mean {fmt.format(statistics.mean(values))})"


def summarize(all_metrics):
    """all_metrics: list of per-run metrics dicts (one entry when --repeat 1)."""
    runs = len(all_metrics)
    print("\n" + "=" * 62 + f"\nSUMMARY  ({runs} run{'s' if runs > 1 else ''})\n" + "=" * 62)
    for kind in sorted({k for m in all_metrics for k in m}):
        ms = [m[kind] for m in all_metrics if kind in m]
        print(f"\n{kind.upper()}: {ms[0]['n']} questions per run, errors per run: {[m['errors'] for m in ms]}")
        print(f"  answer correctness  : {fmt_range([m['correctness'] for m in ms])}")
        scored_txt = [str(m["scored"]) + "/" + str(m["n"]) for m in ms]
        print(f"  scored questions    : {scored_txt}")
        if any(m["tool"] is not None for m in ms):
            print(f"  tool appropriateness: {fmt_range([m['tool'] for m in ms])}")
        print(f"  latency p50 (ms)    : {fmt_range([m['p50'] for m in ms], '{:.0f}')}")
        print(f"  latency p95 (ms)    : {fmt_range([m['p95'] for m in ms], '{:.0f}')}")
        print(f"  latency max (ms)    : {fmt_range([m['max'] for m in ms], '{:.0f}')}")
        cats = sorted({c for m in ms for c in m["categories"]})
        for c in cats:
            vals = [m["categories"][c][0] for m in ms if c in m["categories"]]
            n = next(m["categories"][c][1] for m in ms if c in m["categories"])
            print(f"    {c:<14} {fmt_range(vals, '{:.2f}')}  (n={n})")
        unscored = sorted({q for m in ms for q in m["unscored"]})
        if unscored:
            print(f"  UNSCORED (judge off/failed): {unscored}")


def main():
    ap = argparse.ArgumentParser(description="Live end-to-end eval runner for QueryMind")
    ap.add_argument("--base-url", default=os.getenv("QUERYMIND_URL", "http://localhost:8000"))
    ap.add_argument("--mode", choices=["csv", "pdf", "all"], default="all")
    ap.add_argument("--csv-path", default=DEFAULT_CSV)
    ap.add_argument("--pdf-path", default=DEFAULT_PDF)
    ap.add_argument("--csv-sets", nargs="+", default=DEFAULT_CSV_SETS)
    ap.add_argument("--pdf-set", default=DEFAULT_PDF_SET)
    ap.add_argument("--limit", type=int, default=None, help="max questions per set")
    ap.add_argument("--only", nargs="+", default=None, help="run only these question ids")
    ap.add_argument("--repeat", type=int, default=1, help="run the whole selection N times and report ranges")
    ap.add_argument("--delay", type=float, default=3.0, help="seconds between questions (rate limits)")
    ap.add_argument("--read-timeout", type=float, default=300.0)
    ap.add_argument("--no-judge", action="store_true", help="skip LLM-judge scoring (offline / cheaper)")
    ap.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%d-%H%M"))
    args = ap.parse_args()

    base = args.base_url.rstrip("/")
    use_judge = not args.no_judge
    timeout = httpx.Timeout(connect=60.0, read=args.read_timeout, write=60.0, pool=60.0)
    all_metrics, paths, run_ids = [], [], []

    print(f"Run ID: {args.run_id}   repeats: {args.repeat}\nTarget: {base}\n")
    with httpx.Client(timeout=timeout) as client:
        client.get(f"{base}/")  # wake a sleeping free-tier instance
        for i in range(1, args.repeat + 1):
            run_id = args.run_id if args.repeat == 1 else f"{args.run_id}-r{i}"
            run_ids.append(run_id)
            print(f"\n######## run {i}/{args.repeat}: {run_id} ########")
            results, path = run_once(client, base, args, run_id, use_judge)
            all_metrics.append(metrics(results))
            paths.append(path)

    summarize(all_metrics)
    print("\nSaved:\n  " + "\n  ".join(paths))
    print("\nIn Grafana (Explore -> Code):")
    for rid in run_ids:
        print(f'  {rid}: {{service="query-mind"}} | json | req_id=~"eval-{rid}-.*"')


if __name__ == "__main__":
    main()