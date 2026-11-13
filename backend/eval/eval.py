import json
import os
import re
import time
from datetime import datetime

import httpx

# ---------- settings ----------
BASE = "https://query-mind.onrender.com"
GATEWAY_URL = "https://llm-gateway-qewa.onrender.com/"   # put your gateway address here
UPLOADS = "../uploads/"
PAUSE_S = 5          # seconds to wait between questions (helps avoid 429s)
CATEGORY = None      # e.g. "normal" to run one category only; None = all

# (csv file, eval file holding its questions)
RUNS = [
    ("phpB0xrNj.csv", "eval_set01.json"),
    ("phpB0xrNj_injected.csv", "eval_set01.json"),
    # ("Tweets.csv", "eval_set02.json"),
    # ("Tweets_injected.csv", "eval_set02.json"),
]


# ---------- 1. pick the questions ----------
def load_questions(path, dataset_file, category=None):
    with open(path) as f:
        data = json.load(f)

    result = []
    for q in data:
        if q["dataset_file"] != dataset_file:
            continue
        if category is not None and q["category"] != category:
            continue
        result.append(q)
    return result


# ---------- 2. talk to the server ----------
def wake_up(client, url, tries=30):
    for _ in range(tries):
        try:
            client.get(url)          # any reply at all means it is awake
            return True
        except httpx.HTTPError:
            time.sleep(5)
    return False


def upload_and_wait(client, csv_path):
    name = os.path.basename(csv_path)
    try:
        with open(csv_path, "rb") as fh:
            r = client.post(f"{BASE}/upload", files=[("files", (name, fh, "text/csv"))])
    except httpx.HTTPError as e:
        print("upload error:", e)
        return False
    if r.status_code != 200:
        print("upload status:", r.status_code)
        return False
    for _ in range(60):
        status = client.get(f"{BASE}/documents").json().get(name)
        if status == "ready":
            return True
        if status == "failed":
            return False
        time.sleep(3)
    return False


def ask(client, question):
    start = time.perf_counter()
    parts = []
    error = None
    try:
        with client.stream("POST", f"{BASE}/query/stream", json={"question": question}) as r:
            if r.status_code >= 400:
                error = f"http_{r.status_code}"
            else:
                for line in r.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    event = json.loads(payload)
                    if event["type"] == "token":
                        parts.append(event["content"])
    except httpx.HTTPError as e:
        error = f"{type(e).__name__}: {e}"
    return {
        "answer": "".join(parts).strip(),
        "error": error,
        "total_ms": round((time.perf_counter() - start) * 1000),
    }


# ---------- 3. read what the agent did ----------
def get_trace(client, question):
    try:
        r = client.get(f"{BASE}/dashboard/traces", timeout=60)
        traces = [t for t in r.json() if t.get("query") == question]
    except (httpx.HTTPError, ValueError):
        return None
    if not traces:
        return None
    return max(traces, key=lambda t: t["created_at"])


def summarize(trace):
    steps = trace["steps"]
    agent_stop = next((s["elapsed_ms"] for s in steps if s["type"] == "agent_stop"), None)
    total = steps[-1]["elapsed_ms"]
    return {
        "generator_calls": sum(1 for s in steps if "llm_call_" in s["type"]),
        "reviewer_calls": sum(1 for s in steps if "review_call_" in s["type"]),
        "generator_tools": sum(1 for s in steps if s["type"].endswith("tool_input")),
        "reviewer_tools": sum(1 for s in steps if s["type"] == "review_tool_call"),
        "agent_ms": round(agent_stop) if agent_stop is not None else None,
        "review_ms": round(total - agent_stop) if agent_stop is not None else None,
        "server_total_ms": round(total),
    }


# ---------- 4. score ----------
def extract_numbers(text):
    found = re.findall(r"-?\d[\d,]*\.?\d*", text)
    numbers = []
    for item in found:
        try:
            numbers.append(float(item.replace(",", "")))
        except ValueError:
            pass
    return numbers


def check_answer(answer, golden, tolerance):
    if isinstance(golden, (int, float)):
        allowed = tolerance or 0
        return any(abs(n - golden) <= allowed for n in extract_numbers(answer))
    return str(golden).lower() in answer.lower()


# ---------- 5. one question, start to finish ----------
def run_question(client, item):
    question = item["question"]
    result = ask(client, question)
    trace = get_trace(client, question)
    record = {
        "id": item["id"],
        "category": item["category"],
        "expected_behavior": item["expected_behavior"],
        "question": question,
        "golden_answer": item["golden_answer"],
        "answer": result["answer"],
        "error": result["error"],
        "total_ms": result["total_ms"],
        "summary": summarize(trace) if trace else None,
        "passed": None,
    }
    if result["error"] is None and item["expected_behavior"] == "answer":
        record["passed"] = check_answer(result["answer"], item["golden_answer"], item["tolerance"])
    return record


# ---------- 6. save ----------
def save_record(path, rec):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


# ---------- run everything ----------
def main():
    results_path = f"results_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    with httpx.Client(timeout=120) as client:
        print("backend awake:", wake_up(client, BASE))
        print("gateway awake:", wake_up(client, GATEWAY_URL))
        for csv_name, eval_file in RUNS:
            print("---", csv_name)
            if not upload_and_wait(client, UPLOADS + csv_name):
                print("upload failed, skipping", csv_name)
                continue
            for item in load_questions(eval_file, csv_name, CATEGORY):
                rec = run_question(client, item)
                rec["dataset_file"] = csv_name
                save_record(results_path, rec)
                print(rec["id"], "|", rec["category"], "| passed:", rec["passed"],
                      "| ms:", rec["total_ms"], "| error:", rec["error"])
                time.sleep(PAUSE_S)
    print("saved to", results_path)


if __name__ == "__main__":
    main()