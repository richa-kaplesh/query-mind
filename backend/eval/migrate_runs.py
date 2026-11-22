import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.eval_runs import RUNS_DIR, diff_config, write_run

HERE = os.path.dirname(os.path.abspath(__file__))
BEHAVIOR_CATEGORIES = ["normal", "unanswerable", "false_premise", "out_of_context", "prompt_injection"]

NAMES = {
    "2026-08-26T09:10:01.508885": "Baseline",
    "2026-08-28T08:46:09.058007": "Baseline",
    "2026-08-28T09:02:30.244029": "Baseline rerun 2",
    "2026-08-28T09:17:19.063567": "Baseline rerun 3",
    "2026-08-29T08:02:52.268578": "Baseline rerun 4",
    "2026-08-29T08:13:42.839233": "Baseline rerun 5",
    "2026-08-29T08:21:39.897403": "Baseline rerun 6",
    "2026-08-29T11:47:42.842062": "Reranker threshold fix",
    "2026-09-16T15:53:14.589278": "Reranker threshold fix rerun",
    "2026-09-01T10:28:22.373895": "CSV baseline after fixes",
    "2026-09-08T13:01:50.406025": "Scoring fixes",
    "2026-09-10T10:22:15.818004": "Schema and data cleaning fixes",
    "2026-09-24T09:03:33.888853": "Live gateway pass",
    "2026-09-24T09:30:27.810274": "Live gateway pass, tool scoring corrected",
    "2026-09-24T09:46:19.452836": "Live gateway pass, re-scored",
}

NOTES = {
    "2026-08-29T11:47:42.842062": {
        "problem": "Q13 (IFSCA registration number) was never retrieved, so context recall failed on it.",
        "root_cause": "The fixed-size chunker merged the registration line into an unrelated chunk, and the reranker's hard min_score of 0.3 dropped the right chunk at 0.2991.",
        "fix": "Heading-aware, line-based chunking; removed the min_score filter and rely on top_k rank; removed a duplicate rerank HTTP call that double-billed every request.",
        "result": "Q13's chunk now lands in the final top 5 (rank 4, score 0.251).",
    },
    "2026-09-01T10:28:22.373895": {
        "problem": "The CSV eval could not run end to end.",
        "root_cause": "The eval generator had no tools registered, the schema prompt printed all 617 columns (~40k tokens), Call 2 still allowed tool calls, and the scorer could not compare (min, max) answers.",
        "fix": "Registered the pandas sandbox tool in the eval, cut the schema to 20 detailed columns plus a summary, rebuilt Call 2 without tools, added range scoring.",
        "result": "First working run on 15 questions. Still open: schema_02 fails because the label column is hidden by the schema cutoff.",
    },
    "2026-09-08T13:01:50.406025": {
        "problem": "Answer correctness was 20% although many answers were visibly right.",
        "root_cause": "Number extraction matched digits inside column names (the 1 in f1), and a 0.0001 tolerance demanded 4+ decimal places.",
        "fix": "Letter-boundary check in the number regex, tolerance loosened to 0.001, Call 2 falls back to the raw tool result when it fails.",
        "result": "Correct answers now score as correct. Found next: columns 21-617 never reach the model's prompt.",
    },
    "2026-09-10T10:22:15.818004": {
        "problem": "Questions failed with 'column does not exist', and class filters matched zero rows.",
        "root_cause": "The schema summary was built but never appended to the prompt; the class column held literal quote characters and the sandbox loaded the CSV without cleaning.",
        "fix": "Append the summary lines; one shared quote-stripping utility for extractor and sandbox; golden code updated for numeric class; number extraction anchored on linking words.",
        "result": "Final 15-question pass. Still open: count_05 is a two-part question scored as one value; count_06 has tied classes.",
    },
    "2026-09-24T09:03:33.888853": {
        "problem": "The first run through the live gateway leaked answers between questions and restarted workers for no reason.",
        "root_cause": "The gateway's semantic cache cached Call 2 synthesis, the worker queue had no lock, and a duplicate route() definition hid the retry code.",
        "fix": "An is_tool_related flag skips the cache, a lock wraps the worker exchange, the old route() was deleted, and each provider retries with backoff.",
        "result": "13 of 15 correct. The 2 failures are scoring gaps (schema_06 format, count_05 two-part question).",
    },
    "2026-09-24T09:30:27.810274": {
        "problem": "Tool appropriateness was penalised on three schema questions answered correctly without the tool.",
        "root_cause": "golden_csv_dataset.json had needs_computation=true for schema_02, stat_mean_f10 and stat_range_f10.",
        "fix": "Set needs_computation=false for those three and re-scored the same answers with no new LLM calls.",
        "result": "Only tool appropriateness changed; answer correctness is unchanged.",
    },
    "2026-11-22T16:35:07": {
        "problem": "The behavior baseline failed on false premises, unanswerable questions, prompt injections and off-topic questions.",
        "root_cause": "The reviewer skip rule let unchecked answers through, and the prompt had no rules for premises, scope or the label column and did not isolate the question as data.",
        "fix": "Prompt rules to check claims and stay on the dataset, a reviewer guard for words not in the schema, the question wrapped in <question> tags with a non-answer check, the label column named, and capped reviews marked 'could not be fully verified'.",
        "result": "All 50 questions: 41/50 vs 21/50. Subset: 35 questions answered only by gpt-oss-20b. 28/35 vs 14/35. Still open: tweet false premises, one injection, a regression on d2_normal_05.",
    },
}

CHANGES = {
    "2026-08-29T11:47:42.842062": ["chunker: heading-aware line split", "reranker min_score 0.3 removed", "duplicate rerank call removed"],
    "2026-09-16T15:53:14.589278": [],
    "2026-09-08T13:01:50.406025": ["number regex: letter-boundary check", "tolerance 0.0001 -> 0.001", "Call 2 falls back to tool result", "tracer wired into eval"],
    "2026-09-10T10:22:15.818004": ["schema summary appended to prompt", "shared quote stripping for CSV", "number extraction on linking words", "Call 1 retry x3"],
    "2026-09-24T09:03:33.888853": ["live gateway path", "gateway cache skips tool calls", "lock on pandas worker", "retry with backoff per provider"],
    "2026-09-24T09:30:27.810274": ["needs_computation fixed for 3 questions", "re-score only, no new LLM calls"],
    "2026-09-24T09:46:19.452836": ["re-score only, no new LLM calls"],
    "2026-11-22T16:35:07": [
        "prompt: check claims in the question",
        "prompt: only questions about this dataset",
        "reviewer: no skip on words not in schema",
        "question passed in <question> tags",
        "label column named in schema prompt",
        "capped review marked unverified",
    ],
}


def iso_z(ts):
    return ts.split(".")[0] + "Z"


def load(name):
    with open(os.path.join(HERE, name), "r", encoding="utf-8") as f:
        return json.load(f)


def clean_scores(scores):
    return {k: float(v) for k, v in (scores or {}).items() if isinstance(v, (int, float))}


def pdf_questions(rec, golden):
    out = []
    for i, q in enumerate(rec["per_question"]):
        idx = golden.index(q["question"]) if q["question"] in golden else i
        out.append({"id": f"q{idx + 1:02d}", "text": q["question"], "scores": clean_scores(q.get("scores"))})
    return out


def csv_questions(rec):
    return [{"id": q["id"], "text": q["question"], "scores": clean_scores(q.get("scores"))} for q in rec["per_question"]]


def behavior_record(meta):
    records = list(load(meta["file"]).values())
    ran = [r for r in records if r.get("status") == "ran"]
    averages = {}
    for cat in BEHAVIOR_CATEGORIES:
        rows = [r for r in ran if r.get("category") == cat]
        if rows:
            averages[cat] = sum(1 for r in rows if r.get("passed")) / len(rows)
    averages["overall"] = sum(1 for r in ran if r.get("passed")) / len(ran) if ran else 0.0
    questions = [
        {
            "id": r["id"],
            "text": r.get("question", ""),
            "scores": {"passed": 1.0 if r.get("passed") else 0.0} if r.get("status") == "ran" else {},
        }
        for r in records
    ]
    return averages, questions, len(records)


def main():
    if os.path.isdir(RUNS_DIR) and any(n.endswith(".json") for n in os.listdir(RUNS_DIR)):
        print(f"{RUNS_DIR} already has run files, nothing migrated")
        return
    golden = [q["question"] for q in load("golden_dataset.json")]
    history = load("eval_history.json") + [load("eval_results.json")]
    jobs = []
    for i, rec in enumerate(history):
        ts = rec["timestamp"]
        if rec.get("type") == "csv":
            jobs.append((ts, "csv_golden_v1", rec["averages"], csv_questions(rec), rec.get("config"), None, None))
        elif i == 0:
            jobs.append((ts, "pdf_workout_v1", rec["averages"], pdf_questions(rec, []), rec.get("config"), None, None))
        else:
            jobs.append((ts, "pdf_hdfc_v1", rec["averages"], pdf_questions(rec, golden), rec.get("config"), None, None))
    for meta in load("behavior_runs.json"):
        averages, questions, count = behavior_record(meta)
        jobs.append((meta["timestamp"], "csv_behavior_v1", averages, questions, None, count, meta))
    jobs.sort(key=lambda j: j[0])

    last_config = {}
    for ts, set_id, averages, questions, config, count, meta in jobs:
        name = meta["label"] if meta else NAMES.get(ts, "Run")
        if ts in CHANGES:
            changes = CHANGES[ts]
        else:
            changes = diff_config(last_config.get(set_id), config)
        last_config[set_id] = config
        path = write_run(
            name=name,
            eval_set_id=set_id,
            averages=averages,
            questions=questions,
            config=config,
            config_changes=changes,
            notes=NOTES.get(ts),
            created_at=iso_z(ts),
            question_count=count,
        )
        print("wrote", os.path.basename(path), set_id, name)


if __name__ == "__main__":
    main()
