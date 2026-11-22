import json
import os
import re
from datetime import datetime, timezone

RUNS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "eval", "runs"
)

METRIC_DEFS = {
    "faithfulness": ("Faithfulness", True),
    "answer_relevancy": ("Answer relevancy", True),
    "context_precision": ("Context precision", True),
    "context_recall": ("Context recall", True),
    "tool_appropriateness": ("Tool appropriateness", True),
    "answer_correctness": ("Answer correctness", True),
    "normal": ("Normal questions", True),
    "unanswerable": ("Unanswerable", True),
    "false_premise": ("False premise", True),
    "out_of_context": ("Out of context", True),
    "prompt_injection": ("Prompt injection", True),
    "overall": ("Overall pass rate", True),
}

EVAL_SETS = {
    "pdf_workout_v1": {"id": "pdf_workout_v1", "name": "Workout plan PDF v1", "doc_type": "pdf"},
    "pdf_hdfc_v1": {"id": "pdf_hdfc_v1", "name": "HDFC fund leaflet PDF v1", "doc_type": "pdf"},
    "csv_golden_v1": {"id": "csv_golden_v1", "name": "CSV golden set v1", "doc_type": "csv"},
    "csv_behavior_v1": {"id": "csv_behavior_v1", "name": "CSV behavior set v1", "doc_type": "csv"},
}

SUMMARY_KEYS = ("id", "name", "created_at", "doc_type", "eval_set", "baseline_id", "metrics")
DETAIL_KEYS = SUMMARY_KEYS + ("questions", "config_changes", "notes")
_ID_RE = re.compile(r"^run_\d{3,}$")


def metric_entry(key, value):
    label, higher = METRIC_DEFS.get(key, (key.replace("_", " ").capitalize(), True))
    return {"key": key, "label": label, "value": round(float(value), 4), "higher_is_better": higher}


def _read_all():
    if not os.path.isdir(RUNS_DIR):
        return []
    runs = []
    for name in sorted(os.listdir(RUNS_DIR)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(RUNS_DIR, name), "r", encoding="utf-8") as f:
            runs.append(json.load(f))
    return runs


def list_runs():
    runs = sorted(_read_all(), key=lambda r: r["created_at"], reverse=True)
    return [{k: r.get(k) for k in SUMMARY_KEYS} for r in runs]


def get_run(run_id):
    if not _ID_RE.match(run_id):
        return None
    path = os.path.join(RUNS_DIR, f"{run_id}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        run = json.load(f)
    return {k: run.get(k) for k in DETAIL_KEYS}


def _next_id(runs):
    nums = [int(r["id"].split("_")[1]) for r in runs if _ID_RE.match(r.get("id", ""))]
    return f"run_{(max(nums) + 1 if nums else 1):03d}"


def _flatten(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flatten(v, f"{prefix}{k}."))
    else:
        out[prefix[:-1]] = obj
    return out


def _short(v, n=48):
    s = str(v)
    return s if len(s) <= n else s[: n - 1] + "…"


def _pair(old, new):
    a, b = str(old), str(new)
    if len(a) <= 24 and len(b) <= 24:
        return f"{a} -> {b}"
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    tail = b[i:].strip(" ,(")
    return _short(tail or b)


def diff_config(old, new):
    if not old or not new:
        return []
    a, b = _flatten(old), _flatten(new)
    changes = []
    for k in b:
        if k not in a:
            changes.append(f"{k}: {_short(b[k])}")
        elif a[k] != b[k]:
            changes.append(f"{k}: {_pair(a[k], b[k])}")
    for k in a:
        if k not in b and not any(n.startswith(k + ".") for n in b):
            changes.append(f"removed {k}")
    return changes


def write_run(name, eval_set_id, averages, questions, config=None, config_changes=None,
              notes=None, created_at=None, baseline_id="auto", question_count=None):
    eval_set = EVAL_SETS[eval_set_id]
    runs = _read_all()
    same_set = sorted(
        [r for r in runs if r["eval_set"]["id"] == eval_set_id and r["doc_type"] == eval_set["doc_type"]],
        key=lambda r: r["created_at"],
    )
    if baseline_id == "auto":
        baseline_id = same_set[0]["id"] if same_set else None
    if config_changes is None:
        config_changes = diff_config(same_set[-1].get("config") if same_set else None, config)
    run_id = _next_id(runs)
    record = {
        "id": run_id,
        "name": name,
        "created_at": created_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "doc_type": eval_set["doc_type"],
        "eval_set": {
            "id": eval_set_id,
            "name": eval_set["name"],
            "question_count": question_count if question_count is not None else len(questions),
        },
        "baseline_id": baseline_id,
        "metrics": [metric_entry(k, v) for k, v in averages.items()],
        "questions": questions,
        "config_changes": config_changes,
        "notes": notes,
        "config": config,
    }
    os.makedirs(RUNS_DIR, exist_ok=True)
    path = os.path.join(RUNS_DIR, f"{run_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    return path


def meta_from_env(default_name):
    name = os.environ.get("EVAL_RUN_NAME", default_name)
    notes, changes = None, None
    path = os.environ.get("EVAL_NOTES_FILE")
    if path:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        changes = data.get("config_changes")
        picked = {k: data.get(k) for k in ("problem", "root_cause", "fix", "result")}
        notes = picked if any(picked.values()) else None
    return name, notes, changes


def scored_questions(items, id_of):
    return [
        {
            "id": id_of(i, q),
            "text": q.get("question", ""),
            "scores": {k: float(v) for k, v in (q.get("scores") or {}).items() if isinstance(v, (int, float))},
        }
        for i, q in enumerate(items)
    ]
