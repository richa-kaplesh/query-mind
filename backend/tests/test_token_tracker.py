import json
import core.token_tracker as tt


def test_log_call_records_totals_and_persists(tmp_path, monkeypatch):
    log_file = tmp_path / "logs" / "token_log.json"
    monkeypatch.setattr(tt, "_LOG_PATH", str(log_file))

    tracker = tt.TokenTracker()
    assert tracker.get_all() == []

    tracker.log_call("model-x", prompt_tokens=100, completion_tokens=20, purpose="answer")

    entry = tracker.get_all()[0]
    assert entry["total_tokens"] == 120
    assert entry["model"] == "model-x" and entry["estimated"] is False
    assert json.loads(log_file.read_text())[0]["total_tokens"] == 120
    assert len(tt.TokenTracker().get_all()) == 1          # a new tracker reloads it


def test_corrupt_log_file_starts_fresh(tmp_path, monkeypatch):
    log_file = tmp_path / "token_log.json"
    log_file.write_text("{ this is not json")
    monkeypatch.setattr(tt, "_LOG_PATH", str(log_file))
    assert tt.TokenTracker().get_all() == []