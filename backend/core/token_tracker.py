import json
import os
import time
from threading import Lock

_LOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "logs", "token_log.json"
)

class TokenTracker:
    def __init__(self):
        self.lock = Lock()
        self._entries = self._load()

    def _load(self):
        if os.path.exists(_LOG_PATH):
            try:
                with open(_LOG_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError):
                return []
        return []
    
    def _save(self):
        os.makedirs(os.path.dirname(_LOG_PATH), exist_ok = True)
        with open(_LOG_PATH, "w") as f:
            json.dump(self._entries, f , indent = 2)

    def log_call(self, model: str, prompt_tokens: int, completion_tokens: int, purpose: str):
        entry = {
            "timestamp": time.time(),
            "model": model,
            "purpose": purpose,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        with self.lock:
            self._entries.append(entry)
            self._save()

    def get_all(self):
        return self._entries
    