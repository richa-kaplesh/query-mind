import secrets, shutil, threading, time
from dataclasses import dataclass, field
from pathlib import Path
from config import settings
from core.indexer import Indexer

@dataclass
class Session:
    id: str
    conversation_id: str
    indexer: Indexer                       # one index PER session
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    filename: str | None = None            # display name only, never a path
    file_path: str | None = None
    file_type: str | None = None           # ".pdf" | ".csv"
    status: str = "empty"                  # empty | processing | ready | failed
    error: str | None = None               # machine code, e.g. "password_protected"
    schema: object | None = None           # CSV schema

class SessionCapacityError(Exception): ...

class SessionStore:
    def __init__(self):
        self._s: dict[str, Session] = {}
        self._lock = threading.Lock()

    def create(self) -> Session:
        with self._lock:
            self._sweep_locked()
            if len(self._s) >= settings.max_sessions:
                raise SessionCapacityError
            sid = secrets.token_urlsafe(32)
            s = Session(id=sid, conversation_id=secrets.token_hex(16), indexer=Indexer())
            self._s[sid] = s
            return s

    def get(self, sid: str) -> Session | None:
        with self._lock:
            s = self._s.get(sid)
            if s: s.last_used = time.time()
            return s

    def delete(self, sid: str) -> None:
        with self._lock:
            s = self._s.pop(sid, None)
        if s: shutil.rmtree(Path(settings.upload_dir) / s.id, ignore_errors=True)

    def _sweep_locked(self) -> None:          # lazy TTL cleanup, runs on create()
        now = time.time()
        for k in [k for k, v in self._s.items() if now - v.last_used > settings.session_ttl_seconds]:
            self._s.pop(k)
            shutil.rmtree(Path(settings.upload_dir) / k, ignore_errors=True)