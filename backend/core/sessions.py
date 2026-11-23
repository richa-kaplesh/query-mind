import logging, os, secrets, shutil, threading, time
from dataclasses import dataclass, field
from pathlib import Path
from config import settings
from core.indexer import Indexer
from core.tools import data_inspect_tool
from core.tools.pandas_worker_manager import worker_manager

log = logging.getLogger("sessions")

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
            expired = self._sweep_locked()
            self._remove_orphan_folders_locked()
            if len(self._s) >= settings.max_sessions:
                full = True
            else:
                full = False
                sid = secrets.token_urlsafe(32)
                s = Session(id=sid, conversation_id=secrets.token_hex(16), indexer=Indexer())
                self._s[sid] = s
        for old in expired:
            self._release_memory(old)
        if full:
            raise SessionCapacityError
        return s

    def get(self, sid: str) -> Session | None:
        with self._lock:
            s = self._s.get(sid)
            if s: s.last_used = time.time()
        if s:
            try:
                os.utime(Path(settings.upload_dir) / s.id)
            except OSError:
                pass
        return s

    def count(self) -> int:
        with self._lock:
            return len(self._s)

    def delete(self, sid: str) -> None:
        with self._lock:
            s = self._s.pop(sid, None)
        if s:
            self._release_memory(s)
            shutil.rmtree(Path(settings.upload_dir) / s.id, ignore_errors=True)

    def _sweep_locked(self) -> list[Session]:  # lazy TTL cleanup, runs on create()
        now = time.time()
        expired = []
        for k in [k for k, v in self._s.items() if now - v.last_used > settings.session_ttl_seconds]:
            expired.append(self._s.pop(k))
            shutil.rmtree(Path(settings.upload_dir) / k, ignore_errors=True)
        return expired

    def _remove_orphan_folders_locked(self) -> None:
        root = Path(settings.upload_dir)
        if not root.is_dir():
            return
        cutoff = time.time() - settings.session_ttl_seconds
        for folder in root.iterdir():
            try:
                idle = folder.is_dir() and folder.stat().st_mtime < cutoff
            except OSError:
                continue
            if idle and folder.name not in self._s:
                shutil.rmtree(folder, ignore_errors=True)

    @staticmethod
    def _release_memory(s: Session) -> None:
        if s.file_type != ".csv" or not s.file_path:
            return
        try:
            worker_manager.unload(s.file_path)
            data_inspect_tool.evict(s.file_path)
        except Exception:
            log.warning("could not release dataset memory", exc_info=True)
