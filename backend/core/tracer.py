import threading
import uuid
import time
from datetime import datetime, timezone

class TraceStore:
    def __init__(self):
        self._traces = {}
        self._order = []
        self._version = 0
        self._lock = threading.Lock()

    def start_trace(self, filename: str, query: str) -> str:
        trace_id = uuid.uuid4().hex[:10]
        now_ts = time.time()
        
        with self._lock:
            trace = {
                "id": trace_id,
                "filename": filename,
                "query": query,
                "status": "running",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "total_ms": None,
                "tool_used": None,
                "final_answer": None,
                "steps": [],
                "_start": now_ts
            }
            self._traces[trace_id] = trace
            self._order.append(trace_id)
            
            # Evict oldest if beyond 200 traces
            if len(self._order) > 200:
                oldest_id = self._order.pop(0)
                if oldest_id in self._traces:
                    del self._traces[oldest_id]
            
            self._version += 1
            
        return trace_id

    def add_step(self, trace_id: str, step_type: str, data: dict):
        with self._lock:
            if trace_id not in self._traces:
                return
            
            trace = self._traces[trace_id]
            now_ts = time.time()
            elapsed_ms = (now_ts - trace["_start"]) * 1000
            
            step = {
                "type": step_type,
                "elapsed_ms": elapsed_ms,
                "data": data
            }
            trace["steps"].append(step)
            self._version += 1

    def finish_trace(self, trace_id: str, final_answer: str, tool_used: str, status: str = "success"):
        with self._lock:
            if trace_id not in self._traces:
                return
                
            trace = self._traces[trace_id]
            now_ts = time.time()
            total_ms = (now_ts - trace["_start"]) * 1000
            
            trace["status"] = status
            trace["final_answer"] = final_answer
            trace["tool_used"] = tool_used
            trace["total_ms"] = total_ms
            
            self._version += 1

    def get_all(self) -> list:
        with self._lock:
            return [self._public(self._traces[trace_id]) for trace_id in reversed(self._order)]

    def get(self, trace_id: str) -> dict | None:
        with self._lock:
            if trace_id in self._traces:
                return self._public(self._traces[trace_id])
            return None

    def get_version(self) -> int:
        with self._lock:
            return self._version

    def _public(self, trace: dict) -> dict:
        return {k: v for k, v in trace.items() if not k.startswith('_')}
