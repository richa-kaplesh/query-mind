import json
import logging
import os
import queue
import sys
import threading
import time
from contextvars import ContextVar
from datetime import datetime, timezone

import httpx

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

SHIPPER_THREAD_NAME = "loki-shipper"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "req_id": request_id_var.get(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


class LokiHandler(logging.Handler):
    """Collects log lines in a queue and sends them to Grafana Loki in batches
    from a background thread, so logging never slows down or breaks a request."""

    def __init__(self, url: str, user: str, token: str, labels: dict,
                 batch_size: int = 50, flush_interval: float = 2.0):
        super().__init__()
        self.url = url
        self.auth = (user, token)
        self.labels = labels
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.queue: queue.Queue = queue.Queue(maxsize=10000)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._worker, name=SHIPPER_THREAD_NAME, daemon=True)
        self._thread.start()

    def emit(self, record: logging.LogRecord) -> None:
        # Ignore logs created by the shipper itself (httpx logs every request),
        # otherwise sending logs would create more logs, forever.
        if threading.current_thread().name == SHIPPER_THREAD_NAME:
            return
        try:
            line = self.format(record)
            timestamp_ns = str(int(record.created * 1e9))
            self.queue.put_nowait([timestamp_ns, line])
        except queue.Full:
            pass  # better to drop a log line than to block the app
        except Exception:
            self.handleError(record)

    def _worker(self) -> None:
        client = httpx.Client(timeout=10.0)
        batch: list = []
        last_flush = time.time()
        while not self._stop_event.is_set():
            try:
                batch.append(self.queue.get(timeout=0.5))
            except queue.Empty:
                pass
            due = time.time() - last_flush >= self.flush_interval
            if batch and (len(batch) >= self.batch_size or due):
                self._send(client, batch)
                batch = []
                last_flush = time.time()
        # shutting down: send whatever is left
        while True:
            try:
                batch.append(self.queue.get_nowait())
            except queue.Empty:
                break
        if batch:
            self._send(client, batch)
        client.close()

    def _send(self, client: httpx.Client, batch: list) -> None:
        body = {"streams": [{"stream": self.labels, "values": batch}]}
        try:
            r = client.post(self.url, json=body, auth=self.auth)
            if r.status_code >= 300:
                print(f"[loki] push failed {r.status_code}: {r.text[:200]}", file=sys.stderr)
        except Exception as e:
            print(f"[loki] push error: {e}", file=sys.stderr)

    def close(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=5)
        super().close()


def setup_logging(level: int = logging.INFO) -> None:
    formatter = JsonFormatter()

    console = logging.StreamHandler()
    console.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(console)
    root.setLevel(level)

    # Only turn on Grafana shipping if the env vars exist (so local runs still work)
    loki_url = os.getenv("LOKI_URL")
    if loki_url:
        loki = LokiHandler(
            url=loki_url,
            user=os.getenv("LOKI_USER", ""),
            token=os.getenv("LOKI_TOKEN", ""),
            labels={"service": "query-mind", "env": os.getenv("APP_ENV", "production")},
        )
        loki.setFormatter(formatter)
        root.addHandler(loki)