import multiprocessing
import queue as queue_module
import logging
import platform
import pandas as pd
from core.utils.data_cleaning import strip_stray_quotes
from core.tools.safe_namespace import build_safe_pd, build_safe_df

log = logging.getLogger("pandas_worker")

SAFE_BUILTINS = {
    "len": len, "str": str, "int": int, "float": float, "bool": bool,
    "round": round, "sum": sum, "min": min, "max": max, "abs": abs,
    "sorted": sorted, "list": list, "dict": dict, "tuple": tuple, "set": set,
    "range": range, "enumerate": enumerate, "zip": zip, "print": print,
}


def _apply_resource_limits():
    """resource is POSIX-only — real protection on Render (Linux), silent
    no-op on Windows dev machines, which don't have this module at all."""
    if platform.system() == "Windows":
        return
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (500 * 1024 * 1024, 500 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
        resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))
    except Exception as e:
        log.warning(f"Could not apply resource limits: {e}")


def _persistent_worker(task_queue: multiprocessing.Queue, result_queue: multiprocessing.Queue):
    """Lives for the whole backend process. pandas is imported ONCE, here —
    never again, no matter how many datasets get loaded/swapped over its life."""
    _apply_resource_limits()
    log.info("Persistent pandas worker booting — importing pandas once")

    df = None

    while True:
        msg_type, payload = task_queue.get()

        if msg_type == "shutdown":
            log.info("Shutdown signal received, exiting")
            break

        elif msg_type == "load":
            try:
                loaded = pd.read_csv(payload, low_memory=False)
                df = strip_stray_quotes(loaded)
                log.info(f"Loaded {payload}, shape={df.shape}")
                result_queue.put(("ok", None))
            except Exception as e:
                df = None
                result_queue.put(("error", str(e)))

        elif msg_type == "unload":
            df = None
            log.info("Dataset unloaded, worker idle")
            result_queue.put(("ok", None))

        elif msg_type == "exec":
            if df is None:
                result_queue.put(("error", "No dataset currently loaded"))
                continue
            # Code arriving here was already validated by CodeValidator in
            # the calling process (PandasSandboxTool) — the safe proxies
            # below are the SECOND, independent gate, not the first.
            try:
                local_vars = {
                    "pd": build_safe_pd(),
                    "df": build_safe_df(df.copy()),  # copy — one query's mutation can't corrupt the next
                    "result": None,
                }
                restricted_globals = {"__builtins__": SAFE_BUILTINS}
                exec(payload, restricted_globals, local_vars)
                result_str = str(local_vars["result"])
                if len(result_str) > 5000:
                    result_str = result_str[:5000] + "... (truncated)"
                result_queue.put(("ok", result_str))
            except Exception as e:
                result_queue.put(("error", str(e)))


class PandasWorkerManager:
    """Owns process lifecycle + IPC only. Knows nothing about validation
    rules or pandas internals — those live elsewhere by design."""

    def __init__(self):
        self.process: multiprocessing.Process | None = None
        self.task_queue: multiprocessing.Queue | None = None
        self.result_queue: multiprocessing.Queue | None = None
        self.current_file_path: str | None = None

    def boot(self):
        self.task_queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue()
        self.process = multiprocessing.Process(target=_persistent_worker, args=(self.task_queue, self.result_queue))
        self.process.daemon = True  # ensures the process dies with its parent, even if nothing
                                     # explicitly sends a shutdown message (e.g. standalone scripts)
        self.process.start()
        log.info("Worker manager: process started")
    def is_alive(self) -> bool:
        return self.process is not None and self.process.is_alive()

    def _ensure_alive(self):
        if self.is_alive():
            return
        log.warning("Worker found dead — respawning")
        self.boot()
        if self.current_file_path:
            self._send("load", self.current_file_path, timeout=30)

    def _send(self, msg_type: str, payload, timeout: int):
        self.task_queue.put((msg_type, payload))
        try:
            return self.result_queue.get(timeout=timeout)
        except queue_module.Empty:
            return ("error", f"No response within {timeout}s")

    def load_file(self, file_path: str):
        self._ensure_alive()
        if self.current_file_path == file_path:
            return "ok", None  # already loaded — no-op, preserves the whole point of the persistent worker
        status, err = self._send("load", file_path, timeout=30)
        if status == "ok":
            self.current_file_path = file_path
        return status, err

    def unload(self):
        self.current_file_path = None
        if self.is_alive():
            self._send("unload", None, timeout=5)

    def run_query(self, code: str, timeout_seconds: int = 15) -> str:
        self._ensure_alive()
        status, payload = self._send("exec", code, timeout=timeout_seconds)
        if status == "error" and str(payload).startswith("No response"):
            self.process.kill()
            self.process.join()
            return f"Error: Code execution exceeded {timeout_seconds} seconds. Worker will recover on next query."
        if status == "error":
            return f"Error executing Pandas code: {payload}"
        return payload


worker_manager = PandasWorkerManager()