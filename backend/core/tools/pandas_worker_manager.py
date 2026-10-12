import gc
import math
import multiprocessing
import queue as queue_module
import logging
import platform
import threading
import time
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


def _current_vmsize_bytes() -> int:
    """Address space this process already holds (inherited from the parent)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmSize:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


# How much the worker may allocate ON TOP of what it inherited from the main app.
WORKER_MEMORY_HEADROOM_MB = 600


def _apply_resource_limits():
    """resource is POSIX-only — real protection on Render (Linux), silent
    no-op on Windows dev machines, which don't have this module at all."""
    if platform.system() == "Windows":
        return
    try:
        import resource
        limit = _current_vmsize_bytes() + WORKER_MEMORY_HEADROOM_MB * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        # NOTE: no fixed RLIMIT_CPU here. That limit counts CPU time over the WHOLE life of the
        # process, so a fixed 10 s killed a perfectly healthy worker after ~10 s of total work
        # (every ~11 describe() queries on a big CSV), forcing a respawn + full CSV reload each
        # time. The CPU budget is re-armed per task by _set_cpu_budget() below instead.
    except Exception as e:
        log.warning(f"Could not apply resource limits: {e}")


CPU_SECONDS_PER_LOAD = 30
CPU_SECONDS_PER_QUERY = 10


def _set_cpu_budget(seconds: int):
    """Allow this ONE task `seconds` of CPU on top of what the process already used."""
    if platform.system() == "Windows":
        return
    try:
        import resource
        soft = math.ceil(time.process_time()) + seconds
        resource.setrlimit(resource.RLIMIT_CPU, (soft, resource.RLIM_INFINITY))
    except Exception as e:
        log.warning(f"Could not set per-task CPU budget: {e}")


def _persistent_worker(task_queue: multiprocessing.Queue, result_queue: multiprocessing.Queue):
    """Lives for the whole backend process. pandas is imported ONCE, here —
    never again, no matter how many datasets get loaded/swapped over its life."""
    # In a spawned subprocess the parent's logging handlers do not exist — set
    # up a fresh console handler so log calls here are not silently dropped.
    from core.logging_setup import setup_logging
    setup_logging()
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
                df = None          # drop the previous dataset BEFORE parsing the new one (never hold two)
                gc.collect()
                _set_cpu_budget(CPU_SECONDS_PER_LOAD)
                loaded = pd.read_csv(payload, low_memory=False)
                df = strip_stray_quotes(loaded)
                del loaded
                gc.collect()
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
            local_vars = restricted_globals = None
            _set_cpu_budget(CPU_SECONDS_PER_QUERY)
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
            finally:
                # Without this the per-query df.copy() (a full second copy of the dataset) stayed
                # alive after the answer was sent, until the NEXT query overwrote it.
                local_vars = restricted_globals = None
                gc.collect()


class PandasWorkerManager:
    """Owns process lifecycle + IPC only. Knows nothing about validation
    rules or pandas internals — those live elsewhere by design."""

    def __init__(self):
        self.process: multiprocessing.Process | None = None
        self.task_queue: multiprocessing.Queue | None = None
        self.result_queue: multiprocessing.Queue | None = None
        self.current_file_path: str | None = None
        self._boot_lock = threading.Lock()   # prevents concurrent respawns

    def boot(self):
        ctx = multiprocessing.get_context("spawn")   # avoid fork-after-threads deadlock (faiss/OpenMP
                                                        # threads are already alive by the time this runs)
        self.task_queue = ctx.Queue()
        self.result_queue = ctx.Queue()
        self.process = ctx.Process(target=_persistent_worker, args=(self.task_queue, self.result_queue))
        self.process.daemon = True
        self.process.start()
        log.info("Worker manager: process started")

    def is_alive(self) -> bool:
        return self.process is not None and self.process.is_alive()

    def _ensure_alive(self):
        with self._boot_lock:
            if self.is_alive():
                return
            log.warning("Worker found dead — respawning")
            self.boot()
            if self.current_file_path:
                self._send("load", self.current_file_path, timeout=60)

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
        status, err = self._send("load", file_path, timeout=60)
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