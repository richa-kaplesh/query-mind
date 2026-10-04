from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import time
from core.generator import Generator
from core.tracer import TraceStore
from core.embedder import Embedder
from core.indexer import Indexer
from core.retriever import HybridRetriever
from core.reranker import Reranker
from api.router import router
from api.dashboard_routes import router as dashboard_router
from core.token_tracker import TokenTracker
from core.tools.pandas_worker_manager import worker_manager
import uuid
from core.logging_setup import setup_logging, request_id_var

setup_logging()
log = logging.getLogger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──
    log.info("Booting persistent pandas worker...")
    worker_manager.boot()
    yield
    # ── Shutdown ──
    log.info("Shutting down persistent pandas worker...")
    if worker_manager.is_alive():
        worker_manager.task_queue.put(("shutdown", None))
        worker_manager.process.join(timeout=5)


app = FastAPI(title="QueryMind - CSV Engine", lifespan=lifespan)

# CORS setup for frontend connectivity
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
async def root():
    return {"status": "ok", "service": "QueryMind API"}

@app.middleware("http")
async def log_requests(request, call_next):
    req_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    token = request_id_var.set(req_id)
    start = time.time()
    try:
        log.info(f"→ {request.method} {request.url.path}")
        response = await call_next(request)
        duration = (time.time() - start) * 1000
        log.info(f"← {request.method} {request.url.path} | {response.status_code} | {duration:.1f}ms")
        response.headers["X-Request-ID"] = req_id
        return response
    finally:
        request_id_var.reset(token)

log.info("Loading components...")

# ── Shared stateless components ───────────────────────────────────────────────

# CSV path: tool-calling generator
generator = Generator(tools=[])
app.state.generator = generator

# PDF path: retrieval stack (embedder → indexer → retriever → reranker)
embedder  = Embedder()
indexer   = Indexer()
retriever = HybridRetriever(indexer=indexer)
reranker  = Reranker()

app.state.embedder  = embedder
app.state.indexer   = indexer
app.state.retriever = retriever
app.state.reranker  = reranker

app.state.token_tracker = TokenTracker()

# Global trace store for the debug dashboard
trace_store = TraceStore()
app.state.trace_store = trace_store


log.info("All components loaded ✓")

app.include_router(router)
app.include_router(dashboard_router)