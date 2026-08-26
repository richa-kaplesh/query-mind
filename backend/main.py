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
from api.routes import router
from api.dashboard_routes import router as dashboard_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("main")

app = FastAPI(title="QueryMind - CSV Engine")

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
    start = time.time()
    log.info(f"→ {request.method} {request.url.path}")
    response = await call_next(request)
    duration = (time.time() - start) * 1000
    log.info(f"← {request.method} {request.url.path} | {response.status_code} | {duration:.1f}ms")
    return response

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

# Global trace store for the debug dashboard
trace_store = TraceStore()
app.state.trace_store = trace_store

log.info("All components loaded ✓")

app.include_router(router)
app.include_router(dashboard_router)