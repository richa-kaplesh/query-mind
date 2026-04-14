from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import time

from core.embedder import Embedder
from core.indexer import Indexer
from core.retriever import HybridRetriever
from core.reranker import Reranker
from core.generator import Generator
from api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("main")

app = FastAPI(title="QueryMind")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.middleware("http")
async def log_requests(request, call_next):
    start = time.time()
    log.info(f"→ {request.method} {request.url.path}")
    response = await call_next(request)
    duration = (time.time() - start) * 1000
    log.info(f"← {request.method} {request.url.path} | {response.status_code} | {duration:.1f}ms")
    return response

log.info("Loading components...")

log.info("  [1/5] Loading Embedder...")
embedder = Embedder()

log.info("  [2/5] Loading Indexer...")
indexer = Indexer()

log.info("  [3/5] Loading HybridRetriever...")
retriever = HybridRetriever(indexer=indexer)

log.info("  [4/5] Loading Reranker...")
reranker = Reranker()

log.info("  [5/5] Loading Generator...")
generator = Generator()

log.info("All components loaded ✓")

app.state.embedder = embedder
app.state.indexer = indexer
app.state.retriever = retriever
app.state.reranker = reranker
app.state.generator = generator

app.include_router(router)