from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os

from core.embedder import Embedder
from core.indexer import Indexer
from core.retriever import HybridRetriever
from core.reranker import Reranker
from core.generator import Generator
from api.router import router

app = FastAPI(title="QueryMind")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"]
)


embedder = Embedder()
indexer = Indexer()
retriever = HybridRetriever(indexer = indexer)
reranker = Reranker()
generator = Generator()

app.state.embedder = embedder
app.state.indexer = indexer
app.state.retriever = retriever
app.state.reranker = reranker
app.state.generator = generator

app.include_router(router)
