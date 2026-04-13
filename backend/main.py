# backend/main.py

from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os

from core.pdf_extractor import PDFExtractor
from core.chunker import TextChunker
from core.embedder import Embedder
from core.indexer import Indexer
from core.retriever import HybridRetriever
from core.reranker import Reranker
from core.generator import Generator

app = FastAPI(title="QueryMind")

# allow React frontend to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"]
)