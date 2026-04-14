from fastapi import APIRouter, UploadFile, File, BackgroundTasks, Request
from pydantic import BaseModel
import os
import shutil
import logging
import time

from core.pdf_extractor import PDFExtractor
from core.chunker import TextChunker

log = logging.getLogger("routes")

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

documents = {}


class QueryRequest(BaseModel):
    question: str
    conversation_history: list = []


async def ingest_document(file_path: str, filename: str, request: Request):
    log.info(f"[INGEST] Starting ingestion for: {filename}")
    t0 = time.time()

    embedder = request.app.state.embedder
    indexer = request.app.state.indexer

    try:
        log.info(f"[INGEST] Extracting text from PDF...")
        extractor = PDFExtractor()
        chunker = TextChunker()
        pages = extractor.extract(file_path)
        log.info(f"[INGEST] Extracted {len(pages)} pages")

        log.info(f"[INGEST] Chunking text...")
        all_chunks = []
        for page in pages:
            chunks = chunker.chunk_text(page["text"], metadata=page["metadata"])
            all_chunks.extend(chunks)
        log.info(f"[INGEST] Total chunks created: {len(all_chunks)}")

        log.info(f"[INGEST] Embedding chunks...")
        all_chunks = embedder.embed_chunks(all_chunks)
        log.info(f"[INGEST] Embedding done")

        log.info(f"[INGEST] Indexing chunks...")
        indexer.index(all_chunks)
        log.info(f"[INGEST] Indexing done")

        documents[filename] = "ready"
        log.info(f"[INGEST] ✓ Done in {(time.time()-t0):.2f}s | {filename} is ready")

    except Exception as e:
        documents[filename] = "failed"
        log.error(f"[INGEST] ✗ Failed for {filename}: {e}", exc_info=True)


@router.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    request: Request = None
):
    log.info(f"[UPLOAD] Received file: {file.filename}")
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    log.info(f"[UPLOAD] Saved to disk: {file_path}")

    documents[file.filename] = "processing"
    background_tasks.add_task(
        ingest_document,
        file_path=file_path,
        filename=file.filename,
        request=request
    )

    return {
        "message": "document received, processing started",
        "filename": file.filename,
        "status": "processing"
    }


@router.get("/documents")
async def get_documents():
    log.info(f"[DOCUMENTS] Current state: {documents}")
    return {"documents": documents}


@router.post("/query")
async def query_document(body: QueryRequest, request: Request):
    log.info(f"[QUERY] Question: '{body.question}'")
    t0 = time.time()

    try:
        embedder = request.app.state.embedder
        retriever = request.app.state.retriever
        reranker = request.app.state.reranker
        generator = request.app.state.generator

        log.info("[QUERY] Embedding query...")
        query_embedding = embedder.embed_query(body.question)
        log.info("[QUERY] Query embedded ✓")

        log.info("[QUERY] Retrieving top 20 chunks...")
        chunks = retriever.retrieve(
            query=body.question,
            query_embedding=query_embedding,
            top_k=20
        )
        log.info(f"[QUERY] Retrieved {len(chunks)} chunks")

        log.info("[QUERY] Reranking to top 5...")
        chunks = reranker.rerank(
            query=body.question,
            chunks=chunks,
            top_k=5
        )
        log.info(f"[QUERY] Reranked, got {len(chunks)} chunks")

        log.info("[QUERY] Generating answer...")
        result = generator.generate(
            query=body.question,
            chunks=chunks
        )
        log.info(f"[QUERY] ✓ Done in {(time.time()-t0):.2f}s")

        return {
            "answer": result["answer"],
            "sources": result["sources"]
        }

    except Exception as e:
        log.error(f"[QUERY] ✗ Failed: {e}", exc_info=True)
        raise