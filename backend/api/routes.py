from fastapi import APIRouter, UploadFile, File, BackgroundTasks, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os
import shutil
import logging
import time
import json
from typing import List
from pathlib import Path

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
    try:
        pipeline = request.app.state.ingestion
        result = pipeline.ingest(file_path)
        documents[filename]["status"] = "ready"
        log.info(f"[INGEST] ✓ Done in {(time.time()-t0):.2f}s | {result['chunks']} chunks")
    except Exception as e:
        documents[filename]["status"] = "failed"
        log.error(f"[INGEST] ✗ Failed: {e}", exc_info=True)


@router.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    request: Request = None
):
    uploaded = []

    for file in files:
        log.info(f"[UPLOAD] Received file: {file.filename}")
        file_path = os.path.join(UPLOAD_DIR, file.filename)

        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        log.info(f"[UPLOAD] Saved to disk: {file_path}")

        documents[file.filename] = {
            "status": "processing",
            "file_path": file_path,
            "file_type": Path(file.filename).suffix.lower()
        }

        background_tasks.add_task(
            ingest_document,
            file_path=file_path,
            filename=file.filename,
            request=request
        )
        uploaded.append(file.filename)

    return {
        "message": "documents received, processing started",
        "files": uploaded,
        "status": "processing"
    }


@router.get("/documents")
async def get_documents():
    log.info(f"[DOCUMENTS] Current state: {documents}")
    return {"documents": {k: v["status"] for k, v in documents.items()}}


@router.delete("/documents/{filename}")
async def delete_document(filename: str, request: Request):
    import numpy as np
    import faiss
    from rank_bm25 import BM25Okapi

    indexer = request.app.state.indexer

    indexer.chunks = [
        chunk for chunk in indexer.chunks
        if chunk["metadata"]["source"] != filename
    ]

    if indexer.chunks:
        tokenized = [chunk["text"].lower().split() for chunk in indexer.chunks]
        indexer.bm25 = BM25Okapi(tokenized)

        embeddings = np.array([chunk["embedding"] for chunk in indexer.chunks]).astype("float32")
        indexer.faiss_index = faiss.IndexFlatIP(indexer.dimension)
        faiss.normalize_L2(embeddings)
        indexer.faiss_index.add(embeddings)
    else:
        indexer.bm25 = None
        indexer.faiss_index = None

    documents.pop(filename, None)

    return {"message": f"{filename} deleted", "remaining": list(documents.keys())}


@router.post("/query")
async def query_document(body: QueryRequest, request: Request):
    log.info(f"[QUERY] Question: '{body.question}'")
    t0 = time.time()

    try:
        embedder = request.app.state.embedder
        retriever = request.app.state.retriever
        reranker = request.app.state.reranker
        generator = request.app.state.generator

        chunks = []
        current_file = next((v for v in documents.values() if v.get("status") == "ready"), None)
        is_csv = current_file and current_file.get("file_type") == ".csv"
        has_documents = current_file is not None

        if has_documents and not is_csv:
            log.info("[QUERY] PDF found, running retrieval...")
            query_embedding = embedder.embed_query(body.question)
            chunks = retriever.retrieve(query=body.question, query_embedding=query_embedding, top_k=20)
            chunks = reranker.rerank(query=body.question, chunks=chunks, top_k=5)
            log.info(f"[QUERY] Retrieved {len(chunks)} chunks")
            result = generator.generate(query=body.question, chunks=chunks)

        elif is_csv:
            log.info("[QUERY] CSV found, adding stats tool...")
            from core.tools.csv_stats_tool import CSVStatsTool
            generator.registry.register(CSVStatsTool(...)) = CSVStatsTool(current_file["file_path"])
            result = generator.generate(query=body.question, chunks=[])

        else:
            result = generator.generate(query=body.question, chunks=[])

        log.info(f"[QUERY] ✓ Done in {(time.time()-t0):.2f}s | tool used: {result['tool_used']}")
        return result

    except Exception as e:
        log.error(f"[QUERY] ✗ Failed: {e}", exc_info=True)
        raise


@router.post("/query/stream")
async def query_document_stream(body: QueryRequest, request: Request):
    log.info(f"[STREAM] Question: '{body.question}'")

    embedder = request.app.state.embedder
    retriever = request.app.state.retriever
    reranker = request.app.state.reranker
    generator = request.app.state.generator

    chunks = []
    current_file = next((v for v in documents.values() if v.get("status") == "ready"), None)
    is_csv = current_file and current_file.get("file_type") == ".csv"
    has_documents = current_file is not None

    if has_documents and not is_csv:
        query_embedding = embedder.embed_query(body.question)
        chunks = retriever.retrieve(query=body.question, query_embedding=query_embedding, top_k=20)
        chunks = reranker.rerank(query=body.question, chunks=chunks, top_k=5)

    elif is_csv:
        from core.tools.csv_stats_tool import CSVStatsTool
        generator.registry.register(CSVStatsTool(...)) = CSVStatsTool(current_file["file_path"])

    sources = [
        {
            "source": chunk["metadata"].get("source", "unknown"),
            "page": chunk["metadata"].get("page", "N/A"),
            "text": chunk["text"],
            "rerank_score": chunk.get("rerank_score", 0)
        }
        for chunk in chunks
    ]

    def event_stream():
        for token in generator.generate_stream(body.question, chunks):
            data = json.dumps({"type": "token", "content": token})
            yield f"data: {data}\n\n"

        if sources:
            data = json.dumps({"type": "sources", "content": sources})
            yield f"data: {data}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")