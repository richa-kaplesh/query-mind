from fastapi import APIRouter, UploadFile, File, BackgroundTasks, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import logging
import time
import json
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from typing import List

from api.file_handler import documents, save_file, run_ingestion
from api.query_handler import resolve_query, build_sources

log = logging.getLogger("routes")

router = APIRouter()


class QueryRequest(BaseModel):
    question: str
    conversation_history: list = []


@router.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    request: Request = None
):
    uploaded = []

    for file in files:
        log.info(f"[UPLOAD] Received file: {file.filename}")
        file_path, file_type = save_file(file)

        documents[file.filename] = {
            "status": "processing",
            "file_path": file_path,
            "file_type": file_type
        }

        background_tasks.add_task(
            run_ingestion,
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
        result, chunks = resolve_query(request, body.question, documents, body.conversation_history)
        log.info(f"[QUERY] ✓ Done in {(time.time()-t0):.2f}s | tool used: {result['tool_used']}")
        return result
    except Exception as e:
        log.error(f"[QUERY] ✗ Failed: {e}", exc_info=True)
        raise


@router.post("/query/stream")
async def query_document_stream(body: QueryRequest, request: Request):
    log.info(f"[STREAM] Question: '{body.question}'")
    generator = request.app.state.generator

    result, chunks = resolve_query(request, body.question, documents, body.conversation_history)
    sources = build_sources(chunks)

    def event_stream():
        for token in generator.generate_stream(body.question, chunks, body.conversation_history):
            data = json.dumps({"type": "token", "content": token})
            yield f"data: {data}\n\n"

        if sources:
            data = json.dumps({"type": "sources", "content": sources})
            yield f"data: {data}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/reset")
async def reset(request: Request):
    indexer = request.app.state.indexer
    indexer.chunks = []
    indexer.bm25 = None
    indexer.faiss_index = None
    documents.clear()
    return {"message": "reset complete"}