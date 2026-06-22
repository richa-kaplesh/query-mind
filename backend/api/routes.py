from fastapi import APIRouter, UploadFile, File, BackgroundTasks, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os
import shutil
import logging
import time
import json
from typing import List

from core.extractors.pdf_extractor import PDFExtractor
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
    try:
        pipeline = request.app.state.ingestion
        result = pipeline.ingest(file_path)
        documents[filename] = "ready"
        log.info(f"[INGEST] ✓ Done in {(time.time()-t0):.2f}s | {result['chunks']} chunks")
    except Exception as e:
        documents[filename] = "failed"
        log.error(f"[INGEST] ✗ Failed: {e}", exc_info=True)

@router.post("/query")
async def query_document(body: QueryRequest, request: Request):
    log.info(f"[QUERY] Question: '{body.question}'")
    t0 = time.time()
    try:
        generator = request.app.state.generator
        result = generator.generate(query=body.question)
        log.info(f"[QUERY] ✓ Done in {(time.time()-t0):.2f}s | tool used: {result['tool_used']}")
        return result
    except Exception as e:
        log.error(f"[QUERY] ✗ Failed: {e}", exc_info=True)
        raise

@router.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),  # "files", List
    request: Request = None
):
    uploaded = []
    
    for file in files:
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
        uploaded.append(file.filename)

    return {
        "message": "documents received, processing started",
        "files": uploaded,  # frontend expects data.files
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
        generator = request.app.state.generator
        result = generator.generate(query=body.question)
        log.info(f"[QUERY] ✓ Done in {(time.time()-t0):.2f}s | tool used: {result['tool_used']}")
        return result
    except Exception as e:
        log.error(f"[QUERY] ✗ Failed: {e}", exc_info=True)
        raise


@router.post("/query/stream")
async def query_document_stream(body: QueryRequest, request: Request):
    log.info(f"[STREAM] Question: '{body.question}'")
    generator = request.app.state.generator

    def event_stream():
        for token in generator.generate_stream(body.question):
            data = json.dumps({"type": "token", "content": token})
            yield f"data: {data}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@router.delete("/documents/{filename}")
async def delete_document(filename: str, request: Request):
    indexer = request.app.state.indexer

    # Remove chunks belonging to this file
    indexer.chunks = [
        chunk for chunk in indexer.chunks
        if chunk["metadata"]["source"] != filename
    ]

    # Rebuild both indexes without deleted chunks
    if indexer.chunks:
        tokenized = [chunk["text"].lower().split() for chunk in indexer.chunks]
        indexer.bm25 = BM25Okapi(tokenized)

        embeddings = np.array([chunk["embedding"] for chunk in indexer.chunks]).astype("float32")
        indexer.faiss_index = faiss.IndexFlatIP(indexer.dimension)
        faiss.normalize_L2(embeddings)
        indexer.faiss_index.add(embeddings)
    else:
        # No docs left — reset everything
        indexer.bm25 = None
        indexer.faiss_index = None

    # Remove from documents dict
    documents.pop(filename, None)

    return {"message": f"{filename} deleted", "remaining": list(documents.keys())}