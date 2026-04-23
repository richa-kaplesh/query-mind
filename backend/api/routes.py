from fastapi import APIRouter, UploadFile, File, BackgroundTasks, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os
import shutil
import logging
import time
import json
from typing import List

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

@router.post("/query/stream")
async def query_document_stream(body: QueryRequest, request: Request):
    log.info(f"[STREAM] Question: '{body.question}'")

    embedder = request.app.state.embedder
    retriever = request.app.state.retriever
    reranker = request.app.state.reranker
    generator = request.app.state.generator

    query_embedding = embedder.embed_query(body.question)

    chunks = retriever.retrieve(
        query=body.question,
        query_embedding=query_embedding,
        top_k=20
    )

    chunks = reranker.rerank(
        query=body.question,
        chunks=chunks,
        top_k=5
    )

    sources = [
        {
            "source": chunk["metadata"].get("source", "unknown"),
            "page": chunk["metadata"].get("page", "N/A"),
            "text": chunk["text"],
            "rerank_score": chunk["rerank_score"]
        }
        for chunk in chunks
    ]

    def event_stream():
        for token in generator.generate_stream(body.question, chunks):
            data = json.dumps({"type": "token", "content": token})
            yield f"data: {data}\n\n"

        data = json.dumps({"type": "sources", "content": sources})
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