from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os
import shutil
import asyncio
import logging
import time
import json
from typing import List, Optional
from pathlib import Path

from core.extractors.csv_extractor import CSVExtractor
from core.extractors.pdf_extractor import PDFExtractor
from core.ingestion import IngestionPipeline
from core.tools.pandas_sandbox_tool import PandasSandboxTool
from core.tracer import TraceStore

log = logging.getLogger("routes")

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# In-memory document store: { filename → {status, file_path, schema, ...} }
documents: dict = {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_tracer(trace_store: TraceStore, trace_id: str):
    """Return a tracer callable that records steps into the given trace."""
    def tracer(step_type: str, data: dict):
        trace_store.add_step(trace_id, step_type, data)
    return tracer


def _get_active_file() -> dict | None:
    """Return the first document with status='ready', or None."""
    return next((v for v in documents.values() if v.get("status") == "ready"), None)


def _filename_for_path(file_path: str) -> str:
    """Reverse-lookup: find the filename key for a given file_path."""
    return next(
        (k for k, v in documents.items() if v.get("file_path") == file_path),
        file_path
    )


# ── Schema extraction ─────────────────────────────────────────────────────────

def _extract_schema_sync(file_path: str) -> str:
    """
    Synchronous schema extraction — runs in a thread pool via asyncio.to_thread.
    Returns the formatted schema text string.
    """
    extractor = CSVExtractor()
    pages = extractor.extract(file_path)
    return pages[0].text if pages else ""


async def _extract_schema_async(file_path: str) -> str:
    """Async wrapper: runs blocking pandas/IO work in the thread-pool executor."""
    return await asyncio.to_thread(_extract_schema_sync, file_path)


# ── Background ingestion (upload-only path) ───────────────────────────────────

async def ingest_document(file_path: str, filename: str, app_state) -> None:
    """
    Background task: branch on file type and run the appropriate pipeline.

    CSV  → CSVExtractor.extract() → schema text → documents[filename]["schema"]
    PDF  → IngestionPipeline (extract → chunk → embed → index via app.state components)

    app_state is passed explicitly (background tasks run outside request context
    and cannot use request.app.state directly).
    """
    log.info(f"[INGEST] Starting for: {filename}")
    t0 = time.time()
    ext = Path(file_path).suffix.lower()

    try:
        if ext == ".pdf":
            # ── PDF: full RAG ingestion pipeline ─────────────────────────────
            pipeline = IngestionPipeline(
                embedder=app_state.embedder,
                indexer=app_state.indexer,
            )
            result = await asyncio.to_thread(pipeline.ingest, file_path)
            documents[filename]["status"] = "ready"
            log.info(
                f"[INGEST/PDF] Done in {(time.time()-t0):.2f}s — "
                f"{result['pages']} pages, {result['chunks']} chunks — {filename}"
            )

        elif ext == ".csv":
            # ── CSV: schema extraction only (no chunker/embedder) ─────────────
            schema = await asyncio.to_thread(_extract_schema_sync, file_path)
            documents[filename]["status"] = "ready"
            documents[filename]["schema"] = schema
            log.info(f"[INGEST/CSV] Done in {(time.time()-t0):.2f}s — {filename}")

        else:
            raise ValueError(f"Unsupported file type: {ext}")

    except Exception as e:
        documents[filename]["status"] = "failed"
        log.error(f"[INGEST] Failed for {filename}: {e}", exc_info=True)



# ── Request models ────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    conversation_history: list = []


# ─────────────────────────────────────────────────────────────────────────────
# /upload  — save file(s) and kick off schema extraction in background
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/upload")
async def upload_document(
    request: Request,
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
):
    uploaded = []

    for file in files:
        log.info(f"[UPLOAD] Received: {file.filename}")
        file_path = os.path.join(UPLOAD_DIR, file.filename)

        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        log.info(f"[UPLOAD] Saved: {file_path}")

        documents[file.filename] = {
            "status":    "processing",
            "file_path": file_path,
            "file_type": Path(file.filename).suffix.lower(),
            "schema":    ""
        }

        background_tasks.add_task(
            ingest_document,
            file_path=file_path,
            filename=file.filename,
            app_state=request.app.state,
        )
        uploaded.append(file.filename)

    return {
        "message": "Documents received, ingestion started in background",
        "files":   uploaded,
        "status":  "processing"
    }


# ─────────────────────────────────────────────────────────────────────────────
# /chat  — combined upload + query with concurrent extraction and LLM call
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/chat")
async def chat(
    request:  Request,
    question: str                  = Form(...),
    file:     Optional[UploadFile] = File(None)
):
    """
    Primary endpoint for the CSV Q&A pipeline.

    With file  → saves it, extracts schema, then calls LLM (sequential in one task).
    Without file → uses the most-recently-ready document; fires LLM immediately.
    """
    t0 = time.time()
    log.info(f"[CHAT] Question: '{question}'")

    generator   = request.app.state.generator
    trace_store = request.app.state.trace_store

    # ── Branch A: new file uploaded alongside the query ───────────────────────
    if file and file.filename:
        log.info(f"[CHAT] File provided: {file.filename}")
        file_path = os.path.join(UPLOAD_DIR, file.filename)

        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        log.info(f"[CHAT] Saved to disk: {file_path}")

        schema = await _extract_schema_async(file_path)
        log.info(f"[CHAT] Schema ready")

        documents[file.filename] = {
            "status":    "ready",
            "file_path": file_path,
            "file_type": Path(file.filename).suffix.lower(),
            "schema":    schema
        }

        generator.tools = [PandasSandboxTool(file_path=file_path)]

        trace_id = trace_store.start_trace(file.filename, question)
        tracer   = _make_tracer(trace_store, trace_id)

        try:
            result = await generator.agenerate_with_tools(question, schema=schema, tracer=tracer)
            trace_store.finish_trace(trace_id, result.get("answer", ""), result.get("tool_used"))
        except Exception as e:
            trace_store.finish_trace(trace_id, str(e), None, status="error")
            raise

    # ── Branch B: no file — use already-active dataset ────────────────────────
    else:
        current_file = _get_active_file()
        if not current_file:
            raise HTTPException(
                status_code=400,
                detail=(
                    "No active CSV dataset found. "
                    "Upload a file first via /upload, or send it alongside this request."
                )
            )

        schema    = current_file.get("schema", "")
        file_path = current_file["file_path"]
        generator.tools = [PandasSandboxTool(file_path=file_path)]

        log.info(f"[CHAT] Reusing active file: {file_path}")

        trace_id = trace_store.start_trace(_filename_for_path(file_path), question)
        tracer   = _make_tracer(trace_store, trace_id)

        try:
            result = await generator.agenerate_with_tools(question, schema=schema, tracer=tracer)
            trace_store.finish_trace(trace_id, result.get("answer", ""), result.get("tool_used"))
        except Exception as e:
            trace_store.finish_trace(trace_id, str(e), None, status="error")
            raise

    log.info(f"[CHAT] ✓ {(time.time()-t0):.2f}s | tool: {result.get('tool_used')}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# /query  — legacy JSON endpoint
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/query")
async def query_document(body: QueryRequest, request: Request):
    log.info(f"[QUERY] Question: '{body.question}'")
    t0 = time.time()

    current_file = _get_active_file()
    if not current_file:
        raise HTTPException(
            status_code=400,
            detail="No active CSV dataset found. Please upload a file first."
        )

    generator   = request.app.state.generator
    trace_store = request.app.state.trace_store
    file_path   = current_file["file_path"]
    schema      = current_file.get("schema", "")

    generator.tools = [PandasSandboxTool(file_path=file_path)]

    trace_id = trace_store.start_trace(_filename_for_path(file_path), body.question)
    tracer   = _make_tracer(trace_store, trace_id)

    try:
        result = await generator.agenerate_with_tools(body.question, schema=schema, tracer=tracer)
        trace_store.finish_trace(trace_id, result.get("answer", ""), result.get("tool_used"))
    except Exception as e:
        trace_store.finish_trace(trace_id, str(e), None, status="error")
        log.error(f"[QUERY] ✗ Failed: {e}", exc_info=True)
        raise

    log.info(f"[QUERY] ✓ {(time.time()-t0):.2f}s | tool: {result.get('tool_used')}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# /query/stream  — SSE streaming endpoint (used by the frontend)
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/query/stream")
async def query_document_stream(body: QueryRequest, request: Request):
    log.info(f"[STREAM] Question: '{body.question}'")

    current_file = _get_active_file()
    if not current_file:
        raise HTTPException(
            status_code=400,
            detail="No active dataset found. Please upload a file first.",
        )

    generator   = request.app.state.generator
    trace_store = request.app.state.trace_store
    file_path   = current_file["file_path"]
    file_type   = current_file.get("file_type", ".csv")

    trace_id = trace_store.start_trace(_filename_for_path(file_path), body.question)
    tracer   = _make_tracer(trace_store, trace_id)

    # ── PDF path: hybrid-retrieve → rerank → RAG stream ──────────────────────
    if file_type == ".pdf":
        embedder  = request.app.state.embedder
        retriever = request.app.state.retriever
        reranker  = request.app.state.reranker

        query_embedding = await asyncio.to_thread(
            embedder.embed_query, body.question
        )
        raw_chunks = await asyncio.to_thread(
            retriever.retrieve, body.question, query_embedding
        )
        chunks = await asyncio.to_thread(
            reranker.rerank, body.question, raw_chunks
        )
        log.info(f"[STREAM/PDF] Retrieved {len(raw_chunks)} → reranked to {len(chunks)} chunks")

        def event_stream_pdf():
            final_answer_parts: list[str] = []
            try:
                for token in generator.generate_rag_stream(
                    query=body.question,
                    chunks=chunks,
                    tracer=tracer,
                ):
                    final_answer_parts.append(token)
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
                trace_store.finish_trace(trace_id, "".join(final_answer_parts), None)
            except Exception as e:
                log.error(f"[STREAM/PDF] Error: {e}", exc_info=True)
                trace_store.finish_trace(trace_id, str(e), None, status="error")
                yield f"data: {json.dumps({'type': 'token', 'content': f'[Error: {e}]'})}\n\n"
            finally:
                yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream_pdf(), media_type="text/event-stream")

    # ── CSV path: existing tool-calling stream (unchanged) ────────────────────
    schema = current_file.get("schema", "")
    generator.tools = [PandasSandboxTool(file_path=file_path)]

    def event_stream():
        """SSE generator: converts generate_stream tokens into SSE events."""
        final_answer_parts = []
        tool_used = None
        try:
            for token in generator.generate_stream(
                query=body.question,
                schema=schema,
                tracer=tracer
            ):
                if token.startswith("__tool__:"):
                    tool_used = token.split(":", 1)[1]
                    yield f"data: {json.dumps({'type': 'tool', 'content': tool_used})}\n\n"
                else:
                    final_answer_parts.append(token)
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
            trace_store.finish_trace(trace_id, "".join(final_answer_parts), tool_used)
        except Exception as e:
            log.error(f"[STREAM] Error: {e}", exc_info=True)
            trace_store.finish_trace(trace_id, str(e), tool_used, status="error")
            yield f"data: {json.dumps({'type': 'token', 'content': f'[Error: {e}]'})}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ─────────────────────────────────────────────────────────────────────────────
# /documents  — inspect + delete
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/documents")
async def get_documents():
    return {"documents": {k: v["status"] for k, v in documents.items()}}


@router.get("/documents/status")
async def get_documents_status():
    """
    Lightweight poll endpoint used by the frontend to detect when a document
    transitions from 'processing' → 'ready' so it can auto-submit a queued query.
    Returns {filename: status} for every tracked document.
    """
    return {"documents": {k: v["status"] for k, v in documents.items()}}


@router.delete("/documents/{filename}")
async def delete_document(filename: str):
    documents.pop(filename, None)
    return {"message": f"{filename} removed", "remaining": list(documents.keys())}


# ─────────────────────────────────────────────────────────────────────────────
# /reset  — clear all session state server-side
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/reset")
async def reset_session():
    """
    Clears the in-memory document store (all filenames, schemas, statuses).
    Called by the frontend on 'New Conversation' and 'Delete Conversation'.
    Does NOT delete files from disk — re-upload is cheap and avoids race conditions.
    """
    count = len(documents)
    documents.clear()
    log.info(f"[RESET] Session cleared — removed {count} document(s)")
    return {"message": "Session reset", "cleared": count}