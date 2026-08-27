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
from core.models import QueryRequest, CSVSchema, ColumnSchema, ExtractedPage, PageMetadata

from core.extractors.csv_extractor import CSVExtractor
from core.extractors.pdf_extractor import PDFExtractor
from core.ingestion import IngestionPipeline
from core.tools.pandas_sandbox_tool import PandasSandboxTool
from core.tracer import TraceStore

import groq
import httpx as httpx_module


log = logging.getLogger("routes")

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# In-memory document store: { filename → {status, file_path, schema, ...} }
documents: dict = {}



@router.get("/debug/versions")
async def debug_versions():
    return {
        "groq_version": groq.__version__,
        "httpx_version": httpx_module.__version__,
    }

@router.post("/upload")
async def upload_document(
    request: Request,
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
):
    documents.clear()
    uploaded=[]
    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        documents[file.filename]={
            "status":"processing",
            "file_path": file_path,
            "file_type": Path(file.filename).suffix.lower(),
            "schema":""
        }
        # background_tasks.add_task(fn, **kwargs)
        background_tasks.add_task(
            ingest_document,
            file_path=file_path,
            filename= file.filename,
            app_state= request.app.state
        )
        uploaded.append(file.filename)

    return {
    "message": "Documents received, ingestion started in background",
    "files": uploaded,
    "status": "processing"
    }


def _extract_schema_sync(file_path: str) ->str:
        extractor = CSVExtractor()
        pages = extractor.extract(file_path)
        return pages[0].text if pages else ""

async def ingest_document(file_path: str, filename: str, app_state) -> None:
    ext = Path(file_path).suffix.lower()

    try:
        if ext == ".pdf":
            pipeline = IngestionPipeline(
                embedder = app_state.embedder,
                indexer = app_state.indexer,
            )
            # await ... — pauses this function right here
            # asyncio.to_thread() returns something awaitable — a placeholder that says "the thread is running, I'll let you know when it's done."
            result = await asyncio.to_thread(pipeline.ingest, file_path)
            documents[filename]["status"]="ready"
        
        elif ext == ".csv":
            schema = await asyncio.to_thread(_extract_schema_sync, file_path)
            documents[filename]["status"]="ready"
            documents[filename]["schema"]= schema
        else:
            raise ValueError(f"Unsupported file type:{ext}")
    except Exception as e:
        documents[filename]["status"]="failed"
        log.error(f"[INGEST] Failed for {filename}:{e}",exc_info=True)

def _get_active_file() -> dict | None:
    return next((v for v in documents.values() if v.get("status") == "ready"), None)

def _filename_for_path(file_path: str) -> str:
    return next((k for k, v in documents.items() if v.get("file_path") == file_path), file_path)

def _make_tracer(trace_store, trace_id: str):
    def tracer(step_type: str, data: dict):
        trace_store.add_step(trace_id, step_type, data)
    return tracer

@router.post("/query/stream")
async def query_document_stream(body: QueryRequest, request: Request):
    current_file = _get_active_file()
    if not current_file:
        raise HTTPException(status_code=400, detail="No active dataset found. Upload a file first.")

    generator   = request.app.state.generator
    trace_store = request.app.state.trace_store
    file_path   = current_file["file_path"]
    file_type   = current_file.get("file_type", ".csv")

    trace_id = trace_store.start_trace(_filename_for_path(file_path), body.question)
    tracer   = _make_tracer(trace_store, trace_id)

    # ── PDF path ────────────────────────────────────────────────────────
    if file_type == ".pdf":
        embedder  = request.app.state.embedder
        retriever = request.app.state.retriever
        reranker  = request.app.state.reranker

        query_embedding = await asyncio.to_thread(embedder.embed_query, body.question)
        raw_chunks      = await asyncio.to_thread(retriever.retrieve, body.question, query_embedding)
        chunks          = await asyncio.to_thread(reranker.rerank, body.question, raw_chunks)

        def event_stream_pdf():
            final_answer_parts = []
            try:
                for token in generator.generate_rag_stream(
                    query=body.question,
                    chunks=chunks,
                    tracer=tracer,
                    token_tracker=request.app.state.token_tracker,
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

    # ── CSV path ────────────────────────────────────────────────────────
    schema = current_file.get("schema", "")
    generator.tools = [PandasSandboxTool(file_path=file_path)]

    def event_stream():
        final_answer_parts = []
        tool_used = None
        try:
            for token in generator.generate_stream(
                query=body.question,
                schema=schema,
                tracer=tracer,
                token_tracker=request.app.state.token_tracker,
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


@router.get("/documents")
async def get_documents():
    return {k: v["status"] for k, v in documents.items()}


@router.get("/documents/status")
def status():
    return {"documents": {k: v["status"] for k, v in documents.items()}}


@router.delete("/documents/{filename}")
def delete(filename: str):
    documents.pop(filename, None)
    return {"message": f"{filename} removed", "remaining": list(documents.keys())}


@router.post("/reset")
async def reset_session():
    count = len(documents)
    documents.clear()
    log.info(f"[RESET] Session cleared — removed {count} document(s)")
    return {"message": "Session reset", "cleared": count}