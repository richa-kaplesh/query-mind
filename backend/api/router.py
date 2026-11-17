from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os
from config import settings
from core.exceptions import PDFPasswordProtectedError, PDFCorruptError, DocumentExtractionError
from core import gateway_client
from core.tools.rag_tool import RAGTool
import shutil
import asyncio
import logging
import time
import json
from typing import List, Optional
from pathlib import Path
from core.models import QueryRequest, CSVSchema, ColumnSchema, ExtractedPage, PageMetadata
import uuid
from core.extractors.csv_extractor import CSVExtractor
from core.extractors.pdf_extractor import PDFExtractor
from core.ingestion import IngestionPipeline
from core.tools.pandas_sandbox_tool import PandasSandboxTool
from core.tracer import TraceStore
from core.tools.pandas_worker_manager import worker_manager
from core.reviewer import Reviewer,RAGReviewer
from core.sessions import SessionCapacityError
from core.tools.data_inspect_tool import DataInspectTool
from fastapi import Depends
from api.session import get_session_id
from api.session import get_session
from core.sessions import Session
from core.indexer import Indexer
import secrets
from core.generator import Generator
from core.retriever import HybridRetriever
from core.logging_setup import session_id_var, SID_PREFIX_LEN
from core.tools import data_inspect_tool

log = logging.getLogger("router")

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)





@router.post("/session")
def create_session(request: Request):
    try:
        s = request.app.state.sessions.create()
    except SessionCapacityError:
        log.warning("session rejected, server full", extra={"fields": {"max_sessions": settings.max_sessions}})
        raise HTTPException(status_code=503, detail="Server is full, try again later")
    log.info("session created", extra={"fields": {"sid": s.id[:SID_PREFIX_LEN], "active_sessions": request.app.state.sessions.count()}})
    return {"session_id": s.id}



ALLOWED_TYPES = {".csv", ".pdf"}
@router.post("/upload")
async def upload_document(
    request: Request,
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    session:Session = Depends(get_session),
):
    if len(files)!=1:
        raise HTTPException(status_code=400,detail="Please Upload one file")
    file = files[0]
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_TYPES:
        raise HTTPException(status_code = 400,detail ="Only .csv and .pdf files are supported")
    if session.file_path:
        await asyncio.to_thread(_forget_file, session.file_path, session.file_type)
    folder = Path(settings.upload_dir)/session.id
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True, exist_ok=True)
    file_path = folder/ f"{secrets.token_hex(8)}{ext}"
    with open(file_path,"wb") as f:
        shutil.copyfileobj(file.file, f)
    session.filename = Path(file.filename).name
    session.file_path = str(file_path)
    session.file_type = ext
    session.status = "processing"
    session.error = None
    session.schema = None

    log.info("upload received", extra={"fields": {"file_type": ext, "size_bytes": file_path.stat().st_size, "stored_as": file_path.name}})
    background_tasks.add_task(ingest_document, session=session, app_state=request.app.state)
    return {"message":"File received, processing started","file":session.filename,"status":"processing"}


async def ingest_document(session: Session, app_state) -> None:
    session_id_var.set(session.id[:SID_PREFIX_LEN])
    my_path = session.file_path
    my_type = session.file_type
    started = time.perf_counter()

    def elapsed_ms() -> int:
        return int((time.perf_counter() - started) * 1000)

    def still_current() -> bool:
        return session.file_path == my_path

    def log_stale() -> None:
        log.info("ingest stale, skipped write", extra={"fields": {"stored_as": Path(my_path).name, "duration_ms": elapsed_ms()}})

    def done() -> None:
        log.info("ingest done", extra={"fields": {"file_type": my_type, "stored_as": Path(my_path).name, "duration_ms": elapsed_ms()}})

    def fail(code: str) -> None:
        if still_current():
            session.status, session.error = "failed", code
            log.warning("ingest failed", extra={"fields": {"error": code, "stored_as": Path(my_path).name, "duration_ms": elapsed_ms()}})
        else:
            log_stale()

    log.info("ingest start", extra={"fields": {"file_type": my_type, "stored_as": Path(my_path).name}})
    try:
        if my_type == ".csv":
            status, payload = await asyncio.to_thread(worker_manager.load_file, my_path)
            if status == "error":
                log.error(f"[INGEST] Worker failed to load {session.filename}: {payload}")
                return fail("load_error")
            if still_current():
                session.schema = CSVSchema(**payload)
                session.status = "ready"
                done()
            else:
                log_stale()
        else:
            new_indexer = Indexer()
            pipeline = IngestionPipeline(embedder=app_state.embedder, indexer=new_indexer)
            await asyncio.to_thread(pipeline.ingest, my_path)
            if still_current():
                session.indexer = new_indexer
                session.status = "ready"
                done()
            else:
                log_stale()
    except PDFPasswordProtectedError:
        fail("password_protected")
    except PDFCorruptError:
        fail("corrupt_file")
    except DocumentExtractionError:
        fail("no_extractable_text")
    except Exception as e:
        log.error(f"[INGEST] Failed for {session.filename}: {e}", exc_info=True)
        fail("unknown")


def _make_tracer(trace_store, trace_id: str):
    def tracer(step_type: str, data: dict):
        trace_store.add_step(trace_id, step_type, data)
    return tracer

def _log_query_done(file_type: str, started: float, status: str, tool_used: str | None) -> None:
    log.info("query done", extra={"fields": {
        "file_type": file_type,
        "status": status,
        "tool_used": tool_used,
        "duration_ms": int((time.perf_counter() - started) * 1000),
    }})

@router.post("/query/stream")
async def query_document_stream(body: QueryRequest, request: Request,session:Session= Depends(get_session)):
    if session.status != "ready":
        raise HTTPException(status_code = 400, detail=session.error or f"File not ready (status: {session.status})")
   
    file_path = session.file_path
    file_type = session.file_type
    conversation_id = session.conversation_id
    
    trace_store = request.app.state.trace_store

    trace_id = trace_store.start_trace(session.filename, body.question)
    tracer   = _make_tracer(trace_store, trace_id)
    started = time.perf_counter()
    log.info("query start", extra={"fields": {"file_type": file_type, "stored_as": Path(file_path).name}})

    # ── PDF path ────────────────────────────────────────────────────────
        # ── PDF path (agentic — model decides when/what to search) ────────────
    if file_type == ".pdf":
        retriever = HybridRetriever(indexer=session.indexer)
        generator = Generator(tools=[RAGTool(
            retriever=retriever,
            embedder=request.app.state.embedder,
            reranker=request.app.state.reranker,
        )])
        rag_reviewer = RAGReviewer(max_iterations=settings.reviewer_max_iterations)

        async def event_stream_pdf():
            gateway_client.start_call_budget()
            final_answer_parts = []
            tool_used = None
            try:
                async for token in generator.generate_agentic_rag_stream(
                    query=body.question,
                    conversation_id=conversation_id,
                    tracer=tracer,
                    reviewer=rag_reviewer,
                    token_tracker=request.app.state.token_tracker,
                ):
                    if token.startswith("__tool__:"):
                        tool_used = token.split(":", 1)[1]
                        yield f"data: {json.dumps({'type': 'tool', 'content': tool_used})}\n\n"
                    else:
                        final_answer_parts.append(token)
                        yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
                trace_store.finish_trace(trace_id, "".join(final_answer_parts), tool_used)
                _log_query_done(file_type, started, "ok", tool_used)
            except Exception as e:
                log.error(f"[STREAM/PDF] Error: {e}", exc_info=True)
                trace_store.finish_trace(trace_id, str(e), tool_used, status="error")
                _log_query_done(file_type, started, "error", tool_used)
                yield f"data: {json.dumps({'type': 'error', 'status': getattr(e, 'status', None), 'reason': getattr(e, 'reason', None), 'content': str(e)})}\n\n"
                yield f"data: {json.dumps({'type': 'token', 'content': f'[Error: {e}]'})}\n\n"
            finally:
                yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream_pdf(), media_type="text/event-stream")

    # ── CSV path ────────────────────────────────────────────────────────
    schema = session.schema
    generator = Generator(tools=[PandasSandboxTool(file_path=file_path)])
    reviewer = Reviewer(
        inspect_tool=DataInspectTool(file_path=file_path),
        max_iterations=settings.reviewer_max_iterations,
    )

    async def event_stream():
        gateway_client.start_call_budget()
        final_answer_parts = []
        tool_used = None
        try:
            async for token in generator.generate_stream(
                conversation_id=conversation_id,
                query=body.question,
                schema=schema,
                tracer=tracer,
                token_tracker=request.app.state.token_tracker,
                reviewer=reviewer,
            ):
                if token.startswith("__tool__:"):
                    tool_used = token.split(":", 1)[1]
                    yield f"data: {json.dumps({'type': 'tool', 'content': tool_used})}\n\n"
                else:
                    final_answer_parts.append(token)
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
            trace_store.finish_trace(trace_id, "".join(final_answer_parts), tool_used)
            _log_query_done(file_type, started, "ok", tool_used)
        except Exception as e:
            log.error(f"[STREAM] Error: {e}", exc_info=True)
            trace_store.finish_trace(trace_id, str(e), tool_used, status="error")
            _log_query_done(file_type, started, "error", tool_used)
            yield f"data: {json.dumps({'type': 'error', 'status': getattr(e, 'status', None), 'reason': getattr(e, 'reason', None), 'content': str(e)})}\n\n"
            yield f"data: {json.dumps({'type': 'token', 'content': f'[Error: {e}]'})}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _session_documents(session: Session) -> dict:
    return {session.filename: session.status} if session.filename else {}


def _forget_file(file_path: str, file_type: str | None) -> None:
    if file_type == ".csv":
        worker_manager.unload(file_path)
        data_inspect_tool.evict(file_path)


def _clear_file(session: Session) -> int:
    if not session.file_path:
        return 0
    _forget_file(session.file_path, session.file_type)
    shutil.rmtree(Path(settings.upload_dir) / session.id, ignore_errors=True)
    session.filename = None
    session.file_path = None
    session.file_type = None
    session.status = "empty"
    session.error = None
    session.schema = None
    session.indexer = Indexer()
    return 1


@router.get("/documents")
async def get_documents(session: Session = Depends(get_session)):
    return _session_documents(session)


@router.get("/documents/status")
def status(session: Session = Depends(get_session)):
    return {"documents": _session_documents(session)}


@router.delete("/documents/{filename}")
def delete(filename: str, session: Session = Depends(get_session)):
    if filename != session.filename:
        raise HTTPException(status_code=404, detail=f"{filename} not found")
    _clear_file(session)
    log.info("file cleared", extra={"fields": {"cleared": 1}})
    return {"message": f"{filename} removed", "remaining": []}


@router.post("/reset")
async def reset_session(session: Session = Depends(get_session)):
    count = await asyncio.to_thread(_clear_file, session)
    session.conversation_id = secrets.token_hex(16)
    log.info(f"[RESET] Session cleared — removed {count} document(s)", extra={"fields": {"cleared": count}})
    return {"message": "Session reset", "cleared": count}

def _test_worker(queue):
    queue.put("hello from subprocess")