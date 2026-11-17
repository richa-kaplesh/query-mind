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

log = logging.getLogger("router")

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# In-memory document store: { filename → {status, file_path, schema, conversation_id, ...} }
documents: dict = {}




@router.post("/session")
def create_session(request: Request):
    try:
        s = request.app.state.sessions.create()
    except SessionCapacityError:
        raise HTTPException(status_code=503, detail="Server is full, try again later")
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

    background_tasks.add_task(ingest_document, session=session, app_state=request.app.state)
    return {"message":"File received, processing started","file":session.filename,"status":"processing"}


async def ingest_document(session: Session, app_state) -> None:
    my_path = session.file_path

    def still_current() -> bool:
        return session.file_path == my_path

    def fail(code: str) -> None:
        if still_current():
            session.status, session.error = "failed", code

    try:
        if session.file_type == ".csv":
            status, payload = await asyncio.to_thread(worker_manager.load_file, my_path)
            if status == "error":
                log.error(f"[INGEST] Worker failed to load {session.filename}: {payload}")
                return fail("load_error")
            if still_current():
                session.schema = CSVSchema(**payload)
                session.status = "ready"
        else:
            new_indexer = Indexer()
            pipeline = IngestionPipeline(embedder=app_state.embedder, indexer=new_indexer)
            await asyncio.to_thread(pipeline.ingest, my_path)
            if still_current():
                session.indexer = new_indexer
                session.status = "ready"
    except PDFPasswordProtectedError:
        fail("password_protected")
    except PDFCorruptError:
        fail("corrupt_file")
    except DocumentExtractionError:
        fail("no_extractable_text")
    except Exception as e:
        log.error(f"[INGEST] Failed for {session.filename}: {e}", exc_info=True)
        fail("unknown")
        
def _get_active_file() -> dict | None:
    return next((v for v in documents.values() if v.get("status") == "ready"), None)

def _filename_for_path(file_path: str) -> str:
    return next((k for k, v in documents.items() if v.get("file_path") == file_path), file_path)

def _make_tracer(trace_store, trace_id: str):
    def tracer(step_type: str, data: dict):
        trace_store.add_step(trace_id, step_type, data)
    return tracer

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
            except Exception as e:
                log.error(f"[STREAM/PDF] Error: {e}", exc_info=True)
                trace_store.finish_trace(trace_id, str(e), tool_used, status="error")
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
        except Exception as e:
            log.error(f"[STREAM] Error: {e}", exc_info=True)
            trace_store.finish_trace(trace_id, str(e), tool_used, status="error")
            yield f"data: {json.dumps({'type': 'error', 'status': getattr(e, 'status', None), 'reason': getattr(e, 'reason', None), 'content': str(e)})}\n\n"
            yield f"data: {json.dumps({'type': 'token', 'content': f'[Error: {e}]'})}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/documents")
async def get_documents(session_id:str = Depends(get_session_id)):
    print("Session:",session_id)
    return {k: v["status"] for k, v in documents.items()}


@router.get("/documents/status")
def status(session_id:str = Depends(get_session_id)):
    return {"documents": {k: v["status"] for k, v in documents.items()}}


@router.delete("/documents/{filename}")
def delete(filename: str,session_id:str = Depends(get_session_id)):
    documents.pop(filename, None)
    return {"message": f"{filename} removed", "remaining": list(documents.keys())}


@router.post("/reset")
async def reset_session(session_id:str = Depends(get_session_id)):
    count = len(documents)
    documents.clear()
    worker_manager.unload()
    log.info(f"[RESET] Session cleared — removed {count} document(s)")
    return {"message": "Session reset", "cleared": count}

def _test_worker(queue):
    queue.put("hello from subprocess")