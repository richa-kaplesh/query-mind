from fastapi import APIRouter, UploadFile, File, BackgroundTasks, Request
from pydantic import BaseModel
import os
import shutil

from core.pdf_extractor import PDFExtractor
from core.chunker import TextChunker

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR , exist_ok= True)

documents={}

class QueryRequest(BaseModel):
    question: str
    conversation_history: list=[]

async def ingest_document(file_path:str , filename:str , request:Request):
    embedder = request.app.state.embedder
    indexer = request.app.state.indexer

    extractor = PDFExtractor()
    chunker = TextChunker()
    pages = extractor.extract(file_path)

    all_chunks=[]
    for page in pages:
        chunks = chunker.chunk_text(page["text"],metadata= page["metadata"])
        all_chunks.extend(chunks)
    
    all_chunks = embedder.embed_chunks(all_chunks)
    indexer.index(all_chunks)

    documents[filename]="ready"

@router.post("/upload")
async def upload_document(
    background_tasks : BackgroundTasks,
    file: UploadFile = File(...),
    request: Request = None
    ):
    file_path = os.path.join(UPLOAD_DIR , file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file , f)

    documents[file.filename]= "processing"

    background_tasks.add_task(
            ingest_document,
            file_path=file_path,
            filename = file.filename,
            request = request
    )
    return {
            "message": "document recieved , processing started",
            "filename": file.filename,
            "status":"processing"
    }
@router.get("/documents")
async def get_documents():
    return {"documents":documents}

@router.post("/query")
async def query_document(body: QueryRequest, request: Request):
    embedder = request.app.state.embedder
    retriever = request.app.state.retriever
    reranker = request.app.state.reranker
    generator = request.app.state.generator

    query_embedding = embedder.embed_query(body.question)

    chunks = retriever.retrieve(
        query= body.question,
        query_embedding = query_embedding,
        top_k=20
    )
    chunks = reranker.rerank(
        query=body.question,
        chunks = chunks,
        top_k=5
    )
    result= generator.generate(
        query=body.question,
        chunks = chunks
    )
    return {
        "answer": result["answer"],
        "sources": result["sources"]
    }
