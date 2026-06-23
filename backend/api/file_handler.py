import os
import shutil
import logging
from pathlib import Path

log = logging.getLogger("file_handler")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

documents = {}


def save_file(file) -> tuple[str, str]:
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    log.info(f"[FILE] Saved to disk: {file_path}")
    return file_path, Path(file.filename).suffix.lower()


async def run_ingestion(file_path: str, filename: str, request):
    log.info(f"[INGEST] Starting ingestion for: {filename}")
    import time
    t0 = time.time()
    try:
        pipeline = request.app.state.ingestion
        result = pipeline.ingest(file_path)
        documents[filename]["status"] = "ready"
        log.info(f"[INGEST] ✓ Done in {(time.time()-t0):.2f}s | {result['chunks']} chunks")
    except Exception as e:
        documents[filename]["status"] = "failed"
        log.error(f"[INGEST] ✗ Failed: {e}", exc_info=True)