# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — builder
#   Installs all Python dependencies into an isolated venv so the final stage
#   can copy just the venv without build tools, pip cache, or wheel artefacts.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# System build deps needed to compile wheels (faiss-cpu, numpy, scipy, Pillow)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create and activate a venv in a predictable location
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip first — avoids "legacy resolver" warnings that slow installs
RUN pip install --upgrade pip

# Copy requirements and install — no src code yet so this layer is cached
# as long as requirements.txt doesn't change.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — runtime
#   Copies only the venv + application source.  No build tools, no pip,
#   no wheel cache — keeps the final image as small as possible.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Runtime system deps:
#   libgomp1    — required by faiss-cpu (OpenMP)
#   tesseract   — required by pytesseract (PDF OCR fallback in pdf_extractor.py)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Copy the venv from the builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# ── Application setup ─────────────────────────────────────────────────────────

WORKDIR /app

# Copy source code (everything not excluded by .dockerignore)
COPY . /app

# Create the uploads directory at build time.
# At runtime, mount a host volume here for persistence across restarts:
#   docker run -v /host/uploads:/app/uploads ...
RUN mkdir -p /app/uploads

# ── Runtime configuration ─────────────────────────────────────────────────────

# Port the server will listen on — override via -e PORT=... at runtime
ENV PORT=8000

# All secrets are injected at runtime via environment variables or
# a Docker secrets / compose env_file — never baked into the image:
#   GROQ_API_KEY   — Groq LLM inference
#   JINA_API_KEY   — Jina embeddings + reranking
ENV GROQ_API_KEY=""
ENV JINA_API_KEY=""

EXPOSE ${PORT}

# ── Entrypoint ────────────────────────────────────────────────────────────────

# --host 0.0.0.0   : bind to all interfaces (required inside a container)
# --port $PORT     : from env var, defaults to 8000
# --workers 1      : single worker — FAISS index is in-process memory and
#                    cannot be shared across multiple worker processes safely.
#                    Use a proper vector store (e.g. Qdrant) if you need >1.
# --no-access-log  : request logging is handled by the FastAPI middleware
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1 --no-access-log"]
