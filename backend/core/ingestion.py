import logging
from core.extractors.pdf_extractor import PDFExtractor
from core.extractors.csv_extractor import CSVExtractor
from core.chunker import TextChunker
from core.embedder import Embedder
from core.indexer import Indexer
from pathlib import Path


class IngestionPipeline:

    def __init__(self, embedder: Embedder, indexer: Indexer):
        self.chunker = TextChunker()
        self.embedder = embedder
        self.indexer = indexer
        self.log = logging.getLogger("ingestion")

    def ingest(self, file_path: str) -> dict:
        ext = Path(file_path).suffix.lower()

        if ext != ".pdf":
            raise ValueError(f"IngestionPipeline only handles PDFs, got: {ext}")

        extractor = PDFExtractor()

        self.log.info("Extracting pages...")
        pages = extractor.extract(file_path)
        self.log.info(f"Extracted {len(pages)} pages")

        self.log.info("Chunking...")
        chunks = self.chunker.chunk_pages(pages)
        self.log.info(f"Created {len(chunks)} chunks")

        self.log.info("Embedding...")
        chunks = self.embedder.embed_chunks(chunks)
        self.log.info("Embedding done")

        self.log.info("Indexing...")
        self.indexer.index(chunks)
        self.log.info("Indexing done")

        return {
            "file": file_path,
            "pages": len(pages),
            "chunks": len(chunks)
        }