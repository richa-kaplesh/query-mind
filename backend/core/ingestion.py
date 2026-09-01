import logging
from core.extractors.pdf_extractor import PDFExtractor
from core.extractors.csv_extractor import CSVExtractor
from core.models import PDFExtractionResult, CSVExtractionResult
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

        if ext == ".pdf":
            extractor = PDFExtractor()
        elif ext == ".csv":
            extractor = CSVExtractor()
        else:
            raise ValueError(f"IngestionPipeline does not support file type: {ext}")

        self.log.info(f"Extracting {ext} file...")
        result = extractor.extract(file_path)

        if isinstance(result, PDFExtractionResult):
            self.log.info(f"Extracted {len(result.pages)} pages")

            self.log.info("Chunking...")
            chunks = self.chunker.chunk_pages(result.pages)
            self.log.info(f"Created {len(chunks)} chunks")

            self.log.info("Embedding...")
            chunks = self.embedder.embed_chunks(chunks)
            self.log.info("Embedding done")

            self.log.info("Indexing...")
            self.indexer.index(chunks)
            self.log.info("Indexing done")

            return {
                "file": file_path,
                "type": "pdf",
                "pages": len(result.pages),
                "chunks": len(chunks)
            }
        elif isinstance(result, CSVExtractionResult):
            self.log.info(f"Extracted CSVSchema for {result.schema.source}")
            return {
                "file": file_path,
                "type": "csv",
                "schema": result.schema
            }
        else:
            raise TypeError(f"Unexpected extraction result type: {type(result)}")