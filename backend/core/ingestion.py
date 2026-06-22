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

    def ingest(self, file_path: str) -> dict:
        ext = Path(file_path).suffix.lower()

        if ext == ".pdf":
            extractor = PDFExtractor()
        elif ext == ".csv":
            extractor = CSVExtractor()
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        pages = extractor.extract(file_path)
        chunks = self.chunker.chunk_pages(pages)
        chunks = self.embedder.embed_chunks(chunks)
        self.indexer.index(chunks)

        return {
            "file": file_path,
            "pages": len(pages),
            "chunks": len(chunks)
        }