from backend.core.extractors.pdf_extractor import PDFExtractor
from backend.core.extractors.csv_extractor import CSVExtractor
from backend.core.chunker import TextChunker
from backend.core.embedder import Embedder
from backend.core.indexer import Indexer
from pathlib import Path


class IngestionPipeline:

    def __init__(self):
        self.chunker = TextChunker()
        self.embedder = Embedder()
        self.indexer = Indexer()

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