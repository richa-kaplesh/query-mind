import os
import sys
import argparse
from pathlib import Path

# Add backend directory to sys.path so imports like `core.extractors` work
backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

# Ensure UTF-8 output in Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from core.extractors.pdf_extractor import PDFExtractor
from core.chunker import TextChunker


def get_default_pdf() -> Path | None:
    uploads_dir = backend_dir / "uploads"
    if not uploads_dir.exists():
        return None
    
    pdfs = list(uploads_dir.glob("*.pdf"))
    if not pdfs:
        return None
    
    # Prefer sample_report.pdf or workout_plan.pdf if present, otherwise first pdf
    for preferred in ["sample_report.pdf", "workout_plan.pdf"]:
        for p in pdfs:
            if p.name.lower() == preferred.lower():
                return p
    return pdfs[0]


def inspect_chunks(pdf_path: str, chunk_size: int = 500, chunk_overlap: int = 50, max_display: int | None = None):
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        print(f"Error: File not found at {pdf_file}")
        sys.exit(1)

    print("=" * 80)
    print(f"📄 Processing PDF: {pdf_file.name}")
    print(f"📁 Path: {pdf_file.resolve()}")
    print(f"⚙️ Chunker Settings: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    print("=" * 80)

    # 1. Extraction
    print("\n[1/2] Extracting text and tables with PDFExtractor...")
    extractor = PDFExtractor()
    extraction_result = extractor.extract(str(pdf_file))
    print(f"✓ Extracted {len(extraction_result.pages)} page(s)")

    for idx, page in enumerate(extraction_result.pages):
        warnings = page.metadata.warnings if hasattr(page.metadata, "warnings") else []
        if warnings:
            for w in warnings:
                print(f"  ⚠️  [Page {idx + 1}] Warning: {w}")

    # 2. Chunking
    print("\n[2/2] Chunking extracted pages with TextChunker...")
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunker.chunk_pages(extraction_result.pages)
    total_chunks = len(chunks)
    print(f"✓ Created {total_chunks} total chunk(s)")

    print("\n" + "=" * 80)
    print("🔍 CHUNK INSPECTION RESULTS")
    print("=" * 80)

    display_limit = total_chunks if max_display is None else min(max_display, total_chunks)

    for i in range(display_limit):
        chunk = chunks[i]
        text = chunk.get("text", "")
        metadata = chunk.get("metadata", {})

        page_num = metadata.get("page", "N/A")
        if isinstance(page_num, int):
            page_num = page_num + 1  # 1-based display
        heading = metadata.get("heading") or "(None)"
        chunk_idx = metadata.get("chunk_index", i)

        print(f"\n--- [CHUNK {i + 1} / {total_chunks}] (Page: {page_num} | Heading: '{heading}' | Index: {chunk_idx}) ---")
        print(f"📏 Length: {len(text)} chars | ~{len(text.split())} words")
        print("-" * 80)
        print(text)
        print("-" * 80)

    if max_display and max_display < total_chunks:
        print(f"\n... and {total_chunks - max_display} more chunks not shown (use --all or --max to show more).")

    # Summary Statistics
    if chunks:
        lengths = [len(c.get("text", "")) for c in chunks]
        print("\n" + "=" * 80)
        print("📊 SUMMARY STATISTICS")
        print(f"• Total Chunks: {total_chunks}")
        print(f"• Min Chunk Length: {min(lengths)} chars")
        print(f"• Max Chunk Length: {max(lengths)} chars")
        print(f"• Avg Chunk Length: {sum(lengths) / total_chunks:.1f} chars")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Inspect chunks generated from a PDF by TextChunker.")
    parser.add_argument("pdf_path", nargs="?", help="Path to PDF file (optional; defaults to first PDF in backend/uploads/)")
    parser.add_argument("--chunk-size", type=int, default=500, help="Chunk size in chars (default: 500)")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="Chunk overlap in chars (default: 50)")
    parser.add_argument("--max", type=int, default=None, help="Maximum number of chunks to display (default: all)")
    parser.add_argument("--list", action="store_true", help="List available PDFs in uploads folder")

    args = parser.parse_args()

    uploads_dir = backend_dir / "uploads"

    if args.list:
        print(f"\n📂 PDFs found in {uploads_dir}:")
        if uploads_dir.exists():
            for p in uploads_dir.glob("*.pdf"):
                print(f"  - {p.name}")
        return

    target_pdf = args.pdf_path
    if not target_pdf:
        default_pdf = get_default_pdf()
        if not default_pdf:
            print(f"Error: No PDF files found in {uploads_dir}. Please provide a path to a PDF file.")
            sys.exit(1)
        target_pdf = str(default_pdf)
        print(f"ℹ️  No PDF specified. Defaulting to: {default_pdf.name}\n")

    inspect_chunks(
        pdf_path=target_pdf,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        max_display=args.max
    )


if __name__ == "__main__":
    main()
