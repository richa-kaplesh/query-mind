import os
import sys
import argparse
from pathlib import Path

# Add backend directory to sys.path so imports work properly
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


def get_default_pdf() -> Path | None:
    uploads_dir = backend_dir / "uploads"
    if not uploads_dir.exists():
        return None
    
    pdfs = list(uploads_dir.glob("*.pdf"))
    if not pdfs:
        return None
    
    for preferred in ["sample_report.pdf", "workout_plan.pdf"]:
        for p in pdfs:
            if p.name.lower() == preferred.lower():
                return p
    return pdfs[0]


def inspect_pdf_extraction(pdf_path: str, target_page: int | None = None, show_raw_tags: bool = True):
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        print(f"Error: File not found at {pdf_file}")
        sys.exit(1)

    print("=" * 80)
    print(f"📄 Inspecting PDF Extraction: {pdf_file.name}")
    print(f"📁 Path: {pdf_file.resolve()}")
    print("=" * 80)

    extractor = PDFExtractor()
    result = extractor.extract(str(pdf_file))

    total_pages = len(result.pages)
    print(f"\n✓ Successfully extracted {total_pages} page(s)\n")

    pages_to_show = result.pages
    if target_page is not None:
        if 1 <= target_page <= total_pages:
            pages_to_show = [result.pages[target_page - 1]]
        else:
            print(f"Error: Invalid page number {target_page}. PDF has {total_pages} page(s).")
            sys.exit(1)

    for p in pages_to_show:
        meta = p.metadata
        page_idx = meta.page + 1 if isinstance(meta.page, int) else "N/A"
        heading = meta.heading or "(None)"
        warnings = meta.warnings if meta.warnings else []
        text = p.text

        print("=" * 80)
        print(f"📖 PAGE {page_idx} of {meta.total_pages}")
        print(f"• Source File:    {meta.source}")
        print(f"• Last Heading:   {heading}")
        print(f"• Text Length:    {len(text)} characters ({len(text.split())} words, {len(text.splitlines())} lines)")
        if warnings:
            print(f"• Warnings:       {', '.join(warnings)}")
        print("-" * 80)

        if not text.strip():
            print("[EMPTY OR IMAGE-ONLY PAGE - NO EXTRACTED TEXT]")
        else:
            if not show_raw_tags:
                # Strip internal [[HEADING:...]] tags for a clean reading view
                import re
                clean_text = re.sub(r"^\[\[HEADING:.*?\]\]", "", text, flags=re.MULTILINE)
                print(clean_text)
            else:
                print(text)

        print("-" * 80)

    # Document Summary
    total_chars = sum(len(p.text) for p in result.pages)
    total_words = sum(len(p.text.split()) for p in result.pages)
    all_warnings = [w for p in result.pages for w in (p.metadata.warnings or [])]

    print("\n" + "=" * 80)
    print("📊 EXTRACTION SUMMARY")
    print(f"• Total Pages:    {total_pages}")
    print(f"• Total Chars:    {total_chars}")
    print(f"• Total Words:    {total_words}")
    print(f"• Total Warnings: {len(all_warnings)}")
    if all_warnings:
        for w in all_warnings:
            print(f"  ⚠️  {w}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Inspect raw extracted text and metadata from a PDF using PDFExtractor.")
    parser.add_argument("pdf_path", nargs="?", help="Path to PDF file (optional; defaults to first PDF in backend/uploads/)")
    parser.add_argument("--page", type=int, default=None, help="Inspect only a specific page number (1-based index)")
    parser.add_argument("--clean", action="store_true", help="Hide internal [[HEADING:...]] tags and display clean text")
    parser.add_argument("--list", action="store_true", help="List available PDFs in backend/uploads/ folder")

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

    inspect_pdf_extraction(
        pdf_path=target_pdf,
        target_page=args.page,
        show_raw_tags=not args.clean
    )


if __name__ == "__main__":
    main()
