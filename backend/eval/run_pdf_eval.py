import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
from groq import Groq
from core.extractors.pdf_extractor import PDFExtractor
from core.chunker import TextChunker
from core.embedder import Embedder
from core.indexer import Indexer
from core.retriever import HybridRetriever
from core.reranker import Reranker
from core.generator import Generator
from config import settings

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = r"D:\query-mind\backend\eval\Leaflet - HDFC International Funds - GIFT Outbound Retail - Class B.pdf"
DATASET_PATH = os.path.join(BASE_DIR, "golden_dataset.json")

SAVE_TO_HISTORY = True  # flip to False while debugging so runs don't get permanently recorded

embedder  = Embedder()
indexer   = Indexer()
retriever = HybridRetriever(indexer=indexer)
reranker  = Reranker()
generator = Generator()
groq_client = Groq(api_key=settings.groq_api_key)


def setup_pipeline(pdf_path: str):
    extractor = PDFExtractor()
    chunker = TextChunker()

    pages = extractor.extract(pdf_path)
    all_chunks = chunker.chunk_pages(pages)
    all_chunks = embedder.embed_chunks(all_chunks)
    indexer.index(all_chunks)

    print(f"Indexed {len(all_chunks)} chunks from {pdf_path}")


def run_query(question: str) -> dict:
    query_embedding = embedder.embed_query(question)
    chunks = retriever.retrieve(query=question, query_embedding=query_embedding, top_k=20)
    chunks = reranker.rerank(query=question, chunks=chunks, top_k=5)

    print(f"    Chunk scores: {[round(c.get('rerank_score', 0), 3) for c in chunks]}")

    result = generator.generate_rag(query=question, chunks=chunks)
    return {"answer": result["answer"], "chunks": chunks}


def score_with_llm(question: str, ground_truth: str, actual_answer: str, chunks: list) -> dict:
    context = "\n\n".join([chunk["text"] for chunk in chunks])

    prompt = f"""You are an evaluation judge for a RAG system.
Question:{question}
Ground Truth Answer:{ground_truth}
Generated Answer:{actual_answer}
Retrieved Context:{context}

Score the following 4 metrics from 0.0 to 1.0:
1. Faithfulness - is every claim in the generated answer supported by the retrieved context?
   IMPORTANT: If the generated answer correctly says the information is not in the document,
   and the context indeed lacks that information, score Faithfulness as 1.0.
2. Answer Relevancy - does the generated answer actually address the question asked?
   IMPORTANT: If the question asks about something not in the document and the answer
   correctly states this, score Answer Relevancy as 1.0.
3. Context Precision - were the retrieved chunks relevant to the question?
4. Context Recall - did the retrieved chunks contain enough information to answer completely?
   IMPORTANT: If the document genuinely does not contain the answer, score Context Recall as 1.0.

Return ONLY a JSON object like this, nothing else:

{{
    "faithfulness":0.0,
    "answer_relevancy":0.0,
    "context_precision":0.0,
    "context_recall":0.0
}}"""

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        scores = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"\n❌ JSON Parse Error: {e}")
        print(f"Raw response: {raw[:200]}...")
        scores = {"faithfulness": 0.0, "answer_relevancy": 0.0, "context_precision": 0.0, "context_recall": 0.0}

    return scores


def run_all_evals():
    with open(DATASET_PATH, "r") as f:
        golden_data = json.load(f)

    results = []
    print(f"\n{'='*60}\nRunning evals on {len(golden_data)} questions...\n{'='*60}\n")

    for i, item in enumerate(golden_data, 1):
        question = item["question"]
        ground_truth = item["ground_truth"]

        print(f"{i}/{len(golden_data)}] {question[:60]}...")

        result = run_query(question)
        answer = result["answer"]
        chunks = result["chunks"]

        scores = score_with_llm(question, ground_truth, answer, chunks)
        results.append({"question": question, "answer": answer, "scores": scores})

        print(f"  ✓ Faithfulness: {scores['faithfulness']:.2f} | "
              f"Relevancy: {scores['answer_relevancy']:.2f} | "
              f"Precision: {scores['context_precision']:.2f} | "
              f"Recall: {scores['context_recall']:.2f}")

    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    averages = {
        metric: sum(r["scores"][metric] for r in results) / len(results)
        for metric in metrics
    }

    print(f"\n{'='*60}\nFinal Eval Report - {len(results)} questions\n{'='*60}")
    for metric, score in averages.items():
        bar = "█" * int(score * 20)
        print(f"  {metric:<22} {score:.3f}  {bar}")
    print(f"{'='*60}\n")

    return {"averages": averages, "per_question": results}


if __name__ == "__main__":
    setup_pipeline(PDF_PATH)
    output = run_all_evals()

    if SAVE_TO_HISTORY:
        from datetime import datetime
        run_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "label": "reranker-threshold-fix",
            "description": (
                "Investigated Q13 (IFSCA registration number) recall failure. "
                "Root cause chain: (1) fixed-size char chunker was merging the "
                "registration number line into an unrelated marketing-bullets chunk, "
                "diluting its retrievability -- fixed via heading-aware line-based "
                "chunking in TextChunker/PDFExtractor. (2) Even after chunking fix, "
                "Reranker.rerank() had a hardcoded absolute min_score=0.3 threshold "
                "that silently dropped the registration-number chunk at score 0.2991 "
                "-- just 0.0009 below cutoff, despite ranking #4 of 20 candidates. "
                "Removed min_score filtering entirely; rely on top_k rank from Jina's "
                "already-sorted results instead. Also fixed a duplicate httpx.post() "
                "call in Reranker.rerank() that was double-billing every rerank request. "
                "Confirmed fix: Q13's chunk now lands in final top-5 (rank 4, score 0.251)."
            ),
            "config": {
                "pdf_extractor": "PyMuPDF (layout-aware, font-size heading detection, heading-tagged lines)",
                "csv_extractor": "pandas schema-only (CSVSchema/ColumnSchema)",
                "chunker": {
                    "method": "heading-aware, line-boundary-respecting split (no mid-line cuts)",
                    "chunk_size": settings.chunk_size,
                    "chunk_overlap": settings.chunk_overlap,
                },
                "embedding_model": settings.jina_embed_model,
                "retriever": {
                    "method": "hybrid BM25 + FAISS",
                    "alpha": settings.retriever_alpha,
                    "top_k": settings.retriever_top_k,
                },
                "reranker": {
                    "model": settings.jina_rerank_model,
                    "min_score_filter": "removed (was 0.3, caused false negatives)",
                },
                "generator_model": settings.model_name,
            },
            "averages": output["averages"],
            "per_question": output["per_question"],
        }
        history_path = os.path.join(BASE_DIR, "eval_history.json")
        history = []
        if os.path.exists(history_path):
            with open(history_path, "r") as f:
                history = json.load(f)
        history.append(run_record)
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        print(f"Saved run to {history_path}")
    else:
        print("SAVE_TO_HISTORY is False — run not saved.")