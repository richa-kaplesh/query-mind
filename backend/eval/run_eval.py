import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import time
from groq import Groq
from core.pdf_extractor import PDFExtractor
from core.chunker import TextChunker
from core.embedder import Embedder
from core.indexer import Indexer
from core.retriever import HybridRetriever
from core.reranker import Reranker
from core.generator import Generator

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_DIR, "workout_plan.pdf")
DATASET_PATH = os.path.join(BASE_DIR, "golden_dataset.json")  # Add this


embedder = Embedder()
indexer = Indexer()
retriever = HybridRetriever(indexer = indexer)
reranker = Reranker()
generator = Generator()
groq_client = Groq()

def setup_pipeline(pdf_path: str):
    extractor = PDFExtractor()
    chunker= TextChunker()

    pages = extractor.extract(pdf_path)

    all_chunks = []
    for page in pages:
        chunks= chunker.chunk_text(page["text"],metadata= page["metadata"])
        all_chunks.extend(chunks)

    all_chunks= embedder.embed_chunks(all_chunks)
    indexer.index(all_chunks)

    print(f"indexed {len(all_chunks)} chunks from {pdf_path}")

def run_all_evals():
    print(f"Total chunks indexed: {len(indexer.chunks)}")
    print(f"Sample chunk: {indexer.chunks[0]['text'][:200]}")
    
def run_query(question:str)-> dict:
    query_embedding = embedder.embed_query(question)

    chunks = retriever.retrieve(
        query = question, 
        query_embedding = query_embedding,
        top_k=20
        )
    
    chunks = reranker.rerank(
        query=question,
        chunks = chunks,
        top_k=5
    )
    result = generator.generate(
        query=question,
        chunks = chunks
    )

    return {
        "answer": result["answer"],
        "chunks":chunks
    }

def score_with_llm(question:str , ground_truth:str , actual_answer:str, chunks:list) -> dict:
    context = "\n\n".join([chunk["text"] for chunk in chunks])

    prompt = f"""You are an evaluation judge for a RAG system.
Question:{question}
Ground Truth Answer:{ground_truth}
Generated Answer:{actual_answer}
Retrieved Context:{context}
     
Score the following 4 metrics from 0.0 to 1.0:
1. Faithfulness - is every claim in the generated answer supported by the retrieved context?
2. Answer Relevancy - does the generated answer actually address the question asked?
3. Context Precision - were the retrieved chunks relevant to the question?
4. Context Recall - did the retrieved chunks contain enough information to answer the question completely?

Return ONLY a JSON object like this, nothing else:

{{
    "faithfulness":0.0,
    "answer_relevancy":0.0,
    "context_precision":0.0,
    "context_recall":0.0
}}"""
    
    response = groq_client.chat.completions.create(
        model = "llama-3.3-70b-versatile",
        messages=[
            {
                "role":"user",
                "content":prompt
            }
        ],
        temperature=0.0,
        response_format={"type": "json_object"}  # Force JSON output
    )

    raw = response.choices[0].message.content
    
    # Strip markdown code blocks if present
    raw = raw.strip()
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
        # Return default scores on error
        scores = {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0
        }

    return scores

def run_all_evals():
    with open(DATASET_PATH, "r") as f:
        golden_data = json.load(f)

    results = []

    print(f"\n{'='*60}")
    print(f"Running evals on {len(golden_data)} questions...")
    print(f"{'='*60}\n")

    for i , item in enumerate(golden_data, 1):
        question = item["question"]
        ground_truth = item["ground_truth"]

        print(f"{i}/{len(golden_data)}] {question[:60]}...")

        result = run_query(question)
        answer = result["answer"]
        chunks= result["chunks"]

        scores = score_with_llm(question, ground_truth , answer, chunks)
        results.append({
            "question": question,
            "scores":scores
        })
        
        print(f"  ✓ Faithfulness: {scores['faithfulness']:.2f} | "
              f"Relevancy: {scores['answer_relevancy']:.2f} | "
              f"Precision: {scores['context_precision']:.2f} | "
              f"Recall: {scores['context_recall']:.2f}")
        
        metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        averages = {
            metric: sum(r["scores"][metric] for r in results)/len(results)
            for metric in metrics
        }

        print(f"\n{'='*60}")
        print(f"Final Eval Report - {len(results)} questions")
        print(f"{'='*60}")
        for metric, score in averages.items():
            bar = "█" * int(score * 20)
            print(f"  {metric:<22} {score:.3f}  {bar}")
        print(f"{'='*60}\n")

    return averages

if __name__ == "__main__":
    setup_pipeline(PDF_PATH)
    averages = run_all_evals()