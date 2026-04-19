from groq import Groq
from typing import List

class Generator:

    def __init__(self, model:str="llama-3.3-70b-versatile"):
        self.client = Groq()
        self.model = model

    def _build_context(self, chunks: List[dict]) ->str:

        context_parts = []
        for i , chunk in enumerate(chunks):
            source = chunk["metadata"]["source"]
            page = chunk["metadata"]["page"]
            text = chunk["text"]

            part = f"[SOURCE {i+1} - {source},page {page}]\n{text}"
            context_parts.append(part)

        return "\n".join(context_parts)
    
    def build_prompt(self, query: str, context: str) -> str:
          return f"""You are a research assistant. Answer using ONLY the context provided below.

Rules:
- Only use information explicitly stated in the provided sources
- Always cite which SOURCE number your answer came from
- If the context mentions a related topic but does NOT directly answer the question, 
  say "The document mentions [topic] but does not directly address [question]"
- If the answer is completely absent, say "I cannot find this in the provided documents"
- Never infer, extrapolate, or assume beyond what is written

CONTEXT:
{context}

QUESTION:
{query}

ANSWER (with citations):"""
    def generate(self , query: str, chunks: List[dict]) -> dict:
        context = self._build_context(chunks)
        prompt =  self.build_prompt(query, context)

        response = self.client.chat.completions.create(
            model = self.model,
            messages =[
                {
                    "role":"system",
                    "content": "You are a precise reasearch assistant that always cites sources."
                
                },
                {
                    "role":"user",
                    "content": prompt
                }
            ],
            temperature=0.1
        )

        answer = response.choices[0].message.content

        return {
            "answer": answer,
            "sources": [
                    {
                        "source": chunk["metadata"].get("source", "unknown"),
                        "page": chunk["metadata"].get("page", "N/A"),
                        "text": chunk["text"],
                        "rerank_score": chunk["rerank_score"]
                    }
                    for chunk in chunks
                ]                 
        }

        

        