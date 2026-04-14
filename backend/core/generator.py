from groq import Groq
from typing import List

class Generator:

    def __init__(self, model:str="llama3-70b-8192"):
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
    
    def build_prompt(self, query: str , context: str) -> str:
        return f"""You are a reasearch assistant. Answer the user's ONLY the context provided below.
    
        Rules:
- Only use information from the provided sources
- Always cite which SOURCE number your answer came from
- If the answer is not in the context, say "I cannot find this in the provided documents"
- Never make up information

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
            "sources":[
                {
                    "source": chunk["metadata"]["source"],
                    "pages": chunk["metadata"]["page"],
                    "text":chunk["text"],
                    "rerank_score": chunk["rerank_score"]  
                }
                for chunk in chunks
            ] 
        }

        

        