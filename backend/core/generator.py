from groq import Groq
from typing import List
from core.tools.base_tool import BaseTool
from core.prompt_builder import PromptBuilder
from core.tool_registry import ToolRegistry
from config import settings
import json
import asyncio
import logging

log = logging.getLogger("generator")

# ── RAG system prompt (PDF context-stuffing path) ─────────────────────────────
RAG_SYSTEM_PROMPT = """You are a precise document assistant. You are given extracted passages \
from a PDF document, each preceded by its citation (source file, page number, and section \
heading when available).

Rules you MUST follow:
1. Answer ONLY from the provided context passages. Do not use prior knowledge.
2. Cite your sources inline using the format [source, p.N] or [source, p.N \u2014 Heading] \
   when a heading is available.
3. If multiple passages support the answer, cite all of them.
4. If the answer cannot be found in the provided context, respond exactly with: \
   "The information was not found in the document."
5. Be concise, accurate, and structured. Use bullet points or numbered lists when helpful.
"""


SYSTEM_PROMPT = """You are a data analyst assistant. You are given a CSV dataset schema and access to tools.

Follow these rules strictly:

1. ANSWER FROM SCHEMA if the question is about structure (column names, data types, row count, null counts, value ranges, unique values shown in the schema). Do NOT call any tool for these.

2. USE `pandas_sandbox` TOOL if the question requires computing something from the actual data rows (e.g. averages, sums, counts, filters, groupby, correlations, custom calculations that go beyond what the schema already shows).
   - Write Python/Pandas code that operates on a pre-loaded DataFrame called `df`
   - Assign your final computed answer to a variable named `result`
   - Do NOT include any import statements — pandas is already available as `pd`
   - Do NOT explain the code to the user; only show the final answer

3. Always give a clean, human-readable final answer to the user.
"""

# ── Per-tool parameter specs ──────────────────────────────────────────────────
TOOL_PARAMS: dict[str, dict] = {
    "pandas_sandbox": {
        "code": {
            "type": "string",
            "description": (
                "Python/Pandas code to execute on DataFrame `df`. "
                "Assign your final answer to variable `result`. "
                "No imports allowed. "
                "Example: result = df['salary'].mean()"
            )
        }
    },
    "get_csv_stats": {
        "query": {
            "type": "string",
            "description": (
                "Natural-language description of the statistical analysis to perform "
                "(e.g. 'correlation matrix', 'distribution of age column')."
            )
        }
    },
}


def _safe_serialize(obj):
    """Recursively convert an object to a JSON-serializable form."""
    if isinstance(obj, dict):
        return {k: _safe_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_safe_serialize(i) for i in obj]
    elif hasattr(obj, "model_dump"):        # pydantic v2
        return _safe_serialize(obj.model_dump())
    elif hasattr(obj, "__dict__"):
        return _safe_serialize(vars(obj))
    else:
        try:
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return str(obj)


class Generator:

    def __init__(self, tools: List[BaseTool] = None):
        self.api_key    = settings.groq_api_key
        self.model_name = settings.model_name
        self.client     = Groq(api_key=self.api_key)
        self.tools: List[BaseTool] = tools if tools is not None else []

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_tool_schema(self, tools: List[BaseTool]) -> list[dict]:
        """Convert a BaseTool list → Groq function-calling schema."""
        schema = []
        for tool in tools:
            params = TOOL_PARAMS.get(
                tool.name,
                {"input": {"type": "string", "description": "Input to pass to the tool."}}
            )
            schema.append({
                "type": "function",
                "function": {
                    "name":        tool.name,
                    "description": tool.description,
                    "parameters":  {
                        "type":       "object",
                        "properties": params,
                        "required":   list(params.keys())
                    }
                }
            })
        return schema

    def _get_tool_by_name(self, tool_name: str) -> BaseTool | None:
        """Look up a registered tool by name."""
        for tool in self.tools:
            if tool.name == tool_name:
                return tool
        return None

    def _build_messages(self, query: str, schema: str = None) -> list[dict]:
        """Return the [system, user] message list, injecting schema when provided."""
        user_content = (
            f"Dataset Schema:\n{schema}\n\nQuestion: {query}" if schema else query
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ]

    def _execute_tool_call(self, tool_call) -> tuple[str, str, str]:
        """
        Execute a tool-call object returned by the LLM.

        Returns (tool_name, tool_input, tool_result_str).
        Raises ValueError if the requested tool is not registered — callers
        are responsible for catching this and handling it gracefully.

        Returns tool_input as the second element so callers can trace it
        without re-parsing tool_call.function.arguments themselves.
        """
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)

        tool = self._get_tool_by_name(tool_name)
        if tool is None:
            raise ValueError(
                f"LLM requested unknown tool '{tool_name}' — not registered"
            )

        # Dispatch correct argument key per tool type
        tool_input = (
            tool_args.get("code")       # pandas_sandbox
            or tool_args.get("query")   # get_csv_stats
            or tool_args.get("input", "")
        )

        log.info(f"[TOOL] Executing '{tool_name}'…")
        tool_result = tool.run(tool_input)
        log.info(f"[TOOL] Result preview: {str(tool_result)[:300]}")

        return tool_name, tool_input, str(tool_result)

    # ── Core agentic loop ─────────────────────────────────────────────────────

    def generate_with_tools(self, query: str, schema: str = None, tracer=None) -> dict:
        """
        Agentic tool-calling pipeline (synchronous).

        Flow:
          Call 1 (schema + query + tool defs)
            ├─ Direct answer  →  return immediately
            └─ Tool call  →  _execute_tool_call  →  Call 2  →  final answer

        tracer: optional callable(step_type, data) for observability.
        """
        tool_schema = self._build_tool_schema(self.tools)
        messages    = self._build_messages(query, schema)

        def record(step_type, data):
            if tracer:
                try:
                    tracer(step_type, _safe_serialize(data))
                except Exception:
                    pass

        record("schema_context", {"schema": schema or "(none provided)"})
        record("llm_call_1", {"model": self.model_name, "messages": messages, "tools": tool_schema})

        # ── Call 1 ───────────────────────────────────────────────────────────
        log.info("[LLM] → Call 1 (schema + query + tool defs)")
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tool_schema if tool_schema else None,
            tool_choice="auto" if tool_schema else None,
        )
        message = response.choices[0].message

        record("llm_response_1", {
            "content":       message.content,
            "tool_calls":    [{"name": tc.function.name, "arguments": tc.function.arguments}
                              for tc in (message.tool_calls or [])],
            "finish_reason": response.choices[0].finish_reason,
        })

        # ── Direct answer ─────────────────────────────────────────────────────
        if not message.tool_calls:
            log.info("[LLM] ✓ Direct answer (no tool used)")
            record("final_answer", {"answer": message.content, "tool_used": None})
            return {"answer": message.content, "tool_used": None}

        # ── Tool execution ────────────────────────────────────────────────────
        tool_call = message.tool_calls[0]
        try:
            tool_name, tool_input, tool_result = self._execute_tool_call(tool_call)
        except ValueError as e:
            log.error(f"[LLM] Tool error: {e}")
            return {"answer": str(e), "tool_used": None}

        record("tool_input",  {"tool_name": tool_name, "input":  tool_input})
        record("tool_output", {"tool_name": tool_name, "result": tool_result})

        messages.append({"role": "assistant", "content": None, "tool_calls": [tool_call]})
        messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": tool_result})

        # ── Call 2: synthesise ────────────────────────────────────────────────
        log.info("[LLM] → Call 2 (synthesise tool result → final answer)")
        record("llm_call_2", {"model": self.model_name, "messages": messages})
        final_response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
        )
        answer = final_response.choices[0].message.content
        record("final_answer", {"answer": answer, "tool_used": tool_name})

        return {"answer": answer, "tool_used": tool_name}

    # ── Async wrapper ─────────────────────────────────────────────────────────

    async def agenerate_with_tools(self, query: str, schema: str = None, tracer=None) -> dict:
        """
        Async wrapper — runs generate_with_tools in a thread-pool executor so
        the FastAPI event loop stays free during blocking Groq HTTP calls.
        """
        return await asyncio.to_thread(self.generate_with_tools, query, schema, tracer)

    # ── Streaming ─────────────────────────────────────────────────────────────

    def generate_stream(self, query: str, schema: str = None, tracer=None):
        """
        Token-streaming generator for the SSE endpoint.

        Strategy:
          Call 1 (non-streaming) decides routing:
            ├─ Direct answer  →  yield content from Call 1, return.
            └─ Tool call  →  _execute_tool_call  →  Call 2 (streaming).
        """
        tool_schema = self._build_tool_schema(self.tools)
        messages    = self._build_messages(query, schema)

        def record(step_type, data):
            if tracer:
                try:
                    tracer(step_type, _safe_serialize(data))
                except Exception:
                    pass

        record("schema_context", {"schema": schema or "(none)"})
        record("llm_call_1", {"model": self.model_name, "messages": messages, "tools": tool_schema})

        # ── Call 1 — routing decision ─────────────────────────────────────────
        log.info("[STREAM] → Call 1 (routing decision)")
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tool_schema   if tool_schema else None,
            tool_choice="auto"  if tool_schema else None,
        )
        message = response.choices[0].message

        record("llm_response_1", {
            "content":       message.content,
            "tool_calls":    [{"name": tc.function.name, "arguments": tc.function.arguments}
                              for tc in (message.tool_calls or [])],
            "finish_reason": response.choices[0].finish_reason,
        })

        # ── Direct answer: yield Call 1 content, no second call ───────────────
        if not message.tool_calls:
            log.info("[STREAM] Direct answer — yielding Call 1 content")
            record("final_answer", {"answer": message.content, "tool_used": None})
            yield message.content or ""
            return

        # ── Tool execution ────────────────────────────────────────────────────
        tool_call = message.tool_calls[0]
        try:
            tool_name, tool_input, tool_result = self._execute_tool_call(tool_call)
        except ValueError as e:
            log.error(f"[STREAM] Tool error: {e}")
            yield str(e)
            return

        record("tool_input",  {"tool_name": tool_name, "input":  tool_input})
        record("tool_output", {"tool_name": tool_name, "result": tool_result})

        messages.append({"role": "assistant", "content": None, "tool_calls": [tool_call]})
        messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": tool_result})

        yield f"__tool__:{tool_name}"

        # ── Call 2: stream synthesis ──────────────────────────────────────────
        log.info("[STREAM] → Call 2 (stream tool-result synthesis)")
        record("llm_call_2_stream", {"model": self.model_name, "messages": messages})
        stream = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    # ── RAG streaming (PDF context-stuffing path) ─────────────────────────────

    def _build_rag_context(self, chunks: list[dict]) -> str:
        """
        Format retrieved chunks into a cited context block for the LLM prompt.

        Each chunk is preceded by a citation header:
          [Source: filename.pdf | Page: 3 | Section: Introduction]

        Chunks without a heading omit the Section part.
        Metadata is read from chunk["metadata"] (produced by TextChunker which
        carries through PageMetadata fields: source, page, heading).
        """
        parts: list[str] = []
        for i, chunk in enumerate(chunks, start=1):
            meta    = chunk.get("metadata", {})
            source  = meta.get("source", "unknown")
            page    = meta.get("page")
            heading = meta.get("heading")

            # Build citation header
            citation = f"[Source: {source}"
            if page is not None:
                citation += f" | Page: {page}"
            if heading:
                citation += f" | Section: {heading}"
            citation += "]"

            parts.append(f"{citation}\n{chunk['text'].strip()}")

        return "\n\n---\n\n".join(parts)

    def generate_rag_stream(self, query: str, chunks: list[dict], tracer=None):
        """
        Pure context-stuffing RAG stream for the PDF path.

        Strategy:
          - Build a cited context block from retrieved chunks.
          - Single Groq call with streaming=True — no tool-calling.
          - Yields tokens using the same protocol as generate_stream()
            so the SSE event_stream() wrapper in routes.py is reused unchanged.

        tracer: optional callable(step_type, data) for observability.
        """
        def record(step_type, data):
            if tracer:
                try:
                    tracer(step_type, _safe_serialize(data))
                except Exception:
                    pass

        context = self._build_rag_context(chunks)
        user_content = (
            f"Context passages:\n\n{context}\n\n"
            f"Question: {query}"
        )
        messages = [
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ]

        record("rag_context", {"chunk_count": len(chunks), "context_preview": context[:500]})
        record("rag_llm_call", {"model": self.model_name, "query": query})

        log.info(f"[RAG] → Streaming call | {len(chunks)} chunks | query: '{query[:80]}'")

        stream = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=True,
        )

        full_answer: list[str] = []
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                full_answer.append(delta.content)
                yield delta.content

        record("rag_final_answer", {"answer": "".join(full_answer)})
        log.info("[RAG] ✓ Stream complete")

    # ── Legacy RAG utility (non-streaming, kept for scripts/tests) ────────────

    def generate_rag(self, query: str, chunks: list[dict]) -> dict:
        """Non-streaming RAG — kept for test scripts. Uses same context builder."""
        context = self._build_rag_context(chunks)
        messages = [
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user",   "content": f"Context passages:\n\n{context}\n\nQuestion: {query}"},
        ]
        result = self.client.chat.completions.create(model=self.model_name, messages=messages)
        return {"answer": result.choices[0].message.content}

    def generate(self, query: str, schema: str = None) -> dict:
        """
        Synchronous entry point for tests / scripts.
        Delegates directly to generate_with_tools.
        Empty tools list → plain LLM call (no function-calling overhead).
        """
        return self.generate_with_tools(query, schema=schema)