from groq import Groq
from typing import List
from core.tools.base_tool import BaseTool
from core import gateway_client
from core.token_utils import estimate_tokens
from core.models import CSVSchema
from config import settings
import json
import asyncio
import logging
import time

log = logging.getLogger("generator")

RAG_SYSTEM_PROMPT = """You are a precise document assistant. You are given extracted passages \
from a PDF document, each preceded by its citation (source file, page number, and section \
heading when available).

Rules you MUST follow:
1. Answer ONLY from the provided context passages. Do not use prior knowledge.
2. Cite your sources inline using the format [source, p.N] or [source, p.N — Heading] \
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
    if isinstance(obj, dict):
        return {k: _safe_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_safe_serialize(i) for i in obj]
    elif hasattr(obj, "model_dump"):
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

    def _build_tool_schema(self, tools: List[BaseTool]) -> list[dict]:
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
        for tool in self.tools:
            if tool.name == tool_name:
                return tool
        return None

    def _build_messages(self, query: str, schema: str | CSVSchema = None) -> list[dict]:
        if isinstance(schema, CSVSchema):
            schema_str = schema.to_prompt_string()
        else:
            schema_str = schema
        user_content = (
            f"Dataset Schema:\n{schema_str}\n\nQuestion: {query}" if schema_str else query
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ]

    def _execute_tool_call(self, tool_call) -> tuple[str, str, str]:
        if isinstance(tool_call, dict):
            tool_name = tool_call["function"]["name"]
            tool_args = json.loads(tool_call["function"]["arguments"])
        else:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

        tool = self._get_tool_by_name(tool_name)
        if tool is None:
            raise ValueError(f"LLM requested unknown tool '{tool_name}' — not registered")

        tool_input = tool_args.get("code") or tool_args.get("query") or tool_args.get("input", "")

        log.info(f"[TOOL] Executing '{tool_name}'…")
        tool_result = tool.run(tool_input)
        log.info(f"[TOOL] Result preview: {str(tool_result)[:300]}")

        return tool_name, tool_input, str(tool_result)

    # ── Core agentic loop ─────────────────────────────────────────────────────

    def generate_with_tools(self, query: str, schema: str | CSVSchema = None, tracer=None, token_tracker=None) -> dict:
        tool_schema = self._build_tool_schema(self.tools)
        messages    = self._build_messages(query, schema)

        def record(step_type, data):
            if tracer:
                try:
                    tracer(step_type, _safe_serialize(data))
                
                except Exception as e: print(f"[TRACER ERROR] {e}")

        schema_str = schema.to_prompt_string() if isinstance(schema, CSVSchema) else schema
        record("schema_context", {"schema": schema_str or "(none provided)"})
        record("llm_call_1", {"model": self.model_name, "messages": messages, "tools": tool_schema})


        MAX_RETRIES = 3
        RETRY_DELAY_SECONDS = 1

        log.info("[LLM] → Call 1 (schema + query + tool defs)")
        response = None
        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    tools=tool_schema if tool_schema else None,
                    tool_choice="auto" if tool_schema else None,
                )
                break
            except Exception as e:
                last_error = e
                log.warning(f"[LLM] Call 1 attempt {attempt}/{MAX_RETRIES} failed: {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY_SECONDS)

        if response is None:
            log.error(f"[LLM] Call 1 failed after {MAX_RETRIES} attempts: {last_error}")
            record("final_answer", {"answer": f"LLM call failed after {MAX_RETRIES} attempts: {last_error}", "tool_used": None})
            return {"answer": f"(Call 1 failed after {MAX_RETRIES} attempts: {last_error})", "tool_used": None}

        message = response.choices[0].message
        if token_tracker and response.usage:
            token_tracker.log_call(
                model=self.model_name,
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                purpose="csv_call1",
            )

        record("llm_response_1", {
            "content":       message.content,
            "tool_calls":    [{"name": tc.function.name, "arguments": tc.function.arguments}
                              for tc in (message.tool_calls or [])],
            "finish_reason": response.choices[0].finish_reason,
        })

        if not message.tool_calls:
            log.info("[LLM] ✓ Direct answer (no tool used)")
            record("final_answer", {"answer": message.content, "tool_used": None})
            return {"answer": message.content, "tool_used": None}

        tool_call = message.tool_calls[0]
        try:
            tool_name, tool_input, tool_result = self._execute_tool_call(tool_call)
        except ValueError as e:
            log.error(f"[LLM] Tool error: {e}")
            return {"answer": str(e), "tool_used": None}

        record("tool_input",  {"tool_name": tool_name, "input":  tool_input})
        record("tool_output", {"tool_name": tool_name, "result": tool_result})

        # Fresh, tool-free message list for synthesis — avoids replaying tool_calls
        # and avoids passing `tools`/`tool_choice` at all, which is what previously
        # let the model attempt a second tool call and get rejected by Groq
        # ("tool choice is none, but model called a tool").
        original_user_content = messages[1]["content"]
        synthesis_messages = [
            {"role": "system", "content": messages[0]["content"]},
            {
                "role": "user",
                "content": (
                    f"{original_user_content}\n\n"
                    f"Tool used: {tool_name}\n"
                    f"Tool result: {tool_result}\n\n"
                    "The computation has already been done — the result above is final and correct. "
                    "Do not call any tool or function, do not write or execute any code, and do not "
                    "attempt further computation of any kind. "
                    "Simply state the answer to the original question in plain language, "
                    "using only the tool result provided above."
                ),
            },
        ]

        log.info("[LLM] → Call 2 (synthesise tool result → final answer)")
        record("llm_call_2", {"model": self.model_name, "messages": synthesis_messages})
        try:
            final_response = self.client.chat.completions.create(
                model=self.model_name,
                messages=synthesis_messages,
            )
            answer = final_response.choices[0].message.content

            if token_tracker and final_response.usage:
                token_tracker.log_call(
                    model=self.model_name,
                    prompt_tokens=final_response.usage.prompt_tokens,
                    completion_tokens=final_response.usage.completion_tokens,
                    purpose="csv_call2",
                )
        except Exception as e:
            log.error(f"[LLM] Call 2 synthesis failed: {e}")
            # Fall back to the raw tool result so we don't lose the computation that already succeeded
            answer = f"(Synthesis failed, raw tool result: {tool_result})"
        record("final_answer", {"answer": answer, "tool_used": tool_name})
        return {"answer": answer, "tool_used": tool_name}
    
    async def agenerate_with_tools(self, query: str, schema: str | CSVSchema = None, tracer=None, token_tracker=None) -> dict:
        return await asyncio.to_thread(self.generate_with_tools, query, schema, tracer, token_tracker)

    # ── Streaming ─────────────────────────────────────────────────────────────

    async def generate_stream(self, query: str, schema: str | CSVSchema = None, conversation_id: str = None,
                           user_id: str = "query_mind_user", tracer=None, token_tracker=None):
        tool_schema = self._build_tool_schema(self.tools)
        messages = self._build_messages(query, schema)

        def record(step_type, data):
            if tracer:
                try:
                    tracer(step_type, _safe_serialize(data))
                except Exception:
                    pass

        schema_str = schema.to_prompt_string() if isinstance(schema, CSVSchema) else schema
        record("schema_context", {"schema": schema_str or "(none)"})
        record("llm_call_1", {"messages": messages, "tools": tool_schema})
        log.info("[STREAM] → Gateway Call 1 (routing decision)")

        result = await gateway_client.complete(
            conversation_id=conversation_id, user_id=user_id, messages=messages,
            tools=tool_schema if tool_schema else None,
            tool_choice="auto" if tool_schema else None,
        )

        record("llm_response_1", {
            "content": result.get("content"),
            "tool_calls": result.get("tool_calls"),
            "finish_reason": result.get("finish_reason"),
        })
        if token_tracker:
            token_tracker.log_call(
                model=result.get("model_used", "gateway"),
                prompt_tokens=estimate_tokens(str(messages)),
                completion_tokens=estimate_tokens(result.get("content") or ""),
                purpose="csv_call1", estimated=True,
            )

        tool_calls = result.get("tool_calls")
        if not tool_calls:
            answer = result.get("content") or ""
            log.info("[STREAM] Direct answer — faking stream from gateway content")
            record("final_answer", {"answer": answer, "tool_used": None})
            for word in answer.split(" "):
                yield word + " "
                await asyncio.sleep(0.02)
            return

        tool_call = tool_calls[0]
        try:
            tool_name, tool_input, tool_result = self._execute_tool_call(tool_call)
        except ValueError as e:
            log.error(f"[STREAM] Tool error: {e}")
            yield str(e)
            return

        record("tool_input", {"tool_name": tool_name, "input": tool_input})
        record("tool_output", {"tool_name": tool_name, "result": tool_result})
        yield f"__tool__:{tool_name}"

        # Fresh, tool-free synthesis messages — same fix generate_with_tools already
        # applies, backported here since the streaming path had the same latent bug.
        original_user_content = messages[1]["content"]
        synthesis_messages = [
            {"role": "system", "content": messages[0]["content"]},
            {"role": "user", "content": (
                f"{original_user_content}\n\nTool used: {tool_name}\nTool result: {tool_result}\n\n"
                "The computation has already been done — the result above is final and correct. "
                "Do not call any tool or function, do not write or execute any code, and do not "
                "attempt further computation of any kind. Simply state the answer to the original "
                "question in plain language, using only the tool result provided above."
            )},
        ]

        log.info("[STREAM] → Gateway Call 2 (tool-result synthesis)")
        record("llm_call_2_stream", {"messages": synthesis_messages})
        result2 = await gateway_client.complete(conversation_id=conversation_id, user_id=user_id, messages=synthesis_messages)
        answer = result2.get("content") or ""

        for word in answer.split(" "):
            yield word + " "
            await asyncio.sleep(0.02)

        if token_tracker:
            token_tracker.log_call(
                model=result2.get("model_used", "gateway"),
                prompt_tokens=estimate_tokens(str(synthesis_messages)),
                completion_tokens=estimate_tokens(answer),
                purpose="csv_call2", estimated=True,
            )
    # ── RAG streaming (PDF context-stuffing path) ─────────────────────────────

    def _build_rag_context(self, chunks: list[dict]) -> str:
        parts: list[str] = []
        for i, chunk in enumerate(chunks, start=1):
            meta    = chunk.get("metadata", {})
            source  = meta.get("source", "unknown")
            page    = meta.get("page")
            heading = meta.get("heading")

            citation = f"[Source: {source}"
            if page is not None:
                citation += f" | Page: {page}"
            if heading:
                citation += f" | Section: {heading}"
            citation += "]"

            parts.append(f"{citation}\n{chunk['text'].strip()}")

        return "\n\n---\n\n".join(parts)

    async def generate_rag_stream(self, query: str, chunks: list[dict], conversation_id: str,
                                user_id: str = "query_mind_user", tracer=None, token_tracker=None):
        def record(step_type, data):
            if tracer:
                try:
                    tracer(step_type, _safe_serialize(data))
                except Exception:
                    pass

        context = self._build_rag_context(chunks)
        user_content = f"Context passages:\n\n{context}\n\nQuestion: {query}"
        messages = [
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        record("rag_context", {"chunk_count": len(chunks), "context_preview": context[:500]})
        record("rag_llm_call", {"query": query})
        log.info(f"[RAG] → Gateway call | {len(chunks)} chunks | query: '{query[:80]}'")

        result = await gateway_client.complete(conversation_id=conversation_id, user_id=user_id, messages=messages)
        answer = result.get("content") or ""

        for i, word in enumerate(answer.split(" ")):
            yield word if i == 0 else " " + word
            await asyncio.sleep(0.02)

        if token_tracker:
            token_tracker.log_call(
                model=result.get("model_used", "gateway"),
                prompt_tokens=estimate_tokens(user_content),
                completion_tokens=estimate_tokens(answer),
                purpose="rag",
                estimated=True,
            )

        record("rag_final_answer", {"answer": answer})
        log.info("[RAG] ✓ Gateway call complete")

    # ── Legacy RAG utility (non-streaming, kept for scripts/tests) ────────────

    def generate_rag(self, query: str, chunks: list[dict]) -> dict:
        context = self._build_rag_context(chunks)
        messages = [
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user",   "content": f"Context passages:\n\n{context}\n\nQuestion: {query}"},
        ]
        result = self.client.chat.completions.create(model=self.model_name, messages=messages)
        return {"answer": result.choices[0].message.content}

    def generate(self, query: str, schema: str | CSVSchema = None) -> dict:
        return self.generate_with_tools(query, schema=schema)