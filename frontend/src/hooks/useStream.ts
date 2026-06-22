import { useState, useRef, useCallback } from "react";

// ─── Types ────────────────────────────────────────────────────────────────────

export interface ConversationTurn {
  role: "user" | "assistant";
  content: string;
}

export interface Source {
  source: string;
  page: string | number;
  text: string;
  rerank_score: number;
}

interface StreamEvent {
  type: "token" | "sources";
  content: string | Source[];
}

interface UseStreamReturn {
  streamedAnswer: string;
  streamedSources: Source[];
  isStreaming: boolean;
  startStream: (question: string, conversationHistory?: ConversationTurn[]) => Promise<void>;
  reset: () => void;
}

// ─── Hook ─────────────────────────────────────────────────────────────────────

export function useStream(apiBase: string): UseStreamReturn {
  const [streamedAnswer, setStreamedAnswer] = useState("");
  const [streamedSources, setStreamedSources] = useState<Source[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);

  const abortRef = useRef<AbortController | null>(null);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    setStreamedAnswer("");
    setStreamedSources([]);
    setIsStreaming(false);
  }, []);

  const startStream = useCallback(
    async (question: string, conversationHistory: ConversationTurn[] = []) => {
      abortRef.current?.abort();
      abortRef.current = new AbortController();

      setStreamedAnswer("");
      setStreamedSources([]);
      setIsStreaming(true);

      try {
        const res = await fetch(`${apiBase}/query/stream`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            question,
            conversation_history: conversationHistory,
          }),
          signal: abortRef.current.signal,
        });

        if (!res.ok) throw new Error(`Stream failed: ${res.status}`);
        if (!res.body) throw new Error("No response body");

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });

          // Split on double-newline SSE boundaries
          const lines = buffer.split("\n");
          buffer = lines.pop() ?? "";

          for (const raw of lines) {
            const line = raw.trim();
            if (!line) continue;

            const text = line.startsWith("data:") ? line.slice(5).trim() : line;

            if (text === "[DONE]") {
              setIsStreaming(false);
              return;
            }

            let event: StreamEvent;
            try {
              event = JSON.parse(text);
            } catch {
              continue;
            }

            if (event.type === "token" && typeof event.content === "string") {
              setStreamedAnswer((prev) => prev + event.content);
            } else if (event.type === "sources" && Array.isArray(event.content)) {
              setStreamedSources(event.content as Source[]);
            }
          }
        }
      } catch (err) {
        if ((err as Error).name === "AbortError") return;
        throw err;
      } finally {
        setIsStreaming(false);
      }
    },
    [apiBase]
  );

  return { streamedAnswer, streamedSources, isStreaming, startStream, reset };
}

// ─── Non-streaming query (POST /query) ────────────────────────────────────────

interface QueryResponse {
  answer: string;
  tool_used: string | null;
}

interface UseQueryReturn {
  answer: string;
  toolUsed: string | null;
  isLoading: boolean;
  sendQuery: (question: string, conversationHistory?: ConversationTurn[]) => Promise<void>;
  reset: () => void;
}

export function useQuery(apiBase: string): UseQueryReturn {
  const [answer, setAnswer] = useState("");
  const [toolUsed, setToolUsed] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const abortRef = useRef<AbortController | null>(null);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    setAnswer("");
    setToolUsed(null);
    setIsLoading(false);
  }, []);

  const sendQuery = useCallback(
    async (question: string, conversationHistory: ConversationTurn[] = []) => {
      abortRef.current?.abort();
      abortRef.current = new AbortController();

      setAnswer("");
      setToolUsed(null);
      setIsLoading(true);

      try {
        const res = await fetch(`${apiBase}/query`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            question,
            conversation_history: conversationHistory,
          }),
          signal: abortRef.current.signal,
        });

        if (!res.ok) {
          const err = await res.json().catch(() => ({ detail: `Query failed: ${res.status}` }));
          throw new Error(err.detail || `Query failed: ${res.status}`);
        }

        const data: QueryResponse = await res.json();
        setAnswer(data.answer ?? "");
        setToolUsed(data.tool_used ?? null);
      } catch (err) {
        if ((err as Error).name === "AbortError") return;
        throw err;
      } finally {
        setIsLoading(false);
      }
    },
    [apiBase]
  );

  return { answer, toolUsed, isLoading, sendQuery, reset };
}