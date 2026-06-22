import { useState, useRef, useCallback } from "react";

// ─── Types ────────────────────────────────────────────────────────────────────

export interface Source {
  source: string;
  page: number;
  text: string;
  rerank_score: number;
}

interface ConversationTurn {
  role: "user" | "assistant";
  content: string;
}

// Response shape from POST /query
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

// ─── Hook ─────────────────────────────────────────────────────────────────────

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