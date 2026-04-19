import { useState, useRef, useCallback } from "react";

// ─── Types ────────────────────────────────────────────────────────────────────

export interface Source {
  source: string;
  page: number;
  text: string;
  rerank_score: number;
}

interface StreamEvent {
  type: "token" | "sources";
  content: string | Source[];
}

interface UseStreamReturn {
  streamedAnswer: string;
  sources: Source[];
  isStreaming: boolean;
  startStream: (question: string) => Promise<void>;
  reset: () => void;
}

// ─── Hook ─────────────────────────────────────────────────────────────────────

export function useStream(apiBase: string): UseStreamReturn {
  const [streamedAnswer, setStreamedAnswer] = useState("");
  const [sources, setSources] = useState<Source[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);

  // Abort controller so we can cancel mid-stream if needed
  const abortRef = useRef<AbortController | null>(null);

  const reset = useCallback(() => {
    setStreamedAnswer("");
    setSources([]);
    setIsStreaming(false);
  }, []);

  const startStream = useCallback(async (question: string) => {
    // Cancel any in-flight stream
    abortRef.current?.abort();
    abortRef.current = new AbortController();

    setStreamedAnswer("");
    setSources([]);
    setIsStreaming(true);

    try {
      const res = await fetch(`${apiBase}/query/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
        signal: abortRef.current.signal,
      });

      if (!res.ok) throw new Error(`Stream request failed: ${res.status}`);
      if (!res.body) throw new Error("No response body");

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      // SSE chunks may split across reads — buffer incomplete lines
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // SSE lines are separated by \n. Split and process complete lines only.
        const lines = buffer.split("\n");
        // Last element may be an incomplete line — keep it in buffer
        buffer = lines.pop() ?? "";

        for (const raw of lines) {
          const line = raw.trim();
          if (!line) continue;

          // SSE "data:" prefix
          const text = line.startsWith("data:") ? line.slice(5).trim() : line;

          // Terminal signal
          if (text === "[DONE]") {
            setIsStreaming(false);
            return;
          }

          // Parse JSON event
          let event: StreamEvent;
          try {
            event = JSON.parse(text);
          } catch {
            // Malformed line — skip
            continue;
          }

          if (event.type === "token" && typeof event.content === "string") {
            setStreamedAnswer((prev) => prev + event.content);
          } else if (event.type === "sources" && Array.isArray(event.content)) {
            setSources(event.content as Source[]);
          }
        }
      }
    } catch (err) {
      if ((err as Error).name === "AbortError") return; // intentional cancel
      throw err; // let caller handle real errors
    } finally {
      setIsStreaming(false);
    }
  }, [apiBase]);

  return { streamedAnswer, sources, isStreaming, startStream, reset };
}