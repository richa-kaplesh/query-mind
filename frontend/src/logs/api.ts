import { useCallback, useEffect, useState } from "react";

export const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

export async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(`Request failed (${res.status})`);
  return (await res.json()) as T;
}

export interface FetchState<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
  reload: () => void;
}

export function useJson<T>(path: string | null): FetchState<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(path !== null);
  const [tick, setTick] = useState(0);

  useEffect(() => {
    if (path === null) {
      setData(null);
      setLoading(false);
      return;
    }
    let alive = true;
    setLoading(true);
    setError(null);
    getJson<T>(path)
      .then(d => { if (alive) setData(d); })
      .catch(e => { if (alive) setError(e instanceof Error ? e.message : "Could not load data"); })
      .finally(() => { if (alive) setLoading(false); });
    return () => { alive = false; };
  }, [path, tick]);

  const reload = useCallback(() => setTick(t => t + 1), []);
  return { data, error, loading, reload };
}
