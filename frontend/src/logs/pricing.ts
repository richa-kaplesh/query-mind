export interface ModelPrice {
  prompt_per_1k: number;
  completion_per_1k: number;
  confirmed: boolean;
}

export const MODEL_PRICES: Record<string, ModelPrice> = {
  "openai/gpt-oss-20b": { prompt_per_1k: 0, completion_per_1k: 0, confirmed: false },
};

export const DEFAULT_PRICE: ModelPrice = { prompt_per_1k: 0, completion_per_1k: 0, confirmed: false };

export function priceFor(model: string): ModelPrice {
  return MODEL_PRICES[model] ?? DEFAULT_PRICE;
}

export function costUsd(model: string, promptTokens: number, completionTokens: number): number {
  const p = priceFor(model);
  return (promptTokens / 1000) * p.prompt_per_1k + (completionTokens / 1000) * p.completion_per_1k;
}
