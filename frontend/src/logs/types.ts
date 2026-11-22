export type DocType = "pdf" | "csv";

export interface EvalSetRef {
  id: string;
  name: string;
  question_count: number;
}

export interface Metric {
  key: string;
  label: string;
  value: number;
  higher_is_better: boolean;
}

export interface RunSummary {
  id: string;
  name: string;
  created_at: string;
  doc_type: DocType;
  eval_set: EvalSetRef;
  baseline_id: string | null;
  metrics: Metric[];
}

export interface RunQuestion {
  id: string;
  text: string;
  scores: Record<string, number>;
}

export interface RunNotes {
  problem: string;
  root_cause: string;
  fix: string;
  result: string;
}

export interface RunDetail extends RunSummary {
  questions: RunQuestion[];
  config_changes: string[];
  notes: RunNotes | null;
}

export interface CostCall {
  call_type: string;
  model: string;
  prompt_tokens: number;
  completion_tokens: number;
  timestamp: string;
}
