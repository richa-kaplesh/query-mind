export interface CallKind {
  id: string;
  label: string;
  color: string;
}

export const CALL_KINDS: Record<string, CallKind> = {
  tool_selection: { id: "tool_selection", label: "Tool selection", color: "var(--call-tool)" },
  synthesis: { id: "synthesis", label: "Answer synthesis", color: "var(--call-synthesis)" },
  agent: { id: "agent", label: "Agent step", color: "var(--call-agent)" },
  review: { id: "review", label: "Review", color: "var(--call-review)" },
  other: { id: "other", label: "Other", color: "var(--call-other)" },
};

export const KIND_ORDER = ["tool_selection", "synthesis", "agent", "review", "other"];

export function kindOf(callType: string): CallKind {
  if (callType.includes("review")) return CALL_KINDS.review;
  if (callType === "csv_call1") return CALL_KINDS.tool_selection;
  if (callType === "csv_call2" || callType.endsWith("_synthesis")) return CALL_KINDS.synthesis;
  if (/_call\d+$/.test(callType)) return CALL_KINDS.agent;
  return CALL_KINDS.other;
}
