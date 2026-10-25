/** Domain types used across the app. The backend payload is mapped into these by services/normalize.ts. */

export type MetricKey =
  | "factuality"
  | "hallucination_risk"
  | "completeness"
  | "logical_consistency"
  | "relevance"
  | "evidence_coverage";

/** "ratio" = normalized 0–1 score. "percent" = 0–100 percentage. Never mixed in a comparison. */
export type MetricUnit = "ratio" | "percent";

export interface MetricValue {
  value: number;
  unit: MetricUnit;
}

/** Canonical metrics keyed by MetricKey, plus any extra metrics the backend returned under their own names. */
export interface MetricSet {
  known: Partial<Record<MetricKey, MetricValue>>;
  extra: Array<{ name: string; value: MetricValue }>;
}

export type ClaimStatus =
  | "verified"
  | "unsupported"
  | "contradicted"
  | "partially_supported"
  | "unable_to_verify"
  | "unknown"; // the backend gave no usable status

export interface Claim {
  id: string;
  text: string;
  status: ClaimStatus;
  corrected?: boolean;
  correctedText?: string;
  explanation?: string;
  evidence: string[];
  citationRefs: string[]; // citation ids / urls / indexes referenced by the backend
}

export interface Citation {
  id: string;
  title?: string;
  url?: string;
  domain?: string;
  excerpt?: string;
  claimRefs: string[]; // claim ids / texts referenced by the backend
}

export interface CorrectionSummary {
  text?: string;
  changes: string[];
}

export type VerificationStatus =
  | "processing"
  | "verified" // completed, nothing needed correcting
  | "corrected" // completed and the final answer differs from the initial one
  | "failed" // request or verification failed
  | "review"; // backend quality gate reported not-passed

export interface VerificationResult {
  responseId: string;
  conversationId?: string;
  query: string;
  initialResponse?: string;
  finalResponse?: string;
  /** Undefined when the backend result does not support any status. */
  status?: VerificationStatus;
  qualityPass?: boolean;
  metrics: { initial: MetricSet; final: MetricSet };
  claims: Claim[];
  citations: Citation[];
  correctionSummary?: CorrectionSummary;
  notes?: string;
  source: "backend" | "demo";
}

export interface UserMessage {
  id: string;
  role: "user";
  content: string;
  createdAt: number;
}

export interface AssistantMessage {
  /** Doubles as the response ID that all analysis is stored under. */
  id: string;
  role: "assistant";
  replyTo: string;
  /** The final corrected answer. Never the unverified answer. */
  content: string;
  state: "processing" | "done" | "failed";
  error?: string;
  analysis?: VerificationResult;
  createdAt: number;
}

export type Message = UserMessage | AssistantMessage;

export interface Conversation {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  messages: Message[];
}

export type DataMode = "live" | "demo";
