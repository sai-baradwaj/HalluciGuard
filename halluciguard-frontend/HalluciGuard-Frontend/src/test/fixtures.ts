import type { AssistantMessage, Conversation, UserMessage, VerificationResult } from "@/types";
import { normalizeBackendResult } from "@/services/normalize";
import type { HalluciGuardApi } from "@/services/api";

export function payload(n: number) {
  return {
    initial_response: `INITIAL-${n}: the first draft answer number ${n}.`,
    final_response: `FINAL-${n}: the corrected answer number ${n}.`,
    verification_status: "completed",
    metrics: { initial: { factuality: 0.5 + n / 100, hallucination_risk: 0.7 }, final: { factuality: 0.9, hallucination_risk: 0.1 } },
    claims: [{ id: `k${n}`, claim: `CLAIM-${n} is stated in the first draft answer`, status: "unsupported", corrected: true, explanation: `EXPLAIN-${n}`, evidence: [`EVIDENCE-${n}`], citation_ids: [`s${n}`] }],
    citations: [{ id: `s${n}`, title: `SOURCE-${n}`, url: `https://example.com/source-${n}`, excerpt: `EXCERPT-${n}`, claims: [`k${n}`] }],
  };
}

export function analysisFor(n: number, convId: string): VerificationResult {
  return normalizeBackendResult(payload(n), { responseId: `r${n}`, conversationId: convId, query: `Question ${n}?`, source: "backend" });
}

export function turn(n: number, convId: string, analysis: VerificationResult | null = analysisFor(n, convId)): [UserMessage, AssistantMessage] {
  const user: UserMessage = { id: `u${n}`, role: "user", content: `Question ${n}?`, createdAt: 1000 * n };
  const assistant: AssistantMessage = {
    id: `r${n}`,
    role: "assistant",
    replyTo: user.id,
    content: analysis ? (analysis.finalResponse ?? "") : "",
    state: analysis ? "done" : "failed",
    error: analysis ? undefined : "boom: backend unavailable",
    analysis: analysis ?? undefined,
    createdAt: 1000 * n + 500,
  };
  return [user, assistant];
}

export function conversation(id: string, title: string, turns: number[], extra?: (n: number) => [UserMessage, AssistantMessage]): Conversation {
  return { id, title, createdAt: 1, updatedAt: 2, messages: turns.flatMap((n) => (extra ? extra(n) : turn(n, id))) };
}

export function fakeApi(impl?: HalluciGuardApi["verify"]): HalluciGuardApi & { calls: string[] } {
  const calls: string[] = [];
  return {
    mode: "demo",
    calls,
    verify:
      impl ??
      (async ({ query, responseId, conversationId }) => {
        calls.push(query);
        return normalizeBackendResult(payload(9), { responseId, conversationId, query, source: "backend" });
      }),
  };
}
