/**
 * Demo-mode adapter. Everything produced here is DEMO DATA and is flagged `source: "demo"`
 * so the UI labels it. It goes through the same normalizer as live responses, so the rest of the app
 * cannot tell the difference. Replace by using the live API (mode "live").
 *
 * Tip: a prompt containing the word "fail" makes the demo adapter simulate a backend error.
 */
import type { Conversation, Message } from "@/types";
import type { HalluciGuardApi, VerifyRequest } from "./api";
import { ApiError } from "./errors";
import { normalizeBackendResult } from "./normalize";

const SRC = "https://example.org/demo";
const demoSources = [
  { id: "s1", title: "Demo source: Urban emissions overview", url: `${SRC}/urban-emissions`, excerpt: "[Demo excerpt] Road transport is a major but not universal leading source of urban PM2.5; shares vary by city." },
  { id: "s2", title: "Demo source: Industrial emissions inventory", url: `${SRC}/industrial-emissions`, excerpt: "[Demo excerpt] Power generation and industry emit sulfur dioxide and nitrogen oxides." },
  { id: "s3", title: "Demo source: Household solid-fuel use", url: `${SRC}/household-fuel`, excerpt: "[Demo excerpt] Burning solid fuels for heating and cooking is a significant source of fine particles." },
  { id: "s4", title: "Demo source: Clean Air Act history", url: `${SRC}/clean-air-act`, excerpt: "[Demo excerpt] Smoke pollution fell sharply after 1956 legislation but did not disappear." },
  { id: "s5", title: "Demo source: Air pollution and health", url: `${SRC}/health`, excerpt: "[Demo excerpt] Long-term exposure to fine particles is associated with respiratory and cardiovascular disease." },
];

type Payload = Record<string, unknown>;

function causesPayload(): Payload {
  const initial = `## Main causes of urban air pollution

Urban air pollution comes from several overlapping sources:

- Vehicle emissions are the single largest source of PM2.5 in every city worldwide.
- Industrial plants and power generation release sulfur dioxide and nitrogen oxides.
- Burning solid fuels for heating and cooking adds fine particles to city air.
- Air pollution in London was eliminated after the Clean Air Act of 1956.

In short, tackling traffic alone would solve urban air quality.`;
  const final = `## Main causes of urban air pollution

Urban air pollution comes from several overlapping sources, and their shares differ from city to city:

- Road transport is a major source of nitrogen dioxide and a significant contributor to PM2.5 in many cities, but it is not the largest source everywhere.
- Industrial plants and power generation release sulfur dioxide and nitrogen oxides.
- Burning solid fuels for heating and cooking adds fine particles to city air.
- London's air quality improved dramatically after the Clean Air Act of 1956, although pollution was reduced rather than eliminated.

Traffic controls help, but effective plans also need to address industry, heating and regional sources.`;
  return {
    initial_response: initial,
    final_response: final,
    verification_status: "completed",
    metrics: {
      initial: { factuality: 0.58, hallucination_risk: 0.64, completeness: 0.71, logical_consistency: 0.77, relevance: 0.9, evidence_coverage: 0.32 },
      final: { factuality: 0.93, hallucination_risk: 0.09, completeness: 0.86, logical_consistency: 0.92, relevance: 0.94, evidence_coverage: 0.88 },
    },
    claims: [
      { id: "c1", claim: "Vehicle emissions are the single largest source of PM2.5 in every city worldwide.", status: "contradicted", corrected: true, corrected_claim: "Road transport is a major source of nitrogen dioxide and a significant contributor to PM2.5 in many cities, but it is not the largest source everywhere.", explanation: "Evidence shows the leading PM2.5 source varies by city; the universal claim is too strong.", evidence: ["[Demo excerpt] Road transport is a major but not universal leading source of urban PM2.5; shares vary by city."], citation_ids: ["s1"] },
      { id: "c2", claim: "Industrial plants and power generation release sulfur dioxide and nitrogen oxides.", status: "verified", corrected: false, explanation: "Directly supported by an emissions inventory source.", evidence: ["[Demo excerpt] Power generation and industry emit sulfur dioxide and nitrogen oxides."], citation_ids: ["s2"] },
      { id: "c3", claim: "Burning solid fuels for heating and cooking adds fine particles to city air.", status: "verified", corrected: false, explanation: "Supported by household-fuel evidence.", evidence: ["[Demo excerpt] Burning solid fuels for heating and cooking is a significant source of fine particles."], citation_ids: ["s3"] },
      { id: "c4", claim: "Air pollution in London was eliminated after the Clean Air Act of 1956.", status: "contradicted", corrected: true, corrected_claim: "London's air quality improved dramatically after the Clean Air Act of 1956, although pollution was reduced rather than eliminated.", explanation: "Pollution fell sharply but did not disappear.", evidence: ["[Demo excerpt] Smoke pollution fell sharply after 1956 legislation but did not disappear."], citation_ids: ["s4"] },
      { id: "c5", claim: "tackling traffic alone would solve urban air quality.", status: "unsupported", corrected: true, corrected_claim: "Traffic controls help, but effective plans also need to address industry, heating and regional sources.", explanation: "No retrieved evidence supports a single-source solution; sources above point to several contributors.", evidence: [], citation_ids: ["s1", "s2", "s3"] },
    ],
    citations: demoSources.slice(0, 4).map((s) => ({ ...s, claims: s.id === "s1" ? ["c1", "c5"] : s.id === "s2" ? ["c2", "c5"] : s.id === "s3" ? ["c3", "c5"] : ["c4"] })),
    correction_summary: { summary: "3 of 5 claims were rewritten to match the retrieved evidence.", changes: ["Removed the claim that vehicles are the largest PM2.5 source in every city.", "Changed 'eliminated' to 'reduced' for London after 1956.", "Replaced the single-source conclusion with a multi-source one."] },
  };
}

function healthPayload(): Payload {
  const text = `## Air pollution and health

Long-term exposure to fine particles (PM2.5) is associated with respiratory and cardiovascular disease.

| Group | Common concern |
|---|---|
| Children | Reduced lung development |
| Older adults | Heart and lung disease |

Short spikes in pollution can worsen asthma symptoms.`;
  return {
    initial_response: text,
    final_response: text,
    verification_status: "completed",
    metrics: {
      initial: { factuality: 0.91, hallucination_risk: 0.11, completeness: 0.74, logical_consistency: 0.95, relevance: 0.96, evidence_coverage: 0.8 },
      final: { factuality: 0.91, hallucination_risk: 0.11, completeness: 0.74, logical_consistency: 0.95, relevance: 0.96, evidence_coverage: 0.8 },
    },
    claims: [
      { id: "h1", claim: "Long-term exposure to fine particles (PM2.5) is associated with respiratory and cardiovascular disease.", status: "verified", corrected: false, explanation: "Supported by the retrieved health source.", evidence: ["[Demo excerpt] Long-term exposure to fine particles is associated with respiratory and cardiovascular disease."], citation_ids: ["s5"] },
      { id: "h2", claim: "Short spikes in pollution can worsen asthma symptoms.", status: "partially_supported", corrected: false, explanation: "The retrieved source covers long-term exposure; the short-term effect was not directly addressed.", evidence: [], citation_ids: [] },
    ],
    citations: [{ ...demoSources[4], claims: ["h1"] }],
  };
}

function genericPayload(query: string): Payload {
  const initial = `## Demo answer\n\nThis is **demo data** generated locally for the question:\n\n> ${query}\n\nThe live backend is not being used, so no real model answer or evaluation exists for this prompt.`;
  const final = `## Demo answer (verified)\n\nThis is **demo data** generated locally for the question:\n\n> ${query}\n\nThe live backend is not being used, so no real model answer or evaluation exists for this prompt. Switch the header to *Live backend* to verify real answers.`;
  return {
    base_answer: initial,
    rag_answer: final,
    quality_pass: true,
    base_metrics: { factuality: 55, hallucination_risk: 40, completeness: 60, relevance: 80 },
    final_metrics: { factuality: 82, hallucination_risk: 14, completeness: 78, relevance: 90 },
    claims: [
      { claim: "The live backend is not being used.", status: "verified", explanation: "Demo adapter is active." },
      { claim: "no real model answer or evaluation exists for this prompt.", status: "unable_to_verify", explanation: "Demo data cannot be verified against evidence." },
    ],
    citations: [],
  };
}

export function demoPayloadFor(query: string): Payload {
  const q = query.toLowerCase();
  if (q.includes("cause")) return causesPayload();
  if (q.includes("health")) return healthPayload();
  return genericPayload(query);
}

const sleep = (ms: number, signal?: AbortSignal) =>
  new Promise<void>((resolve, reject) => {
    const t = setTimeout(resolve, ms);
    signal?.addEventListener("abort", () => {
      clearTimeout(t);
      reject(new ApiError("The request was cancelled."));
    });
  });

export function createMockApi(latencyMs = 1400): HalluciGuardApi {
  return {
    mode: "demo",
    async verify({ query, conversationId, responseId, signal }: VerifyRequest) {
      await sleep(latencyMs, signal);
      if (/\bfail/i.test(query)) throw new ApiError("Demo mode: simulated backend failure (HTTP 500).", 500);
      return normalizeBackendResult(demoPayloadFor(query), { responseId, conversationId, query, source: "demo" });
    },
  };
}

/** Seed conversations so history, scrolling and the verification view can be explored in demo mode. */
export function createDemoConversations(): Conversation[] {
  const now = Date.now();
  const topics: Array<{ title: string; prompts: string[] }> = [
    { title: "Urban air pollution", prompts: ["What are the main causes of urban air pollution?", "What is the impact of air pollution on health?", "How do industrial emissions compare with household sources of pollution?", "Which cities have reduced air pollution the most?", "What are the health effects of PM2.5 on children and the elderly in large cities?", "Do low-emission zones work?", "How is air quality measured?"] },
    { title: "Cleaner cities", prompts: ["How can cities reduce air pollution?"] },
    { title: "Health and the environment", prompts: ["What is the impact of air pollution on health?"] },
    { title: "Causes of smog", prompts: ["What causes smog in winter?"] },
    { title: "Public transport", prompts: ["Does public transport reduce emissions?"] },
    { title: "Indoor air quality", prompts: ["How do I improve indoor air quality?"] },
    { title: "Wildfire smoke", prompts: ["What are the causes of wildfire smoke episodes?"] },
  ];
  return topics.map((topic, ci) => {
    const convoId = `demo-conv-${ci + 1}`;
    const created = now - (ci + 1) * 3_600_000;
    const messages: Message[] = [];
    topic.prompts.forEach((prompt, pi) => {
      const uid = `${convoId}-u${pi + 1}`;
      const rid = `${convoId}-r${pi + 1}`;
      const ts = created + pi * 90_000;
      const analysis = normalizeBackendResult(demoPayloadFor(prompt), { responseId: rid, conversationId: convoId, query: prompt, source: "demo" });
      messages.push({ id: uid, role: "user", content: prompt, createdAt: ts });
      messages.push({ id: rid, role: "assistant", replyTo: uid, content: analysis.finalResponse ?? analysis.initialResponse ?? "", state: "done", analysis, createdAt: ts + 5_000 });
    });
    return { id: convoId, title: topic.title, createdAt: created, updatedAt: created + topic.prompts.length * 90_000, messages };
  });
}
