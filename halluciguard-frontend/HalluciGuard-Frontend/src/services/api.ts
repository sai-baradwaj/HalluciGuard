/**
 * Centralised API layer. Components never call fetch(); they use a HalluciGuardApi.
 * Swap or extend `createLiveApi` if the backend contract changes (field names live in normalize.ts).
 */
import type { DataMode, VerificationResult } from "@/types";
import { normalizeBackendResult } from "./normalize";
import { createMockApi } from "./mock";
import { ApiError } from "./errors";

export interface VerifyRequest {
  query: string;
  conversationId: string;
  responseId: string;
  signal?: AbortSignal;
}

export interface HalluciGuardApi {
  mode: DataMode;
  verify(request: VerifyRequest): Promise<VerificationResult>;
}

export { ApiError };

export const config = {
  baseUrl: (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, ""),
  apiKey: import.meta.env.VITE_API_KEY || "halluciguard-dev-key",
  startInDemo: import.meta.env.VITE_USE_MOCK === "false",
  timeoutMs: 30 * 60 * 1000, // 30-minute verification timeout // verification runs an LLM + retrieval pipeline and can be slow
};

function describeHttpError(status: number, detail: string): string {
  if (status === 401 || status === 403) return `The backend rejected the API key (HTTP ${status}). Check VITE_API_KEY.`;
  if (status === 404) return "The backend has no /verify endpoint at this URL (HTTP 404). Check VITE_API_BASE_URL.";
  if (status === 422) return `The backend could not accept this request (HTTP 422): ${detail}`;
  if (status >= 500) return `The backend hit an error while verifying (HTTP ${status}): ${detail}`;
  return `Backend returned HTTP ${status}: ${detail}`;
}

function detailFrom(data: unknown, fallback: string): string {
  if (data && typeof data === "object") {
    const d = (data as Record<string, unknown>).detail ?? (data as Record<string, unknown>).message;
    if (typeof d === "string") return d;
    if (d !== undefined) {
      try {
        return JSON.stringify(d);
      } catch {
        /* fall through */
      }
    }
  }
  return fallback;
}

export function createLiveApi(): HalluciGuardApi {
  return {
    mode: "live",
    async verify({ query, conversationId, responseId,chatHistory = [], signal }) {
      const timeout = new AbortController();
      const timer = setTimeout(() => timeout.abort(), config.timeoutMs);
      const onAbort = () => timeout.abort();
      signal?.addEventListener("abort", onAbort);

      let response: Response;
      try {
        response = await fetch(`${config.baseUrl}/verify`, {
          method: "POST",
          headers: { "Content-Type": "application/json", "x-api-key": config.apiKey },
          // The existing contract is exactly { query }. Do not add fields the backend has not agreed to.
          body: JSON.stringify({query, chat_history: chatHistory ,}),
          signal: timeout.signal,
        });
      } catch (err) {
        if (signal?.aborted) throw new ApiError("The request was cancelled.");
        if (timeout.signal.aborted) throw new ApiError("The backend took too long to respond. Try again, or check the server logs.");
        throw new ApiError(
          `Can't reach the HalluciGuard backend at ${config.baseUrl}. Check that the server is running and that CORS allows this origin and the x-api-key header.`,
        );
      } finally {
        clearTimeout(timer);
        signal?.removeEventListener("abort", onAbort);
      }

      const text = await response.text();
      let data: unknown;
      try {
        data = text ? JSON.parse(text) : {};
      } catch {
        throw new ApiError(`The backend returned non-JSON data (HTTP ${response.status}). Check the API server and endpoint.`, response.status);
      }
      if (!response.ok) throw new ApiError(describeHttpError(response.status, detailFrom(data, text || response.statusText)), response.status);

      const result = normalizeBackendResult(data, { responseId, conversationId, query, source: "backend" });
      if (!result.finalResponse && !result.initialResponse) {
        throw new ApiError("The backend returned a result, but it contained no recognized answer field.", response.status);
      }
      return result;
    },
  };
}

export function createApi(mode: DataMode): HalluciGuardApi {
  return mode === "demo" ? createMockApi() : createLiveApi();
}
export interface ChatHistoryMessage {
  role: "user" | "assistant";
  content: string;
}

export interface VerifyRequest {
  query: string;
  conversationId: string;
  responseId: string;
  chatHistory?: ChatHistoryMessage[];
  signal?: AbortSignal;
}
