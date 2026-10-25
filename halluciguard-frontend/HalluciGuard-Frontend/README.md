# HalluciGuard Frontend

Chat-style interface for **HalluciGuard** (detect · verify · trust). Ask a question, read the final corrected answer, and open **Verify Response** on any answer to compare it with the initial LLM answer: metrics, claim-by-claim verification, and citations.

React + Vite + TypeScript + Tailwind CSS v4, Lucide icons, shadcn/ui-style components (Radix primitives), `react-markdown`. The navy/cyan theme of the original frontend is preserved (dark by default, light available).

## Run

```bash
npm install
cp .env.example .env.local     # optional; defaults shown below
npm run dev                    # http://localhost:3000
```

| Command | Purpose |
|---|---|
| `npm run dev` | Dev server |
| `npm run build` | Typecheck + production build to `dist/` |
| `npm test` | Vitest suite (35 tests) |
| `npm run typecheck` | TypeScript only |

### Environment

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
VITE_API_KEY=halluciguard-dev-key
VITE_USE_MOCK=false
```

Restart Vite after changing them. The key is visible in the browser bundle, so only use a development key here.

## Backend contract

Unchanged from the original frontend:

- `POST {VITE_API_BASE_URL}/verify`
- Headers: `Content-Type: application/json`, `x-api-key`
- Body: `{ "query": "..." }`

The backend must allow the frontend origin (normally `http://localhost:3000`) and the `Content-Type` and `x-api-key` headers via CORS.

**All field-name knowledge lives in `src/services/normalize.ts`.** It accepts the existing names and the suggested contract from the product brief:

| Meaning | Accepted fields (first match wins) |
|---|---|
| Initial answer | `initial_response`, `base_answer`, `initial_answer`, `original_answer`, `llm_response`, `response`, `answer` |
| Final answer | `final_response`, `rag_answer`, `final_answer`, `verified_answer`, `corrected_answer`, `output` |
| Initial / final metrics | `base_metrics` / `initial_metrics`, `final_metrics` / `rag_metrics`, or `metrics.initial` / `metrics.final` (object or `[{name, score}]`) |
| Claims | `claims`, `claim_verification`, `claim_results`, `verified_claims` |
| Citations | `citations`, `sources`, `references` (objects or URL strings) |
| Correction summary | `correction_summary`, `corrections`, `correction_notes` |
| Status | `verification_status` / `status` (`completed`, `failed`, `processing`...) and `quality_pass` |

If your backend uses other names, add them to those lists. I could not inspect your FastAPI/LangGraph code (the upload contained only the old frontend), so verify these against your real `/verify` response. Anything not matched is shown as unavailable, never guessed.

### What is and is not shown

- **Chat** always shows the final corrected answer. If the backend returns no corrected answer, the initial one is shown with an explicit "unverified" note.
- **Status badge** (Verified / Corrected / In progress / Failed / Quality check not passed) appears only when the response supports it: an explicit backend status or `quality_pass`. *Corrected* means the backend passed it and the final answer differs from the initial one (or claims/summary report corrections).
- **Metrics**: the six standard rows are always listed; missing values show `N/A`. Changes are computed only when both values exist and share a unit. Normalized scores (0–1) and percentages (0–100) are labelled separately and never mixed. Hallucination risk is marked "lower is better". Extra metrics the backend returns are appended.
- **Claims** are statuses mapped to Verified / Unsupported / Contradicted / Partially Supported / Unable to Verify; a missing status reads "Status not provided".
- **Citations**: only what the backend returned. Only `http(s)` URLs become links (`target="_blank" rel="noopener noreferrer"`). Empty → "No citations available for this response."
- Claim highlights in the response panels appear only where the claim text matches the answer word for word.

## Demo mode

Switch **Live backend ↔ Demo** in the header (or `VITE_USE_MOCK=true`). Demo mode uses `src/services/mock.ts`, seeds sample conversations, and labels every demo message and verification view with **Demo data**. Prompts containing "fail" simulate a backend error. Demo sources use `example.org` URLs and `[Demo excerpt]` text; none of it is real evaluation data.

## Structure

```
src/
  services/   api.ts (live client + factory), normalize.ts, mock.ts, storage.ts
  hooks/      useChat (state, per-response storage), useVerification, useTheme, useMediaQuery
  lib/        metrics.ts (comparison logic), linking.ts, share.ts, utils.ts
  components/ Sidebar, ConversationHistory, PromptHistory (right-edge rail), ChatHeader, ChatMessage,
              MessageActions, MessageComposer, VerifyResponseButton, Markdown, ui/*
    verification/ VerificationModal, ResponseComparison, MetricsComparison,
                  ClaimVerification, CitationsPanel
  test/       normalize, metrics, api, chat, verification
```

## How each response gets its own analysis

Each assistant message's `id` is its response ID, and the analysis is stored on that message. **Verify Response** stores only `{conversationId, responseId}`; the sheet re-resolves the analysis from live state on every render and is keyed by response ID, so state (tabs, expanded claims) resets on switch. The prev/next arrows move between responses in the same conversation. `src/test/verification.test.tsx` opens the second response before the first and asserts the first's data never appears.

## Behaviour notes

- Conversations and results persist in `localStorage` (this browser only). A request interrupted by reload is marked failed with a Retry button.
- The sidebar lists conversations (five rows visible, scroll for more). **Prompt history is a slim rail on the right edge of the chat, vertically centred**: it shows only short lines (the five newest; scroll for older ones). Hovering a line shows the first words of that prompt, and clicking it scrolls to that prompt's response. On touch devices, tap a line to reveal the words, then tap the preview to jump. The rail belongs to the active conversation only.
- **Share** uses the native share sheet where available, otherwise copies Markdown to the clipboard. There is no server-side share link because the backend has no such API.
- The attachment (+) button from the wireframe is omitted because `/verify` accepts text only.
- The header selector is **Live backend / Demo** rather than a model picker, since the contract has no model parameter.
- `Esc`, the close button, or a backdrop click closes the verification sheet. On phones it is full screen.
