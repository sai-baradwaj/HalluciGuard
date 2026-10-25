import { createLiveApi, config } from "@/services/api";

const req = { query: "What causes smog?", conversationId: "c", responseId: "r" };

afterEach(() => vi.restoreAllMocks());

describe("live API client", () => {
  it("POSTs exactly { query } to /verify with the x-api-key header and normalizes the reply", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ base_answer: "draft", rag_answer: "final", quality_pass: true }), { status: 200 }),
    );
    const result = await createLiveApi().verify(req);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`${config.baseUrl}/verify`);
    expect(init?.method).toBe("POST");
    expect((init?.headers as Record<string, string>)["x-api-key"]).toBe(config.apiKey);
    expect(JSON.parse(init?.body as string)).toEqual({ query: "What causes smog?" });
    expect(result).toMatchObject({ responseId: "r", initialResponse: "draft", finalResponse: "final", source: "backend" });
  });

  it("turns HTTP errors, network failures and unusable payloads into readable errors", async () => {
    const f = vi.spyOn(globalThis, "fetch");
    f.mockResolvedValueOnce(new Response(JSON.stringify({ detail: "Invalid API key" }), { status: 401 }));
    await expect(createLiveApi().verify(req)).rejects.toThrow(/API key/);
    f.mockRejectedValueOnce(new TypeError("Failed to fetch"));
    await expect(createLiveApi().verify(req)).rejects.toThrow(/Can't reach the HalluciGuard backend/);
    f.mockResolvedValueOnce(new Response(JSON.stringify({ unrelated: 1 }), { status: 200 }));
    await expect(createLiveApi().verify(req)).rejects.toThrow(/no recognized answer/);
  });
});
