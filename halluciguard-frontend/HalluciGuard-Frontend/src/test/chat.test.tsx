import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import App from "@/App";
import { conversation, fakeApi, payload } from "./fixtures";
import { normalizeBackendResult } from "@/services/normalize";
import { mockMatchMedia } from "./setup";

describe("message composer", () => {
  it("Enter submits once, Shift+Enter adds a line, and duplicate submissions are blocked while verifying", async () => {
    let release!: () => void;
    const gate = new Promise<void>((r) => (release = r));
    const api = fakeApi(async ({ query, responseId, conversationId }) => {
      api.calls.push(query);
      await gate;
      return normalizeBackendResult(payload(7), { responseId, conversationId, query, source: "backend" });
    });
    const user = userEvent.setup();
    render(<App persist={false} api={api} />);

    const box = screen.getByPlaceholderText(/ask halluciguard/i);
    await user.type(box, "line one{Shift>}{Enter}{/Shift}line two");
    expect(box).toHaveValue("line one\nline two");
    expect(api.calls).toHaveLength(0);

    await user.keyboard("{Enter}");
    await waitFor(() => expect(api.calls).toEqual(["line one\nline two"]));
    expect(box).toHaveValue("");

    // While verifying: the status is shown, Send is disabled, and a second Enter does nothing.
    expect(screen.getByText(/checking it against evidence/i)).toBeInTheDocument();
    await user.type(box, "second prompt{Enter}");
    expect(api.calls).toHaveLength(1);
    expect(screen.getByRole("button", { name: "Verifying" })).toBeDisabled(); // composer send button
    expect(screen.getByRole("button", { name: /verifying…/i })).toBeDisabled(); // this response's Verify button

    release();
    await screen.findByText(/FINAL-7/);
    expect(screen.queryByText(/INITIAL-7/)).not.toBeInTheDocument(); // the chat shows the corrected answer
    expect(screen.getByTestId("status-badge")).toHaveTextContent("Corrected");
  });

  it("shows a readable error and a retry control when the backend request fails", async () => {
    const api = fakeApi(async () => {
      throw new Error("Can't reach the HalluciGuard backend");
    });
    const user = userEvent.setup();
    render(<App persist={false} api={api} />);
    await user.type(screen.getByPlaceholderText(/ask halluciguard/i), "hello{Enter}");
    expect(await screen.findByRole("alert")).toHaveTextContent("Can't reach the HalluciGuard backend");
    expect(screen.getByTestId("status-badge")).toHaveTextContent("Verification failed");
    expect(screen.getByRole("button", { name: /retry/i })).toBeInTheDocument();
  });
});

describe("prompt history rail", () => {
  const convoA = conversation("a", "Alpha chat", [1, 2, 3, 4, 5, 6, 7]);
  const convoB = conversation("b", "Beta chat", [8]);

  it("is generated from real data (no placeholders), lists older prompts too, and is independent per conversation", async () => {
    const user = userEvent.setup();
    render(<App persist={false} initialConversations={[convoA, convoB]} api={fakeApi()} />);
    expect(screen.queryAllByTestId("prompt-row")).toHaveLength(0);

    await user.click(screen.getByRole("button", { name: "Alpha chat" }));
    const rows = screen.getAllByTestId("prompt-row");
    expect(rows).toHaveLength(7); // five visible at a time, the rest reachable by scrolling the rail
    expect(rows[6]).toHaveTextContent(""); // lines only: no prompt text is rendered until hover
    expect(within(rows[6]).getByRole("button")).toHaveAccessibleName("Question 7?"); // newest at the bottom

    await user.click(screen.getByRole("button", { name: "Beta chat" }));
    const other = screen.getAllByTestId("prompt-row");
    expect(other).toHaveLength(1);
    expect(within(other[0]).getByRole("button")).toHaveAccessibleName("Question 8?");
  });

  it("is not part of the sidebar and shows no text until hovered", async () => {
    const user = userEvent.setup();
    render(<App persist={false} initialConversations={[convoA]} api={fakeApi()} />);
    await user.click(screen.getByRole("button", { name: "Alpha chat" }));
    const rail = screen.getByRole("navigation", { name: "Prompt history" });
    expect(screen.getByRole("complementary").contains(rail)).toBe(false);
    expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();
  });

  it("reveals some words of the prompt on hover", async () => {
    const long = "What are the main causes of urban air pollution and how do they differ between megacities and small towns in different regions of the world today?";
    const c = conversation("a", "Alpha chat", [1]);
    if (c.messages[0].role === "user") c.messages[0].content = long;
    const user = userEvent.setup();
    render(<App persist={false} initialConversations={[c]} api={fakeApi()} />);
    await user.click(screen.getByRole("button", { name: "Alpha chat" }));
    await user.hover(within(screen.getByTestId("prompt-row")).getByRole("button"));
    const tip = await screen.findByRole("tooltip");
    expect(tip).toHaveTextContent("What are the main causes of urban air pollution");
    expect(tip.textContent).toMatch(/…$/); // shortened, not the whole prompt
  });

  it("clicking a line scrolls to that prompt's response", async () => {
    const spy = vi.spyOn(Element.prototype, "scrollIntoView");
    const user = userEvent.setup();
    render(<App persist={false} initialConversations={[convoA]} api={fakeApi()} />);
    await user.click(screen.getByRole("button", { name: "Alpha chat" }));
    spy.mockClear();
    await user.click(within(screen.getAllByTestId("prompt-row")[2]).getByRole("button")); // prompt 3
    expect(spy).toHaveBeenCalledTimes(1);
    expect((spy.mock.instances[0] as unknown as HTMLElement).id).toBe("message-r3"); // the assistant response, not the prompt
    spy.mockRestore();
  });

  it("on touch devices a tap reveals the words and a second tap on the preview navigates", async () => {
    mockMatchMedia((q) => q.includes("hover: none") || q.includes("min-width"));
    const spy = vi.spyOn(Element.prototype, "scrollIntoView");
    const user = userEvent.setup();
    render(<App persist={false} initialConversations={[conversation("a", "Alpha chat", [1, 2])]} api={fakeApi()} />);
    await user.click(screen.getByRole("button", { name: "Alpha chat" }));
    spy.mockClear();
    const row = screen.getAllByTestId("prompt-row")[0];
    await user.click(within(row).getByRole("button", { expanded: false }));
    expect(spy).not.toHaveBeenCalled(); // first tap only reveals
    await user.click(within(row).getByText("Question 1?")); // the revealed preview
    expect((spy.mock.instances[0] as unknown as HTMLElement).id).toBe("message-r1");
    spy.mockRestore();
    mockMatchMedia((q) => q.includes("min-width"));
  });
});
