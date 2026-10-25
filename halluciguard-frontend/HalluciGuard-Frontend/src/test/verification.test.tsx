import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import App from "@/App";
import { conversation, fakeApi, turn } from "./fixtures";

function renderTwoResponses() {
  const convo = conversation("c1", "Two answers", [1, 2]);
  const user = userEvent.setup();
  render(<App persist={false} initialConversations={[convo]} api={fakeApi()} />);
  return { user };
}

async function openConversation(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByRole("button", { name: "Two answers" }));
}

describe("Verify Response", () => {
  it("does not open the verification view on page load or when a conversation is opened", async () => {
    const { user } = renderTwoResponses();
    expect(screen.queryByTestId("verification-modal")).not.toBeInTheDocument();
    await openConversation(user);
    expect(screen.queryByTestId("verification-modal")).not.toBeInTheDocument();
  });

  it("shows the FINAL corrected answer in the chat, never the initial one", async () => {
    const { user } = renderTwoResponses();
    await openConversation(user);
    expect(screen.getByText(/FINAL-1/)).toBeInTheDocument();
    expect(screen.getByText(/FINAL-2/)).toBeInTheDocument();
    expect(screen.queryByText(/INITIAL-1/)).not.toBeInTheDocument();
  });

  it("each response's button opens that response's own analysis (and only that one)", async () => {
    const { user } = renderTwoResponses();
    await openConversation(user);
    const buttons = screen.getAllByRole("button", { name: /verify response/i });
    expect(buttons).toHaveLength(2);
    expect(buttons.map((b) => b.dataset.responseId)).toEqual(["r1", "r2"]);

    // Open the SECOND response first, to catch implementations that always show the first/last analysis.
    await user.click(buttons[1]);
    let modal = screen.getByTestId("verification-modal");
    expect(modal).toHaveAttribute("data-response-id", "r2");
    expect(within(modal).getByText(/INITIAL-2/)).toBeInTheDocument();
    expect(within(modal).getByText(/FINAL-2/)).toBeInTheDocument();
    expect(within(modal).getByText("SOURCE-2")).toBeInTheDocument();
    expect(within(modal).queryByText(/INITIAL-1/)).not.toBeInTheDocument();
    expect(within(modal).queryByText("SOURCE-1")).not.toBeInTheDocument();

    await user.keyboard("{Escape}");
    expect(screen.queryByTestId("verification-modal")).not.toBeInTheDocument();

    await user.click(screen.getAllByRole("button", { name: /verify response/i })[0]);
    modal = screen.getByTestId("verification-modal");
    expect(modal).toHaveAttribute("data-response-id", "r1");
    expect(within(modal).getByText(/INITIAL-1/)).toBeInTheDocument();
    expect(within(modal).queryByText(/INITIAL-2/)).not.toBeInTheDocument();
  });

  it("switching responses inside the view never leaves stale data behind", async () => {
    const { user } = renderTwoResponses();
    await openConversation(user);
    await user.click(screen.getAllByRole("button", { name: /verify response/i })[0]);
    const modal = screen.getByTestId("verification-modal");
    expect(within(modal).getByText(/INITIAL-1/)).toBeInTheDocument();

    await user.click(within(modal).getByRole("button", { name: /next response/i }));
    expect(modal).toHaveAttribute("data-response-id", "r2");
    expect(within(modal).getByText(/INITIAL-2/)).toBeInTheDocument();
    expect(within(modal).queryByText(/INITIAL-1/)).not.toBeInTheDocument();
    expect(within(modal).queryByText("EXCERPT-1")).not.toBeInTheDocument();
  });

  it("renders real metrics with calculated changes, claims, and safe citation links", async () => {
    const { user } = renderTwoResponses();
    await openConversation(user);
    await user.click(screen.getAllByRole("button", { name: /verify response/i })[0]);
    const modal = screen.getByTestId("verification-modal");

    const table = within(modal).getByTestId("metrics-table");
    const factuality = within(table).getByText("Factuality").closest("tr")!;
    expect(within(factuality).getByText("0.51")).toBeInTheDocument(); // initial: 0.5 + 1/100
    expect(within(factuality).getByText("0.90")).toBeInTheDocument();
    expect(within(factuality).getByText("+0.39")).toBeInTheDocument();
    const completeness = within(table).getByText("Completeness").closest("tr")!;
    expect(within(completeness).getAllByText("N/A").length).toBeGreaterThanOrEqual(3); // omitted by backend: not invented

    // Claim details are collapsed until expanded
    expect(within(modal).queryByText("EXPLAIN-1")).not.toBeInTheDocument();
    await user.click(within(modal).getByRole("button", { name: /CLAIM-1/ }));
    expect(within(modal).getByText("EXPLAIN-1")).toBeInTheDocument();
    expect(within(modal).getByText("EVIDENCE-1")).toBeInTheDocument();

    const citation = within(modal).getByTestId("citation");
    const link = within(citation).getAllByRole("link")[0];
    expect(link).toHaveAttribute("href", "https://example.com/source-1");
    expect(link).toHaveAttribute("target", "_blank");
    expect(link).toHaveAttribute("rel", expect.stringContaining("noopener"));
    expect(within(citation).getByText(/CLAIM-1/)).toBeInTheDocument(); // the claim this source supports
  });

  it("shows the exact empty message when there are no citations", async () => {
    const convo = conversation("c1", "Two answers", [1]);
    const msg = convo.messages[1];
    if (msg.role === "assistant" && msg.analysis) msg.analysis = { ...msg.analysis, citations: [] };
    const user = userEvent.setup();
    render(<App persist={false} initialConversations={[convo]} api={fakeApi()} />);
    await user.click(screen.getByRole("button", { name: "Two answers" }));
    await user.click(screen.getByRole("button", { name: /verify response/i }));
    expect(screen.getByText("No citations available for this response.")).toBeInTheDocument();
  });

  it("shows a clear failure state for a failed response instead of claiming success", async () => {
    const convo = conversation("c1", "Two answers", [1, 2], (n) => turn(n, "c1", n === 2 ? null : undefined));
    const user = userEvent.setup();
    render(<App persist={false} initialConversations={[convo]} api={fakeApi()} />);
    await user.click(screen.getByRole("button", { name: "Two answers" }));
    await user.click(screen.getAllByRole("button", { name: /verify response/i })[1]);
    const modal = screen.getByTestId("verification-modal");
    expect(within(modal).getByText("Verification failed", { selector: "p" })).toBeInTheDocument();
    expect(within(modal).getByText(/boom: backend unavailable/)).toBeInTheDocument();
    expect(within(modal).queryByTestId("citation")).not.toBeInTheDocument();
  });
});
