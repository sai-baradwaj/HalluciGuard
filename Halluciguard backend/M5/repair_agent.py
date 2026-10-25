
from M5.llm_32b import generate_32b


def repair_response(question, response, failed_claims):
    """
    Repair an answer using the supplied verification results and evidence.

    Verification statuses:
        SUPPORTED     - Evidence supports the claim.
        CONTRADICTED  - Evidence conflicts with the claim.
        UNCERTAIN     - Evidence is insufficient or inconclusive.
    """

    evidence_text = ""

    for index, item in enumerate(failed_claims, start=1):
        claim = str(item.get("claim", "")).strip()
        verification = str(
            item.get("verification", "UNCERTAIN")
        ).strip().upper()

        if verification not in {
            "SUPPORTED",
            "CONTRADICTED",
            "UNCERTAIN",
        }:
            verification = "UNCERTAIN"

        evidence_text += f"""
Claim {index}:
{claim}

Verification Status:
{verification}

Evidence:
"""

        claim_evidence = item.get("evidence", [])

        if not isinstance(claim_evidence, list) or not claim_evidence:
            evidence_text += "- No evidence was provided for this claim.\n"
            continue

        for evidence in claim_evidence:
            if not isinstance(evidence, dict):
                continue

            title = str(evidence.get("title", "")).strip()
            snippet = str(evidence.get("snippet", "")).strip()
            url = str(evidence.get("url", "")).strip()

            if not snippet:
                continue

            evidence_text += f"- {title}: {snippet}"

            if url:
                evidence_text += f"\n  Source URL: {url}"

            evidence_text += "\n"

    if not evidence_text.strip():
        evidence_text = (
            "No claim-specific evidence was provided. "
            "Do not invent facts or claim that verification succeeded."
        )

    prompt = f"""
You are the Repair Agent in HalluciGuard, a high-quality factual
verification and response-correction system.

Your task is to repair the ORIGINAL ANSWER using the verification
results and evidence provided below. Preserve the answer's useful
reasoning, detail, technical accuracy, and explanatory depth.

USER QUESTION:
{question}

ORIGINAL ANSWER:
{response}

CLAIMS THAT FAILED VERIFICATION:
{evidence_text}

CORE REPAIR OBJECTIVES:

1. FULL-ANSWER ANALYSIS
   Review the entire original answer, not just the listed claims.
   Identify contradictions, unsupported factual assertions, misleading
   wording, invalid conclusions, and inconsistencies between sentences.

2. EVIDENCE-GROUNDED CORRECTION
   Use the supplied evidence to determine what can reasonably be stated.
   Correct contradicted claims and qualify claims that lack sufficient
   support. Do not invent facts, sources, statistics, quotations, or
   discoveries.

3. INTERPRET VERIFICATION STATUS CORRECTLY
   - SUPPORTED: The supplied evidence supports the claim. Preserve it
     unless another supplied result reveals a genuine contradiction.
   - CONTRADICTED: The supplied evidence conflicts with the claim.
     Correct or remove the conflicting statement.
   - UNCERTAIN: The evidence is insufficient, ambiguous, or inconclusive.
     Do not automatically treat the claim as false. Qualify it or state
     that it cannot be verified from the available evidence.

4. HANDLE FALSE PREMISES
   Never agree with or endorse a false premise in the user's question.
   Correct the premise clearly when reliable supplied evidence supports
   that correction. Remove misleading statements that affirm the premise,
   even if later sentences attempt to correct it.

5. PRESERVE CORRECT INFORMATION
   Retain correct information, relevant context, useful explanations,
   technical details, and sound reasoning from the original answer.
   Do not unnecessarily shorten the response or discard valid content.
   Rewrite only what is needed to improve factual reliability and
   logical consistency.

6. HANDLE COMPOUND CLAIMS
   If a statement contains multiple factual assertions, assess them
   separately. Do not present the entire statement as established when
   important parts remain unsupported or contradicted.

7. HANDLE INSUFFICIENT OR IRRELEVANT EVIDENCE
   Evidence about a related topic does not automatically establish the
   claim being checked. Do not use unrelated evidence to justify a
   conclusion. When the supplied evidence is insufficient, explicitly
   communicate the limitation instead of guessing.

8. MAINTAIN LOGICAL CONSISTENCY
   Ensure the opening, body, and conclusion agree with one another.
   Remove unsupported certainty and resolve contradictions throughout
   the answer.

9. CITATIONS AND SOURCES
   Do not fabricate citations or claim to have consulted sources that
   were not provided. Do not imply that a source proves a claim unless
   its supplied content actually supports that claim. Citation formatting
   will be handled by the appropriate downstream component.

10. FINAL QUALITY CHECK
    Before returning the answer, check that:
    - The original question has been addressed.
    - Incorrect or misleading claims have been corrected.
    - Uncertain claims are appropriately qualified.
    - Supported information and useful explanations are preserved.
    - No new unsupported factual assertions have been introduced.
    - The entire answer is internally consistent.

IMPORTANT CONSTRAINTS:
- Treat the supplied evidence as the factual basis for corrections.
- Do not interpret missing evidence as proof that a claim is false.
- Do not invent evidence to force a claim into SUPPORTED status.
- If the evidence cannot establish a definitive answer, explain that
  limitation clearly.
- Return only the complete corrected answer, without analysis of your
  repair process, status labels, or introductory commentary.
"""

    repaired_answer = generate_32b(
        prompt,
        max_new_tokens=1000,
        temperature=0.1,
    )

    if not isinstance(repaired_answer, str):
        raise TypeError(
            "generate_32b() must return a string, "
            f"received {type(repaired_answer).__name__}."
        )

    repaired_answer = repaired_answer.strip()

    if not repaired_answer:
        raise ValueError(
            "The Repair Agent returned an empty answer."
        )

    return repaired_answer
