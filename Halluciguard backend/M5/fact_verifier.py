from M5.llm_32b import generate_32b

def verify_claim(claim, evidence):
    """
    Verify a claim against retrieved evidence.

    Returns:
        SUPPORTED
        CONTRADICTED
        UNCERTAIN
    """

    if not evidence:
        return "UNCERTAIN"

    evidence_text = ""

    for i, item in enumerate(evidence, 1):
        snippet = item.get("snippet", "")

        evidence_text += f"""
Evidence {i}:
{snippet}
"""

    verification_prompt = f"""
You are a strict factual evidence classifier.

Your task is to compare ONE CLAIM against the provided EVIDENCE.

CLAIM:
{claim}

EVIDENCE:
{evidence_text}

==================================================
DECISION RULES
==================================================

SUPPORTED:
Choose SUPPORTED when the provided evidence directly supports
the claim or when multiple evidence snippets together clearly
establish all important parts of the claim.

CONTRADICTED:
Choose CONTRADICTED when reliable evidence directly conflicts
with an important factual part of the claim.

UNCERTAIN:
Choose UNCERTAIN when the evidence is insufficient, ambiguous,
irrelevant, or does not establish whether the claim is true.

==================================================
VERY IMPORTANT RULES
==================================================

1. Evaluate the exact factual meaning of the claim, not just
   matching keywords.

2. Evidence may support a claim across multiple snippets.
   Consider all provided evidence together.

3. A claim can be supported by a clear equivalent statement,
   even if the wording differs.

4. Related-topic evidence alone is not sufficient.

5. Do not label a claim CONTRADICTED merely because the evidence
   does not mention it.

6. For compound claims, evaluate every important factual part.
   Choose SUPPORTED only if all important parts are supported.

7. If evidence supports one part but leaves another important
   part unresolved, choose UNCERTAIN unless evidence directly
   contradicts that unresolved part.

8. Choose CONTRADICTED only when evidence provides a clear
   factual conflict, not merely an absence of support.

9. Use ONLY the provided evidence. Do not rely on outside
   knowledge or invent facts.

10. If the claim is a correction of a false premise, evaluate
    the corrected factual statement itself against the evidence.
    Do not assume the original premise is true.

==================================================
**NEGATION AND MEANING RULE:**
Classify the meaning of the entire claim, including words such as "not", "incorrect", "false",
 and "does not". Do not classify the claim by matching only its topic or keywords.

Example:
- Evidence: Pure water boils at approximately 100°C at standard atmospheric pressure.
- Claim: "The statement that water boils at 50°C under standard atmospheric pressure is incorrect."
- Correct label: SUPPORTED, because the evidence supports the claim that the 50°C statement is incorrect.

A claim is CONTRADICTED only when the evidence conflicts with the meaning of the claim itself. 
If the evidence does not establish whether the complete claim is true or false, return UNCERTAIN.
==================================================

Now classify the CLAIM using ONLY the EVIDENCE.

Return ONLY ONE WORD:

SUPPORTED
CONTRADICTED
UNCERTAIN
"""

    
    result = generate_32b(
        verification_prompt,
        max_new_tokens=50,
        temperature=0.0
    )

    print(f"\n[FactVerifier] Claim: {claim}")
    print(f"[FactVerifier] Raw model output: {result!r}")

    # Normalize the model output.
    result = result.strip().upper()

    # Accept only a single valid classification.
    valid_labels = {
        "SUPPORTED",
        "CONTRADICTED",
        "UNCERTAIN"
    }

    if result in valid_labels:
        return result

    # If the model adds explanations or multiple labels,
    # fail safely instead of guessing.
    print(
        f"[FactVerifier] Invalid classification output: "
        f"{result!r}"
    )

    return "UNCERTAIN"
