import os

# Reduce unnecessary Hugging Face / Transformers output
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import json
import re
from M5.llm_8b import generate_8b


def clean_claim(claim):
    claim = claim.strip()

    # Remove code fences
    claim = re.sub(r"^```(?:json)?\s*", "", claim, flags=re.IGNORECASE)
    claim = re.sub(r"\s*```$", "", claim)

    # Remove bullet and numbered prefixes
    claim = re.sub(r"^[-*•]\s*", "", claim)
    claim = re.sub(r"^\d+[.)]\s*", "", claim)

    # Remove common introductory phrases
    intro_patterns = [
        r"^here'?s\s+(?:a\s+)?simple explanation\s*:\s*",
        r"^here\s+is\s+(?:a\s+)?simple explanation\s*:\s*",
        r"^in\s+simple\s+terms\s*:\s*",
        r"^the\s+answer\s+is\s*:\s*",
        r"^explanation\s*:\s*",
        r"^for\s+example\s*:\s*",
        r"^step\s+\d+\s*[:\-]\s*",
    ]

    changed = True
    while changed:
        old_claim = claim

        for pattern in intro_patterns:
            claim = re.sub(
                pattern,
                "",
                claim,
                flags=re.IGNORECASE,
            )

        claim = re.sub(r"^[-*•]\s*", "", claim)
        claim = re.sub(r"^\d+[.)]\s*", "", claim)

        changed = claim != old_claim

    # Remove Markdown emphasis
    claim = re.sub(r"[*_]{1,3}", "", claim)

    # Normalize whitespace and punctuation
    claim = re.sub(r"\s+", " ", claim).strip()
    claim = re.sub(r"[.!?]+$", "", claim).strip()

    return claim



def remove_duplicates(claims):
    """
    Remove exact duplicates and claims that are already
    completely covered by a longer claim.
    """

    # First remove exact duplicates
    unique_claims = []

    for claim in claims:

        if not claim:
            continue

        duplicate = False

        for existing in unique_claims:

            if claim.lower() == existing.lower():
                duplicate = True
                break

        if not duplicate:
            unique_claims.append(claim)

    # -----------------------------------------
    # Remove claims contained inside longer claims
    # -----------------------------------------

    final_claims = []

    for i, claim in enumerate(unique_claims):

        claim_words = set(
            re.findall(
                r"\b[a-zA-Z0-9]+\b",
                claim.lower()
            )
        )

        if not claim_words:
            continue

        covered = False

        for j, other in enumerate(unique_claims):

            if i == j:
                continue

            other_words = set(
                re.findall(
                    r"\b[a-zA-Z0-9]+\b",
                    other.lower()
                )
            )

            if not other_words:
                continue

            # If almost all words of the current claim
            # are already present in a longer claim,
            # treat it as redundant.
            overlap = len(
                claim_words.intersection(other_words)
            ) / len(claim_words)

            if overlap >= 0.85 and len(other_words) > len(claim_words):

                covered = True
                break

        if not covered:
            final_claims.append(claim)

    return final_claims

def extract_list_items(response):
    """
    Extract useful top-level numbered/bullet claims.

    Nested bullets, headings, examples, and fragments are ignored.
    """

    lines = response.splitlines()

    items = []

    for line in lines:

        # Match ONLY top-level list items.
        # Nested items such as "  - Words..." are ignored.
        match = re.match(
            r"^(?:[-*•]|\d+[\.\)])\s+(.*)$",
            line
        )

        if not match:
            continue

        item = match.group(1).strip()

        if not item:
            continue

        # Clean markdown
        item = clean_claim(item)
        lower = item.lower()

        if lower.startswith((
            "let me know",
            "feel free to ask",
            "if you need help",
            "if you have any questions",
            "hope this helps"
        )):
            continue

        if not item:
            continue

        # Ignore heading-like list items
        if item.endswith(":"):
            continue

        # Ignore example/output labels
        if re.match(
            r"^(text|tokens|example|output|input)\s*:",
            item,
            flags=re.IGNORECASE
        ):
            continue

        # Ignore very short fragments
        words = re.findall(
            r"\b[a-zA-Z0-9]+\b",
            item
        )

        if len(words) < 5:
            continue

        items.append(item)

    return items


def validate_claims(response, claims):

    response_lower = response.lower()

    valid_claims = []

    response_words = set(
        re.findall(
            r"\b[a-zA-Z0-9]+\b",
            response_lower
        )
    )

    for claim in claims:

        claim_lower = claim.lower().strip()

        if not claim_lower:
            continue

        claim_words = set(
            re.findall(
                r"\b[a-zA-Z0-9]+\b",
                claim_lower
            )
        )

        if not claim_words:
            continue

        common_words = claim_words.intersection(
            response_words
        )

        overlap = len(common_words) / len(claim_words)

        if overlap >= 0.40:

            valid_claims.append(claim)

    return valid_claims


def extract_sentence_claims(response):
    """
    Extract candidate factual sentences from normal paragraphs.

    This acts as a backup for claims that the LLM may miss.
    It does not decide whether the statements are true.
    """

    claims = []

    # Process line by line
    for line in response.splitlines():

        line = line.strip()

        if not line:
            continue

        # Skip headings
        if line.startswith("#"):
            continue

        # Skip numbered/bullet list items.
        # They are handled by extract_list_items().
        if re.match(r"^\s*(?:[-*•]|\d+[\.\)])\s+", line):
            continue

        # Split normal paragraph text into sentences
        sentences = re.split(
            r"(?<=[.!?])\s+",
            line
        )

        for sentence in sentences:

            sentence = sentence.strip()

            if not sentence:
                continue

            # Ignore questions
            if sentence.endswith("?"):
                continue

            # Ignore common instruction/introductory sentences
            lower = sentence.lower()

            if lower.startswith((
                "let me know",
                "feel free to ask",
                "if you need help",
                "if you have any questions",
                "hope this helps"
            )):
                continue

                    # Ignore example/output lines
            if re.match(
                r"^(text|tokens|example|output|input)\s*:",
                sentence,
                flags=re.IGNORECASE
            ):
                continue

            # Ignore heading-like fragments
            if sentence.endswith(":"):
                continue

            ignored_starts = (
                "here is ",
                "here's ",
                "in simple terms",
                "for example",
                "please ",
                "note that",
                "remember that"
            )

            if lower.startswith(ignored_starts):
                continue

            # Ignore very short fragments
            words = re.findall(
                r"\b[a-zA-Z0-9]+\b",
                sentence
            )

            if len(words) < 4:
                continue

            sentence = clean_claim(sentence)

            if sentence:
                claims.append(sentence)

    return claims


def extract_claims(response):

    extraction_prompt = f"""
You are a factual claim extraction system.

Your ONLY task is to extract factual claims from the ANSWER.

ANSWER:
{response}

STRICT RULES:

1. Extract ONLY information explicitly stated in the ANSWER.
2. NEVER use your own knowledge.
3. NEVER add information that is missing from the ANSWER.
4. Extract independently verifiable factual statements.
5. If one sentence contains multiple factual statements,
   split them into separate claims when appropriate.
6. Do NOT include headings.
7. Do NOT include introductory phrases.
8. Do NOT include labels such as Step 1 or Layer 1
   unless the label is part of the factual statement.
9. Do NOT include opinions, questions, greetings,
   instructions, or formatting text.
10. Do NOT repeat the same claim.
11. Each claim must be understandable on its own.
12. Every claim MUST be directly supported by text
    in the ANSWER.
13. Do NOT correct the ANSWER.
14. Do NOT complete missing information.
15. Do NOT determine whether a claim is true or false.
16. If there are no factual claims, return [].

IMPORTANT:

The ANSWER is the ONLY source.

Do not use outside knowledge.

Return ONLY a JSON array of strings.

Now extract the claims from the ANSWER.
"""

    raw_output = generate_8b(
        extraction_prompt,
        max_new_tokens=300,
        do_sample=False
    )

    raw_output = raw_output.strip()

    # Remove markdown JSON fences
    raw_output = re.sub(
        r"```json\s*",
        "",
        raw_output,
        flags=re.IGNORECASE
    )

    raw_output = re.sub(
        r"```\s*",
        "",
        raw_output
    )

    raw_output = raw_output.strip()

    start = raw_output.find("[")
    end = raw_output.rfind("]")

    llm_claims = []

    if start != -1 and end != -1:

        json_text = raw_output[start:end + 1]

        try:

            claims = json.loads(json_text)

            if isinstance(claims, list):

                llm_claims = [
                    claim.strip()
                    for claim in claims
                    if isinstance(claim, str)
                    and claim.strip()
                ]

        except json.JSONDecodeError:

            llm_claims = []

    # Clean LLM claims
    # Clean LLM claims
    llm_claims = [
        clean_claim(claim)
        for claim in llm_claims
    ]

    # Extract numbered/bullet list claims
    list_claims = extract_list_items(response)

    # Extract claims from normal paragraphs
    sentence_claims = extract_sentence_claims(response)

    # Combine all sources
    # Combine all extraction sources
    claims = llm_claims + list_claims + sentence_claims

    # Remove obvious conversational filler and non-factual commentary

    # Filter conversational filler and verification-process commentary.
    ignored_exact = {
        "you're absolutely correct",
        "you are absolutely correct",
        "hope this helps",
    }

    verification_commentary_starts = (
        "the verification process indicates",
        "the verification process shows",
        "the claim remains unverified",
        "the claim remains uncertain",
        "this claim remains unverified",
        "this claim remains uncertain",
        "this premise is false",
        "this premise is not supported",
        "the premise is false",
        "the premise is not supported",
        "in conclusion, the claim remains",
        "in summary, the claim remains",
    )

    filtered_claims = []

    for claim in claims:
        cleaned = claim.strip()
        normalized = re.sub(
            r"^[^\w]+", "", cleaned
        ).strip().casefold()
        normalized = re.sub(
            r"[.!?]+$", "", normalized
        ).strip()

        if not normalized:
            continue

        if normalized in ignored_exact:
            continue

        if normalized.startswith(verification_commentary_starts):
            continue

        if normalized.startswith((
            "this is based on well-established scientific principles",
            "this is based on established scientific principles",
        )):
            continue

        filtered_claims.append(cleaned)

    claims = filtered_claims

    # Remove duplicates
    claims = remove_duplicates(claims)

    # Make sure claims actually occur in the answer
    claims = validate_claims(
        response,
        claims
    )

    return claims


if __name__ == "__main__":

    print("\n" + "=" * 40)
    print("HALLUCIGUARD CLAIM EXTRACTOR")
    print("=" * 40)

    print("\nEnter the answer to analyze.")
    print("Type END on a new line when finished.\n")

    lines = []

    while True:

        line = input()

        if line.strip().upper() == "END":
            break

        lines.append(line)

    response = "\n".join(lines).strip()

    if not response:

        print("\nNo answer provided.")

    else:

        claims = extract_claims(response)

        print("\n" + "=" * 40)
        print("EXTRACTED CLAIMS")
        print("=" * 40)

        if claims:

            for i, claim in enumerate(claims, 1):
                print(f"{i}. {claim}")

            print(f"\nTotal claims: {len(claims)}")

        else:

            print("No factual claims detected.")