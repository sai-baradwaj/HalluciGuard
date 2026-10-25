from ddgs import DDGS
import re


# ==========================================
# CREATE SEARCH QUERY
# ==========================================

def create_search_query(claim):

    # Remove punctuation
    query = re.sub(
        r"[^\w\s]",
        " ",
        claim
    )

    # Convert multiple spaces into one
    query = re.sub(
        r"\s+",
        " ",
        query
    ).strip()

    words = query.split()

    # Words that usually add little value
    stop_words = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "this",
        "that",
        "it",
        "they",
        "them",
        "he",
        "she",
        "we",
        "you",
        "i",
        "to",
        "of",
        "in",
        "on",
        "for",
        "with",
        "from",
        "and",
        "or",
        "as",
        "at",
        "by",
        "into",
        "through",
        "how",
        "like"
    }

    important_words = [
        word
        for word in words
        if word.lower() not in stop_words
    ]

    # Keep a reasonable query length
    important_words = important_words[:10]

    return " ".join(important_words)


# ==========================================
# RUN DDGS SEARCH
# ==========================================

def run_search(query, max_results):
    results = DDGS().text(
        query,
        region="us-en",
        safesearch="moderate",
        max_results=max_results
    )

    evidence = []
    seen_urls = set()

    for result in results:
        title = str(result.get("title") or "").strip()
        url = str(result.get("href") or "").strip()
        snippet = str(result.get("body") or "").strip()

        # Reject results without actual evidence text.
        if not snippet:
            continue

        # Reject malformed or non-HTTP source URLs.
        from urllib.parse import urlparse

        try:
            parsed = urlparse(url)

            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.hostname
                or parsed.username is not None
                or parsed.password is not None
            ):
                continue

        except (ValueError, TypeError):
            continue

        # Remove duplicate source URLs.
        normalized_url = url.rstrip("/").casefold()

        if normalized_url in seen_urls:
            continue

        seen_urls.add(normalized_url)

        evidence.append({
            "title": title or "Untitled source",
            "url": url,
            "snippet": snippet
        })

    return evidence



# ==========================================
# SEARCH ONE CLAIM
# ==========================================

# def search_claim(claim, max_results=5):

#     try:
#         evidence = run_search(claim, max_results)

#         if evidence:
#             return evidence

#     except Exception:
#         pass

#     short_query = create_search_query(claim)

#     try:
#         evidence = run_search(short_query, max_results)
#         return evidence

#     except Exception:
#         return []



def search_claim(claim, max_results=5):
    claim = str(claim or "").strip()

    if not claim:
        return []

    short_query = create_search_query(claim)

    # Try the original claim, then a simplified query.
    queries = []
    for query in (claim, short_query):
        query = query.strip()
        if query and query.casefold() not in {
            existing.casefold() for existing in queries
        }:
            queries.append(query)

    for index, query in enumerate(queries):
        try:
            evidence = run_search(query, max_results)

            print(
                f"\n[EvidenceSearch] Query {index + 1}: {query}"
            )
            print(
                f"[EvidenceSearch] Results: {len(evidence)}"
            )

            for item in evidence:
                print(
                    f"[EvidenceSearch] {item.get('title', '')} | "
                    f"{item.get('snippet', '')[:180]}"
                )

            if evidence:
                return evidence
                                                                                           
        except Exception as exc:
            message = str(exc).lower()

            # No results is a normal search outcome, not a system error.
            if "no results found" in message:
                continue

            # Log genuine provider or network errors.
            print(
                f"[EvidenceSearch] Search attempt "
                f"{index + 1} failed: {exc}"
            )

    # No usable evidence was found by either query.
    return []


# ==========================================
# SEARCH ALL CLAIMS
# ==========================================

def search_claims(claims, max_results=5):

    all_evidence = []

    for claim in claims:

        evidence = search_claim(
            claim,
            max_results
        )

        all_evidence.append({
            "claim": claim,
            "evidence": evidence
        })

    return all_evidence


# ==========================================
# TEST
# ==========================================

if __name__ == "__main__":

    test_claims = [

        "The Earth revolves around the Sun.",

        "The Moon revolves around the Earth.",

        "The OSI model has seven layers.",

        "TCP provides reliable delivery of data."
    ]

    results = search_claims(
        test_claims,
        max_results=5
    )

    print("\n")
    print("==========================================")
    print("EVIDENCE SEARCH RESULTS")
    print("==========================================")

    for item in results:

        print("\n------------------------------------------")

        print("CLAIM:")
        print(item["claim"])

        print("\nSEARCH RESULTS:")

        if not item["evidence"]:

            print("No evidence found.")

        else:

            for i, evidence in enumerate(
                item["evidence"],
                start=1
            ):

                print(
                    f"\n{i}. {evidence['title']}"
                )

                print(
                    "URL:",
                    evidence["url"]
                )

                print(
                    "SNIPPET:",
                    evidence["snippet"]
                )