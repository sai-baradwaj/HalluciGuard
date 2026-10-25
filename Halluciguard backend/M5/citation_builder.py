# import re


# def normalize_text(text):
#     """Remove Markdown formatting and normalize whitespace."""
#     text = re.sub(r"\*\*", "", text)
#     text = re.sub(r"\s+", " ", text)
#     return text.strip()


# def build_citations(response, verification_results):
#     """
#     Add citations to successfully verified claims.
#     """

#     cited_response = response

#     for item in verification_results:

#         if item["verification"] != "SUPPORTED":
#             continue

#         evidence = item.get("evidence", [])

#         if not evidence:
#             continue

#         source = evidence[0]

#         title = source.get("title", "").strip()
#         url = source.get("url", "").strip()

#         if not url:
#             continue

#         claim = item["claim"].strip()

#         citation = f" [{title}]({url})"

#         # --------------------------------
#         # Exact match
#         # --------------------------------

#         if claim in cited_response:
#             cited_response = cited_response.replace(
#                 claim,
#                 claim + citation,
#                 1
#             )
#             continue

#         # --------------------------------
#         # Normalized match
#         # --------------------------------

#         normalized_claim = normalize_text(claim)
#         normalized_response = normalize_text(cited_response)

#         if normalized_claim in normalized_response:

#             # Find the important words from the claim
#             words = normalized_claim.split()

#             # Use first several words to locate the sentence
#             search_text = " ".join(words[:8])

#             match = re.search(
#                 re.escape(search_text),
#                 cited_response,
#                 re.IGNORECASE
#             )

#             if match:

#                 end_position = match.end()

#                 cited_response = (
#                     cited_response[:end_position]
#                     + citation
#                     + cited_response[end_position:]
#                 )

#     return cited_response


import re
from urllib.parse import urlparse


def _verified_url(value):
    """Accept only absolute HTTP/HTTPS URLs."""
    value = str(value or "").strip()
    parsed = urlparse(value)

    if (
        parsed.scheme.lower() in {"http", "https"}
        and parsed.netloc
        and not parsed.username
        and not parsed.password
    ):
        return value

    return ""


def normalize_text(text):
    """Remove basic Markdown formatting and normalize whitespace."""
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", str(text or ""))
    text = re.sub(r"__(.*?)__", r"\1", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_citations(response, verification_results):
    """
    Add citations only to exact claim matches marked SUPPORTED.

    A SUPPORTED label alone does not prove that every source
    supports the claim. Evidence relevance must be checked by
    the verifier before the result reaches this function.
    """
    cited_response = str(response or "")
    citations_added = set()

    for item in verification_results or []:
        if str(item.get("verification", "")).strip().upper() != "SUPPORTED":
            continue

        claim = normalize_text(item.get("claim", ""))
        if not claim:
            continue

        # Select a URL-valid source. Relevance must already have
        # been assessed by the verification pipeline.
        source = None
        url = ""

        for candidate in item.get("evidence", []) or []:
            candidate_url = _verified_url(candidate.get("url", ""))
            if candidate_url:
                source = candidate
                url = candidate_url
                break

        if not source:
            continue

        title = str(source.get("title", "")).strip()
        if not title:
            title = urlparse(url).netloc

        citation_key = (claim.casefold(), url)
        if citation_key in citations_added:
            continue

        citation = f" [{title}]({url})"

        # Use exact text matching only. Do not guess the location
        # from a partial phrase, which can cite the wrong sentence.
        position = cited_response.find(claim)

        if position == -1:
            continue

        insert_at = position + len(claim)
        cited_response = (
            cited_response[:insert_at]
            + citation
            + cited_response[insert_at:]
        )

        citations_added.add(citation_key)

    return cited_response
