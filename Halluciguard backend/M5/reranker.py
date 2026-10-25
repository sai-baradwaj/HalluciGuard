from FlagEmbedding import FlagReranker


# ==========================================
# LOAD RERANKER MODEL
# ==========================================

model_name = "BAAI/bge-reranker-v2-m3"

reranker = FlagReranker(
    model_name,
    use_fp16=True
)


# ==========================================
# RERANK EVIDENCE FOR ONE CLAIM
# ==========================================

def rerank_evidence(claim, evidence, top_k=3):

    if not evidence:
        return []

    # ------------------------------------------
    # Create claim-evidence pairs
    # ------------------------------------------

    pairs = []

    for item in evidence:

        title = item.get("title", "")
        snippet = item.get("snippet", "")

        text = f"{title}. {snippet}"

        pairs.append([
            claim,
            text
        ])

    # ------------------------------------------
    # Calculate relevance scores
    # ------------------------------------------

    scores = reranker.compute_score(
        pairs,
        normalize=True
    )

    # If there is only one pair,
    # FlagReranker may return a single float
    if isinstance(scores, float):
        scores = [scores]

    # ------------------------------------------
    # Attach scores to evidence
    # ------------------------------------------

    ranked_evidence = []

    for item, score in zip(
        evidence,
        scores
    ):

        ranked_item = item.copy()

        ranked_item["score"] = float(score)

        ranked_evidence.append(
            ranked_item
        )

    # ------------------------------------------
    # Sort highest score first
    # ------------------------------------------
    
    
    print(
    f"\n[Reranker] Claim: {claim}"
    )

    print(
        "[Reranker] Scores:",
        [
            (item["title"], round(item["score"], 4))
            for item in ranked_evidence
        ]
    )

    MIN_RELEVANCE_SCORE = 0.01

    filtered_evidence = [
        item
        for item in ranked_evidence
        if item["score"] >= MIN_RELEVANCE_SCORE
    ]

    print(
        f"[Reranker] Kept {len(filtered_evidence)} "
        f"of {len(ranked_evidence)} sources "
        f"(threshold={MIN_RELEVANCE_SCORE})"
    )

    return filtered_evidence[:top_k]



# ==========================================
# RERANK EVIDENCE FOR MULTIPLE CLAIMS
# ==========================================

def rerank_claims(
    evidence_results,
    top_k=3
):

    all_ranked = []

    for item in evidence_results:

        claim = item["claim"]
        evidence = item["evidence"]

        ranked = rerank_evidence(
            claim,
            evidence,
            top_k
        )

        all_ranked.append({
            "claim": claim,
            "evidence": ranked
        })

    return all_ranked


# ==========================================
# TEST
# ==========================================

if __name__ == "__main__":

    test_claim = (
        "The OSI model has seven layers."
    )

    test_evidence = [

        {
            "title": "OSI Model - Wikipedia",
            "url": "https://en.wikipedia.org/wiki/OSI_model",
            "snippet": (
                "The OSI reference model is "
                "a seven-layer model for network communication."
            )
        },

        {
            "title": "TCP Protocol",
            "url": "https://example.com/tcp",
            "snippet": (
                "TCP provides reliable and ordered "
                "delivery of data."
            )
        },

        {
            "title": "OSI Layers - GeeksforGeeks",
            "url": "https://www.geeksforgeeks.org/",
            "snippet": (
                "The OSI model is a conceptual framework "
                "that divides network communication into "
                "seven functional layers."
            )
        },

        {
            "title": "Python Programming",
            "url": "https://example.com/python",
            "snippet": (
                "Python is a programming language "
                "used for many applications."
            )
        }
    ]

    ranked = rerank_evidence(
        test_claim,
        test_evidence,
        top_k=3
    )

    print("\n")
    print("==========================================")
    print("RERANKED EVIDENCE")
    print("==========================================")

    print("\nCLAIM:")
    print(test_claim)

    for i, item in enumerate(
        ranked,
        start=1
    ):

        print("\n------------------------------------------")

        print(
            f"{i}. SCORE: "
            f"{item['score']:.4f}"
        )

        print(
            "TITLE:",
            item["title"]
        )

        print(
            "URL:",
            item["url"]
        )

        print(
            "SNIPPET:",
            item["snippet"]
        )