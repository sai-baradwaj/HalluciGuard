from fact_verifier import verify_claim


claim = "NLP is a subfield of artificial intelligence."

evidence = [
    {
        "title": "What is Natural Language Processing?",
        "snippet": "Natural language processing (NLP) is a subset of artificial intelligence that allows computers to understand human language."
    }
]

result = verify_claim(claim, evidence)

print("Claim:", claim)
print("Verification:", result)