from citation_builder import build_citations


response = "The largest planet in our solar system is Jupiter."

verification_results = [
    {
        "claim": "The largest planet in our solar system is Jupiter",
        "verification": "SUPPORTED",
        "evidence": [
            {
                "title": "NASA",
                "url": "https://www.nasa.gov/",
                "snippet": "Jupiter is the largest planet in our solar system."
            }
        ]
    }
]


result = build_citations(
    response,
    verification_results
)

print("\n==============================")
print("CITED RESPONSE")
print("==============================")
print(result)