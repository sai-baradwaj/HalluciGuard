from repair_agent import repair_response


question = "What is the largest planet in our solar system?"

response = "The largest planet in our solar system is Mars."

failed_claims = [
    {
        "claim": "The largest planet in our solar system is Mars",
        "evidence": [
            {
                "title": "NASA - Jupiter",
                "snippet": "Jupiter is the largest planet in our solar system."
            }
        ]
    }
]


corrected = repair_response(
    question,
    response,
    failed_claims
)

print("\n==============================")
print("REPAIRED RESPONSE")
print("==============================")
print(corrected)