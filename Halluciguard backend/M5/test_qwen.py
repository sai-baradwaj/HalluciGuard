# from transformers import AutoTokenizer, AutoModelForCausalLM
# def generate_response(prompt):
#     model_name="Qwen/Qwen3-0.6B"
#     tokenizer=AutoTokenizer.from_pretrained(model_name)
#     model=AutoModelForCausalLM.from_pretrained(model_name,dtype="auto",device_map="auto")
#     # model_inputs=tokenizer(["The secret to baking a good cake is "],return_tensors="pt")
#     # print(model_inputs)
#     # print("----------------------------------------------------------------------------")
#     chat=[{"role":"user","content":prompt}]
#     # print(model_inputs["input_ids"])
#     # model_inputs=tokenizer.apply_chat_template(chat,continue_final_message=True,return_tensors="pt")
#     model_inputs=tokenizer.apply_chat_template(chat,add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors="pt").to(model.device)
#     generated_ids=model.generate(**model_inputs,max_new_tokens=200,do_sample=True,top_p=0.7,temperature=0.7)
#     print("----------------------------------------------------------------------------")
#     print(generated_ids)
#     print(tokenizer.batch_decode(generated_ids)[0])    

# prompt="The secret to baking a good cake is "
# generate_response(prompt)









# from transformers import AutoTokenizer, AutoModelForCausalLM

# # -----------------------------
# # Load model only once
# # -----------------------------
# model_name = "Qwen/Qwen3-0.6B"

# tokenizer = AutoTokenizer.from_pretrained(model_name)

# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype="auto",
#     device_map="auto"
# )


# # -----------------------------
# # Generate response
# # -----------------------------
# def generate_response(prompt):

#     messages = [
#         {
#             "role": "user",
#             "content": prompt
#         }
#     ]

#     model_inputs = tokenizer.apply_chat_template(
#         messages,
#         add_generation_prompt=True,
#         tokenize=True,
#         return_dict=True,
#         return_tensors="pt",
#         enable_thinking=False
#     ).to(model.device)

#     generated_ids = model.generate(
#         **model_inputs,
#         max_new_tokens=200,
#         do_sample=True,
#         temperature=0.7,
#         top_p=0.8,
#         top_k=20
#     )

#     # Remove the original prompt tokens
#     input_length = model_inputs["input_ids"].shape[-1]

#     generated_ids = generated_ids[0][input_length:]

#     response = tokenizer.decode(
#         generated_ids,
#         skip_special_tokens=True
#     )

#     return response.strip()


# def quality_evaluation(prompt, response):

#     scores = {}

#     # -------------------------
#     # 1. Relevance
#     # -------------------------
#     prompt_words = set(prompt.lower().split())
#     response_words = set(response.lower().split())

#     common_words = prompt_words.intersection(response_words)

#     if len(prompt_words) > 0:
#         relevance = len(common_words) / len(prompt_words) * 10
#     else:
#         relevance = 0

#     relevance = min(relevance, 10)

#     scores["relevance"] = relevance


#     # -------------------------
#     # 2. Completeness
#     # -------------------------
#     if len(response.strip()) == 0:
#         completeness = 0

#     elif len(response.split()) < 10:
#         completeness = 4

#     elif len(response.split()) < 30:
#         completeness = 7

#     else:
#         completeness = 10

#     scores["completeness"] = completeness


#     # -------------------------
#     # 3. Factuality
#     # -------------------------
#     # Basic version:
#     # We cannot truly verify facts yet.
#     # So give a neutral score.
#     factuality = 7

#     scores["factuality"] = factuality


#     # -------------------------
#     # 4. Clarity
#     # -------------------------
#     if len(response.strip()) == 0:
#         clarity = 0
#     elif len(response.split()) < 10:
#         clarity = 5
#     else:
#         clarity = 8

#     scores["clarity"] = clarity


#     # -------------------------
#     # 5. Confidence
#     # -------------------------
#     if len(response.strip()) == 0:
#         confidence = 0
#     else:
#         confidence = 7

#     scores["confidence"] = confidence


#     # -------------------------
#     # Overall Score
#     # -------------------------
#     overall_score = sum(scores.values()) / len(scores)

#     # -------------------------
#     # Quality Pass
#     # -------------------------
#     quality_pass = overall_score >= 6

#     return scores, overall_score, quality_pass


# # -----------------------------
# # Test
# # -----------------------------
# prompt = "The secret to baking a good cake is"

# response = generate_response(prompt)

# print("--------------------------------")
# print("INITIAL RESPONSE:")
# print(response)


# scores, overall_score, quality_pass = quality_evaluation(
#     prompt,
#     response
# )

# print("--------------------------------")
# print("QUALITY EVALUATION")

# print("Relevance:", scores["relevance"])
# print("Completeness:", scores["completeness"])
# print("Factuality:", scores["factuality"])
# print("Clarity:", scores["clarity"])
# print("Confidence:", scores["confidence"])

# print("--------------------------------")
# print("Overall Score:", overall_score)

# if quality_pass:
#     print("Quality Pass: YES")
# else:
#     print("Quality Pass: NO")
















# ==========================================
# LOAD MODEL
# ==========================================

from llm_32b import generate_32b
from llm_8b import generate_8b
from claim_extractor import extract_claims
from evidence_search import search_claims
from reranker import rerank_claims
from fact_verifier import verify_claim
from repair_agent import repair_response
from citation_builder import build_citations
import re

# # ==========================================
# # GENERATE RESPONSE
# # ==========================================

# def generate_response(
#     prompt,
#     max_new_tokens=500,
#     do_sample=True,
#     temperature=0.7,
#     top_p=0.8,
#     top_k=20
# ):

#     messages = [
#         {
#             "role": "user",
#             "content": prompt
#         }
#     ]

#     model_inputs = tokenizer.apply_chat_template(
#         messages,
#         add_generation_prompt=True,
#         tokenize=True,
#         return_dict=True,
#         return_tensors="pt",
#         enable_thinking=False
#     ).to(model.device)

#     generated_ids = model.generate(
#     **model_inputs,
#     max_new_tokens=max_new_tokens,
#     do_sample=do_sample,
#     temperature=temperature if do_sample else None,
#     top_p=top_p if do_sample else None,
#     top_k=top_k if do_sample else None
# )

#     input_length = model_inputs["input_ids"].shape[-1]

#     generated_ids = generated_ids[0][input_length:]

#     response = tokenizer.decode(
#         generated_ids,
#         skip_special_tokens=True
#     )

#     return response.strip()


# ==========================================
# QUALITY EVALUATION
# ==========================================
def evaluate_quality(query, response):

    evaluation_prompt = f"""
You are a strict quality evaluator.

Evaluate the following answer based ONLY on the user's query
and the provided answer.

USER QUERY:
{query}

ANSWER:
{response}

Give a score from 0 to 10 for each category.

1. Relevance
Does the answer directly answer the user's question?

2. Completeness
Does the answer include all important information required
by the question?

3. Factuality
Are the statements in the answer factually correct?

4. Clarity
Is the answer clear and easy to understand?

5. Confidence
Does the answer avoid unsupported, misleading,
contradictory, or suspicious claims?

IMPORTANT:

Be VERY strict.

Do not give 10 unless the answer is excellent.

Look carefully for:
- repeated information
- contradictory information
- duplicated list items
- incorrect numbering
- missing information
- unsupported claims
- obvious inconsistencies
- statements that conflict with other statements
- suspicious or hallucinated content

If the answer contains obvious structural problems,
reduce the score.

Return ONLY these five lines:

Relevance: <score>
Completeness: <score>
Factuality: <score>
Clarity: <score>
Confidence: <score>
"""

    evaluation = generate_8b(
      evaluation_prompt,
      max_new_tokens=100,
      do_sample=False
  )

    return evaluation.strip()




# ==========================================
# EXTRACT OVERALL SCORE
# ==========================================

def get_quality_score(evaluation, response):

    scores = {}

    # ------------------------------------------
    # Extract LLM scores
    # ------------------------------------------

    patterns = {
        "Relevance": r"Relevance:\s*(\d+(?:\.\d+)?)",
        "Completeness": r"Completeness:\s*(\d+(?:\.\d+)?)",
        "Factuality": r"Factuality:\s*(\d+(?:\.\d+)?)",
        "Clarity": r"Clarity:\s*(\d+(?:\.\d+)?)",
        "Confidence": r"Confidence:\s*(\d+(?:\.\d+)?)"
    }

    for category, pattern in patterns.items():

        match = re.search(
            pattern,
            evaluation,
            re.IGNORECASE
        )

        if match:

            scores[category] = float(
                match.group(1)
            )

        else:

            scores[category] = 0.0

    # ------------------------------------------
    # LLM average
    # ------------------------------------------

    llm_score = sum(
        scores.values()
    ) / len(scores)

    # ------------------------------------------
    # Deterministic structural score
    # ------------------------------------------

    structural_score, issues = (
        structural_quality_score(
            response
        )
    )

    # ------------------------------------------
    # Combine scores
    # ------------------------------------------

    final_score = (
        (llm_score * 0.7)
        +
        (structural_score * 0.3)
    )

    final_score = round(
        final_score,
        2
    )

    return scores, final_score, issues


# ==========================================
# MAIN PIPELINE
# ==========================================

# prompt = "Who invented the telephone?"
# prompt = "What is the largest planet in our solar system?"
# prompt = "Explain the OSI model in simple words."
# prompt = "A train travels 60 km in 1 hour. How far will it travel in 3 hours?"
# prompt = "What is the boiling point of water?"



import re


def check_numbering(response):
    """
    Detect suspicious numbering in structured answers.

    Example:
    1. Physical
    2. Data Link
    ...
    7. Application
    8. Transport
    9. Application

    This indicates duplicated / suspicious content.
    """

    numbers = re.findall(
        r"(?m)^\s*(\d+)[\.\)]\s+",
        response
    )

    numbers = [int(n) for n in numbers]

    if not numbers:
        return {
            "issue": False,
            "message": "No numbered list detected."
        }

    issues = []

    # Check whether numbering starts at 1
    if numbers[0] != 1:
        issues.append(
            "Numbered list does not start at 1."
        )

    # Check sequential numbering
    expected = list(
        range(1, len(numbers) + 1)
    )

    if numbers != expected:
        issues.append(
            f"Unexpected numbering detected: {numbers}"
        )

    # Check suspiciously large numbered list
    if len(numbers) > 7:
        issues.append(
            f"Answer contains {len(numbers)} numbered items."
        )

    if issues:
        return {
            "issue": True,
            "message": " ".join(issues)
        }

    return {
        "issue": False,
        "message": "Numbering looks consistent."
    }


# ==========================================
# DUPLICATE CONTENT CHECK
# ==========================================

def check_duplicate_content(response):
    """
    Detect repeated sentences or repeated list items.
    """

    lines = response.splitlines()

    cleaned_lines = []

    for line in lines:

        line = line.strip()

        # Remove numbering
        line = re.sub(
            r"^\d+[\.\)]\s*",
            "",
            line
        )

        # Remove markdown formatting
        line = re.sub(
            r"[*_]",
            "",
            line
        )

        line = line.strip()

    
        if line and line not in {"$$", "$", "```"}:
            cleaned_lines.append(line.lower())

    duplicates = []

    for i in range(len(cleaned_lines)):

        for j in range(i + 1, len(cleaned_lines)):

            if cleaned_lines[i] == cleaned_lines[j]:

                duplicates.append(
                    cleaned_lines[i]
                )

    if duplicates:

        return {
            "issue": True,
            "message": (
                "Duplicate content detected: "
                + "; ".join(duplicates[:3])
            )
        }

    return {
        "issue": False,
        "message": "No exact duplicate content detected."
    }


# ==========================================
# STRUCTURAL QUALITY SCORE
# ==========================================

def structural_quality_score(response):

    score = 10.0

    issues = []

    # ------------------------------------------
    # Numbering check
    # ------------------------------------------

    numbering_result = check_numbering(
        response
    )

    if numbering_result["issue"]:

        score -= 2

        issues.append(
            numbering_result["message"]
        )

    # ------------------------------------------
    # Duplicate check
    # ------------------------------------------

    duplicate_result = check_duplicate_content(
        response
    )

    if duplicate_result["issue"]:

        score -= 2

        issues.append(
            duplicate_result["message"]
        )

    # ------------------------------------------
    # Prevent negative score
    # ------------------------------------------

    score = max(
        0,
        min(10, score)
    )

    return score, issues

# ==========================================
# MAIN PIPELINE
# ==========================================

if __name__ == "__main__":

    print("\n================================")
    print("HALLUCIGUARD")
    print("================================")

    # --------------------------------------
    # USER INPUT
    # --------------------------------------

    prompt = input("\nEnter your question:\n> ").strip()

    if not prompt:
        print("\nNo question entered.")
        exit()

    # --------------------------------------
    # 1. Generate Initial Response
    # --------------------------------------

    response = generate_32b(prompt)
    print("\n================================")
    print("INITIAL RESPONSE")
    print("================================")
    print(response)

    # --------------------------------------
    # 2. Quality Evaluation
    # --------------------------------------

    evaluation = evaluate_quality(
        prompt,
        response
    )

    scores, score, issues = get_quality_score(
        evaluation,
        response
    )

    print("\n================================")
    print("QUALITY")
    print("================================")
    print(f"Score: {score:.2f}")

    if score >= 7:
        print("Pass: YES")
    else:
        print("Pass: NO")

    # --------------------------------------
    # 3. Claim Extraction
    # --------------------------------------

    claims = extract_claims(response)

    print("\n================================")
    print("CLAIMS")
    print("================================")

    if not claims:

        print("No factual claims found.")

    else:

        for i, claim in enumerate(claims, 1):
            print(f"{i}. {claim}")

    # --------------------------------------
    # 4. Evidence Search
    # --------------------------------------

    evidence_results = search_claims(
        claims,
        max_results=5
    )

    # --------------------------------------
    # 5. Reranking
    # --------------------------------------

    ranked_results = rerank_claims(
        evidence_results,
        top_k=3
    )

    print("\n================================")
    print("EVIDENCE")
    print("================================")

    
verification_results = []

for item in ranked_results:
    claim = item["claim"]
    evidence = item.get("evidence", [])

    if not evidence:
        verification = "UNCERTAIN"
        best_score = None
        supporting_evidence = []

    else:
        best_score = evidence[0].get("score")

        verification = verify_claim(claim, evidence)

        # Do not attach citations to uncertain or contradicted claims.
        supporting_evidence = (
            evidence if verification == "SUPPORTED" else []
        )

    verification_results.append({
        "claim": claim,
        "evidence": supporting_evidence,
        "reranker_score": best_score,
        "verification": verification
    })

    print(f"\nClaim: {claim}")
    print(f"Verification: {verification}")
    print(f"Evidence score: {best_score}")
    print(f"Evidence available for citation: {len(supporting_evidence)}")


    print("\n================================")
    print("VERIFICATION")
    print("================================")

    for i, item in enumerate(verification_results, 1):
        print(f"\n{i}. Claim: {item['claim']}")

        if item["reranker_score"] is not None:
            print(
                f"   Evidence score: "
                f"{item['reranker_score']:.4f}"
            )

            print(
                f"   Evidence title: "
                f"{item['evidence'][0].get('title', '')}"
            )

            print(
                f"   Evidence: "
                f"{item['evidence'][0].get('snippet', '')}"
            )

        else:
            print("   Evidence score: N/A")

        print(
            f"   Verification: "
            f"{item['verification']}"
        )


# ==========================================
# 6. REPAIR FAILED CLAIMS
# ==========================================

failed_claims = [
    item
    for item in verification_results
    if item["verification"] == "CONTRADICTED"
]

if failed_claims:

    print("\n================================")
    print("REPAIR")
    print("================================")

    repaired_response = repair_response(
        prompt,
        response,
        failed_claims
    )

    print(repaired_response)

else:

    repaired_response = response

    print("\n================================")
    print("REPAIR")
    print("================================")

    print("No repair required.")



# ==========================================
# 7. RE-VERIFICATION
# ==========================================

if failed_claims:

    print("\n================================")
    print("RE-VERIFICATION")
    print("================================")

    repaired_claims = extract_claims(repaired_response)

    repaired_evidence = search_claims(
        repaired_claims,
        max_results=5
    )

    repaired_ranked = rerank_claims(
        repaired_evidence,
        top_k=3
    )

    re_verification_results = []

    for item in repaired_ranked:

        claim = item["claim"]
        evidence = item["evidence"]

        if not evidence:
            verification = "UNCERTAIN"
            best_score = None

        else:
            best_score = evidence[0]["score"]
            verification = verify_claim(
                claim,
                evidence
            )

        re_verification_results.append({
            "claim": claim,
            "verification": verification,
            "evidence": evidence,
            "reranker_score": best_score
        })

    for i, item in enumerate(
        re_verification_results, 1
    ):

        print(
            f"{i}. Claim: {item['claim']}"
        )

        print(
            f"   Verification: "
            f"{item['verification']}"
        )

    final_response = repaired_response

else:

    final_response = response




# ================================
# CITATION BUILDER
# ================================

print("\n================================")
print("FINAL RESPONSE")
print("================================")

print("\nDEBUG VERIFICATION RESULTS")
print(verification_results)

final_response_with_citations = build_citations(
    final_response,
    re_verification_results if failed_claims else verification_results
)

print(final_response_with_citations)