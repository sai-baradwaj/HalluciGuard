import json
import numpy as np
from pathlib import Path

from embeddings.embedding_model import EmbeddingModel


# =========================================================
# CONFIGURATION
# =========================================================

DATASETS = [
    "dialogue_data",
    "general_data",
    "qa_data",
    "summarization_data",
    "TruthfulQA"
]

EMBEDDING_DIR = Path("outputs/embeddings/full")

TOP_K = 5


# =========================================================
# LOAD ALL EMBEDDINGS
# =========================================================

print()
print("=" * 70)
print("SEMANTIC EMBEDDING QUALITY TEST")
print("=" * 70)

all_records = []
all_vectors = []

for dataset in DATASETS:

    json_file = EMBEDDING_DIR / f"{dataset}.json"
    npy_file = EMBEDDING_DIR / f"{dataset}.npy"

    print(f"\nLoading {dataset}...")

    with open(json_file, "r", encoding="utf-8") as f:
        records = json.load(f)

    vectors = np.load(npy_file)

    all_records.extend(records)
    all_vectors.append(vectors)

    print(f"Records : {len(records)}")
    print(f"Shape   : {vectors.shape}")


# Combine all embeddings

embedding_matrix = np.vstack(all_vectors)

print()
print("-" * 70)
print(f"TOTAL EMBEDDINGS LOADED : {len(all_records)}")
print(f"COMBINED MATRIX SHAPE   : {embedding_matrix.shape}")
print("-" * 70)


# =========================================================
# LOAD SAME BGE MODEL
# =========================================================

print("\nLoading BGE model...")

embedding_model = EmbeddingModel()

# IMPORTANT:
# EmbeddingModel is a wrapper.
# get_model() returns the actual SentenceTransformer.

model = embedding_model.get_model()

print("BGE model loaded successfully.")


# =========================================================
# TEST QUERIES
# =========================================================

queries = [
    "Who played Iron Man?",
    "What is the capital of France?",
    "How does photosynthesis work?",
    "What is machine learning?",
    "What causes earthquakes?"
]


# =========================================================
# RUN SEMANTIC SEARCH
# =========================================================

for query in queries:

    print()
    print("=" * 70)
    print(f"QUERY: {query}")
    print("=" * 70)

    # -----------------------------------------------------
    # Generate query embedding
    # -----------------------------------------------------

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]

    print(
        f"Query embedding dimension : "
        f"{len(query_embedding)}"
    )

    # -----------------------------------------------------
    # Cosine similarity
    #
    # Stored embeddings are already normalized.
    # Query embedding is normalized above.
    # Therefore dot product = cosine similarity.
    # -----------------------------------------------------

    scores = np.dot(
        embedding_matrix,
        query_embedding
    )

    # -----------------------------------------------------
    # Get Top-K
    # -----------------------------------------------------

    top_indices = np.argsort(scores)[::-1][:TOP_K]

    print("\nTOP RESULTS:\n")

    for rank, index in enumerate(
        top_indices,
        start=1
    ):

        record = all_records[index]

        print(
            f"{rank}. Similarity Score : "
            f"{scores[index]:.4f}"
        )

        print(
            f"   Dataset : "
            f"{record['dataset']}"
        )

        print(
            f"   Chunk   : "
            f"{record['chunk_uid']}"
        )

        print(
            f"   Text    : "
            f"{record['chunk_text'][:500]}"
        )

        print()


print("=" * 70)
print("SEMANTIC EMBEDDING TEST COMPLETED")
print("=" * 70)