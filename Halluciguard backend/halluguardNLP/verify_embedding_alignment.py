import json
import numpy as np
from pathlib import Path


DATASETS = [
    "dialogue_data",
    "general_data",
    "qa_data",
    "summarization_data",
    "TruthfulQA"
]

CHUNK_DIR = Path("outputs/chunked")
EMBEDDING_DIR = Path("outputs/embeddings/full")

all_ok = True


print()
print("=" * 70)
print("FINAL EMBEDDING VERIFICATION")
print("=" * 70)


for dataset in DATASETS:

    print()
    print("-" * 70)
    print(f"Checking: {dataset}")
    print("-" * 70)

    chunk_file = CHUNK_DIR / f"{dataset}_chunked.json"
    embedding_file = EMBEDDING_DIR / f"{dataset}.json"
    numpy_file = EMBEDDING_DIR / f"{dataset}.npy"

    # --------------------------------------------------
    # Load files
    # --------------------------------------------------

    with open(chunk_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    with open(embedding_file, "r", encoding="utf-8") as f:
        embeddings = json.load(f)

    matrix = np.load(numpy_file)

    # --------------------------------------------------
    # Basic counts
    # --------------------------------------------------

    count_match = len(chunks) == len(embeddings)

    # --------------------------------------------------
    # Check embedding dimension
    # --------------------------------------------------

    dimension_match = (
        len(embeddings) > 0
        and all(
            len(record["embedding"]) == 384
            for record in embeddings
        )
    )

    # --------------------------------------------------
    # Check NumPy shape
    # --------------------------------------------------

    numpy_shape_match = (
        matrix.shape == (len(embeddings), 384)
    )

    # --------------------------------------------------
    # Check chunk UID alignment
    # --------------------------------------------------

    uid_match = (
        count_match
        and all(
            chunk["chunk_uid"]
            == embedding["chunk_uid"]
            for chunk, embedding
            in zip(chunks, embeddings)
        )
    )

    # --------------------------------------------------
    # IMPORTANT:
    # Check actual chunk text alignment
    # --------------------------------------------------

    text_match = (
        count_match
        and all(
            chunk["chunk_text"]
            == embedding["chunk_text"]
            for chunk, embedding
            in zip(chunks, embeddings)
        )
    )

    # --------------------------------------------------
    # Print results
    # --------------------------------------------------

    print(f"Chunks                  : {len(chunks)}")
    print(f"Embedding records       : {len(embeddings)}")
    print(f"NumPy shape             : {matrix.shape}")
    print(f"Count match             : {count_match}")
    print(f"384 dimensions          : {dimension_match}")
    print(f"NumPy shape correct     : {numpy_shape_match}")
    print(f"Chunk UID match         : {uid_match}")
    print(f"Chunk text match        : {text_match}")

    dataset_ok = (
        count_match
        and dimension_match
        and numpy_shape_match
        and uid_match
        and text_match
    )

    if dataset_ok:
        print()
        print("✓ DATASET PASSED")
    else:
        print()
        print("✗ DATASET FAILED")
        all_ok = False


# ======================================================
# FINAL RESULT
# ======================================================

print()
print("=" * 70)
print("FINAL RESULT")
print("=" * 70)

if all_ok:
    print()
    print("✓ ALL EMBEDDINGS ARE CORRECT")
    print()
    print("✓ All chunk counts match")
    print("✓ All embedding counts match")
    print("✓ All embeddings are 384-dimensional")
    print("✓ All NumPy matrices have correct shape")
    print("✓ All chunk UIDs match")
    print("✓ All chunk text matches the embedded records")
else:
    print()
    print("✗ VERIFICATION FAILED")

print()
print("=" * 70)