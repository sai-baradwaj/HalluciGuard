"""
HalluciGuard Module 3
Final Embedding Verification

Independently verifies the exported embedding files.

Checks:
    1. All expected JSON and NPY files exist
    2. JSON record count matches NPY row count
    3. Embedding dimension is 384
    4. Required fields are present
    5. JSON embeddings contain 384 values
    6. Chunk IDs are present
    7. Total records equal 74,347
    8. Total NumPy embeddings equal 74,347
"""

import json
import os
import numpy as np


# =====================================================
# CONFIGURATION
# =====================================================

OUTPUT_DIRECTORY = "outputs/embeddings/full"

DATASETS = {
    "dialogue_data": 10116,
    "general_data": 4618,
    "qa_data": 10079,
    "summarization_data": 48735,
    "TruthfulQA": 799
}

EXPECTED_TOTAL = 74347

EXPECTED_DIMENSION = 384


# =====================================================
# VERIFICATION
# =====================================================

print("\n")
print("=" * 70)
print("HalluciGuard Final Embedding Verification")
print("=" * 70)

total_json_records = 0
total_numpy_embeddings = 0

all_passed = True


# =====================================================
# VERIFY EACH DATASET
# =====================================================

for dataset_name, expected_count in DATASETS.items():

    print("\n")
    print("-" * 70)
    print(f"VERIFYING DATASET : {dataset_name}")
    print("-" * 70)

    json_path = os.path.join(
        OUTPUT_DIRECTORY,
        f"{dataset_name}.json"
    )

    numpy_path = os.path.join(
        OUTPUT_DIRECTORY,
        f"{dataset_name}.npy"
    )

    # =================================================
    # FILE EXISTENCE
    # =================================================

    if not os.path.exists(json_path):

        print(
            f"✗ Missing JSON file : {json_path}"
        )

        all_passed = False
        continue

    print("✓ JSON file found")

    if not os.path.exists(numpy_path):

        print(
            f"✗ Missing NumPy file : {numpy_path}"
        )

        all_passed = False
        continue

    print("✓ NumPy file found")

    # =================================================
    # LOAD JSON
    # =================================================

    try:

        with open(
            json_path,
            "r",
            encoding="utf-8"
        ) as file:

            records = json.load(file)

    except Exception as error:

        print(
            f"✗ Failed to load JSON : {error}"
        )

        all_passed = False
        continue

    # =================================================
    # LOAD NUMPY
    # =================================================

    try:

        embeddings = np.load(
            numpy_path
        )

    except Exception as error:

        print(
            f"✗ Failed to load NumPy : {error}"
        )

        all_passed = False
        continue

    # =================================================
    # DISPLAY COUNTS
    # =================================================

    json_count = len(records)

    numpy_count = embeddings.shape[0]

    print(
        f"\nExpected Chunks : "
        f"{expected_count:,}"
    )

    print(
        f"JSON Records : "
        f"{json_count:,}"
    )

    print(
        f"NumPy Embeddings : "
        f"{numpy_count:,}"
    )

    print(
        f"NumPy Shape : "
        f"{embeddings.shape}"
    )

    # =================================================
    # CHECK EXPECTED RECORD COUNT
    # =================================================

    if json_count == expected_count:

        print(
            "✓ JSON Record Count Verified"
        )

    else:

        print(
            "✗ JSON Record Count Mismatch"
        )

        print(
            f"  Expected : {expected_count:,}"
        )

        print(
            f"  Found    : {json_count:,}"
        )

        all_passed = False

    # =================================================
    # CHECK NUMPY COUNT
    # =================================================

    if numpy_count == expected_count:

        print(
            "✓ NumPy Embedding Count Verified"
        )

    else:

        print(
            "✗ NumPy Embedding Count Mismatch"
        )

        print(
            f"  Expected : {expected_count:,}"
        )

        print(
            f"  Found    : {numpy_count:,}"
        )

        all_passed = False

    # =================================================
    # JSON COUNT = NUMPY COUNT
    # =================================================

    if json_count == numpy_count:

        print(
            "✓ JSON Count = NumPy Count"
        )

    else:

        print(
            "✗ JSON Count != NumPy Count"
        )

        all_passed = False

    # =================================================
    # CHECK EMBEDDING DIMENSION
    # =================================================

    if (
        len(embeddings.shape) == 2
        and embeddings.shape[1]
        == EXPECTED_DIMENSION
    ):

        print(
            f"✓ Embedding Dimension = "
            f"{EXPECTED_DIMENSION}"
        )

    else:

        print(
            "✗ Embedding Dimension Mismatch"
        )

        print(
            f"  Expected : "
            f"(N, {EXPECTED_DIMENSION})"
        )

        print(
            f"  Found    : "
            f"{embeddings.shape}"
        )

        all_passed = False

    # =================================================
    # VERIFY JSON RECORD STRUCTURE
    # =================================================

    required_fields = {
        "chunk_id",
        "chunk_text",
        "embedding"
    }

    invalid_structure_count = 0

    invalid_dimension_count = 0

    missing_id_count = 0

    # -------------------------------------------------
    # Check every JSON record
    # -------------------------------------------------

    for record in records:

        # ---------------------------------------------
        # Required fields
        # ---------------------------------------------

        if not required_fields.issubset(
            record.keys()
        ):

            invalid_structure_count += 1

            continue

        # ---------------------------------------------
        # Chunk ID
        # ---------------------------------------------

        if (
            record["chunk_id"]
            is None
        ):

            missing_id_count += 1

        # ---------------------------------------------
        # Embedding dimension
        # ---------------------------------------------

        embedding = record[
            "embedding"
        ]

        if len(embedding) != EXPECTED_DIMENSION:

            invalid_dimension_count += 1

    # =================================================
    # REPORT STRUCTURE
    # =================================================

    if invalid_structure_count == 0:

        print(
            "✓ All JSON records contain "
            "required fields"
        )

    else:

        print(
            f"✗ Invalid JSON records : "
            f"{invalid_structure_count:,}"
        )

        all_passed = False

    # =================================================
    # REPORT CHUNK IDs
    # =================================================

    if missing_id_count == 0:

        print(
            "✓ All chunk IDs are present"
        )

    else:

        print(
            f"✗ Missing chunk IDs : "
            f"{missing_id_count:,}"
        )

        all_passed = False

    # =================================================
    # REPORT JSON EMBEDDING DIMENSION
    # =================================================

    if invalid_dimension_count == 0:

        print(
            f"✓ All JSON embeddings "
            f"have {EXPECTED_DIMENSION} dimensions"
        )

    else:

        print(
            f"✗ JSON embeddings with "
            f"incorrect dimension : "
            f"{invalid_dimension_count:,}"
        )

        all_passed = False

    # =================================================
    # ADD TO TOTALS
    # =================================================

    total_json_records += json_count

    total_numpy_embeddings += numpy_count


# =====================================================
# FINAL TOTAL VERIFICATION
# =====================================================

print("\n")
print("=" * 70)
print("FINAL VERIFICATION SUMMARY")
print("=" * 70)

print(
    f"\nTotal JSON Records : "
    f"{total_json_records:,}"
)

print(
    f"Total NumPy Embeddings : "
    f"{total_numpy_embeddings:,}"
)

print(
    f"Expected Total : "
    f"{EXPECTED_TOTAL:,}"
)

# =====================================================
# TOTAL JSON COUNT
# =====================================================

if total_json_records == EXPECTED_TOTAL:

    print(
        "✓ Total JSON Record Count Verified"
    )

else:

    print(
        "✗ Total JSON Record Count Mismatch"
    )

    all_passed = False


# =====================================================
# TOTAL NUMPY COUNT
# =====================================================

if total_numpy_embeddings == EXPECTED_TOTAL:

    print(
        "✓ Total NumPy Embedding Count Verified"
    )

else:

    print(
        "✗ Total NumPy Embedding Count Mismatch"
    )

    all_passed = False


# =====================================================
# TOTAL JSON = NUMPY
# =====================================================

if (
    total_json_records
    == total_numpy_embeddings
):

    print(
        "✓ Total JSON Count = "
        "Total NumPy Count"
    )

else:

    print(
        "✗ Total JSON Count != "
        "Total NumPy Count"
    )

    all_passed = False


# =====================================================
# FINAL RESULT
# =====================================================

print("\n")

if all_passed:

    print(
        "=" * 70
    )

    print(
        "✓✓✓ ALL EMBEDDING FILES "
        "VERIFIED SUCCESSFULLY ✓✓✓"
    )

    print(
        "=" * 70
    )

    print(
        "\nModule 3 embedding output "
        "is ready for the next stage."
    )

else:

    print(
        "=" * 70
    )

    print(
        "✗✗✗ EMBEDDING VERIFICATION FAILED ✗✗✗"
    )

    print(
        "=" * 70
    )

    print(
        "\nDo NOT proceed until the "
        "verification issues are resolved."
    )