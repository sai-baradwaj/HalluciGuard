from data_preparation.loader import DatasetLoader
from data_preparation.inspector import DatasetInspector
from data_preparation.cleaner import DatasetCleaner
from data_preparation.validator import DatasetValidator
from data_preparation.normalizer import DatasetNormalizer

from chunking.tokenizer import DatasetTokenizer
from chunking.chunker import DatasetChunker
from chunking.chunk_utils import ChunkUtils


def main():

    print("=" * 70)
    print("HalluciGuard NLP Pipeline")
    print("=" * 70)

    # =====================================================
    # STEP 1 : Load Datasets
    # =====================================================

    loader = DatasetLoader()

    print("\nLoading HaluEval Dataset...\n")

    halueval = loader.load_halueval()

    print("Loaded HaluEval Files:")

    for dataset_name in halueval.keys():
        print(f"✓ {dataset_name}")

    print("\nLoading TruthfulQA Dataset...\n")

    truthfulqa = loader.load_truthfulqa()

    print("✓ TruthfulQA Loaded Successfully.")

    # =====================================================
    # STEP 2 : Inspect Datasets
    # =====================================================

    inspector = DatasetInspector()

    print("\n")
    print("=" * 70)
    print("Starting Dataset Inspection...")
    print("=" * 70)

    inspector.generate_report(
        halueval,
        truthfulqa
    )

    print("\n")
    print("=" * 70)
    print("Dataset Inspection Completed Successfully.")
    print("=" * 70)

    # =====================================================
    # STEP 3 : Clean Datasets
    # =====================================================

    cleaner = DatasetCleaner()

    print("\n")
    print("=" * 70)
    print("Cleaning Datasets...")
    print("=" * 70)

    cleaned_halueval, cleaned_truthfulqa = (
        cleaner.generate_clean_dataset(
            halueval,
            truthfulqa
        )
    )

    print("\nCleaning Completed Successfully.")

    print("\nCleaned TruthfulQA Dataset Shape:")
    print(cleaned_truthfulqa.shape)

    print("\nNumber of Cleaned HaluEval Datasets:")
    print(len(cleaned_halueval))

    print("\nCleaned HaluEval Files:")

    for dataset_name in cleaned_halueval.keys():
        print(f"✓ {dataset_name}")

    # =====================================================
    # STEP 4 : Validate Datasets
    # =====================================================

    validator = DatasetValidator()

    print("\n")
    print("=" * 70)
    print("Validating Datasets...")
    print("=" * 70)

    validated_halueval, validated_truthfulqa = (
        validator.generate_valid_dataset(
            cleaned_halueval,
            cleaned_truthfulqa
        )
    )

    print("\nValidation Completed Successfully.")

    print("\nValidated TruthfulQA Dataset Shape:")
    print(validated_truthfulqa.shape)

    print("\nNumber of Validated HaluEval Datasets:")
    print(len(validated_halueval))

    print("\nValidated HaluEval Files:")

    for dataset_name in validated_halueval.keys():
        print(f"✓ {dataset_name}")

    # =====================================================
    # STEP 5 : Normalize Datasets
    # =====================================================

    normalizer = DatasetNormalizer()

    print("\n")
    print("=" * 70)
    print("Normalizing Datasets...")
    print("=" * 70)

    normalized_halueval, normalized_truthfulqa = (
        normalizer.generate_normalized_dataset(
            validated_halueval,
            validated_truthfulqa
        )
    )

    print("\nNormalization Completed Successfully.")

    print("\nNormalized TruthfulQA Dataset Shape:")
    print(normalized_truthfulqa.shape)

    print("\nNumber of Normalized HaluEval Datasets:")
    print(len(normalized_halueval))

    print("\nNormalized HaluEval Files:")

    for dataset_name in normalized_halueval.keys():
        print(f"✓ {dataset_name}")

    # =====================================================
    # STEP 6 : Tokenization
    # =====================================================

    tokenizer = DatasetTokenizer()

    print("\n")
    print("=" * 70)
    print("Generating Token Statistics...")
    print("=" * 70)

    halueval_token_stats, truthfulqa_token_stats = (
        tokenizer.generate_token_statistics(
            normalized_halueval,
            normalized_truthfulqa
        )
    )

    print("\nTokenization Completed Successfully.")

    print("\nHaluEval Token Statistics Generated:")
    print(len(halueval_token_stats))

    print("\nTruthfulQA Token Statistics Generated:")
    print(len(truthfulqa_token_stats))

    # =====================================================
    # STEP 7 : Chunking
    # =====================================================

    chunker = DatasetChunker(
        tokenizer
    )

    print("\n")
    print("=" * 70)
    print("Starting Dataset Chunking...")
    print("=" * 70)

    chunked_halueval, chunked_truthfulqa = (
        chunker.generate_chunked_dataset(
            normalized_halueval,
            normalized_truthfulqa
        )
    )

    print("\nChunking Completed Successfully.")

    print("\nNumber of Chunked HaluEval Datasets:")
    print(len(chunked_halueval))

    print("\nChunked HaluEval Files:")

    for dataset_name in chunked_halueval.keys():
        print(f"✓ {dataset_name}")

    print("\nNumber of TruthfulQA Chunks:")
    print(len(chunked_truthfulqa))

    # =====================================================
    # STEP 8 : Chunk Utilities Validation
    # =====================================================

    chunk_utils = ChunkUtils()

    print("\n")
    print("=" * 70)
    print("Validating Saved Chunks...")
    print("=" * 70)

    dataset_names = [
        "dialogue_data",
        "general_data",
        "qa_data",
        "summarization_data",
        "TruthfulQA"
    ]

    validation_results = (
        chunk_utils.validate_all_datasets(
            dataset_names
        )
    )

    for dataset_name, result in validation_results.items():

        print(f"\n{dataset_name}")

        print(
            f"Total Chunks   : "
            f"{result['total_chunks']}"
        )

        print(
            f"Valid Chunks   : "
            f"{result['valid_chunks']}"
        )

        print(
            f"Invalid Chunks : "
            f"{result['invalid_chunks']}"
        )

        if result["is_valid"]:

            print(
                "Status         : ✓ VALID"
            )

        else:

            print(
                "Status         : ✗ INVALID"
            )

    # =====================================================
    # MODULE 2 COMPLETION
    # =====================================================

    print("\n")
    print("=" * 70)
    print("Module 2 - Tokenization & Chunking")
    print("Completed Successfully.")
    print("=" * 70)

    print("\n")
    print("=" * 70)
    print("HalluciGuard NLP Pipeline Completed Successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()