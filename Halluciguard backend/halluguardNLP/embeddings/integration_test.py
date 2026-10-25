"""
HalluciGuard Module 3 Integration Test

Tests the complete real-data embedding pipeline:

Module 2 Chunks
        ↓
Dataset Embedder
        ↓
Embedding Utilities
        ↓
Embedding Exporter
        ↓
Reload
        ↓
Verification

Only 10 real chunks are processed.
"""

from embeddings.embedding_model import EmbeddingModel
from embeddings.embedder import DatasetEmbedder
from embeddings.embedding_utils import EmbeddingUtils
from embeddings.exporter import EmbeddingExporter
from chunking.chunk_utils import ChunkUtils


def main():

    print("\n")
    print("=" * 70)
    print("HALLUCIGUARD MODULE 3 INTEGRATION TEST")
    print("=" * 70)

    # =====================================================
    # STEP 1 : LOAD REAL MODULE 2 CHUNKS
    # =====================================================

    print("\n")
    print("=" * 70)
    print("STEP 1 : Loading Real Module 2 Chunks")
    print("=" * 70)

    chunk_utils = ChunkUtils()

    chunks = chunk_utils.load_dataset(
        "dialogue_data"
    )

    print(
        f"\nTotal Saved Chunks : "
        f"{len(chunks)}"
    )

    # -----------------------------------------------------
    # Validate Module 2 chunks
    # -----------------------------------------------------

    validation = (
        chunk_utils.validate_chunks(
            chunks
        )
    )

    print(
        f"Valid Chunks : "
        f"{validation['valid_chunks']}"
    )

    print(
        f"Invalid Chunks : "
        f"{validation['invalid_chunks']}"
    )

    if not validation["is_valid"]:

        raise ValueError(
            "Module 2 chunks are invalid."
        )

    # -----------------------------------------------------
    # Select only 10 chunks
    # -----------------------------------------------------

    test_chunks = chunks[:10]

    print(
        f"\nChunks Selected For Test : "
        f"{len(test_chunks)}"
    )

    # =====================================================
    # STEP 2 : LOAD EMBEDDING MODEL
    # =====================================================

    print("\n")
    print("=" * 70)
    print("STEP 2 : Loading Embedding Model")
    print("=" * 70)

    embedding_model = EmbeddingModel()

    # =====================================================
    # STEP 3 : INITIALIZE EMBEDDER
    # =====================================================

    embedder = DatasetEmbedder(
        embedding_model
    )

    # =====================================================
    # STEP 4 : GENERATE EMBEDDINGS
    # =====================================================

    print("\n")
    print("=" * 70)
    print("STEP 3 : Generating Embeddings")
    print("=" * 70)

    embedding_records = (
        embedder.embed_chunks(
            test_chunks,
            show_progress=True
        )
    )

    print(
        f"\nEmbedding Records Generated : "
        f"{len(embedding_records)}"
    )

    # =====================================================
    # STEP 5 : VALIDATE EMBEDDINGS
    # =====================================================

    print("\n")
    print("=" * 70)
    print("STEP 4 : Validating Embeddings")
    print("=" * 70)

    validation_result = (
        EmbeddingUtils.validate_records(
            embedding_records
        )
    )

    print(
        f"\nTotal Records : "
        f"{validation_result['total_records']}"
    )

    print(
        f"Valid Records : "
        f"{validation_result['valid_records']}"
    )

    print(
        f"Invalid Records : "
        f"{validation_result['invalid_records']}"
    )

    if not validation_result["is_valid"]:

        raise ValueError(
            "Embedding validation failed."
        )

    print(
        "\n✓ Embedding Validation Passed"
    )

    # -----------------------------------------------------
    # Generate statistics
    # -----------------------------------------------------

    EmbeddingUtils.print_statistics(
        embedding_records
    )

    # -----------------------------------------------------
    # Check embedding matrix
    # -----------------------------------------------------

    embedding_matrix = (
        EmbeddingUtils.get_embedding_matrix(
            embedding_records
        )
    )

    print(
        f"\nEmbedding Matrix Shape : "
        f"{embedding_matrix.shape}"
    )

    if embedding_matrix.shape != (
        len(embedding_records),
        embedding_model.get_embedding_dimension()
    ):

        raise ValueError(
            "Embedding matrix dimension mismatch."
        )

    print(
        "✓ Embedding Matrix Verified"
    )

    # =====================================================
    # STEP 6 : EXPORT EMBEDDINGS
    # =====================================================

    print("\n")
    print("=" * 70)
    print("STEP 5 : Exporting Embeddings")
    print("=" * 70)

    exporter = EmbeddingExporter()

    output_name = (
        "integration_test"
    )

    exported_files = (
        exporter.export_embeddings(
            embedding_records,
            output_name
        )
    )

    print(
        f"\nJSON File : "
        f"{exported_files['json']}"
    )

    print(
        f"NumPy File : "
        f"{exported_files['numpy']}"
    )

    # =====================================================
    # STEP 7 : RELOAD EXPORTED DATA
    # =====================================================

    print("\n")
    print("=" * 70)
    print("STEP 6 : Reloading Exported Data")
    print("=" * 70)

    loaded_records = (
        exporter.load_json(
            "integration_test.json"
        )
    )

    loaded_embeddings = (
        exporter.load_numpy(
            "integration_test.npy"
        )
    )

    print(
        f"\nOriginal Records : "
        f"{len(embedding_records)}"
    )

    print(
        f"Loaded Records : "
        f"{len(loaded_records)}"
    )

    print(
        f"Loaded Embedding Shape : "
        f"{loaded_embeddings.shape}"
    )

    # =====================================================
    # STEP 8 : FINAL VERIFICATION
    # =====================================================

    print("\n")
    print("=" * 70)
    print("FINAL INTEGRATION VERIFICATION")
    print("=" * 70)

    # -----------------------------------------------------
    # Verify record count
    # -----------------------------------------------------

    records_match = (
        len(loaded_records)
        == len(embedding_records)
    )

    if records_match:

        print(
            "✓ Record Count Verified"
        )

    else:

        print(
            "✗ Record Count Mismatch"
        )

    # -----------------------------------------------------
    # Verify embedding count
    # -----------------------------------------------------

    embedding_count_match = (
        loaded_embeddings.shape[0]
        == len(embedding_records)
    )

    if embedding_count_match:

        print(
            "✓ Embedding Count Verified"
        )

    else:

        print(
            "✗ Embedding Count Mismatch"
        )

    # -----------------------------------------------------
    # Verify embedding dimension
    # -----------------------------------------------------

    embedding_dimension_match = (
        loaded_embeddings.shape[1]
        == embedding_model.get_embedding_dimension()
    )

    if embedding_dimension_match:

        print(
            "✓ Embedding Dimension Verified"
        )

    else:

        print(
            "✗ Embedding Dimension Mismatch"
        )

    # -----------------------------------------------------
    # Verify chunk IDs survived export
    # -----------------------------------------------------

    original_ids = (
        EmbeddingUtils.get_chunk_ids(
            embedding_records
        )
    )

    loaded_ids = (
        EmbeddingUtils.get_chunk_ids(
            loaded_records
        )
    )

    chunk_ids_match = (
        original_ids == loaded_ids
    )

    if chunk_ids_match:

        print(
            "✓ Chunk IDs Preserved"
        )

    else:

        print(
            "✗ Chunk IDs Changed"
        )

    # -----------------------------------------------------
    # Verify chunk text survived export
    # -----------------------------------------------------

    original_texts = (
        EmbeddingUtils.get_chunk_texts(
            embedding_records
        )
    )

    loaded_texts = (
        EmbeddingUtils.get_chunk_texts(
            loaded_records
        )
    )

    chunk_text_match = (
        original_texts == loaded_texts
    )

    if chunk_text_match:

        print(
            "✓ Chunk Text Preserved"
        )

    else:

        print(
            "✗ Chunk Text Changed"
        )

    # =====================================================
    # FINAL RESULT
    # =====================================================

    if (
        records_match
        and embedding_count_match
        and embedding_dimension_match
        and chunk_ids_match
        and chunk_text_match
    ):

        print("\n")
        print("=" * 70)
        print(
            "✓ MODULE 3 INTEGRATION TEST PASSED"
        )
        print("=" * 70)

        print(
            "\n10 real Module 2 chunks successfully "
            "completed the complete embedding pipeline."
        )

    else:

        print("\n")
        print("=" * 70)
        print(
            "✗ MODULE 3 INTEGRATION TEST FAILED"
        )
        print("=" * 70)


if __name__ == "__main__":

    main()