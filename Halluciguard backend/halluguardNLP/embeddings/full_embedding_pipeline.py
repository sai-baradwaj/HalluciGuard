"""
Full Embedding Pipeline

Loads all chunked datasets, generates embeddings,
and exports JSON + NumPy files.

This module connects:
    ChunkUtils
        ↓
    DatasetEmbedder
        ↓
    EmbeddingExporter
"""

from chunking.chunk_utils import ChunkUtils
from embeddings.embedding_model import EmbeddingModel
from embeddings.embedder import DatasetEmbedder
from embeddings.exporter import EmbeddingExporter


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

OUTPUT_DIRECTORY = "outputs/embeddings/full"


# =========================================================
# MAIN PIPELINE
# =========================================================

def main():

    print("\n")
    print("=" * 70)
    print("FULL EMBEDDING PIPELINE")
    print("=" * 70)

    # -----------------------------------------------------
    # Initialize components
    # -----------------------------------------------------

    chunk_utils = ChunkUtils()

    embedding_model = EmbeddingModel()

    embedder = DatasetEmbedder(
        embedding_model
    )

    exporter = EmbeddingExporter(
        OUTPUT_DIRECTORY
    )

    # -----------------------------------------------------
    # Overall statistics
    # -----------------------------------------------------

    total_chunks_loaded = 0
    total_embeddings_generated = 0

    # -----------------------------------------------------
    # Process every dataset
    # -----------------------------------------------------

    for dataset_name in DATASETS:

        print("\n")
        print("=" * 70)
        print(
            f"PROCESSING DATASET : "
            f"{dataset_name}"
        )
        print("=" * 70)

        # -------------------------------------------------
        # Load chunks
        # -------------------------------------------------

        chunks = chunk_utils.load_dataset(
            dataset_name
        )

        print(
            f"\nChunks Loaded : "
            f"{len(chunks)}"
        )

        total_chunks_loaded += len(chunks)

        # -------------------------------------------------
        # Validate chunks
        # -------------------------------------------------

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
                f"Dataset '{dataset_name}' "
                f"contains invalid chunks."
            )

        # -------------------------------------------------
        # Generate embeddings
        # -------------------------------------------------

        print("\nGenerating embeddings...")

        embedding_records = (
            embedder.embed_chunks(
                chunks,
                show_progress=True
            )
        )

        print(
            f"\nEmbedding Records Generated : "
            f"{len(embedding_records)}"
        )

        total_embeddings_generated += (
            len(embedding_records)
        )

        # -------------------------------------------------
        # Verify embedding count
        # -------------------------------------------------

        if len(embedding_records) != len(chunks):

            raise ValueError(
                f"Embedding count mismatch "
                f"for {dataset_name}."
            )

        # -------------------------------------------------
        # Verify embedding dimensions
        # -------------------------------------------------

        for index, record in enumerate(
            embedding_records
        ):

            if len(
                record["embedding"]
            ) != embedder.embedding_dimension:

                raise ValueError(
                    f"Embedding dimension mismatch "
                    f"in {dataset_name}, "
                    f"record {index}."
                )

        # -------------------------------------------------
        # Export JSON + NPY
        # -------------------------------------------------

        exported_files = (
            exporter.export_embeddings(
                embedding_records,
                dataset_name
            )
        )

        # -------------------------------------------------
        # Verify exported JSON
        # -------------------------------------------------

        loaded_records = (
            exporter.load_json(
                f"{dataset_name}.json"
            )
        )

        # -------------------------------------------------
        # Verify exported NPY
        # -------------------------------------------------

        loaded_embeddings = (
            exporter.load_numpy(
                f"{dataset_name}.npy"
            )
        )

        # -------------------------------------------------
        # Final dataset verification
        # -------------------------------------------------

        json_count = len(
            loaded_records
        )

        numpy_rows = (
            loaded_embeddings.shape[0]
        )

        numpy_dimension = (
            loaded_embeddings.shape[1]
        )

        print("\n")
        print("-" * 70)
        print(
            f"Verification : "
            f"{dataset_name}"
        )
        print("-" * 70)

        print(
            f"Total Chunks Loaded       : "
            f"{len(chunks)}"
        )

        print(
            f"Embedding Records         : "
            f"{len(embedding_records)}"
        )

        print(
            f"JSON Records              : "
            f"{json_count}"
        )

        print(
            f"NumPy Rows                : "
            f"{numpy_rows}"
        )

        print(
            f"Embedding Dimension       : "
            f"{numpy_dimension}"
        )

        # -------------------------------------------------
        # Verify everything matches
        # -------------------------------------------------

        if not (
            len(chunks)
            == len(embedding_records)
            == json_count
            == numpy_rows
        ):

            raise ValueError(
                f"Count verification failed "
                f"for {dataset_name}."
            )

        if numpy_dimension != (
            embedder.embedding_dimension
        ):

            raise ValueError(
                f"Dimension verification failed "
                f"for {dataset_name}."
            )

        print(
            "\n✓ DATASET EMBEDDING VERIFIED"
        )

    # =====================================================
    # FINAL PIPELINE VERIFICATION
    # =====================================================

    print("\n")
    print("=" * 70)
    print("FINAL EMBEDDING PIPELINE VERIFICATION")
    print("=" * 70)

    print(
        f"\nTotal Chunks Loaded : "
        f"{total_chunks_loaded}"
    )

    print(
        f"Total Embeddings Generated : "
        f"{total_embeddings_generated}"
    )

    if (
        total_chunks_loaded
        == total_embeddings_generated
    ):

        print(
            "\n✓ FULL EMBEDDING PIPELINE "
            "COMPLETED SUCCESSFULLY"
        )

    else:

        raise ValueError(
            "\n✗ FINAL COUNT MISMATCH"
        )


# =========================================================
# RUN PIPELINE
# =========================================================

if __name__ == "__main__":

    main()