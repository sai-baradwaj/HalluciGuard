"""
Embedding Utilities Module

Provides utility functions for inspecting,
validating, and managing generated embeddings.

Responsibilities:
    - Validate embedding dimensions
    - Validate embedding records
    - Check embedding values
    - Generate embedding statistics
    - Prepare embedding records for export

This module DOES NOT:
    - Generate embeddings
    - Load the embedding model
    - Perform retrieval
    - Save final embedding files
"""


import numpy as np


class EmbeddingUtils:

    # =====================================================
    # CONFIGURATION
    # =====================================================

    EXPECTED_DIMENSION = 384

    # =====================================================
    # VALIDATE SINGLE EMBEDDING
    # =====================================================

    @staticmethod
    def validate_embedding(embedding):
        """
        Validate a single embedding vector.

        Returns
        -------
        bool
            True if the embedding is valid.
        """

        if embedding is None:
            return False

        if len(embedding) != (
            EmbeddingUtils.EXPECTED_DIMENSION
        ):
            return False

        try:

            values = np.asarray(
                embedding,
                dtype=np.float32
            )

        except (TypeError, ValueError):

            return False

        if not np.all(
            np.isfinite(values)
        ):
            return False

        return True

    # =====================================================
    # VALIDATE EMBEDDING RECORD
    # =====================================================

    @staticmethod
    def validate_record(record):
        """
        Validate one complete embedding record.

        Required fields:
            chunk_id
            chunk_text
            embedding
        """

        if not isinstance(
            record,
            dict
        ):
            return False

        if "chunk_id" not in record:
            return False

        if "chunk_text" not in record:
            return False

        if "embedding" not in record:
            return False

        if not record["chunk_id"]:
            return False

        if not isinstance(
            record["chunk_text"],
            str
        ):
            return False

        if not record["chunk_text"].strip():
            return False

        if not EmbeddingUtils.validate_embedding(
            record["embedding"]
        ):
            return False

        return True

    # =====================================================
    # VALIDATE ALL RECORDS
    # =====================================================

    @staticmethod
    def validate_records(records):
        """
        Validate a collection of embedding records.

        Returns
        -------
        dict
            Validation summary.
        """

        total_records = len(records)

        valid_records = 0

        invalid_records = 0

        invalid_indices = []

        for index, record in enumerate(
            records
        ):

            if EmbeddingUtils.validate_record(
                record
            ):

                valid_records += 1

            else:

                invalid_records += 1

                invalid_indices.append(
                    index
                )

        return {

            "total_records":
                total_records,

            "valid_records":
                valid_records,

            "invalid_records":
                invalid_records,

            "invalid_indices":
                invalid_indices,

            "is_valid":
                invalid_records == 0

        }

    # =====================================================
    # EMBEDDING STATISTICS
    # =====================================================

    @staticmethod
    def generate_statistics(
        records
    ):
        """
        Generate statistics about embeddings.
        """

        if not records:

            return {

                "total_embeddings": 0,

                "embedding_dimension":
                    EmbeddingUtils.EXPECTED_DIMENSION,

                "valid_embeddings": 0,

                "invalid_embeddings": 0

            }

        validation = (
            EmbeddingUtils.validate_records(
                records
            )
        )

        return {

            "total_embeddings":
                validation[
                    "total_records"
                ],

            "embedding_dimension":
                EmbeddingUtils.EXPECTED_DIMENSION,

            "valid_embeddings":
                validation[
                    "valid_records"
                ],

            "invalid_embeddings":
                validation[
                    "invalid_records"
                ]

        }

    # =====================================================
    # EXTRACT EMBEDDING MATRIX
    # =====================================================

    @staticmethod
    def get_embedding_matrix(
        records
    ):
        """
        Convert embedding records into
        a NumPy matrix.

        Returns
        -------
        numpy.ndarray

            Shape:

                (number_of_records, 384)
        """

        if not records:

            return np.empty(
                (
                    0,
                    EmbeddingUtils.EXPECTED_DIMENSION
                ),
                dtype=np.float32
            )

        embeddings = [

            record["embedding"]

            for record in records

        ]

        return np.asarray(
            embeddings,
            dtype=np.float32
        )

    # =====================================================
    # GET CHUNK IDs
    # =====================================================

    @staticmethod
    def get_chunk_ids(records):
        """
        Extract chunk IDs from embedding records.
        """

        return [

            record["chunk_id"]

            for record in records

        ]

    # =====================================================
    # GET CHUNK TEXT
    # =====================================================

    @staticmethod
    def get_chunk_texts(records):
        """
        Extract chunk text from embedding records.
        """

        return [

            record["chunk_text"]

            for record in records

        ]

    # =====================================================
    # PRINT STATISTICS
    # =====================================================

    @staticmethod
    def print_statistics(
        records
    ):
        """
        Print embedding statistics.
        """

        statistics = (
            EmbeddingUtils.generate_statistics(
                records
            )
        )

        print("\nEmbedding Statistics")
        print("-" * 40)

        print(
            f"Total Embeddings : "
            f"{statistics['total_embeddings']}"
        )

        print(
            f"Embedding Dimension : "
            f"{statistics['embedding_dimension']}"
        )

        print(
            f"Valid Embeddings : "
            f"{statistics['valid_embeddings']}"
        )

        print(
            f"Invalid Embeddings : "
            f"{statistics['invalid_embeddings']}"
        )

        if statistics[
            "invalid_embeddings"
        ] == 0:

            print(
                "Status : ✓ VALID"
            )

        else:

            print(
                "Status : ✗ INVALID"
            )


# =========================================================
# DIRECT TEST
# =========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("Testing Embedding Utilities")
    print("=" * 70)

    # -----------------------------------------------------
    # Create a small valid test embedding
    # -----------------------------------------------------

    test_embedding = (
        np.random.rand(
            EmbeddingUtils.EXPECTED_DIMENSION
        ).astype(
            np.float32
        )
    )

    test_record = {

        "chunk_id":
            "test_chunk_001",

        "chunk_text":
            "The capital of France is Paris.",

        "embedding":
            test_embedding.tolist()

    }

    # -----------------------------------------------------
    # Test Single Embedding
    # -----------------------------------------------------

    is_valid = (
        EmbeddingUtils.validate_embedding(
            test_embedding
        )
    )

    print(
        f"\nSingle Embedding Valid : "
        f"{is_valid}"
    )

    # -----------------------------------------------------
    # Test Record
    # -----------------------------------------------------

    record_valid = (
        EmbeddingUtils.validate_record(
            test_record
        )
    )

    print(
        f"Embedding Record Valid : "
        f"{record_valid}"
    )

    # -----------------------------------------------------
    # Test Statistics
    # -----------------------------------------------------

    test_records = [
        test_record
    ]

    EmbeddingUtils.print_statistics(
        test_records
    )

    # -----------------------------------------------------
    # Test Matrix
    # -----------------------------------------------------

    matrix = (
        EmbeddingUtils.get_embedding_matrix(
            test_records
        )
    )

    print(
        f"\nEmbedding Matrix Shape : "
        f"{matrix.shape}"
    )

    print("\nEmbedding Utilities Test Completed.")