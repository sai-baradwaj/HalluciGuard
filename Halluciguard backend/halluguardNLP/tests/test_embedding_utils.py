import numpy as np

from embeddings.embedding_utils import EmbeddingUtils


class TestEmbeddingUtils:

    # =====================================================
    # TEST SINGLE VALID EMBEDDING
    # =====================================================

    def test_validate_valid_embedding(self):

        embedding = np.ones(
            EmbeddingUtils.EXPECTED_DIMENSION,
            dtype=np.float32
        )

        result = EmbeddingUtils.validate_embedding(
            embedding
        )

        assert result is True

    # =====================================================
    # TEST INVALID EMBEDDING DIMENSION
    # =====================================================

    def test_validate_invalid_embedding_dimension(self):

        embedding = np.ones(
            100,
            dtype=np.float32
        )

        result = EmbeddingUtils.validate_embedding(
            embedding
        )

        assert result is False

    # =====================================================
    # TEST INVALID EMBEDDING VALUES
    # =====================================================

    def test_validate_invalid_embedding_values(self):

        embedding = np.ones(
            EmbeddingUtils.EXPECTED_DIMENSION,
            dtype=np.float32
        )

        embedding[0] = np.nan

        result = EmbeddingUtils.validate_embedding(
            embedding
        )

        assert result is False

    # =====================================================
    # TEST VALID RECORD
    # =====================================================

    def test_validate_valid_record(self):

        record = {
            "chunk_id": "chunk_001",

            "chunk_text":
                "The capital of France is Paris.",

            "embedding":
                np.ones(
                    EmbeddingUtils.EXPECTED_DIMENSION,
                    dtype=np.float32
                ).tolist()
        }

        result = EmbeddingUtils.validate_record(
            record
        )

        assert result is True

    # =====================================================
    # TEST INVALID RECORD
    # =====================================================

    def test_validate_invalid_record(self):

        record = {
            "chunk_id": "chunk_001",

            "chunk_text":
                "The capital of France is Paris."
        }

        result = EmbeddingUtils.validate_record(
            record
        )

        assert result is False

    # =====================================================
    # TEST RECORD COLLECTION VALIDATION
    # =====================================================

    def test_validate_records(self):

        valid_embedding = np.ones(
            EmbeddingUtils.EXPECTED_DIMENSION,
            dtype=np.float32
        ).tolist()

        records = [

            {
                "chunk_id": "chunk_001",

                "chunk_text":
                    "The capital of France is Paris.",

                "embedding":
                    valid_embedding
            },

            {
                "chunk_id": "chunk_002",

                "chunk_text":
                    "Python is a programming language.",

                "embedding":
                    valid_embedding
            }
        ]

        result = EmbeddingUtils.validate_records(
            records
        )

        assert result["total_records"] == 2

        assert result["valid_records"] == 2

        assert result["invalid_records"] == 0

        assert result["is_valid"] is True

    # =====================================================
    # TEST EMBEDDING STATISTICS
    # =====================================================

    def test_generate_statistics(self):

        embedding = np.ones(
            EmbeddingUtils.EXPECTED_DIMENSION,
            dtype=np.float32
        ).tolist()

        records = [

            {
                "chunk_id": "chunk_001",

                "chunk_text":
                    "First test chunk.",

                "embedding":
                    embedding
            },

            {
                "chunk_id": "chunk_002",

                "chunk_text":
                    "Second test chunk.",

                "embedding":
                    embedding
            }
        ]

        statistics = (
            EmbeddingUtils.generate_statistics(
                records
            )
        )

        assert statistics[
            "total_embeddings"
        ] == 2

        assert statistics[
            "embedding_dimension"
        ] == 384

        assert statistics[
            "valid_embeddings"
        ] == 2

        assert statistics[
            "invalid_embeddings"
        ] == 0

    # =====================================================
    # TEST EMPTY STATISTICS
    # =====================================================

    def test_empty_statistics(self):

        statistics = (
            EmbeddingUtils.generate_statistics(
                []
            )
        )

        assert statistics[
            "total_embeddings"
        ] == 0

        assert statistics[
            "embedding_dimension"
        ] == 384

        assert statistics[
            "valid_embeddings"
        ] == 0

        assert statistics[
            "invalid_embeddings"
        ] == 0

    # =====================================================
    # TEST EMBEDDING MATRIX
    # =====================================================

    def test_get_embedding_matrix(self):

        embedding = np.ones(
            EmbeddingUtils.EXPECTED_DIMENSION,
            dtype=np.float32
        ).tolist()

        records = [

            {
                "chunk_id": "chunk_001",

                "chunk_text":
                    "First chunk.",

                "embedding":
                    embedding
            },

            {
                "chunk_id": "chunk_002",

                "chunk_text":
                    "Second chunk.",

                "embedding":
                    embedding
            }
        ]

        matrix = (
            EmbeddingUtils.get_embedding_matrix(
                records
            )
        )

        assert isinstance(
            matrix,
            np.ndarray
        )

        assert matrix.shape == (
            2,
            384
        )

        assert matrix.dtype == np.float32

    # =====================================================
    # TEST CHUNK ID EXTRACTION
    # =====================================================

    def test_get_chunk_ids(self):

        records = [

            {
                "chunk_id": "chunk_001",

                "chunk_text":
                    "First chunk.",

                "embedding":
                    np.ones(
                        384,
                        dtype=np.float32
                    ).tolist()
            },

            {
                "chunk_id": "chunk_002",

                "chunk_text":
                    "Second chunk.",

                "embedding":
                    np.ones(
                        384,
                        dtype=np.float32
                    ).tolist()
            }
        ]

        chunk_ids = (
            EmbeddingUtils.get_chunk_ids(
                records
            )
        )

        assert chunk_ids == [
            "chunk_001",
            "chunk_002"
        ]

    # =====================================================
    # TEST CHUNK TEXT EXTRACTION
    # =====================================================

    def test_get_chunk_texts(self):

        records = [

            {
                "chunk_id": "chunk_001",

                "chunk_text":
                    "First chunk.",

                "embedding":
                    np.ones(
                        384,
                        dtype=np.float32
                    ).tolist()
            },

            {
                "chunk_id": "chunk_002",

                "chunk_text":
                    "Second chunk.",

                "embedding":
                    np.ones(
                        384,
                        dtype=np.float32
                    ).tolist()
            }
        ]

        chunk_texts = (
            EmbeddingUtils.get_chunk_texts(
                records
            )
        )

        assert chunk_texts == [
            "First chunk.",
            "Second chunk."
        ]