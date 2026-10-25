import numpy as np
import pytest

from retrieval.retriever import Retriever


class TestRetriever:

    # =====================================================
    # SETUP
    # =====================================================

    def setup_method(self):

        self.embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        self.records = [
            {
                "chunk_id": 1,
                "chunk_text": "Information about cats."
            },
            {
                "chunk_id": 2,
                "chunk_text": "Information about dogs."
            },
            {
                "chunk_id": 3,
                "chunk_text": "Information about animals."
            },
            {
                "chunk_id": 4,
                "chunk_text": "Information about computers."
            }
        ]

        self.retriever = Retriever(
            self.embeddings,
            self.records
        )

    # =====================================================
    # TEST INITIALIZATION
    # =====================================================

    def test_retriever_initialization(self):

        assert len(
            self.retriever.records
        ) == 4

        assert self.retriever.embeddings.shape == (
            4,
            3
        )

    # =====================================================
    # TEST SIMILARITY SCORES
    # =====================================================

    def test_get_similarity_scores(self):

        query = np.array([
            1.0,
            0.0,
            0.0
        ], dtype=np.float32)

        scores = (
            self.retriever.get_similarity_scores(
                query
            )
        )

        assert isinstance(
            scores,
            np.ndarray
        )

        assert len(scores) == 4

        assert np.isclose(
            scores[0],
            1.0
        )

    # =====================================================
    # TEST TOP-K RETRIEVAL
    # =====================================================

    def test_retrieve_top_k(self):

        query = np.array([
            1.0,
            0.0,
            0.0
        ], dtype=np.float32)

        results = (
            self.retriever.retrieve(
                query,
                top_k=3
            )
        )

        assert len(results) == 3

        assert results[0]["chunk_id"] == 1

        assert results[0]["rank"] == 1

        assert results[0]["similarity_score"] > 0.99

    # =====================================================
    # TEST RESULT ORDER
    # =====================================================

    def test_results_are_sorted(self):

        query = np.array([
            1.0,
            0.0,
            0.0
        ], dtype=np.float32)

        results = (
            self.retriever.retrieve(
                query,
                top_k=4
            )
        )

        scores = [
            result["similarity_score"]
            for result in results
        ]

        assert scores == sorted(
            scores,
            reverse=True
        )

    # =====================================================
    # TEST RANKING
    # =====================================================

    def test_result_ranks(self):

        query = np.array([
            1.0,
            0.0,
            0.0
        ], dtype=np.float32)

        results = (
            self.retriever.retrieve(
                query,
                top_k=3
            )
        )

        ranks = [
            result["rank"]
            for result in results
        ]

        assert ranks == [1, 2, 3]

    # =====================================================
    # TEST TOP-K LARGER THAN DATASET
    # =====================================================

    def test_top_k_larger_than_dataset(self):

        query = np.array([
            1.0,
            0.0,
            0.0
        ], dtype=np.float32)

        results = (
            self.retriever.retrieve(
                query,
                top_k=10
            )
        )

        assert len(results) == 4

    # =====================================================
    # TEST BEST RESULT
    # =====================================================

    def test_retrieve_best(self):

        query = np.array([
            1.0,
            0.0,
            0.0
        ], dtype=np.float32)

        result = (
            self.retriever.retrieve_best(
                query
            )
        )

        assert isinstance(
            result,
            dict
        )

        assert result["chunk_id"] == 1

        assert result["rank"] == 1

        assert result["similarity_score"] > 0.99

    # =====================================================
    # TEST THRESHOLD
    # =====================================================

    def test_retrieve_with_threshold(self):

        query = np.array([
            1.0,
            0.0,
            0.0
        ], dtype=np.float32)

        results = (
            self.retriever
            .retrieve_with_threshold(
                query,
                top_k=4,
                min_score=0.5
            )
        )

        assert len(results) == 2

        for result in results:

            assert (
                result["similarity_score"]
                >= 0.5
            )

    # =====================================================
    # TEST INVALID TOP-K
    # =====================================================

    def test_invalid_top_k(self):

        query = np.array([
            1.0,
            0.0,
            0.0
        ], dtype=np.float32)

        with pytest.raises(
            ValueError
        ):

            self.retriever.retrieve(
                query,
                top_k=0
            )

    # =====================================================
    # TEST WRONG QUERY DIMENSION
    # =====================================================

    def test_wrong_query_dimension(self):

        query = np.array([
            1.0,
            0.0
        ], dtype=np.float32)

        with pytest.raises(
            ValueError
        ):

            self.retriever.retrieve(
                query,
                top_k=2
            )

    # =====================================================
    # TEST RECORD AND EMBEDDING COUNT MISMATCH
    # =====================================================

    def test_record_embedding_count_mismatch(self):

        wrong_records = self.records[:2]

        with pytest.raises(
            ValueError
        ):

            Retriever(
                self.embeddings,
                wrong_records
            )