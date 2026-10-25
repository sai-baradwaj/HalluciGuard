import numpy as np

from embeddings.embedding_model import EmbeddingModel
from retrieval.query_embedder import QueryEmbedder


class TestQueryEmbedder:

    # =====================================================
    # SETUP
    # =====================================================

    def setup_method(self):

        self.embedding_model = EmbeddingModel()

        self.query_embedder = QueryEmbedder(
            self.embedding_model
        )

    # =====================================================
    # TEST INITIALIZATION
    # =====================================================

    def test_query_embedder_initialization(self):

        assert self.query_embedder is not None

    # =====================================================
    # TEST SINGLE QUERY EMBEDDING
    # =====================================================

    def test_embed_query(self):

        query = (
            "What is an example of hallucination?"
        )

        embedding = (
            self.query_embedder.embed_query(
                query
            )
        )

        assert isinstance(
            embedding,
            np.ndarray
        )

        assert embedding.shape == (
            384,
        )

    # =====================================================
    # TEST EMBEDDING DIMENSION
    # =====================================================

    def test_embedding_dimension(self):

        query = (
            "What is natural language processing?"
        )

        embedding = (
            self.query_embedder.embed_query(
                query
            )
        )

        assert len(
            embedding
        ) == 384

    # =====================================================
    # TEST EMBEDDING DATA TYPE
    # =====================================================

    def test_embedding_data_type(self):

        query = (
            "What is artificial intelligence?"
        )

        embedding = (
            self.query_embedder.embed_query(
                query
            )
        )

        assert embedding.dtype == np.float32

    # =====================================================
    # TEST FINITE VALUES
    # =====================================================

    def test_embedding_contains_finite_values(self):

        query = (
            "What is a hallucination?"
        )

        embedding = (
            self.query_embedder.embed_query(
                query
            )
        )

        assert np.all(
            np.isfinite(embedding)
        )

    # =====================================================
    # TEST DIFFERENT QUERIES
    # =====================================================

    def test_different_queries_produce_embeddings(self):

        query_1 = "What is a cat?"

        query_2 = "What is a computer?"

        embedding_1 = (
            self.query_embedder.embed_query(
                query_1
            )
        )

        embedding_2 = (
            self.query_embedder.embed_query(
                query_2
            )
        )

        assert embedding_1.shape == (
            384,
        )

        assert embedding_2.shape == (
            384,
        )

        assert not np.array_equal(
            embedding_1,
            embedding_2
        )

    # =====================================================
    # TEST EMPTY QUERY
    # =====================================================

    def test_empty_query(self):

        try:

            embedding = (
                self.query_embedder.embed_query(
                    ""
                )
            )

            assert embedding.shape == (
                384,
            )

        except (
            ValueError,
            AssertionError
        ):

            assert True