import numpy as np

from embeddings.embedding_model import EmbeddingModel
from embeddings.embedder import DatasetEmbedder


class TestDatasetEmbedder:

    # =====================================================
    # SETUP
    # =====================================================

    def setup_method(self):

        self.embedding_model = EmbeddingModel()

        self.embedder = DatasetEmbedder(
            self.embedding_model
        )

    # =====================================================
    # TEST SINGLE TEXT EMBEDDING
    # =====================================================

    def test_single_text_embedding(self):

        texts = [
            "This is a test sentence."
        ]

        embeddings = self.embedder.embed_texts(
            texts,
            show_progress=False
        )

        assert embeddings is not None

        assert isinstance(
            embeddings,
            np.ndarray
        )

        assert embeddings.shape == (
            1,
            384
        )

    # =====================================================
    # TEST MULTIPLE TEXT EMBEDDINGS
    # =====================================================

    def test_multiple_text_embeddings(self):

        texts = [
            "The capital of France is Paris.",
            "Python is a programming language.",
            "Machine learning uses data."
        ]

        embeddings = self.embedder.embed_texts(
            texts,
            show_progress=False
        )

        assert embeddings is not None

        assert isinstance(
            embeddings,
            np.ndarray
        )

        assert embeddings.shape == (
            3,
            384
        )

    # =====================================================
    # TEST EMBEDDING DIMENSION
    # =====================================================

    def test_embedding_dimension(self):

        texts = [
            "HalluciGuard detects hallucinations."
        ]

        embeddings = self.embedder.embed_texts(
            texts,
            show_progress=False
        )

        assert embeddings.shape[1] == 384

    # =====================================================
    # TEST EMBEDDING COUNT
    # =====================================================

    def test_embedding_count_matches_input(self):

        texts = [
            "First test sentence.",
            "Second test sentence.",
            "Third test sentence.",
            "Fourth test sentence.",
            "Fifth test sentence."
        ]

        embeddings = self.embedder.embed_texts(
            texts,
            show_progress=False
        )

        assert len(embeddings) == len(texts)

    # =====================================================
    # TEST EMBEDDINGS ARE FINITE
    # =====================================================

    def test_embeddings_are_finite(self):

        texts = [
            "Test embedding validation."
        ]

        embeddings = self.embedder.embed_texts(
            texts,
            show_progress=False
        )

        assert np.all(
            np.isfinite(embeddings)
        )

    # =====================================================
    # TEST NORMALIZED EMBEDDINGS
    # =====================================================

    def test_embeddings_are_normalized(self):

        texts = [
            "Testing normalized embeddings."
        ]

        embeddings = self.embedder.embed_texts(
            texts,
            show_progress=False
        )

        norms = np.linalg.norm(
            embeddings,
            axis=1
        )

        assert np.allclose(
            norms,
            1.0,
            atol=1e-5
        )