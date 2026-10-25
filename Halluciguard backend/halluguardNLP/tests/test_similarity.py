import numpy as np
import pytest

from retrieval.similarity import SimilarityCalculator


class TestSimilarityCalculator:

    # =====================================================
    # SETUP
    # =====================================================

    def setup_method(self, method):

        self.calculator = SimilarityCalculator()

    # =====================================================
    # TEST IDENTICAL VECTORS
    # =====================================================

    def test_identical_vectors(self):

        vector_a = np.array(
            [1.0, 0.0, 0.0]
        )

        vector_b = np.array(
            [1.0, 0.0, 0.0]
        )

        similarity = (
            self.calculator.cosine_similarity(
                vector_a,
                vector_b
            )
        )

        assert np.isclose(
            similarity,
            1.0
        )

    # =====================================================
    # TEST PERPENDICULAR VECTORS
    # =====================================================

    def test_perpendicular_vectors(self):

        vector_a = np.array(
            [1.0, 0.0, 0.0]
        )

        vector_b = np.array(
            [0.0, 1.0, 0.0]
        )

        similarity = (
            self.calculator.cosine_similarity(
                vector_a,
                vector_b
            )
        )

        assert np.isclose(
            similarity,
            0.0
        )

    # =====================================================
    # TEST OPPOSITE VECTORS
    # =====================================================

    def test_opposite_vectors(self):

        vector_a = np.array(
            [1.0, 0.0, 0.0]
        )

        vector_b = np.array(
            [-1.0, 0.0, 0.0]
        )

        similarity = (
            self.calculator.cosine_similarity(
                vector_a,
                vector_b
            )
        )

        assert np.isclose(
            similarity,
            -1.0
        )

    # =====================================================
    # TEST SAME DIMENSION VECTORS
    # =====================================================

    def test_different_dimensions(self):

        vector_a = np.array(
            [1.0, 0.0, 0.0]
        )

        vector_b = np.array(
            [1.0, 0.0]
        )

        with pytest.raises(
            ValueError
        ):

            self.calculator.cosine_similarity(
                vector_a,
                vector_b
            )

    # =====================================================
    # TEST ZERO VECTOR
    # =====================================================

    def test_zero_vector(self):

        vector_a = np.array(
            [0.0, 0.0, 0.0]
        )

        vector_b = np.array(
            [1.0, 0.0, 0.0]
        )

        with pytest.raises(
            ValueError
        ):

            self.calculator.cosine_similarity(
                vector_a,
                vector_b
            )

    # =====================================================
    # TEST MATRIX COMPARISON
    # =====================================================

    def test_compare_with_matrix(self):

        query = np.array(
            [1.0, 0.0, 0.0]
        )

        matrix = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])

        similarities = (
            self.calculator.compare_with_matrix(
                query,
                matrix
            )
        )

        assert isinstance(
            similarities,
            np.ndarray
        )

        assert similarities.shape == (
            4,
        )

        assert np.isclose(
            similarities[0],
            1.0
        )

        assert np.isclose(
            similarities[1],
            0.0
        )

    # =====================================================
    # TEST TOP INDICES
    # =====================================================

    def test_get_top_indices(self):

        similarities = np.array([
            0.2,
            0.9,
            0.5,
            0.8,
            0.3
        ])

        top_indices = (
            self.calculator.get_top_indices(
                similarities,
                top_k=3
            )
        )

        assert list(
            top_indices
        ) == [
            1,
            3,
            2
        ]

    # =====================================================
    # TEST TOP SCORES
    # =====================================================

    def test_get_top_scores(self):

        similarities = np.array([
            0.2,
            0.9,
            0.5,
            0.8,
            0.3
        ])

        top_scores = (
            self.calculator.get_top_scores(
                similarities,
                top_k=3
            )
        )

        assert np.allclose(
            top_scores,
            [
                0.9,
                0.8,
                0.5
            ]
        )

    # =====================================================
    # TEST TOP K GREATER THAN ARRAY SIZE
    # =====================================================

    def test_top_k_larger_than_array(self):

        similarities = np.array([
            0.2,
            0.9,
            0.5
        ])

        top_indices = (
            self.calculator.get_top_indices(
                similarities,
                top_k=10
            )
        )

        assert len(
            top_indices
        ) == 3

    # =====================================================
    # TEST INVALID TOP K
    # =====================================================

    def test_invalid_top_k(self):

        similarities = np.array([
            0.2,
            0.9,
            0.5
        ])

        with pytest.raises(
            ValueError
        ):

            self.calculator.get_top_indices(
                similarities,
                top_k=0
            )