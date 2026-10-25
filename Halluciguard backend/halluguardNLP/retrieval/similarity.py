"""
Similarity Module

Responsible for calculating similarity between
embedding vectors.

Responsibilities:
    - Calculate cosine similarity
    - Compare one query embedding with multiple
      stored embeddings
    - Return similarity scores
    - Rank embeddings by similarity

This module DOES NOT:
    - Generate embeddings
    - Load the embedding model
    - Load embedding files
    - Perform text chunking
    - Perform retrieval itself
"""


import numpy as np


class SimilarityCalculator:

    # =====================================================
    # CONSTRUCTOR
    # =====================================================

    def __init__(self):

        print("\n")
        print("=" * 70)
        print("Similarity Calculator Initialized")
        print("=" * 70)

    # =====================================================
    # COSINE SIMILARITY
    # =====================================================

    def cosine_similarity(
        self,
        vector_a,
        vector_b
    ):
        """
        Calculate cosine similarity between two vectors.

        Cosine similarity measures how similar two
        vectors are based on their direction.

        Formula:

            cosine_similarity =
                (A · B) / (||A|| × ||B||)

        Parameters
        ----------
        vector_a : numpy.ndarray
            First embedding vector.

        vector_b : numpy.ndarray
            Second embedding vector.

        Returns
        -------
        float
            Cosine similarity score.
        """

        vector_a = np.asarray(
            vector_a,
            dtype=np.float32
        )

        vector_b = np.asarray(
            vector_b,
            dtype=np.float32
        )

        # -------------------------------------------------
        # Check dimensions
        # -------------------------------------------------

        if vector_a.ndim != 1:

            raise ValueError(
                "vector_a must be a 1-dimensional vector."
            )

        if vector_b.ndim != 1:

            raise ValueError(
                "vector_b must be a 1-dimensional vector."
            )

        if vector_a.shape != vector_b.shape:

            raise ValueError(
                "Both vectors must have the same dimensions."
            )

        # -------------------------------------------------
        # Calculate vector norms
        # -------------------------------------------------

        norm_a = np.linalg.norm(
            vector_a
        )

        norm_b = np.linalg.norm(
            vector_b
        )

        # -------------------------------------------------
        # Prevent division by zero
        # -------------------------------------------------

        if norm_a == 0 or norm_b == 0:

            raise ValueError(
                "Cosine similarity cannot be calculated "
                "for a zero vector."
            )

        # -------------------------------------------------
        # Calculate cosine similarity
        # -------------------------------------------------

        similarity = (

            np.dot(
                vector_a,
                vector_b
            )

            /

            (
                norm_a
                *
                norm_b
            )

        )

        return float(
            similarity
        )

    # =====================================================
    # COMPARE QUERY WITH EMBEDDING MATRIX
    # =====================================================

    def compare_with_matrix(
        self,
        query_embedding,
        embedding_matrix
    ):
        """
        Compare one query embedding against
        multiple stored embeddings.

        Parameters
        ----------
        query_embedding : numpy.ndarray
            Query embedding.

        embedding_matrix : numpy.ndarray
            Matrix containing stored embeddings.

            Shape:

                (number_of_embeddings, 384)

        Returns
        -------
        numpy.ndarray

            Similarity score for every stored embedding.
        """

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32
        )

        embedding_matrix = np.asarray(
            embedding_matrix,
            dtype=np.float32
        )

        # -------------------------------------------------
        # Validate query
        # -------------------------------------------------

        if query_embedding.ndim != 1:

            raise ValueError(
                "query_embedding must be "
                "a 1-dimensional vector."
            )

        # -------------------------------------------------
        # Validate matrix
        # -------------------------------------------------

        if embedding_matrix.ndim != 2:

            raise ValueError(
                "embedding_matrix must be "
                "a 2-dimensional matrix."
            )

        if (
            embedding_matrix.shape[1]
            != query_embedding.shape[0]
        ):

            raise ValueError(
                "Query embedding dimension must match "
                "embedding matrix dimension."
            )

        # -------------------------------------------------
        # Calculate query norm
        # -------------------------------------------------

        query_norm = np.linalg.norm(
            query_embedding
        )

        if query_norm == 0:

            raise ValueError(
                "Query embedding cannot be a zero vector."
            )

        # -------------------------------------------------
        # Calculate stored embedding norms
        # -------------------------------------------------

        matrix_norms = np.linalg.norm(
            embedding_matrix,
            axis=1
        )

        if np.any(
            matrix_norms == 0
        ):

            raise ValueError(
                "Embedding matrix contains "
                "a zero vector."
            )

        # -------------------------------------------------
        # Dot products
        # -------------------------------------------------

        dot_products = np.dot(
            embedding_matrix,
            query_embedding
        )

        # -------------------------------------------------
        # Cosine similarity
        # -------------------------------------------------

        similarities = (

            dot_products

            /

            (
                matrix_norms
                *
                query_norm
            )

        )

        return similarities.astype(
            np.float32
        )

    # =====================================================
    # GET TOP INDICES
    # =====================================================

    def get_top_indices(
        self,
        similarities,
        top_k=5
    ):
        """
        Return indices of the highest similarity scores.

        Parameters
        ----------
        similarities : numpy.ndarray
            Similarity scores.

        top_k : int
            Number of top results to return.

        Returns
        -------
        numpy.ndarray
            Indices sorted from highest similarity
            to lowest similarity.
        """

        similarities = np.asarray(
            similarities
        )

        if similarities.ndim != 1:

            raise ValueError(
                "similarities must be "
                "a 1-dimensional array."
            )

        if top_k <= 0:

            raise ValueError(
                "top_k must be greater than zero."
            )

        top_k = min(
            top_k,
            len(similarities)
        )

        # -------------------------------------------------
        # Sort from highest to lowest
        # -------------------------------------------------

        top_indices = np.argsort(
            similarities
        )[::-1][:top_k]

        return top_indices

    # =====================================================
    # GET TOP SCORES
    # =====================================================

    def get_top_scores(
        self,
        similarities,
        top_k=5
    ):
        """
        Return the highest similarity scores.

        Parameters
        ----------
        similarities : numpy.ndarray

        top_k : int

        Returns
        -------
        numpy.ndarray
            Highest similarity scores.
        """

        top_indices = self.get_top_indices(
            similarities,
            top_k
        )

        return similarities[
            top_indices
        ]


# =========================================================
# DIRECT TEST
# =========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("Testing Similarity Calculator")
    print("=" * 70)

    calculator = SimilarityCalculator()

    # -----------------------------------------------------
    # Create test vectors
    # -----------------------------------------------------

    vector_a = np.array(
        [1.0, 0.0, 0.0]
    )

    vector_b = np.array(
        [1.0, 0.0, 0.0]
    )

    vector_c = np.array(
        [0.0, 1.0, 0.0]
    )

    # -----------------------------------------------------
    # Test identical vectors
    # -----------------------------------------------------

    similarity_same = (
        calculator.cosine_similarity(
            vector_a,
            vector_b
        )
    )

    print(
        f"\nSimilarity of identical vectors: "
        f"{similarity_same:.4f}"
    )

    # -----------------------------------------------------
    # Test perpendicular vectors
    # -----------------------------------------------------

    similarity_different = (
        calculator.cosine_similarity(
            vector_a,
            vector_c
        )
    )

    print(
        f"Similarity of perpendicular vectors: "
        f"{similarity_different:.4f}"
    )

    # -----------------------------------------------------
    # Test matrix comparison
    # -----------------------------------------------------

    embedding_matrix = np.array([

        [1.0, 0.0, 0.0],

        [0.0, 1.0, 0.0],

        [1.0, 1.0, 0.0],

        [0.0, 0.0, 1.0]

    ])

    similarities = (
        calculator.compare_with_matrix(
            vector_a,
            embedding_matrix
        )
    )

    print(
        "\nSimilarity Scores:"
    )

    print(
        similarities
    )

    # -----------------------------------------------------
    # Test top results
    # -----------------------------------------------------

    top_indices = (
        calculator.get_top_indices(
            similarities,
            top_k=3
        )
    )

    print(
        "\nTop 3 Indices:"
    )

    print(
        top_indices
    )

    top_scores = (
        calculator.get_top_scores(
            similarities,
            top_k=3
        )
    )

    print(
        "\nTop 3 Scores:"
    )

    print(
        top_scores
    )

    # -----------------------------------------------------
    # Final verification
    # -----------------------------------------------------

    if (

        np.isclose(
            similarity_same,
            1.0
        )

        and

        np.isclose(
            similarity_different,
            0.0
        )

        and

        len(top_indices) == 3

    ):

        print(
            "\n✓ Similarity Test Passed"
        )

    else:

        print(
            "\n✗ Similarity Test Failed"
        )