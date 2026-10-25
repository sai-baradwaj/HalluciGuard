"""
Retriever Module

Responsible for retrieving the most relevant chunks
for a given query.

Responsibilities:
    - Accept query embeddings
    - Compare query embedding with stored embeddings
    - Calculate similarity scores
    - Select top-K relevant chunks
    - Return retrieved chunks with scores

This module DOES NOT:
    - Generate embeddings
    - Create chunks
    - Train models
    - Calculate embeddings itself
"""

import numpy as np

from retrieval.similarity import SimilarityCalculator


class Retriever:

    # =====================================================
    # CONSTRUCTOR
    # =====================================================

    def __init__(
        self,
        embeddings,
        records
    ):
        """
        Initialize the retriever.

        Parameters
        ----------
        embeddings : numpy.ndarray
            Matrix containing stored chunk embeddings.

            Shape:
                (number_of_chunks, 384)

        records : list[dict]
            Chunk records corresponding to the embeddings.

        The order of records must match the order
        of embeddings.
        """

        self.embeddings = np.asarray(
            embeddings,
            dtype=np.float32
        )

        self.records = records

        self.similarity_calculator = (
            SimilarityCalculator()
        )

        # -------------------------------------------------
        # Validate embeddings
        # -------------------------------------------------

        if self.embeddings.ndim != 2:

            raise ValueError(
                "embeddings must be a "
                "2-dimensional matrix."
            )

        # -------------------------------------------------
        # Validate record count
        # -------------------------------------------------

        if len(self.records) != len(
            self.embeddings
        ):

            raise ValueError(
                "Number of records must match "
                "number of embeddings."
            )

        # -------------------------------------------------
        # Validate embedding matrix
        # -------------------------------------------------

        if len(self.embeddings) == 0:

            raise ValueError(
                "Embedding matrix cannot be empty."
            )

        print("\n")
        print("=" * 70)
        print("Retriever Initialized")
        print("=" * 70)

        print(
            f"\nNumber of Records : "
            f"{len(self.records)}"
        )

        print(
            f"Embedding Dimension : "
            f"{self.embeddings.shape[1]}"
        )

    # =====================================================
    # GET SIMILARITY SCORES
    # =====================================================

    def get_similarity_scores(
        self,
        query_embedding
    ):
        """
        Calculate similarity between a query embedding
        and all stored chunk embeddings.

        Parameters
        ----------
        query_embedding : numpy.ndarray
            Embedding representing the query.

        Returns
        -------
        numpy.ndarray
            Similarity score for every stored chunk.
        """

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32
        )

        if query_embedding.ndim != 1:

            raise ValueError(
                "query_embedding must be "
                "a 1-dimensional vector."
            )

        if (
            query_embedding.shape[0]
            != self.embeddings.shape[1]
        ):

            raise ValueError(
                "Query embedding dimension must "
                "match stored embedding dimension."
            )

        similarities = (
            self.similarity_calculator
            .compare_with_matrix(
                query_embedding,
                self.embeddings
            )
        )

        return similarities

    # =====================================================
    # RETRIEVE TOP-K RESULTS
    # =====================================================

    def retrieve(
        self,
        query_embedding,
        top_k=5
    ):
        """
        Retrieve the most relevant chunks.

        Parameters
        ----------
        query_embedding : numpy.ndarray
            Query embedding.

        top_k : int
            Number of results to retrieve.

        Returns
        -------
        list[dict]

            Each result contains the original
            chunk information plus:

                "similarity_score"
                "rank"
        """

        if top_k <= 0:

            raise ValueError(
                "top_k must be greater than zero."
            )

        # -------------------------------------------------
        # Calculate similarity scores
        # -------------------------------------------------

        similarities = (
            self.get_similarity_scores(
                query_embedding
            )
        )

        # -------------------------------------------------
        # Get highest-scoring indices
        # -------------------------------------------------

        top_indices = (
            self.similarity_calculator
            .get_top_indices(
                similarities,
                top_k
            )
        )

        # -------------------------------------------------
        # Build retrieval results
        # -------------------------------------------------

        results = []

        for rank, index in enumerate(
            top_indices,
            start=1
        ):

            record = dict(
                self.records[index]
            )

            record["similarity_score"] = (
                float(
                    similarities[index]
                )
            )

            record["rank"] = rank

            results.append(
                record
            )

        return results

    # =====================================================
    # RETRIEVE WITH MINIMUM SCORE
    # =====================================================

    def retrieve_with_threshold(
        self,
        query_embedding,
        top_k=5,
        min_score=0.0
    ):
        """
        Retrieve top-K chunks that also satisfy
        a minimum similarity score.

        Parameters
        ----------
        query_embedding : numpy.ndarray

        top_k : int

        min_score : float
            Minimum acceptable similarity score.

        Returns
        -------
        list[dict]
        """

        if top_k <= 0:

            raise ValueError(
                "top_k must be greater than zero."
            )

        similarities = (
            self.get_similarity_scores(
                query_embedding
            )
        )

        top_indices = (
            self.similarity_calculator
            .get_top_indices(
                similarities,
                top_k
            )
        )

        results = []

        for rank, index in enumerate(
            top_indices,
            start=1
        ):

            score = float(
                similarities[index]
            )

            if score < min_score:

                continue

            record = dict(
                self.records[index]
            )

            record["similarity_score"] = score

            record["rank"] = len(results) + 1

            results.append(
                record
            )

        return results

    # =====================================================
    # GET SINGLE BEST RESULT
    # =====================================================

    def retrieve_best(
        self,
        query_embedding
    ):
        """
        Retrieve the single most relevant chunk.

        Parameters
        ----------
        query_embedding : numpy.ndarray

        Returns
        -------
        dict
            Best matching chunk.
        """

        results = self.retrieve(
            query_embedding,
            top_k=1
        )

        return results[0]


# =========================================================
# DIRECT TEST
# =========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("Testing Retriever")
    print("=" * 70)

    # -----------------------------------------------------
    # Create sample embeddings
    # -----------------------------------------------------

    embeddings = np.array([

        [1.0, 0.0, 0.0],

        [0.0, 1.0, 0.0],

        [1.0, 1.0, 0.0],

        [0.0, 0.0, 1.0]

    ], dtype=np.float32)

    # -----------------------------------------------------
    # Create sample records
    # -----------------------------------------------------

    records = [

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

    # -----------------------------------------------------
    # Initialize retriever
    # -----------------------------------------------------

    retriever = Retriever(
        embeddings,
        records
    )

    # -----------------------------------------------------
    # Create query embedding
    # -----------------------------------------------------

    query_embedding = np.array([
        1.0,
        0.0,
        0.0
    ], dtype=np.float32)

    # -----------------------------------------------------
    # Retrieve top 3
    # -----------------------------------------------------

    results = retriever.retrieve(
        query_embedding,
        top_k=3
    )

    # -----------------------------------------------------
    # Display results
    # -----------------------------------------------------

    print("\n")
    print("=" * 70)
    print("Retrieval Results")
    print("=" * 70)

    for result in results:

        print(
            f"\nRank : "
            f"{result['rank']}"
        )

        print(
            f"Chunk ID : "
            f"{result['chunk_id']}"
        )

        print(
            f"Similarity : "
            f"{result['similarity_score']:.4f}"
        )

        print(
            f"Text : "
            f"{result['chunk_text']}"
        )

    # -----------------------------------------------------
    # Final verification
    # -----------------------------------------------------

    if (
        len(results) == 3
        and results[0]["chunk_id"] == 1
        and results[0]["similarity_score"] > 0.99
    ):

        print(
            "\n✓ Retriever Test Passed"
        )

    else:

        print(
            "\n✗ Retriever Test Failed"
        )