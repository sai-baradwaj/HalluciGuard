"""
Query Embedder Module

Responsible for converting a user's text query
into an embedding vector.

Embedding Model:
BAAI/bge-small-en-v1.5

Responsibilities:
    - Accept a text query
    - Generate its embedding
    - Return a 384-dimensional vector
    - Use the existing EmbeddingModel

This module DOES NOT:
    - Perform retrieval
    - Calculate similarity
    - Create chunks
    - Save embeddings
"""


import numpy as np

from embeddings.embedding_model import EmbeddingModel


class QueryEmbedder:

    # =====================================================
    # CONSTRUCTOR
    # =====================================================

    def __init__(self, embedding_model):

        self.embedding_model = embedding_model

        self.model = (
            embedding_model.get_model()
        )

        self.device = (
            embedding_model.get_device()
        )

        self.embedding_dimension = (
            embedding_model.get_embedding_dimension()
        )

        print("\n")
        print("=" * 70)
        print("Query Embedder Initialized")
        print("=" * 70)

        print(
            f"\nDevice : "
            f"{self.device}"
        )

        print(
            f"Embedding Dimension : "
            f"{self.embedding_dimension}"
        )

    # =====================================================
    # GENERATE QUERY EMBEDDING
    # =====================================================

    def embed_query(self, query):
        """
        Generate an embedding for a text query.

        Parameters
        ----------
        query : str
            User query.

        Returns
        -------
        numpy.ndarray
            384-dimensional query embedding.
        """

        # -------------------------------------------------
        # Validate query
        # -------------------------------------------------

        if not isinstance(query, str):

            raise ValueError(
                "Query must be a string."
            )

        if not query.strip():

            raise ValueError(
                "Query cannot be empty."
            )

        # -------------------------------------------------
        # Generate embedding
        # -------------------------------------------------

        embedding = self.model.encode(
            query,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=self.device
        )

        embedding = np.asarray(
            embedding,
            dtype=np.float32
        )

        # -------------------------------------------------
        # Validate dimension
        # -------------------------------------------------

        if embedding.ndim != 1:

            raise ValueError(
                "Query embedding must be "
                "a 1-dimensional vector."
            )

        if (
            embedding.shape[0]
            != self.embedding_dimension
        ):

            raise ValueError(
                "Query embedding dimension "
                "does not match the expected "
                "embedding dimension."
            )

        return embedding


# =========================================================
# DIRECT TEST
# =========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("Testing Query Embedder")
    print("=" * 70)

    # -----------------------------------------------------
    # Initialize embedding model
    # -----------------------------------------------------

    embedding_model = EmbeddingModel()

    # -----------------------------------------------------
    # Initialize query embedder
    # -----------------------------------------------------

    query_embedder = QueryEmbedder(
        embedding_model
    )

    # -----------------------------------------------------
    # Test query
    # -----------------------------------------------------

    query = (
        "What is an example of hallucination?"
    )

    print(
        f"\nQuery : {query}"
    )

    # -----------------------------------------------------
    # Generate embedding
    # -----------------------------------------------------

    query_embedding = (
        query_embedder.embed_query(
            query
        )
    )

    # -----------------------------------------------------
    # Display result
    # -----------------------------------------------------

    print(
        f"\nEmbedding Shape : "
        f"{query_embedding.shape}"
    )

    print(
        f"Embedding Dimension : "
        f"{len(query_embedding)}"
    )

    print(
        f"Embedding Data Type : "
        f"{query_embedding.dtype}"
    )

    # -----------------------------------------------------
    # Final verification
    # -----------------------------------------------------

    if (
        query_embedding.shape
        == (
            query_embedder.embedding_dimension,
        )

        and

        np.all(
            np.isfinite(
                query_embedding
            )
        )
    ):

        print(
            "\n✓ Query Embedding Test Passed"
        )

    else:

        print(
            "\n✗ Query Embedding Test Failed"
        )