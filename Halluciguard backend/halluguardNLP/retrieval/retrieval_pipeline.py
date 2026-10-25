"""
Retrieval Pipeline Module

Connects the complete retrieval process:

    User Query
        ↓
    Query Embedding
        ↓
    Stored Dataset Embeddings
        ↓
    Similarity Calculation
        ↓
    Top-K Retrieval Results

Responsibilities:
    - Load exported embedding data
    - Load the query embedding model
    - Generate query embeddings
    - Perform retrieval
    - Return relevant chunks

This module does NOT:
    - Create datasets
    - Create chunks
    - Generate stored dataset embeddings
    - Train models
"""

import os
import numpy as np

from embeddings.embedding_model import EmbeddingModel
from retrieval.query_embedder import QueryEmbedder
from retrieval.retriever import Retriever


class RetrievalPipeline:

    # =====================================================
    # CONFIGURATION
    # =====================================================

    DEFAULT_EMBEDDING_DIRECTORY = (
        "outputs/embeddings"
    )

    DATASET_NAMES = [

        "dialogue_data",

        "general_data",

        "qa_data",

        "summarization_data",

        "TruthfulQA"

    ]

    # =====================================================
    # CONSTRUCTOR
    # =====================================================

    def __init__(
        self,
        embedding_directory=None
    ):

        if embedding_directory is None:

            self.embedding_directory = (
                self.DEFAULT_EMBEDDING_DIRECTORY
            )

        else:

            self.embedding_directory = (
                embedding_directory
            )

        print("\n")
        print("=" * 70)
        print("Initializing Retrieval Pipeline")
        print("=" * 70)

        # -------------------------------------------------
        # Load Embedding Model
        # -------------------------------------------------

        self.embedding_model = (
            EmbeddingModel()
        )

        # -------------------------------------------------
        # Initialize Query Embedder
        # -------------------------------------------------

        self.query_embedder = QueryEmbedder(
            self.embedding_model
        )

        # -------------------------------------------------
        # Dataset Storage
        # -------------------------------------------------

        self.retrievers = {}

        # -------------------------------------------------
        # Load All Datasets
        # -------------------------------------------------

        self._load_datasets()

        print("\n")
        print("=" * 70)
        print("Retrieval Pipeline Initialized")
        print("=" * 70)

        print(
            f"\nDatasets Loaded : "
            f"{len(self.retrievers)}"
        )

    # =====================================================
    # LOAD DATASET
    # =====================================================

    def _load_dataset(
        self,
        dataset_name
    ):

        json_path = os.path.join(
            self.embedding_directory,
            f"{dataset_name}.json"
        )

        numpy_path = os.path.join(
            self.embedding_directory,
            f"{dataset_name}.npy"
        )

        # -------------------------------------------------
        # Check Files
        # -------------------------------------------------

        if not os.path.exists(json_path):

            raise FileNotFoundError(
                f"Embedding JSON file not found: "
                f"{json_path}"
            )

        if not os.path.exists(numpy_path):

            raise FileNotFoundError(
                f"Embedding NumPy file not found: "
                f"{numpy_path}"
            )

        # -------------------------------------------------
        # Load Records
        # -------------------------------------------------

        import json

        with open(
            json_path,
            "r",
            encoding="utf-8"
        ) as file:

            records = json.load(file)

        # -------------------------------------------------
        # Load Embedding Matrix
        # -------------------------------------------------

        embeddings = np.load(
            numpy_path
        )

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32
        )

        # -------------------------------------------------
        # Validate
        # -------------------------------------------------

        if embeddings.ndim != 2:

            raise ValueError(
                f"Embeddings for {dataset_name} "
                f"must be a 2-dimensional matrix."
            )

        if len(records) != len(embeddings):

            raise ValueError(
                f"Record count and embedding count "
                f"do not match for {dataset_name}."
            )

        if embeddings.shape[1] != 384:

            raise ValueError(
                f"Expected 384-dimensional embeddings "
                f"for {dataset_name}, but found "
                f"{embeddings.shape[1]}."
            )

        # -------------------------------------------------
        # Create Retriever
        # -------------------------------------------------

        retriever = Retriever(
            embeddings,
            records
        )

        return retriever

    # =====================================================
    # LOAD ALL DATASETS
    # =====================================================

    def _load_datasets(self):

        print("\n")
        print("=" * 70)
        print("Loading Stored Embeddings")
        print("=" * 70)

        for dataset_name in self.DATASET_NAMES:

            print(
                f"\nLoading : "
                f"{dataset_name}"
            )

            retriever = self._load_dataset(
                dataset_name
            )

            self.retrievers[
                dataset_name
            ] = retriever

            print(
                f"✓ Loaded : "
                f"{dataset_name}"
            )

    # =====================================================
    # EMBED QUERY
    # =====================================================

    def embed_query(
        self,
        query
    ):

        return self.query_embedder.embed_query(
            query
        )

    # =====================================================
    # RETRIEVE FROM ONE DATASET
    # =====================================================

    def retrieve_from_dataset(
        self,
        query,
        dataset_name,
        top_k=5
    ):

        if dataset_name not in self.retrievers:

            raise ValueError(
                f"Unknown dataset: "
                f"{dataset_name}"
            )

        query_embedding = (
            self.embed_query(
                query
            )
        )

        results = (
            self.retrievers[
                dataset_name
            ].retrieve(
                query_embedding,
                top_k=top_k
            )
        )

        return results

    # =====================================================
    # RETRIEVE FROM ALL DATASETS
    # =====================================================

    def retrieve(
        self,
        query,
        top_k=5
    ):

        query_embedding = (
            self.embed_query(
                query
            )
        )

        all_results = []

        for dataset_name, retriever in (
            self.retrievers.items()
        ):

            results = retriever.retrieve(
                query_embedding,
                top_k=top_k
            )

            for result in results:

                result["dataset"] = (
                    dataset_name
                )

                all_results.append(
                    result
                )

        # -------------------------------------------------
        # Sort Across All Datasets
        # -------------------------------------------------

        all_results.sort(
            key=lambda result:
                result["similarity_score"],
            reverse=True
        )

        # -------------------------------------------------
        # Keep Global Top-K
        # -------------------------------------------------

        final_results = (
            all_results[:top_k]
        )

        # -------------------------------------------------
        # Reassign Global Ranks
        # -------------------------------------------------

        for rank, result in enumerate(
            final_results,
            start=1
        ):

            result["rank"] = rank

        return final_results

    # =====================================================
    # RETRIEVE WITH THRESHOLD
    # =====================================================

    def retrieve_with_threshold(
        self,
        query,
        top_k=5,
        min_score=0.0
    ):

        query_embedding = (
            self.embed_query(
                query
            )
        )

        all_results = []

        for dataset_name, retriever in (
            self.retrievers.items()
        ):

            results = (
                retriever.retrieve_with_threshold(
                    query_embedding,
                    top_k=top_k,
                    min_score=min_score
                )
            )

            for result in results:

                result["dataset"] = (
                    dataset_name
                )

                all_results.append(
                    result
                )

        # -------------------------------------------------
        # Sort Results
        # -------------------------------------------------

        all_results.sort(
            key=lambda result:
                result["similarity_score"],
            reverse=True
        )

        # -------------------------------------------------
        # Keep Top-K
        # -------------------------------------------------

        final_results = (
            all_results[:top_k]
        )

        # -------------------------------------------------
        # Assign Ranks
        # -------------------------------------------------

        for rank, result in enumerate(
            final_results,
            start=1
        ):

            result["rank"] = rank

        return final_results

    # =====================================================
    # GET BEST RESULT
    # =====================================================

    def retrieve_best(
        self,
        query
    ):

        results = self.retrieve(
            query,
            top_k=1
        )

        if not results:

            return None

        return results[0]

    # =====================================================
    # GET DATASET NAMES
    # =====================================================

    def get_dataset_names(self):

        return list(
            self.retrievers.keys()
        )


# =========================================================
# DIRECT TEST
# =========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("Testing Retrieval Pipeline")
    print("=" * 70)

    # -----------------------------------------------------
    # Initialize Pipeline
    # -----------------------------------------------------

    pipeline = RetrievalPipeline()

    # -----------------------------------------------------
    # Test Query
    # -----------------------------------------------------

    query = (
        "What is an example of hallucination?"
    )

    print("\n")
    print("=" * 70)
    print("Testing Query")
    print("=" * 70)

    print(
        f"\nQuery : {query}"
    )

    # -----------------------------------------------------
    # Retrieve Results
    # -----------------------------------------------------

    results = pipeline.retrieve(
        query,
        top_k=5
    )

    # -----------------------------------------------------
    # Display Results
    # -----------------------------------------------------

    print("\n")
    print("=" * 70)
    print("Top Retrieval Results")
    print("=" * 70)

    for result in results:

        print(
            f"\nRank : "
            f"{result['rank']}"
        )

        print(
            f"Dataset : "
            f"{result['dataset']}"
        )

        print(
            f"Chunk UID : "
            f"{result['chunk_uid']}"
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
    # Verification
    # -----------------------------------------------------

    if len(results) > 0:

        print(
            "\n✓ Retrieval Pipeline Test Passed"
        )

    else:

        print(
            "\n✗ Retrieval Pipeline Test Failed"
        )