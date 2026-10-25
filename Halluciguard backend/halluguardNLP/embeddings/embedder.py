"""
Dataset Embedder Module

Responsible for generating embeddings from
the chunks produced by Module 2.

Embedding Model:
BAAI/bge-small-en-v1.5

Responsibilities:
    - Receive Module 2 chunk dictionaries
    - Extract chunk text
    - Process chunks in batches
    - Generate 384-dimensional embeddings
    - Use GPU when available
    - Preserve chunk IDs and metadata
    - Return embedding records

This module DOES NOT:
    - Save embedding files
    - Export datasets
    - Perform retrieval
    - Validate embeddings
"""

import numpy as np
import pandas as pd

from embeddings.embedding_model import EmbeddingModel


class DatasetEmbedder:
    """
    Generates embeddings for chunked datasets.

    Module 2 produces a list of chunk dictionaries.
    This module converts the chunk_text from those
    dictionaries into 384-dimensional embeddings.

    Embeddings are generated in batches so that the
    complete dataset is never processed by the model
    as one large batch.
    """

    # =====================================================
    # EMBEDDING CONFIGURATION
    # =====================================================

    BATCH_SIZE = 32

    NORMALIZE_EMBEDDINGS = True

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
        print("Dataset Embedder Initialized")
        print("=" * 70)

        print(
            f"\nDevice : {self.device}"
        )

        print(
            f"Batch Size : {self.BATCH_SIZE}"
        )

        print(
            f"Embedding Dimension : "
            f"{self.embedding_dimension}"
        )

        print(
            f"Normalize Embeddings : "
            f"{self.NORMALIZE_EMBEDDINGS}"
        )

    # =====================================================
    # EMBED SINGLE BATCH
    # =====================================================

    def embed_batch(self, texts):
        """
        Generate embeddings for one batch of texts.

        Parameters
        ----------
        texts : list[str]

        Returns
        -------
        numpy.ndarray

            Shape:
                (number_of_texts, 384)
        """

        if not texts:

            return np.empty(
                (
                    0,
                    self.embedding_dimension
                ),
                dtype=np.float32
            )

        embeddings = self.model.encode(
            texts,
            batch_size=self.BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=(
                self.NORMALIZE_EMBEDDINGS
            ),
            device=self.device
        )

        return embeddings.astype(
            np.float32
        )

    # =====================================================
    # EMBED TEXTS IN BATCHES
    # =====================================================

    def embed_texts(
        self,
        texts,
        show_progress=True
    ):
        """
        Generate embeddings for a collection
        of texts using batches.

        Parameters
        ----------
        texts : list[str]

        show_progress : bool

        Returns
        -------
        numpy.ndarray

            Shape:
                (number_of_texts, 384)
        """

        if not texts:

            return np.empty(
                (
                    0,
                    self.embedding_dimension
                ),
                dtype=np.float32
            )

        total_texts = len(texts)

        all_embeddings = []

        total_batches = (
            (total_texts + self.BATCH_SIZE - 1)
            // self.BATCH_SIZE
        )

        for batch_start in range(
            0,
            total_texts,
            self.BATCH_SIZE
        ):

            batch_end = min(
                batch_start + self.BATCH_SIZE,
                total_texts
            )

            batch_texts = texts[
                batch_start:batch_end
            ]

            batch_embeddings = (
                self.embed_batch(
                    batch_texts
                )
            )

            all_embeddings.append(
                batch_embeddings
            )

            if show_progress:

                current_batch = (
                    batch_start
                    // self.BATCH_SIZE
                ) + 1

                print(
                    f"Batch "
                    f"{current_batch}/"
                    f"{total_batches} "
                    f"completed "
                    f"({batch_end}/"
                    f"{total_texts} texts)"
                )

        return np.vstack(
            all_embeddings
        )

    # =====================================================
    # EMBED MODULE 2 CHUNKS
    # =====================================================

    def embed_chunks(
        self,
        chunks,
        show_progress=True
    ):
        """
        Generate embeddings for Module 2 chunks.

        Module 2 produces chunks in this form:

            {
                "chunk_uid": "...",
                "source_type": "...",
                "dataset": "...",
                "record_id": ...,
                "chunk_id": ...,
                "total_chunks": ...,
                "token_count": ...,
                "character_count": ...,
                "chunk_text": "..."
            }

        The complete chunk metadata is preserved.

        A new "embedding" field is added.

        Parameters
        ----------
        chunks : list[dict]

        show_progress : bool

        Returns
        -------
        list[dict]

            Original chunk information plus:

                "embedding": [384 floating-point values]
        """

        if not chunks:

            return []

        # -------------------------------------------------
        # Validate basic chunk structure
        # -------------------------------------------------

        for index, chunk in enumerate(chunks):

            if not isinstance(chunk, dict):

                raise ValueError(
                    f"Chunk at index {index} "
                    f"is not a dictionary."
                )

            if "chunk_text" not in chunk:

                raise ValueError(
                    f"Chunk at index {index} "
                    f"is missing 'chunk_text'."
                )

        # -------------------------------------------------
        # Extract chunk text
        # -------------------------------------------------

        texts = [

            str(
                chunk["chunk_text"]
            )

            for chunk in chunks

        ]

        # -------------------------------------------------
        # Generate embeddings
        # -------------------------------------------------

        embeddings = self.embed_texts(
            texts,
            show_progress=show_progress
        )

        # -------------------------------------------------
        # Preserve metadata and add embeddings
        # -------------------------------------------------

        embedding_records = []

        for index, chunk in enumerate(chunks):

            record = dict(chunk)

            record["embedding"] = (
                embeddings[index]
                .tolist()
            )

            embedding_records.append(
                record
            )

        return embedding_records

    # =====================================================
    # EMBED DATAFRAME
    # =====================================================

    def embed_dataframe(
        self,
        dataframe,
        text_column="chunk_text",
        id_column="chunk_id",
        show_progress=True
    ):
        """
        Generate embeddings for a chunk DataFrame.

        This method is retained for compatibility
        with DataFrame-based workflows.

        Parameters
        ----------
        dataframe : pandas.DataFrame

        text_column : str

        id_column : str

        Returns
        -------
        list[dict]
        """

        if dataframe.empty:

            return []

        if text_column not in dataframe.columns:

            raise ValueError(
                f"Missing text column: "
                f"{text_column}"
            )

        if id_column not in dataframe.columns:

            raise ValueError(
                f"Missing ID column: "
                f"{id_column}"
            )

        texts = (
            dataframe[text_column]
            .fillna("")
            .astype(str)
            .tolist()
        )

        embeddings = self.embed_texts(
            texts,
            show_progress=show_progress
        )

        records = []

        for position, (_, row) in enumerate(
            dataframe.iterrows()
        ):

            record = {

                "chunk_id":
                    row[id_column],

                "chunk_text":
                    row[text_column],

                "embedding":
                    embeddings[
                        position
                    ].tolist()

            }

            records.append(
                record
            )

        return records

    # =====================================================
    # TEST EMBEDDING GENERATION
    # =====================================================

    def test_embedding(
        self,
        texts
    ):
        """
        Test embedding generation on a small
        number of texts.

        This method is intended for testing
        before processing the complete dataset.
        """

        print("\n")
        print("=" * 70)
        print("Testing Embedding Generation")
        print("=" * 70)

        print(
            f"\nNumber of Test Texts : "
            f"{len(texts)}"
        )

        embeddings = self.embed_texts(
            texts,
            show_progress=True
        )

        print(
            "\nEmbedding Test Completed."
        )

        print(
            f"Embedding Array Shape : "
            f"{embeddings.shape}"
        )

        print(
            f"Expected Dimension : "
            f"{self.embedding_dimension}"
        )

        if (
            len(embeddings.shape) == 2
            and embeddings.shape[1]
            == self.embedding_dimension
        ):

            print(
                "✓ Embedding Dimension "
                "Verified"
            )

        else:

            print(
                "✗ Embedding Dimension "
                "Mismatch"
            )

        return embeddings


# =========================================================
# DIRECT TEST
# =========================================================

if __name__ == "__main__":

    from chunking.chunk_utils import ChunkUtils

    # -----------------------------------------------------
    # Initialize Embedding Model
    # -----------------------------------------------------

    embedding_model = EmbeddingModel()

    # -----------------------------------------------------
    # Initialize Embedder
    # -----------------------------------------------------

    embedder = DatasetEmbedder(
        embedding_model
    )

    # -----------------------------------------------------
    # Load Real Module 2 Chunks
    # -----------------------------------------------------

    print("\n")
    print("=" * 70)
    print("Loading Real Module 2 Chunks")
    print("=" * 70)

    chunk_utils = ChunkUtils()

    test_chunks = chunk_utils.load_dataset(
        "dialogue_data"
    )

    print(
        f"\nTotal Saved Chunks : "
        f"{len(test_chunks)}"
    )

    # -----------------------------------------------------
    # Validate Saved Chunks
    # -----------------------------------------------------

    validation = (
        chunk_utils.validate_chunks(
            test_chunks
        )
    )

    print(
        f"Valid Chunks : "
        f"{validation['valid_chunks']}"
    )

    print(
        f"Invalid Chunks : "
        f"{validation['invalid_chunks']}"
    )

    if not validation["is_valid"]:

        raise ValueError(
            "Saved Module 2 chunks "
            "contain invalid records."
        )

    # -----------------------------------------------------
    # Select ONLY 10 chunks
    # -----------------------------------------------------

    test_chunks = test_chunks[:10]

    print(
        f"\nChunks Selected For Test : "
        f"{len(test_chunks)}"
    )

    # -----------------------------------------------------
    # Generate Real Embeddings
    # -----------------------------------------------------

    print("\n")
    print("=" * 70)
    print("Embedding Real Module 2 Chunks")
    print("=" * 70)

    embedding_records = (
        embedder.embed_chunks(
            test_chunks,
            show_progress=True
        )
    )

    # -----------------------------------------------------
    # Verify Results
    # -----------------------------------------------------

    print("\n")
    print("=" * 70)
    print("Integration Test Results")
    print("=" * 70)

    print(
        f"\nInput Chunks : "
        f"{len(test_chunks)}"
    )

    print(
        f"Generated Embedding Records : "
        f"{len(embedding_records)}"
    )

    first_record = (
        embedding_records[0]
    )

    print(
        f"\nFirst Chunk UID : "
        f"{first_record['chunk_uid']}"
    )

    print(
        f"First Chunk ID : "
        f"{first_record['chunk_id']}"
    )

    print(
        f"First Chunk Token Count : "
        f"{first_record['token_count']}"
    )

    print(
        f"First Embedding Length : "
        f"{len(first_record['embedding'])}"
    )

    # -----------------------------------------------------
    # Final Verification
    # -----------------------------------------------------

    all_dimensions_correct = all(

        len(
            record["embedding"]
        )
        == embedder.embedding_dimension

        for record in embedding_records

    )

    metadata_preserved = all(

        "chunk_uid" in record
        and "chunk_text" in record
        and "embedding" in record

        for record in embedding_records

    )

    print("\n")

    if (
        len(embedding_records)
        == len(test_chunks)
        and all_dimensions_correct
        and metadata_preserved
    ):

        print(
            "✓ REAL CHUNK EMBEDDING "
            "INTEGRATION TEST PASSED"
        )

    else:

        print(
            "✗ REAL CHUNK EMBEDDING "
            "INTEGRATION TEST FAILED"
        )