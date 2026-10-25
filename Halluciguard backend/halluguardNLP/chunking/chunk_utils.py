"""
Chunk Utilities Module

Provides reusable utilities for working with
already-generated chunked datasets.

Responsibilities:
    1. Load saved chunks
    2. Validate chunk structure
    3. Find chunks by ID
    4. Extract chunk text and metadata
    5. Generate chunk statistics

This module does NOT:
    - Create chunks
    - Generate embeddings
    - Perform retrieval
    - Calculate similarity
"""

import json
from pathlib import Path


class ChunkUtils:
    """
    Utility functions for managing chunked datasets.
    """

    # =====================================================
    # REQUIRED CHUNK FIELDS
    # =====================================================

    REQUIRED_FIELDS = [
        "chunk_uid",
        "source_type",
        "dataset",
        "record_id",
        "chunk_id",
        "total_chunks",
        "token_count",
        "character_count",
        "chunk_text"
    ]

    # =====================================================
    # CONSTRUCTOR
    # =====================================================

    def __init__(self):

        self.chunk_path = (
            Path("outputs") / "chunked"
        )

    # =====================================================
    # 1. LOAD CHUNKS
    # =====================================================

    def load_chunks(self, file_path):
        """
        Load chunks from a JSON file.

        Parameters
        ----------
        file_path : str | Path
            Path to a chunked JSON file.

        Returns
        -------
        list
            List of chunk dictionaries.
        """

        file_path = Path(file_path)

        if not file_path.exists():

            raise FileNotFoundError(
                f"Chunk file not found: {file_path}"
            )

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:

            chunks = json.load(file)

        if not isinstance(chunks, list):

            raise ValueError(
                "Chunk file must contain a JSON list."
            )

        return chunks

    # =====================================================
    # LOAD CHUNK DATASET BY NAME
    # =====================================================

    def load_dataset(self, dataset_name):
        """
        Load a chunked dataset from outputs/chunked/.

        Parameters
        ----------
        dataset_name : str
            Dataset name, for example:
                dialogue_data
                TruthfulQA

        Returns
        -------
        list
            Chunk records.
        """

        file_path = (
            self.chunk_path
            / f"{dataset_name}_chunked.json"
        )

        return self.load_chunks(
            file_path
        )

    # =====================================================
    # 2. VALIDATE CHUNK STRUCTURE
    # =====================================================

    def validate_chunk(self, chunk):
        """
        Validate a single chunk.

        Parameters
        ----------
        chunk : dict

        Returns
        -------
        bool
            True if the chunk is valid.
        """

        if not isinstance(chunk, dict):
            return False

        # -------------------------------------------------
        # Check required fields
        # -------------------------------------------------

        for field in self.REQUIRED_FIELDS:

            if field not in chunk:
                return False

        # -------------------------------------------------
        # Check important values
        # -------------------------------------------------

        if not isinstance(
            chunk["chunk_uid"],
            str
        ):
            return False

        if not isinstance(
            chunk["chunk_text"],
            str
        ):
            return False

        if not chunk["chunk_text"].strip():
            return False

        if not isinstance(
            chunk["token_count"],
            int
        ):
            return False

        if not isinstance(
            chunk["character_count"],
            int
        ):
            return False

        if chunk["token_count"] <= 0:
            return False

        if chunk["character_count"] <= 0:
            return False

        return True

    # =====================================================
    # VALIDATE ALL CHUNKS
    # =====================================================

    def validate_chunks(self, chunks):
        """
        Validate an entire list of chunks.

        Returns
        -------
        dict
            Validation summary.
        """

        total_chunks = len(chunks)

        valid_chunks = 0
        invalid_chunks = 0

        invalid_indices = []

        for index, chunk in enumerate(chunks):

            if self.validate_chunk(chunk):

                valid_chunks += 1

            else:

                invalid_chunks += 1

                invalid_indices.append(
                    index
                )

        return {
            "total_chunks": total_chunks,
            "valid_chunks": valid_chunks,
            "invalid_chunks": invalid_chunks,
            "invalid_indices": invalid_indices,
            "is_valid": invalid_chunks == 0
        }

    # =====================================================
    # VALIDATE SAVED DATASET
    # =====================================================

    def validate_dataset(self, dataset_name):
        """
        Load and validate a saved chunked dataset.
        """

        chunks = self.load_dataset(
            dataset_name
        )

        validation_result = (
            self.validate_chunks(
                chunks
            )
        )

        return validation_result

    # =====================================================
    # 3. FIND CHUNK BY ID
    # =====================================================

    def get_chunk_by_id(
        self,
        chunks,
        chunk_uid
    ):
        """
        Find a chunk using its unique chunk UID.

        Parameters
        ----------
        chunks : list
            List of chunk dictionaries.

        chunk_uid : str
            Unique chunk identifier.

        Returns
        -------
        dict | None
        """

        for chunk in chunks:

            if chunk.get(
                "chunk_uid"
            ) == chunk_uid:

                return chunk

        return None

    # =====================================================
    # FIND ALL CHUNKS FROM RECORD
    # =====================================================

    def get_record_chunks(
        self,
        chunks,
        record_id
    ):
        """
        Find all chunks belonging to one
        original record.

        Parameters
        ----------
        chunks : list
        record_id : int

        Returns
        -------
        list
        """

        return [

            chunk

            for chunk in chunks

            if chunk.get(
                "record_id"
            ) == record_id

        ]

    # =====================================================
    # 4. EXTRACT CHUNK TEXT
    # =====================================================

    def get_chunk_text(self, chunk):
        """
        Extract only the text from a chunk.
        """

        if not self.validate_chunk(chunk):

            raise ValueError(
                "Invalid chunk structure."
            )

        return chunk["chunk_text"]

    # =====================================================
    # EXTRACT CHUNK METADATA
    # =====================================================

    def get_chunk_metadata(self, chunk):
        """
        Extract metadata from a chunk.

        The chunk text itself is excluded.
        """

        if not self.validate_chunk(chunk):

            raise ValueError(
                "Invalid chunk structure."
            )

        return {

            "chunk_uid":
                chunk["chunk_uid"],

            "source_type":
                chunk["source_type"],

            "dataset":
                chunk["dataset"],

            "record_id":
                chunk["record_id"],

            "chunk_id":
                chunk["chunk_id"],

            "total_chunks":
                chunk["total_chunks"],

            "token_count":
                chunk["token_count"],

            "character_count":
                chunk["character_count"]

        }

    # =====================================================
    # 5. GENERATE CHUNK STATISTICS
    # =====================================================

    def generate_statistics(self, chunks):
        """
        Generate statistics for a list of chunks.

        Returns
        -------
        dict
        """

        if not chunks:

            return {

                "total_chunks": 0,

                "average_tokens": 0,

                "minimum_tokens": 0,

                "maximum_tokens": 0,

                "average_characters": 0,

                "minimum_characters": 0,

                "maximum_characters": 0

            }

        token_counts = [

            chunk["token_count"]

            for chunk in chunks

            if self.validate_chunk(chunk)

        ]

        character_counts = [

            chunk["character_count"]

            for chunk in chunks

            if self.validate_chunk(chunk)

        ]

        if not token_counts:

            return {

                "total_chunks": len(chunks),

                "average_tokens": 0,

                "minimum_tokens": 0,

                "maximum_tokens": 0,

                "average_characters": 0,

                "minimum_characters": 0,

                "maximum_characters": 0

            }

        return {

            "total_chunks": len(chunks),

            "average_tokens": round(
                sum(token_counts)
                / len(token_counts),
                2
            ),

            "minimum_tokens": min(
                token_counts
            ),

            "maximum_tokens": max(
                token_counts
            ),

            "average_characters": round(
                sum(character_counts)
                / len(character_counts),
                2
            ),

            "minimum_characters": min(
                character_counts
            ),

            "maximum_characters": max(
                character_counts
            )

        }

    # =====================================================
    # PRINT STATISTICS
    # =====================================================

    def print_statistics(self, chunks):
        """
        Print chunk statistics in a readable format.
        """

        statistics = (
            self.generate_statistics(
                chunks
            )
        )

        print("\nChunk Statistics")
        print("-" * 40)

        print(
            f"Total Chunks : "
            f"{statistics['total_chunks']}"
        )

        print(
            f"Average Tokens : "
            f"{statistics['average_tokens']}"
        )

        print(
            f"Minimum Tokens : "
            f"{statistics['minimum_tokens']}"
        )

        print(
            f"Maximum Tokens : "
            f"{statistics['maximum_tokens']}"
        )

        print(
            f"Average Characters : "
            f"{statistics['average_characters']}"
        )

        print(
            f"Minimum Characters : "
            f"{statistics['minimum_characters']}"
        )

        print(
            f"Maximum Characters : "
            f"{statistics['maximum_characters']}"
        )

    # =====================================================
    # VALIDATE ALL SAVED DATASETS
    # =====================================================

    def validate_all_datasets(
        self,
        dataset_names
    ):
        """
        Validate multiple saved chunk datasets.

        Parameters
        ----------
        dataset_names : list[str]

        Returns
        -------
        dict
            Validation results for each dataset.
        """

        results = {}

        for dataset_name in dataset_names:

            results[dataset_name] = (
                self.validate_dataset(
                    dataset_name
                )
            )

        return results