"""
Embedding Exporter Module

Responsible for saving generated embeddings
and their associated chunk information.

Responsibilities:
    - Save embedding records
    - Create output directories
    - Preserve chunk IDs
    - Preserve chunk text
    - Preserve 384-dimensional embeddings
    - Load exported embeddings for verification

This module DOES NOT:
    - Generate embeddings
    - Load the embedding model
    - Perform embedding validation
    - Perform retrieval
"""

import json
import os
import numpy as np


class EmbeddingExporter:

    # =====================================================
    # CONFIGURATION
    # =====================================================

    DEFAULT_OUTPUT_DIRECTORY = "outputs/embeddings"

    # =====================================================
    # CONSTRUCTOR
    # =====================================================

    def __init__(
        self,
        output_directory=None
    ):

        if output_directory is None:

            self.output_directory = (
                self.DEFAULT_OUTPUT_DIRECTORY
            )

        else:

            self.output_directory = (
                output_directory
            )

        os.makedirs(
            self.output_directory,
            exist_ok=True
        )

        print("\n")
        print("=" * 70)
        print("Embedding Exporter Initialized")
        print("=" * 70)

        print(
            f"\nOutput Directory : "
            f"{self.output_directory}"
        )

    # =====================================================
    # SAVE JSON
    # =====================================================

    def save_json(
        self,
        records,
        filename
    ):
        """
        Save embedding records as JSON.

        Parameters
        ----------
        records : list[dict]

        filename : str

        Returns
        -------
        str
            Path of saved file.
        """

        output_path = os.path.join(
            self.output_directory,
            filename
        )

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                records,
                file,
                ensure_ascii=False,
                indent=2
            )

        print(
            f"\nSaved JSON : "
            f"{output_path}"
        )

        print(
            f"Records Saved : "
            f"{len(records)}"
        )

        return output_path

    # =====================================================
    # SAVE NUMPY EMBEDDINGS
    # =====================================================

    def save_numpy(
        self,
        records,
        filename
    ):
        """
        Save only the embedding vectors
        as a NumPy array.

        Parameters
        ----------
        records : list[dict]

        filename : str

        Returns
        -------
        str
            Path of saved NumPy file.
        """

        embeddings = [

            record["embedding"]

            for record in records

        ]

        matrix = np.asarray(
            embeddings,
            dtype=np.float32
        )

        output_path = os.path.join(
            self.output_directory,
            filename
        )

        np.save(
            output_path,
            matrix
        )

        print(
            f"\nSaved NumPy Embeddings : "
            f"{output_path}"
        )

        print(
            f"Embedding Matrix Shape : "
            f"{matrix.shape}"
        )

        return output_path

    # =====================================================
    # LOAD JSON
    # =====================================================

    def load_json(
        self,
        filename
    ):
        """
        Load previously exported embedding records.
        """

        input_path = os.path.join(
            self.output_directory,
            filename
        )

        with open(
            input_path,
            "r",
            encoding="utf-8"
        ) as file:

            records = json.load(
                file
            )

        print(
            f"\nLoaded JSON : "
            f"{input_path}"
        )

        print(
            f"Records Loaded : "
            f"{len(records)}"
        )

        return records

    # =====================================================
    # LOAD NUMPY EMBEDDINGS
    # =====================================================

    def load_numpy(
        self,
        filename
    ):
        """
        Load previously exported NumPy embeddings.
        """

        input_path = os.path.join(
            self.output_directory,
            filename
        )

        embeddings = np.load(
            input_path
        )

        print(
            f"\nLoaded NumPy Embeddings : "
            f"{input_path}"
        )

        print(
            f"Embedding Matrix Shape : "
            f"{embeddings.shape}"
        )

        return embeddings

    # =====================================================
    # EXPORT COMPLETE EMBEDDING DATA
    # =====================================================

    def export_embeddings(
        self,
        records,
        base_filename
    ):
        """
        Save both complete embedding records
        and the embedding matrix.

        Parameters
        ----------
        records : list[dict]

        base_filename : str

        Returns
        -------
        dict
            Paths of exported files.
        """

        json_filename = (
            f"{base_filename}.json"
        )

        numpy_filename = (
            f"{base_filename}.npy"
        )

        json_path = self.save_json(
            records,
            json_filename
        )

        numpy_path = self.save_numpy(
            records,
            numpy_filename
        )

        return {

            "json": json_path,

            "numpy": numpy_path

        }


# =========================================================
# DIRECT TEST
# =========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("Testing Embedding Exporter")
    print("=" * 70)

    # -----------------------------------------------------
    # Create small test embeddings
    # -----------------------------------------------------

    test_records = [

        {
            "chunk_id":
                "test_chunk_001",

            "chunk_text":
                "The capital of France is Paris.",

            "embedding":
                np.random.rand(
                    384
                ).astype(
                    np.float32
                ).tolist()
        },

        {
            "chunk_id":
                "test_chunk_002",

            "chunk_text":
                "Python is a programming language.",

            "embedding":
                np.random.rand(
                    384
                ).astype(
                    np.float32
                ).tolist()
        }

    ]

    # -----------------------------------------------------
    # Initialize Exporter
    # -----------------------------------------------------

    exporter = EmbeddingExporter()

    # -----------------------------------------------------
    # Export Test Data
    # -----------------------------------------------------

    exported_files = (
        exporter.export_embeddings(
            test_records,
            "exporter_test"
        )
    )

    # -----------------------------------------------------
    # Load Files Again
    # -----------------------------------------------------

    loaded_records = (
        exporter.load_json(
            "exporter_test.json"
        )
    )

    loaded_embeddings = (
        exporter.load_numpy(
            "exporter_test.npy"
        )
    )

    # -----------------------------------------------------
    # Verify
    # -----------------------------------------------------

    print("\n")
    print("=" * 70)
    print("Exporter Verification")
    print("=" * 70)

    print(
        f"\nOriginal Records : "
        f"{len(test_records)}"
    )

    print(
        f"Loaded Records : "
        f"{len(loaded_records)}"
    )

    print(
        f"Loaded Embedding Shape : "
        f"{loaded_embeddings.shape}"
    )

    if (
        len(test_records)
        == len(loaded_records)
        and loaded_embeddings.shape
        == (2, 384)
    ):

        print(
            "\n✓ Exporter Test Passed"
        )

    else:

        print(
            "\n✗ Exporter Test Failed"
        )