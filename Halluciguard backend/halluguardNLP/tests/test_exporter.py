import os
import numpy as np
import pytest

from embeddings.exporter import EmbeddingExporter


class TestEmbeddingExporter:

    # =====================================================
    # SETUP
    # =====================================================

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):

        self.output_directory = str(
            tmp_path
        )

        self.exporter = EmbeddingExporter(
            self.output_directory
        )

        self.records = [

            {
                "chunk_id":
                    "chunk_001",

                "chunk_text":
                    "The capital of France is Paris.",

                "embedding":
                    np.ones(
                        384,
                        dtype=np.float32
                    ).tolist()
            },

            {
                "chunk_id":
                    "chunk_002",

                "chunk_text":
                    "Python is a programming language.",

                "embedding":
                    np.ones(
                        384,
                        dtype=np.float32
                    ).tolist()
            }
        ]

    # =====================================================
    # TEST SAVE JSON
    # =====================================================

    def test_save_json(self):

        path = self.exporter.save_json(
            self.records,
            "test_embeddings.json"
        )

        assert os.path.exists(path)

        assert path.endswith(
            "test_embeddings.json"
        )

    # =====================================================
    # TEST SAVE NUMPY
    # =====================================================

    def test_save_numpy(self):

        path = self.exporter.save_numpy(
            self.records,
            "test_embeddings.npy"
        )

        assert os.path.exists(path)

        assert path.endswith(
            "test_embeddings.npy"
        )

    # =====================================================
    # TEST LOAD JSON
    # =====================================================

    def test_load_json(self):

        self.exporter.save_json(
            self.records,
            "test_embeddings.json"
        )

        loaded_records = (
            self.exporter.load_json(
                "test_embeddings.json"
            )
        )

        assert isinstance(
            loaded_records,
            list
        )

        assert len(
            loaded_records
        ) == len(
            self.records
        )

        assert (
            loaded_records[0]["chunk_id"]
            == "chunk_001"
        )

        assert (
            loaded_records[1]["chunk_id"]
            == "chunk_002"
        )

    # =====================================================
    # TEST LOAD NUMPY
    # =====================================================

    def test_load_numpy(self):

        self.exporter.save_numpy(
            self.records,
            "test_embeddings.npy"
        )

        loaded_embeddings = (
            self.exporter.load_numpy(
                "test_embeddings.npy"
            )
        )

        assert isinstance(
            loaded_embeddings,
            np.ndarray
        )

        assert loaded_embeddings.shape == (
            2,
            384
        )

        assert loaded_embeddings.dtype == (
            np.float32
        )

    # =====================================================
    # TEST SAVED EMBEDDING VALUES
    # =====================================================

    def test_saved_embedding_values(self):

        self.exporter.save_numpy(
            self.records,
            "test_embeddings.npy"
        )

        loaded_embeddings = (
            self.exporter.load_numpy(
                "test_embeddings.npy"
            )
        )

        expected = np.asarray(
            [
                self.records[0]["embedding"],
                self.records[1]["embedding"]
            ],
            dtype=np.float32
        )

        assert np.array_equal(
            loaded_embeddings,
            expected
        )

    # =====================================================
    # TEST COMPLETE EXPORT
    # =====================================================

    def test_export_embeddings(self):

        exported_files = (
            self.exporter.export_embeddings(
                self.records,
                "complete_test"
            )
        )

        assert isinstance(
            exported_files,
            dict
        )

        assert "json" in exported_files

        assert "numpy" in exported_files

        assert os.path.exists(
            exported_files["json"]
        )

        assert os.path.exists(
            exported_files["numpy"]
        )

    # =====================================================
    # TEST COMPLETE EXPORT CONTENT
    # =====================================================

    def test_exported_files_can_be_loaded(self):

        self.exporter.export_embeddings(
            self.records,
            "verification_test"
        )

        loaded_records = (
            self.exporter.load_json(
                "verification_test.json"
            )
        )

        loaded_embeddings = (
            self.exporter.load_numpy(
                "verification_test.npy"
            )
        )

        assert len(
            loaded_records
        ) == 2

        assert loaded_embeddings.shape == (
            2,
            384
        )

        assert (
            loaded_records[0]["chunk_text"]
            == "The capital of France is Paris."
        )

        assert (
            loaded_records[1]["chunk_text"]
            == "Python is a programming language."
        )