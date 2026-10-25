import os
import numpy as np

from chunking.chunk_utils import ChunkUtils
from embeddings.embedding_model import EmbeddingModel
from embeddings.embedder import DatasetEmbedder
from embeddings.embedding_utils import EmbeddingUtils
from embeddings.exporter import EmbeddingExporter


class TestHalluciGuardPipeline:

    # =====================================================
    # SETUP
    # =====================================================

    def setup_method(self, method):

        self.chunk_utils = ChunkUtils()

        self.embedding_model = EmbeddingModel()

        self.embedder = DatasetEmbedder(
            self.embedding_model
        )

        self.embedding_utils = EmbeddingUtils()

        # Use the existing embedding output directory
        # for pipeline testing.
        self.test_output_directory = (
            "outputs/embeddings/pipeline_test"
        )

        self.exporter = EmbeddingExporter(
            self.test_output_directory
        )

        # -------------------------------------------------
        # Load existing Module 2 chunks
        # -------------------------------------------------

        chunk_path = (
            "outputs/chunked/"
            "TruthfulQA_chunked.json"
        )

        self.chunks = (
            self.chunk_utils.load_chunks(
                chunk_path
            )
        )

        # -------------------------------------------------
        # Keep pipeline test small
        # -------------------------------------------------

        self.test_chunks = self.chunks[:10]

    # =====================================================
    # TEST CHUNKS ARE AVAILABLE
    # =====================================================

    def test_chunks_are_available(self):

        assert self.chunks is not None

        assert len(self.chunks) > 0

        assert len(
            self.test_chunks
        ) == 10

    # =====================================================
    # TEST CHUNK STRUCTURE
    # =====================================================

    def test_chunk_structure(self):

        for chunk in self.test_chunks:

            assert isinstance(
                chunk,
                dict
            )

            assert "chunk_text" in chunk

            assert isinstance(
                chunk["chunk_text"],
                str
            )

            assert len(
                chunk["chunk_text"].strip()
            ) > 0

    # =====================================================
    # TEST CHUNK → EMBEDDING PIPELINE
    # =====================================================

    def test_chunk_to_embedding_pipeline(self):

        embedding_records = (
            self.embedder.embed_chunks(
                self.test_chunks,
                show_progress=False
            )
        )

        assert isinstance(
            embedding_records,
            list
        )

        assert len(
            embedding_records
        ) == 10

        for record in embedding_records:

            assert "chunk_text" in record

            assert "embedding" in record

            assert len(
                record["embedding"]
            ) == 384

    # =====================================================
    # TEST EMBEDDING VALIDATION
    # =====================================================

    def test_embedding_validation(self):

        embedding_records = (
            self.embedder.embed_chunks(
                self.test_chunks,
                show_progress=False
            )
        )

        validation_result = (
            self.embedding_utils.validate_records(
                embedding_records
            )
        )

        assert validation_result["is_valid"] is True

    # =====================================================
    # TEST EMBEDDING COUNT
    # =====================================================

    def test_chunk_embedding_count(self):

        embedding_records = (
            self.embedder.embed_chunks(
                self.test_chunks,
                show_progress=False
            )
        )

        assert len(
            embedding_records
        ) == len(
            self.test_chunks
        )

    # =====================================================
    # TEST EMBEDDING EXPORT
    # =====================================================

    def test_pipeline_export(self):

        embedding_records = (
            self.embedder.embed_chunks(
                self.test_chunks,
                show_progress=False
            )
        )

        exported_files = (
            self.exporter.export_embeddings(
                embedding_records,
                "pipeline_test"
            )
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
    # TEST COMPLETE PIPELINE
    # =====================================================

    def test_complete_pipeline(self):

        # -------------------------------------------------
        # 1. Verify Module 2 chunks
        # -------------------------------------------------

        assert len(
            self.test_chunks
        ) == 10

        # -------------------------------------------------
        # 2. Generate embeddings from chunks
        # -------------------------------------------------

        embedding_records = (
            self.embedder.embed_chunks(
                self.test_chunks,
                show_progress=False
            )
        )

        assert len(
            embedding_records
        ) == 10

        # -------------------------------------------------
        # 3. Verify embedding dimensions
        # -------------------------------------------------

        for record in embedding_records:

            assert len(
                record["embedding"]
            ) == 384

        # -------------------------------------------------
        # 4. Validate embedding records
        # -------------------------------------------------

        validation_result = (
            self.embedding_utils.validate_records(
                embedding_records
            )
        )

        assert validation_result["is_valid"] is True

        # -------------------------------------------------
        # 5. Export embeddings
        # -------------------------------------------------

        exported_files = (
            self.exporter.export_embeddings(
                embedding_records,
                "complete_pipeline_test"
            )
        )

        assert os.path.exists(
            exported_files["json"]
        )

        assert os.path.exists(
            exported_files["numpy"]
        )

        # -------------------------------------------------
        # 6. Load exported embeddings
        # -------------------------------------------------

        loaded_records = (
            self.exporter.load_json(
                "complete_pipeline_test.json"
            )
        )

        loaded_embeddings = (
            self.exporter.load_numpy(
                "complete_pipeline_test.npy"
            )
        )

        # -------------------------------------------------
        # 7. Final verification
        # -------------------------------------------------

        assert len(
            loaded_records
        ) == 10

        assert loaded_embeddings.shape == (
            10,
            384
        )

        assert np.all(
            np.isfinite(
                loaded_embeddings
            )
        )

        # -------------------------------------------------
        # 8. Verify chunk information was preserved
        # -------------------------------------------------

        for original, loaded in zip(
            self.test_chunks,
            loaded_records
        ):

            assert (
                loaded["chunk_text"]
                == original["chunk_text"]
            )

            if "chunk_id" in original:

                assert (
                    loaded["chunk_id"]
                    == original["chunk_id"]
                )

        # -------------------------------------------------
        # 9. Verify embedding records
        # -------------------------------------------------

        for record in loaded_records:

            assert "embedding" in record

            assert len(
                record["embedding"]
            ) == 384