from data_preparation.loader import DatasetLoader
from data_preparation.cleaner import DatasetCleaner
from data_preparation.validator import DatasetValidator
from data_preparation.normalizer import DatasetNormalizer

from chunking.tokenizer import DatasetTokenizer
from chunking.chunker import DatasetChunker


class TestDatasetChunker:

    def setup_method(self):

        self.loader = DatasetLoader()
        self.cleaner = DatasetCleaner()
        self.validator = DatasetValidator()
        self.normalizer = DatasetNormalizer()
        self.tokenizer = DatasetTokenizer()

        self.halueval = (
            self.loader.load_halueval()
        )

        self.truthfulqa = (
            self.loader.load_truthfulqa()
        )

        cleaned_halueval, cleaned_truthfulqa = (
            self.cleaner.generate_clean_dataset(
                self.halueval,
                self.truthfulqa
            )
        )

        validated_halueval, validated_truthfulqa = (
            self.validator.generate_valid_dataset(
                cleaned_halueval,
                cleaned_truthfulqa
            )
        )

        (
            self.normalized_halueval,
            self.normalized_truthfulqa
        ) = (
            self.normalizer.generate_normalized_dataset(
                validated_halueval,
                validated_truthfulqa
            )
        )

        self.chunker = DatasetChunker(
            self.tokenizer
        )

    # =====================================================
    # TEST CHUNK GENERATION
    # =====================================================

    def test_generate_chunks(self):

        chunked_halueval, chunked_truthfulqa = (
            self.chunker.generate_chunked_dataset(
                self.normalized_halueval,
                self.normalized_truthfulqa
            )
        )

        assert chunked_halueval is not None
        assert chunked_truthfulqa is not None

        assert isinstance(
            chunked_halueval,
            dict
        )

        assert isinstance(
            chunked_truthfulqa,
            list
        )

    # =====================================================
    # TEST HALUEVAL DATASETS
    # =====================================================

    def test_halueval_chunked_datasets(self):

        chunked_halueval, _ = (
            self.chunker.generate_chunked_dataset(
                self.normalized_halueval,
                self.normalized_truthfulqa
            )
        )

        expected_datasets = {
            "dialogue_data",
            "general_data",
            "qa_data",
            "summarization_data"
        }

        assert (
            set(chunked_halueval.keys())
            == expected_datasets
        )

        for dataset_name, chunks in (
            chunked_halueval.items()
        ):

            assert isinstance(
                chunks,
                list
            )

            assert len(chunks) > 0

    # =====================================================
    # TEST TRUTHFULQA CHUNKS
    # =====================================================

    def test_truthfulqa_chunks_not_empty(self):

        _, chunked_truthfulqa = (
            self.chunker.generate_chunked_dataset(
                self.normalized_halueval,
                self.normalized_truthfulqa
            )
        )

        assert isinstance(
            chunked_truthfulqa,
            list
        )

        assert len(chunked_truthfulqa) > 0

    # =====================================================
    # TEST CHUNK STRUCTURE
    # =====================================================

    def test_chunk_structure(self):

        chunked_halueval, chunked_truthfulqa = (
            self.chunker.generate_chunked_dataset(
                self.normalized_halueval,
                self.normalized_truthfulqa
            )
        )

        all_datasets = list(
            chunked_halueval.values()
        )

        all_datasets.append(
            chunked_truthfulqa
        )

        for chunks in all_datasets:

            assert len(chunks) > 0

            first_chunk = chunks[0]

            assert isinstance(
                first_chunk,
                dict
            )

            assert len(first_chunk) > 0

    # =====================================================
    # TEST CHUNK IDs
    # =====================================================

    def test_chunk_ids_exist(self):

        chunked_halueval, chunked_truthfulqa = (
            self.chunker.generate_chunked_dataset(
                self.normalized_halueval,
                self.normalized_truthfulqa
            )
        )

        all_datasets = list(
            chunked_halueval.values()
        )

        all_datasets.append(
            chunked_truthfulqa
        )

        for chunks in all_datasets:

            for chunk in chunks:

                assert isinstance(
                    chunk,
                    dict
                )

                assert "chunk_id" in chunk

                assert (
                    chunk["chunk_id"]
                    is not None
                )

                assert (
                    str(chunk["chunk_id"]).strip()
                    != ""
                )