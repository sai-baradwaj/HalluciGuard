from data_preparation.loader import DatasetLoader
from data_preparation.cleaner import DatasetCleaner
from data_preparation.validator import DatasetValidator
from data_preparation.normalizer import DatasetNormalizer

from chunking.tokenizer import DatasetTokenizer


class TestDatasetTokenizer:

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

        self.normalized_halueval, self.normalized_truthfulqa = (
            self.normalizer.generate_normalized_dataset(
                validated_halueval,
                validated_truthfulqa
            )
        )

    # =====================================================
    # TEST TOKEN STATISTICS GENERATION
    # =====================================================

    def test_generate_token_statistics(self):

        halueval_stats, truthfulqa_stats = (
            self.tokenizer.generate_token_statistics(
                self.normalized_halueval,
                self.normalized_truthfulqa
            )
        )

        assert halueval_stats is not None
        assert truthfulqa_stats is not None

        assert isinstance(
            halueval_stats,
            dict
        )

        assert isinstance(
            truthfulqa_stats,
            dict
        )

    # =====================================================
    # TEST HALUEVAL TOKEN STATISTICS
    # =====================================================

    def test_halueval_token_statistics(self):

        halueval_stats, _ = (
            self.tokenizer.generate_token_statistics(
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
            set(halueval_stats.keys())
            == expected_datasets
        )

    # =====================================================
    # TEST TRUTHFULQA TOKEN STATISTICS
    # =====================================================

    def test_truthfulqa_token_statistics(self):

        _, truthfulqa_stats = (
            self.tokenizer.generate_token_statistics(
                self.normalized_halueval,
                self.normalized_truthfulqa
            )
        )

        assert len(truthfulqa_stats) > 0

    # =====================================================
    # TEST TOKEN STATISTICS ARE NOT EMPTY
    # =====================================================

    def test_token_statistics_not_empty(self):

        halueval_stats, truthfulqa_stats = (
            self.tokenizer.generate_token_statistics(
                self.normalized_halueval,
                self.normalized_truthfulqa
            )
        )

        for dataset_name, stats in (
            halueval_stats.items()
        ):

            assert stats is not None

        assert truthfulqa_stats is not None

    # =====================================================
    # TEST TOKEN STATISTICS STRUCTURE
    # =====================================================

    def test_token_statistics_structure(self):

        halueval_stats, truthfulqa_stats = (
            self.tokenizer.generate_token_statistics(
                self.normalized_halueval,
                self.normalized_truthfulqa
            )
        )

        for dataset_name, stats in (
            halueval_stats.items()
        ):

            assert isinstance(
                stats,
                dict
            )

            assert len(stats) > 0

        assert isinstance(
            truthfulqa_stats,
            dict
        )

        assert len(truthfulqa_stats) > 0