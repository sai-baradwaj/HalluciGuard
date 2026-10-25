from data_preparation.loader import DatasetLoader
from data_preparation.cleaner import DatasetCleaner
from data_preparation.validator import DatasetValidator
from data_preparation.normalizer import DatasetNormalizer


class TestDatasetNormalizer:

    def setup_method(self):

        self.loader = DatasetLoader()
        self.cleaner = DatasetCleaner()
        self.validator = DatasetValidator()
        self.normalizer = DatasetNormalizer()

        self.halueval = (
            self.loader.load_halueval()
        )

        self.truthfulqa = (
            self.loader.load_truthfulqa()
        )

        self.cleaned_halueval, self.cleaned_truthfulqa = (
            self.cleaner.generate_clean_dataset(
                self.halueval,
                self.truthfulqa
            )
        )

        self.validated_halueval, self.validated_truthfulqa = (
            self.validator.generate_valid_dataset(
                self.cleaned_halueval,
                self.cleaned_truthfulqa
            )
        )

    # =====================================================
    # TEST HALUEVAL NORMALIZATION
    # =====================================================

    def test_normalize_halueval(self):

        normalized_halueval, normalized_truthfulqa = (
            self.normalizer.generate_normalized_dataset(
                self.validated_halueval,
                self.validated_truthfulqa
            )
        )

        assert normalized_halueval is not None
        assert isinstance(
            normalized_halueval,
            dict
        )

        expected_datasets = {
            "dialogue_data",
            "general_data",
            "qa_data",
            "summarization_data"
        }

        assert (
            set(normalized_halueval.keys())
            == expected_datasets
        )

    # =====================================================
    # TEST NORMALIZED HALUEVAL NOT EMPTY
    # =====================================================

    def test_normalized_halueval_not_empty(self):

        normalized_halueval, _ = (
            self.normalizer.generate_normalized_dataset(
                self.validated_halueval,
                self.validated_truthfulqa
            )
        )

        for dataset_name, dataset in (
            normalized_halueval.items()
        ):

            assert dataset is not None
            assert isinstance(dataset, list)
            assert len(dataset) > 0

    # =====================================================
    # TEST TRUTHFULQA NORMALIZATION
    # =====================================================

    def test_normalize_truthfulqa(self):

        _, normalized_truthfulqa = (
            self.normalizer.generate_normalized_dataset(
                self.validated_halueval,
                self.validated_truthfulqa
            )
        )

        assert normalized_truthfulqa is not None
        assert len(normalized_truthfulqa) > 0

    # =====================================================
    # TEST NORMALIZED RECORD STRUCTURE
    # =====================================================

    def test_normalized_record_structure(self):

        normalized_halueval, _ = (
            self.normalizer.generate_normalized_dataset(
                self.validated_halueval,
                self.validated_truthfulqa
            )
        )

        for dataset_name, dataset in (
            normalized_halueval.items()
        ):

            assert isinstance(dataset, list)

            if dataset:

                first_record = dataset[0]

                assert isinstance(
                    first_record,
                    dict
                )

                assert len(first_record) > 0

    # =====================================================
    # TEST NORMALIZATION DOES NOT INCREASE RECORDS
    # =====================================================

    def test_normalization_does_not_increase_records(self):

        normalized_halueval, normalized_truthfulqa = (
            self.normalizer.generate_normalized_dataset(
                self.validated_halueval,
                self.validated_truthfulqa
            )
        )

        for dataset_name in self.validated_halueval:

            validated_count = len(
                self.validated_halueval[dataset_name]
            )

            normalized_count = len(
                normalized_halueval[dataset_name]
            )

            assert normalized_count <= validated_count

        assert (
            len(normalized_truthfulqa)
            <= len(self.validated_truthfulqa)
        )