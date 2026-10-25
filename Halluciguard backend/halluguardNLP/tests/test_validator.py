from data_preparation.loader import DatasetLoader
from data_preparation.cleaner import DatasetCleaner
from data_preparation.validator import DatasetValidator


class TestDatasetValidator:

    def setup_method(self):

        self.loader = DatasetLoader()
        self.cleaner = DatasetCleaner()
        self.validator = DatasetValidator()

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

    # =====================================================
    # TEST HALUEVAL VALIDATION
    # =====================================================

    def test_validate_halueval(self):

        validated_halueval, validated_truthfulqa = (
            self.validator.generate_valid_dataset(
                self.cleaned_halueval,
                self.cleaned_truthfulqa
            )
        )

        assert validated_halueval is not None
        assert isinstance(
            validated_halueval,
            dict
        )

        expected_datasets = {
            "dialogue_data",
            "general_data",
            "qa_data",
            "summarization_data"
        }

        assert (
            set(validated_halueval.keys())
            == expected_datasets
        )

    # =====================================================
    # TEST VALIDATED DATA IS NOT EMPTY
    # =====================================================

    def test_validated_halueval_not_empty(self):

        validated_halueval, _ = (
            self.validator.generate_valid_dataset(
                self.cleaned_halueval,
                self.cleaned_truthfulqa
            )
        )

        for dataset_name, dataset in (
            validated_halueval.items()
        ):

            assert dataset is not None
            assert isinstance(dataset, list)
            assert len(dataset) > 0

    # =====================================================
    # TEST TRUTHFULQA VALIDATION
    # =====================================================

    def test_validate_truthfulqa(self):

        _, validated_truthfulqa = (
            self.validator.generate_valid_dataset(
                self.cleaned_halueval,
                self.cleaned_truthfulqa
            )
        )

        assert validated_truthfulqa is not None
        assert len(validated_truthfulqa) > 0

    # =====================================================
    # TEST VALIDATED RECORD STRUCTURE
    # =====================================================

    def test_validated_record_structure(self):

        validated_halueval, _ = (
            self.validator.generate_valid_dataset(
                self.cleaned_halueval,
                self.cleaned_truthfulqa
            )
        )

        for dataset_name, dataset in (
            validated_halueval.items()
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
    # TEST VALIDATED DATA DOES NOT INCREASE
    # =====================================================

    def test_validation_does_not_increase_records(self):

        validated_halueval, validated_truthfulqa = (
            self.validator.generate_valid_dataset(
                self.cleaned_halueval,
                self.cleaned_truthfulqa
            )
        )

        for dataset_name in self.cleaned_halueval:

            cleaned_count = len(
                self.cleaned_halueval[dataset_name]
            )

            validated_count = len(
                validated_halueval[dataset_name]
            )

            assert validated_count <= cleaned_count

        assert (
            len(validated_truthfulqa)
            <= len(self.cleaned_truthfulqa)
        )