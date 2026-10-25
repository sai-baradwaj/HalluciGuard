from data_preparation.loader import DatasetLoader
from data_preparation.cleaner import DatasetCleaner


class TestDatasetCleaner:

    def setup_method(self):

        self.loader = DatasetLoader()
        self.cleaner = DatasetCleaner()

        self.halueval = (
            self.loader.load_halueval()
        )

        self.truthfulqa = (
            self.loader.load_truthfulqa()
        )

    # =====================================================
    # TEST CLEANING HALUEVAL
    # =====================================================

    def test_clean_halueval(self):

        cleaned_halueval, cleaned_truthfulqa = (
            self.cleaner.generate_clean_dataset(
                self.halueval,
                self.truthfulqa
            )
        )

        assert cleaned_halueval is not None
        assert isinstance(
            cleaned_halueval,
            dict
        )

        assert len(cleaned_halueval) == 4

        expected_datasets = {
            "dialogue_data",
            "general_data",
            "qa_data",
            "summarization_data"
        }

        assert (
            set(cleaned_halueval.keys())
            == expected_datasets
        )

    # =====================================================
    # TEST CLEANED HALUEVAL DATA
    # =====================================================

    def test_cleaned_halueval_not_empty(self):

        cleaned_halueval, _ = (
            self.cleaner.generate_clean_dataset(
                self.halueval,
                self.truthfulqa
            )
        )

        for dataset_name, dataset in (
            cleaned_halueval.items()
        ):

            assert dataset is not None
            assert isinstance(dataset, list)
            assert len(dataset) > 0

    # =====================================================
    # TEST CLEANING TRUTHFULQA
    # =====================================================

    def test_clean_truthfulqa(self):

        _, cleaned_truthfulqa = (
            self.cleaner.generate_clean_dataset(
                self.halueval,
                self.truthfulqa
            )
        )

        assert cleaned_truthfulqa is not None
        assert len(cleaned_truthfulqa) > 0

    # =====================================================
    # TEST CLEANED HALUEVAL RECORD STRUCTURE
    # =====================================================

    def test_cleaned_halueval_record_structure(self):

        cleaned_halueval, _ = (
            self.cleaner.generate_clean_dataset(
                self.halueval,
                self.truthfulqa
            )
        )

        for dataset_name, dataset in (
            cleaned_halueval.items()
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
    # TEST CLEANING DOES NOT INCREASE RECORDS
    # =====================================================

    def test_cleaning_does_not_increase_records(self):

        cleaned_halueval, cleaned_truthfulqa = (
            self.cleaner.generate_clean_dataset(
                self.halueval,
                self.truthfulqa
            )
        )

        for dataset_name in self.halueval:

            original_count = len(
                self.halueval[dataset_name]
            )

            cleaned_count = len(
                cleaned_halueval[dataset_name]
            )

            assert cleaned_count <= original_count

        assert (
            len(cleaned_truthfulqa)
            <= len(self.truthfulqa)
        )