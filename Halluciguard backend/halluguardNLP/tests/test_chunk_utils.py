from chunking.chunk_utils import ChunkUtils


class TestChunkUtils:

    def setup_method(self):

        self.chunk_utils = ChunkUtils()

        self.dataset_names = [
            "dialogue_data",
            "general_data",
            "qa_data",
            "summarization_data",
            "TruthfulQA"
        ]

    # =====================================================
    # TEST VALIDATE ALL DATASETS
    # =====================================================

    def test_validate_all_datasets(self):

        results = (
            self.chunk_utils.validate_all_datasets(
                self.dataset_names
            )
        )

        assert results is not None
        assert isinstance(
            results,
            dict
        )

        assert (
            set(results.keys())
            == set(self.dataset_names)
        )

    # =====================================================
    # TEST DATASET VALIDATION RESULTS
    # =====================================================

    def test_dataset_validation_results(self):

        results = (
            self.chunk_utils.validate_all_datasets(
                self.dataset_names
            )
        )

        for dataset_name in self.dataset_names:

            result = results[dataset_name]

            assert isinstance(
                result,
                dict
            )

            required_fields = {
                "total_chunks",
                "valid_chunks",
                "invalid_chunks",
                "is_valid"
            }

            assert required_fields.issubset(
                result.keys()
            )

    # =====================================================
    # TEST CHUNK COUNTS
    # =====================================================

    def test_chunk_counts(self):

        results = (
            self.chunk_utils.validate_all_datasets(
                self.dataset_names
            )
        )

        for dataset_name in self.dataset_names:

            result = results[dataset_name]

            assert (
                result["total_chunks"]
                >= 0
            )

            assert (
                result["valid_chunks"]
                >= 0
            )

            assert (
                result["invalid_chunks"]
                >= 0
            )

            assert (
                result["valid_chunks"]
                + result["invalid_chunks"]
                == result["total_chunks"]
            )

    # =====================================================
    # TEST ALL CURRENT DATASETS ARE VALID
    # =====================================================

    def test_all_datasets_are_valid(self):

        results = (
            self.chunk_utils.validate_all_datasets(
                self.dataset_names
            )
        )

        for dataset_name in self.dataset_names:

            result = results[dataset_name]

            assert result["is_valid"] is True

            assert (
                result["invalid_chunks"]
                == 0
            )

    # =====================================================
    # TEST EXPECTED CHUNK COUNTS
    # =====================================================

    def test_expected_chunk_counts(self):

        expected_counts = {
            "dialogue_data": 10116,
            "general_data": 4618,
            "qa_data": 10079,
            "summarization_data": 48735,
            "TruthfulQA": 799
        }

        results = (
            self.chunk_utils.validate_all_datasets(
                self.dataset_names
            )
        )

        for dataset_name, expected_count in (
            expected_counts.items()
        ):

            actual_count = (
                results[dataset_name]["total_chunks"]
            )

            assert (
                actual_count
                == expected_count
            )