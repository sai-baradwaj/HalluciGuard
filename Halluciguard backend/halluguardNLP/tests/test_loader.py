import os
import pytest

from data_preparation.loader import DatasetLoader


class TestDatasetLoader:

    def setup_method(self):
        self.loader = DatasetLoader()

    # =====================================================
    # TEST HALUEVAL LOADING
    # =====================================================

    def test_load_halueval(self):

        halueval = self.loader.load_halueval()

        assert halueval is not None
        assert isinstance(halueval, dict)

        expected_datasets = {
            "dialogue_data",
            "general_data",
            "qa_data",
            "summarization_data"
        }

        assert set(halueval.keys()) == expected_datasets

    # =====================================================
    # TEST INDIVIDUAL HALUEVAL DATASETS
    # =====================================================

    def test_halueval_datasets_are_not_empty(self):

        halueval = self.loader.load_halueval()

        for dataset_name, dataset in halueval.items():

            assert dataset is not None
            assert len(dataset) > 0

    # =====================================================
    # TEST TRUTHFULQA LOADING
    # =====================================================

    def test_load_truthfulqa(self):

        truthfulqa = self.loader.load_truthfulqa()

        assert truthfulqa is not None
        assert len(truthfulqa) > 0

    # =====================================================
    # TEST REQUIRED DATASET FILES EXIST
    # =====================================================

    def test_dataset_files_exist(self):

        halueval_path = "datasets/HaluEval"

        truthfulqa_path = "datasets/TruthfulQA"

        assert os.path.isdir(halueval_path)
        assert os.path.isdir(truthfulqa_path)

        expected_halueval_files = [
            "dialogue_data.json",
            "general_data.json",
            "qa_data.json",
            "summarization_data.json"
        ]

        for filename in expected_halueval_files:

            filepath = os.path.join(
                halueval_path,
                filename
            )

            assert os.path.isfile(filepath)

        truthfulqa_file = os.path.join(
            truthfulqa_path,
            "TruthfulQA.csv"
        )

        assert os.path.isfile(truthfulqa_file)