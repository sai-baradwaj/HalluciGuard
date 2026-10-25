"""
Dataset Loader Module

Responsible for loading the HaluEval and TruthfulQA datasets
into Python objects.
"""

import json
import pandas as pd
from pathlib import Path


class DatasetLoader:
    """
    Responsible only for loading datasets.
    """

    def __init__(self):
        """
        Initialize dataset paths.
        """

        # Root datasets folder
        self.base_path = Path("datasets")

        # Dataset folders
        self.halueval_path = self.base_path / "HaluEval"
        self.truthfulqa_path = self.base_path / "TruthfulQA"

    def load_json(self, file_path):
        """
        Load both JSON and JSONL files.

        Parameters:
            file_path (str | Path)

        Returns:
            list | dict
        """

        file_path = Path(file_path)

        with open(file_path, "r", encoding="utf-8") as file:

            try:
                # Standard JSON
                return json.load(file)

            except json.JSONDecodeError:
                # JSONL (JSON Lines)
                file.seek(0)

                data = []

                for line in file:
                    line = line.strip()

                    if line:
                        data.append(json.loads(line))

                return data

    def load_csv(self, file_path):
        """
        Load any CSV file.

        Parameters:
            file_path (str | Path)

        Returns:
            pandas.DataFrame
        """

        file_path = Path(file_path)

        return pd.read_csv(file_path)

    def load_halueval(self):
        """
        Load all JSON files ins ide the HaluEval folder.

        Returns:
            dict
        """

        datasets = {}

        json_files = sorted(self.halueval_path.glob("*.json"))

        if not json_files:
            raise FileNotFoundError(
                f"No JSON files found in {self.halueval_path}"
            )

        for json_file in json_files:
            datasets[json_file.stem] = self.load_json(json_file)

        return datasets

    def load_truthfulqa(self):
        """
        Load the TruthfulQA CSV dataset.

        Returns:
            pandas.DataFrame
        """

        csv_files = sorted(self.truthfulqa_path.glob("*.csv"))

        if not csv_files:
            raise FileNotFoundError(
                f"No CSV file found in {self.truthfulqa_path}"
            )

        return self.load_csv(csv_files[0])