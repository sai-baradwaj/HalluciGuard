"""
Dataset Inspector Module

Responsible for analyzing datasets and generating
basic statistics before preprocessing.
"""

import pandas as pd


class DatasetInspector:
    """
    Inspects HaluEval and TruthfulQA datasets.
    """

    def inspect_truthfulqa(self, dataframe):
        """
        Display detailed information about the TruthfulQA dataset.
        """

        print("\n" + "=" * 60)
        print("TRUTHFULQA DATASET REPORT")
        print("=" * 60)

        # Dataset Shape
        print(f"\nRows    : {dataframe.shape[0]}")
        print(f"Columns : {dataframe.shape[1]}")

        # Column Names
        print("\nColumn Names:")
        for column in dataframe.columns:
            print(f"• {column}")

        # Missing Values
        print("\nMissing Values:")
        print(dataframe.isnull().sum())

        # Duplicate Rows
        duplicates = dataframe.duplicated().sum()
        print(f"\nDuplicate Rows : {duplicates}")

        # Dataset Info
        print("\nDataset Information:")
        dataframe.info()

        # Category Distribution
        print("\nCategory Distribution:")
        print(dataframe["Category"].value_counts())

        # Sample Data
        print("\nFirst Five Rows:")
        print(dataframe.head())

    def inspect_halueval(self, datasets):
        """
        Display information about every HaluEval dataset.
        """

        print("\n" + "=" * 60)
        print("HALUEVAL DATASET REPORT")
        print("=" * 60)

        for dataset_name, dataset in datasets.items():

            print("\n" + "-" * 60)
            print(f"Dataset : {dataset_name}")
            print("-" * 60)

            print(f"Total Records : {len(dataset)}")

            if len(dataset) > 0:

                print("\nAvailable Fields:")

                for key in dataset[0].keys():
                    print(f"• {key}")

                print("\nSample Record:")

                print(dataset[0])

    def text_statistics(self, dataframe):
        """
        Display question and answer length statistics.
        """

        print("\n" + "=" * 60)
        print("TEXT STATISTICS")
        print("=" * 60)

        question_length = dataframe["Question"].str.split().str.len()

        answer_length = dataframe["Best Answer"].str.split().str.len()

        print(f"\nAverage Question Length : {question_length.mean():.2f} words")

        print(f"Average Answer Length   : {answer_length.mean():.2f} words")

        print(f"Longest Question        : {question_length.max()} words")

        print(f"Shortest Question       : {question_length.min()} words")

    def generate_report(self, halueval, truthfulqa):
        """
        Generate the complete inspection report.
        """

        self.inspect_halueval(halueval)

        self.inspect_truthfulqa(truthfulqa)

        self.text_statistics(truthfulqa)