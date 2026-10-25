"""
Dataset Cleaner Module

Responsible for cleaning datasets and saving the cleaned
datasets for the next pipeline stage.
"""

import json
import re
from pathlib import Path

import pandas as pd


class DatasetCleaner:
    """
    Cleans HaluEval and TruthfulQA datasets.
    """

    def __init__(self):
        """
        Initialize output directory.
        """

        self.output_path = Path("outputs") / "cleaned"

        # Automatically create outputs/cleaned/
        self.output_path.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------
    # Text Cleaning
    # -------------------------------------------------------

    def clean_text(self, text):
        """
        Clean a single text field.
        """

        if pd.isna(text):
            return ""

        text = str(text)

        # Remove leading and trailing spaces
        text = text.strip()

        # Replace multiple spaces/newlines/tabs
        text = re.sub(r"\s+", " ", text)

        return text

    # -------------------------------------------------------
    # TruthfulQA Cleaning
    # -------------------------------------------------------

    def clean_truthfulqa(self, dataframe):
        """
        Clean the TruthfulQA dataset.
        """

        df = dataframe.copy()

        # Remove duplicate rows
        df = df.drop_duplicates()

        # Remove rows missing important fields
        df = df.dropna(subset=["Question", "Best Answer"])

        text_columns = [
            "Question",
            "Best Answer",
            "Best Incorrect Answer",
            "Correct Answers",
            "Incorrect Answers",
        ]

        for column in text_columns:

            if column in df.columns:
                df[column] = df[column].apply(self.clean_text)

        return df

    # -------------------------------------------------------
    # HaluEval Cleaning
    # -------------------------------------------------------

    def clean_halueval(self, datasets):
        """
        Clean every HaluEval dataset.
        """

        cleaned_datasets = {}

        for dataset_name, dataset in datasets.items():

            cleaned_records = []

            for record in dataset:

                cleaned_record = {}

                for key, value in record.items():

                    if isinstance(value, str):
                        cleaned_record[key] = self.clean_text(value)

                    else:
                        cleaned_record[key] = value

                cleaned_records.append(cleaned_record)

            cleaned_datasets[dataset_name] = cleaned_records

        return cleaned_datasets

    # -------------------------------------------------------
    # Save TruthfulQA
    # -------------------------------------------------------

    def save_truthfulqa(self, dataframe):
        """
        Save cleaned TruthfulQA dataset.
        """

        output_file = self.output_path / "TruthfulQA_cleaned.csv"

        dataframe.to_csv(output_file, index=False)

        print(f"✓ Saved: {output_file}")

    # -------------------------------------------------------
    # Save HaluEval
    # -------------------------------------------------------

    def save_halueval(self, datasets):
        """
        Save cleaned HaluEval datasets.
        """

        for dataset_name, dataset in datasets.items():

            output_file = self.output_path / f"{dataset_name}_cleaned.json"

            with open(output_file, "w", encoding="utf-8") as file:

                json.dump(
                    dataset,
                    file,
                    indent=4,
                    ensure_ascii=False,
                )

            print(f"✓ Saved: {output_file}")

    # -------------------------------------------------------
    # Complete Cleaning Pipeline
    # -------------------------------------------------------

    def generate_clean_dataset(self, halueval, truthfulqa):
        """
        Clean and save both datasets.
        """

        print("\nCleaning HaluEval...")

        cleaned_halueval = self.clean_halueval(halueval)

        print("Cleaning TruthfulQA...")

        cleaned_truthfulqa = self.clean_truthfulqa(truthfulqa)

        print("\nSaving cleaned datasets...\n")

        self.save_halueval(cleaned_halueval)

        self.save_truthfulqa(cleaned_truthfulqa)

        print("\nAll cleaned datasets saved successfully.")

        return cleaned_halueval, cleaned_truthfulqa