"""
Dataset Normalizer Module

Responsible for normalizing validated datasets and saving
normalized datasets for the next NLP pipeline stage.
"""

import json
import re
import unicodedata
from pathlib import Path

import pandas as pd


class DatasetNormalizer:
    """
    Normalizes HaluEval and TruthfulQA datasets.
    """

    def __init__(self):

        self.output_path = Path("outputs") / "normalized"
        self.output_path.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------
    # Normalize Single Text
    # -------------------------------------------------------

    def normalize_text(self, text):
        """
        Normalize a single text string.
        """

        if pd.isna(text):
            return ""

        text = str(text)

        # Unicode Normalization
        text = unicodedata.normalize("NFKC", text)

        # Convert to lowercase
        text = text.lower()

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Remove spaces before punctuation
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)

        # Ensure one space after punctuation
        text = re.sub(r"([.,!?;:])([^\s])", r"\1 \2", text)

        return text

    # -------------------------------------------------------
    # Normalize TruthfulQA
    # -------------------------------------------------------

    def normalize_truthfulqa(self, dataframe):

        df = dataframe.copy()

        text_columns = [
            "Question",
            "Best Answer",
            "Best Incorrect Answer",
            "Correct Answers",
            "Incorrect Answers"
        ]

        for column in text_columns:

            if column in df.columns:
                df[column] = df[column].apply(
                    self.normalize_text
                )

        print("\nTruthfulQA Normalization Completed.")

        return df

    # -------------------------------------------------------
    # Normalize HaluEval
    # -------------------------------------------------------

    def normalize_halueval(self, datasets):

        normalized_datasets = {}

        print("\nHaluEval Normalization Report")
        print("------------------------------")

        for dataset_name, dataset in datasets.items():

            normalized_records = []

            for record in dataset:

                normalized_record = {}

                for key, value in record.items():

                    if isinstance(value, str):

                        normalized_record[key] = (
                            self.normalize_text(value)
                        )

                    else:

                        normalized_record[key] = value

                normalized_records.append(normalized_record)

            normalized_datasets[dataset_name] = normalized_records

            print(
                f"{dataset_name} : "
                f"{len(normalized_records)} records normalized"
            )

        return normalized_datasets

    # -------------------------------------------------------
    # Save TruthfulQA
    # -------------------------------------------------------

    def save_truthfulqa(self, dataframe):

        output_file = (
            self.output_path /
            "TruthfulQA_normalized.csv"
        )

        dataframe.to_csv(output_file, index=False)

        print(f"\n✓ Saved: {output_file}")

    # -------------------------------------------------------
    # Save HaluEval
    # -------------------------------------------------------

    def save_halueval(self, datasets):

        for dataset_name, dataset in datasets.items():

            output_file = (
                self.output_path /
                f"{dataset_name}_normalized.json"
            )

            with open(
                output_file,
                "w",
                encoding="utf-8"
            ) as file:

                json.dump(
                    dataset,
                    file,
                    indent=4,
                    ensure_ascii=False
                )

            print(f"✓ Saved: {output_file}")

    # -------------------------------------------------------
    # Complete Normalization Pipeline
    # -------------------------------------------------------

    def generate_normalized_dataset(
        self,
        validated_halueval,
        validated_truthfulqa
    ):  

        print("\nNormalizing HaluEval...")

        normalized_halueval = self.normalize_halueval(
            validated_halueval
        )

        print("\nNormalizing TruthfulQA...")

        normalized_truthfulqa = self.normalize_truthfulqa(
            validated_truthfulqa
        )

        print("\nSaving normalized datasets...\n")

        self.save_halueval(normalized_halueval)

        self.save_truthfulqa(normalized_truthfulqa)

        print("\nAll normalized datasets saved successfully.")

        return (
            normalized_halueval,
            normalized_truthfulqa
        )