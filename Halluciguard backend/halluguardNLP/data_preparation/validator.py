"""
Dataset Validator Module

Responsible for validating cleaned datasets and saving
validated datasets for the next NLP pipeline stage.
"""

import json
from pathlib import Path

import pandas as pd


class DatasetValidator:
    """
    Validates HaluEval and TruthfulQA datasets.
    """

    def __init__(self):

        self.output_path = Path("outputs") / "validated"
        self.output_path.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------
    # TruthfulQA Validation
    # -------------------------------------------------------

    def validate_truthfulqa(self, dataframe):

        required_columns = [
            "Question",
            "Best Answer"
        ]

        # Check required columns
        for column in required_columns:

            if column not in dataframe.columns:
                raise ValueError(
                    f"Missing required column: {column}"
                )

        valid_rows = []
        invalid_count = 0

        for _, row in dataframe.iterrows():

            question = str(row["Question"]).strip()
            answer = str(row["Best Answer"]).strip()

            # Rule 1
            if question == "":
                invalid_count += 1
                continue

            # Rule 2
            if answer == "":
                invalid_count += 1
                continue

            # Rule 3
            if len(question) < 3:
                invalid_count += 1
                continue

            # Rule 4
            if len(answer) < 2:
                invalid_count += 1
                continue

            valid_rows.append(row)

        validated_df = pd.DataFrame(valid_rows)

        print("\nTruthfulQA Validation Report")
        print("----------------------------")
        print(f"Valid Records   : {len(validated_df)}")
        print(f"Invalid Records : {invalid_count}")

        return validated_df

    # -------------------------------------------------------
    # HaluEval Validation
    # -------------------------------------------------------

    def validate_halueval(self, datasets):

        validated_datasets = {}

        print("\nHaluEval Validation Report")
        print("----------------------------")

        for dataset_name, dataset in datasets.items():

            valid_records = []
            invalid_count = 0

            for record in dataset:

                # Must be dictionary
                if not isinstance(record, dict):
                    invalid_count += 1
                    continue

                # Cannot be empty
                if len(record) == 0:
                    invalid_count += 1
                    continue

                valid_records.append(record)

            validated_datasets[dataset_name] = valid_records

            print(f"\n{dataset_name}")
            print(f"Valid Records   : {len(valid_records)}")
            print(f"Invalid Records : {invalid_count}")

        return validated_datasets

    # -------------------------------------------------------
    # Save TruthfulQA
    # -------------------------------------------------------

    def save_truthfulqa(self, dataframe):

        output_file = self.output_path / "TruthfulQA_validated.csv"

        dataframe.to_csv(output_file, index=False)

        print(f"\n✓ Saved: {output_file}")

    # -------------------------------------------------------
    # Save HaluEval
    # -------------------------------------------------------

    def save_halueval(self, datasets):

        for dataset_name, dataset in datasets.items():

            output_file = (
                self.output_path /
                f"{dataset_name}_validated.json"
            )

            with open(output_file, "w", encoding="utf-8") as file:

                json.dump(
                    dataset,
                    file,
                    indent=4,
                    ensure_ascii=False
                )

            print(f"✓ Saved: {output_file}")

    # -------------------------------------------------------
    # Complete Validation Pipeline
    # -------------------------------------------------------

    def generate_valid_dataset(
        self,
        cleaned_halueval,
        cleaned_truthfulqa
    ):

        print("\nValidating HaluEval...")
        validated_halueval = self.validate_halueval(
            cleaned_halueval
        )

        print("\nValidating TruthfulQA...")
        validated_truthfulqa = self.validate_truthfulqa(
            cleaned_truthfulqa
        )

        print("\nSaving validated datasets...\n")

        self.save_halueval(validated_halueval)
        self.save_truthfulqa(validated_truthfulqa)

        print("\nAll validated datasets saved successfully.")

        return validated_halueval, validated_truthfulqa