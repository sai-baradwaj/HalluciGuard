"""
Dataset Tokenizer Module

Responsible for tokenizing normalized datasets,
counting tokens, and providing token information
to the Chunker.

Tokenizer Model:
    BAAI/bge-small-en-v1.5

Important:
    This module performs TOKENIZATION only.
    It does not create embeddings.
    It does not create chunks.
"""

from transformers import AutoTokenizer
import pandas as pd


class DatasetTokenizer:
    """
    Tokenizes HaluEval and TruthfulQA datasets.

    The tokenizer is deliberately configured without
    truncation so that long texts can be measured
    accurately before the Chunker splits them.
    """

    # =====================================================
    # TOKENIZER CONFIGURATION
    # =====================================================

    MODEL_NAME = "BAAI/bge-small-en-v1.5"

    # =====================================================
    # CONSTRUCTOR
    # =====================================================

    def __init__(self):

        print(
            f"\nLoading Tokenizer : {self.MODEL_NAME}"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_NAME
        )

    # =====================================================
    # TOKENIZE TEXT
    # =====================================================

    def tokenize_text(self, text):
        """
        Convert text into token strings.

        No truncation is applied.

        Parameters
        ----------
        text : str

        Returns
        -------
        list
            List of token strings.
        """

        if text is None or pd.isna(text):
            return []

        text = str(text)

        if not text.strip():
            return []

        return self.tokenizer.tokenize(
            text,
            add_special_tokens=False
        )

    # =====================================================
    # GET TOKEN IDs
    # =====================================================

    def encode_text(self, text):
        """
        Convert text into token IDs.

        No truncation is applied.

        This method will be used by the Chunker.

        Parameters
        ----------
        text : str

        Returns
        -------
        list
            List of token IDs.
        """

        if text is None or pd.isna(text):
            return []

        text = str(text)

        if not text.strip():
            return []

        return self.tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=False
        )

    # =====================================================
    # COUNT TOKENS
    # =====================================================

    def count_tokens(self, text):
        """
        Count the tokens in a text.

        No maximum sequence length is imposed.

        Parameters
        ----------
        text : str

        Returns
        -------
        int
            Number of tokens.
        """

        return len(
            self.encode_text(text)
        )

    # =====================================================
    # DECODE TOKEN IDs
    # =====================================================

    def decode_tokens(self, token_ids):
        """
        Convert token IDs back into readable text.

        Parameters
        ----------
        token_ids : list

        Returns
        -------
        str
        """

        if not token_ids:
            return ""

        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        ).strip()

    # =====================================================
    # TRUTHFULQA TOKEN STATISTICS
    # =====================================================

    def tokenize_truthfulqa(self, dataframe):

        text_columns = [
            "Question",
            "Best Answer",
            "Best Incorrect Answer",
            "Correct Answers",
            "Incorrect Answers"
        ]

        statistics = {}

        print("\nTruthfulQA Token Statistics")
        print("-" * 40)

        for column in text_columns:

            if column not in dataframe.columns:
                continue

            token_counts = dataframe[column].apply(
                self.count_tokens
            )

            if len(token_counts) == 0:
                continue

            statistics[column] = {

                "Average Tokens": round(
                    token_counts.mean(),
                    2
                ),

                "Minimum Tokens": int(
                    token_counts.min()
                ),

                "Maximum Tokens": int(
                    token_counts.max()
                )
            }

            print(f"\n{column}")

            print(
                f"Average Tokens : "
                f"{statistics[column]['Average Tokens']}"
            )

            print(
                f"Minimum Tokens : "
                f"{statistics[column]['Minimum Tokens']}"
            )

            print(
                f"Maximum Tokens : "
                f"{statistics[column]['Maximum Tokens']}"
            )

        return statistics

    # =====================================================
    # HALUEVAL TOKEN STATISTICS
    # =====================================================

    def tokenize_halueval(self, datasets):

        statistics = {}

        print("\nHaluEval Token Statistics")
        print("-" * 40)

        for dataset_name, dataset in datasets.items():

            token_counts = []

            for record in dataset:

                total_tokens = 0

                for value in record.values():

                    if isinstance(value, str):

                        total_tokens += self.count_tokens(
                            value
                        )

                token_counts.append(
                    total_tokens
                )

            if not token_counts:
                continue

            statistics[dataset_name] = {

                "Average Tokens": round(
                    sum(token_counts)
                    / len(token_counts),
                    2
                ),

                "Minimum Tokens": min(
                    token_counts
                ),

                "Maximum Tokens": max(
                    token_counts
                )
            }

            print(f"\n{dataset_name}")

            print(
                f"Average Tokens : "
                f"{statistics[dataset_name]['Average Tokens']}"
            )

            print(
                f"Minimum Tokens : "
                f"{statistics[dataset_name]['Minimum Tokens']}"
            )

            print(
                f"Maximum Tokens : "
                f"{statistics[dataset_name]['Maximum Tokens']}"
            )

        return statistics

    # =====================================================
    # COMPLETE TOKENIZATION STATISTICS
    # =====================================================

    def generate_token_statistics(
        self,
        normalized_halueval,
        normalized_truthfulqa
    ):

        print("\nGenerating Token Statistics...")

        halueval_statistics = (
            self.tokenize_halueval(
                normalized_halueval
            )
        )

        truthfulqa_statistics = (
            self.tokenize_truthfulqa(
                normalized_truthfulqa
            )
        )

        print(
            "\nTokenization Completed Successfully."
        )

        return (
            halueval_statistics,
            truthfulqa_statistics
        )