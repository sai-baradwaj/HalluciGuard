"""
Dataset Chunker Module

Responsible for splitting normalized datasets into
retrieval-friendly chunks.

Chunking Strategy:
    - Preserve complete short records as single chunks.
    - For long records, use token-based overlapping chunks.
    - Prefer splitting at logical field/line boundaries.
    - Keep hallucination labels with the corresponding record.
    - Preserve chunk metadata required by embedding/retrieval.

This module does NOT:
    - Generate embeddings
    - Perform retrieval
    - Load embedding models
"""

import json
from pathlib import Path

import pandas as pd


class DatasetChunker:
    """
    Creates retrieval-friendly chunks from normalized
    HaluEval and TruthfulQA datasets.
    """

    # =====================================================
    # CHUNKING CONFIGURATION
    # =====================================================

    CHUNK_SIZE = 256
    CHUNK_OVERLAP = 32

    # =====================================================
    # CONSTRUCTOR
    # =====================================================

    def __init__(self, tokenizer):

        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            raise ValueError(
                "CHUNK_OVERLAP must be smaller than CHUNK_SIZE."
            )

        self.tokenizer = tokenizer

        self.output_path = (
            Path("outputs") / "chunked"
        )

        self.output_path.mkdir(
            parents=True,
            exist_ok=True
        )

        print("\nChunker Configuration")
        print("-" * 40)
        print(
            f"Chunk Size     : "
            f"{self.CHUNK_SIZE} tokens"
        )
        print(
            f"Chunk Overlap  : "
            f"{self.CHUNK_OVERLAP} tokens"
        )

    # =====================================================
    # TOKEN COUNT
    # =====================================================

    def _token_count(self, text):
        """
        Return the number of tokenizer tokens.
        """

        if text is None:
            return 0

        text = str(text).strip()

        if not text:
            return 0

        token_ids = self.tokenizer.encode_text(
            text
        )

        return len(token_ids)

    # =====================================================
    # CORE TOKEN CHUNKING
    # =====================================================

    def create_token_chunks(self, text):
        """
        Split text into overlapping token-based chunks.

        This is the fallback chunking mechanism for long
        pieces of text.

        Short text is preserved as one complete chunk.
        """

        if text is None:
            return []

        text = str(text).strip()

        if not text:
            return []

        token_ids = self.tokenizer.encode_text(
            text
        )

        if not token_ids:
            return []

        # -------------------------------------------------
        # IMPORTANT:
        # If the entire record fits inside one chunk,
        # keep it completely intact.
        # -------------------------------------------------

        if len(token_ids) <= self.CHUNK_SIZE:

            return [
                {
                    "chunk_text": text,
                    "token_count": len(token_ids),
                    "character_count": len(text)
                }
            ]

        chunks = []

        step = (
            self.CHUNK_SIZE
            - self.CHUNK_OVERLAP
        )

        start = 0

        while start < len(token_ids):

            end = min(
                start + self.CHUNK_SIZE,
                len(token_ids)
            )

            chunk_token_ids = (
                token_ids[start:end]
            )

            chunk_text = (
                self.tokenizer.decode_tokens(
                    chunk_token_ids
                )
            )

            chunk_text = (
                str(chunk_text).strip()
            )

            if chunk_text:

                chunks.append(
                    {
                        "chunk_text": chunk_text,

                        "token_count":
                            len(chunk_token_ids),

                        "character_count":
                            len(chunk_text)
                    }
                )

            if end >= len(token_ids):
                break

            start += step

        return chunks

    # =====================================================
    # STRUCTURE-AWARE CHUNKING
    # =====================================================

    def create_structured_chunks(self, fields):
        """
        Create chunks while preserving logical dataset fields.

        Parameters
        ----------
        fields : list[tuple[str, str]]
            Each item contains:
                (field_name, field_value)

        Returns
        -------
        list
            Chunk dictionaries.

        Strategy
        --------
        1. Build logical field blocks.
        2. Keep the complete record together when possible.
        3. If the record is too large, split between fields.
        4. If one field itself is too large, token-chunk that field.
        """

        field_blocks = []

        for field_name, field_value in fields:

            if field_value is None:
                continue

            value = str(field_value).strip()

            if not value:
                continue

            block = (
                f"{field_name}: {value}"
            )

            field_blocks.append(block)

        if not field_blocks:
            return []

        complete_text = "\n".join(
            field_blocks
        )

        # -------------------------------------------------
        # Best case:
        # Entire logical record fits in one chunk.
        # -------------------------------------------------

        if (
            self._token_count(complete_text)
            <= self.CHUNK_SIZE
        ):

            return [
                {
                    "chunk_text": complete_text,
                    "token_count":
                        self._token_count(
                            complete_text
                        ),
                    "character_count":
                        len(complete_text)
                }
            ]

        # -------------------------------------------------
        # Record is too large.
        #
        # Build chunks using complete fields wherever
        # possible.
        # -------------------------------------------------

        chunks = []

        current_blocks = []
        current_text = ""

        for block in field_blocks:

            # -------------------------------------------------
            # If this individual field is too large,
            # flush the current chunk first and then
            # token-chunk the large field.
            # -------------------------------------------------

            if (
                self._token_count(block)
                > self.CHUNK_SIZE
            ):

                if current_blocks:

                    text = "\n".join(
                        current_blocks
                    )

                    chunks.append(
                        {
                            "chunk_text": text,
                            "token_count":
                                self._token_count(
                                    text
                                ),
                            "character_count":
                                len(text)
                        }
                    )

                    current_blocks = []
                    current_text = ""

                large_chunks = (
                    self.create_token_chunks(
                        block
                    )
                )

                chunks.extend(
                    large_chunks
                )

                continue

            # -------------------------------------------------
            # Try adding the next complete field.
            # -------------------------------------------------

            if not current_blocks:

                current_blocks = [block]

                current_text = block

                continue

            candidate_text = (
                current_text
                + "\n"
                + block
            )

            candidate_tokens = (
                self._token_count(
                    candidate_text
                )
            )

            if candidate_tokens <= self.CHUNK_SIZE:

                current_blocks.append(
                    block
                )

                current_text = candidate_text

            else:

                # ---------------------------------------------
                # Current chunk is full enough.
                # Save it before starting another one.
                # ---------------------------------------------

                chunks.append(
                    {
                        "chunk_text":
                            current_text,

                        "token_count":
                            self._token_count(
                                current_text
                            ),

                        "character_count":
                            len(current_text)
                    }
                )

                current_blocks = [block]
                current_text = block

        # -------------------------------------------------
        # Save final chunk.
        # -------------------------------------------------

        if current_blocks:

            chunks.append(
                {
                    "chunk_text":
                        current_text,

                    "token_count":
                        self._token_count(
                            current_text
                        ),

                    "character_count":
                        len(current_text)
                }
            )

        return chunks

    # =====================================================
    # CREATE CHUNK METADATA
    # =====================================================

    def create_chunk_metadata(
        self,
        chunks,
        source_type,
        dataset_name,
        record_id,
        extra_metadata=None
    ):
        """
        Add metadata to generated chunks.

        Retrieval text contains only the actual content.
        Dataset annotations are preserved separately as metadata.
        """

        total_chunks = len(chunks)

        final_chunks = []

        if extra_metadata is None:
            extra_metadata = {}

        for index, chunk in enumerate(
            chunks,
            start=1
        ):

            chunk_data = {

                "chunk_uid":
                    f"{dataset_name}_{record_id}_{index}",

                "source_type":
                    source_type,

                "dataset":
                    dataset_name,

                "record_id":
                    record_id,

                "chunk_id":
                    index,

                "total_chunks":
                    total_chunks,

                "token_count":
                    chunk["token_count"],

                "character_count":
                    chunk["character_count"],

                "chunk_text":
                    chunk["chunk_text"]
            }

            for key, value in extra_metadata.items():

                chunk_data[key] = value

            final_chunks.append(
                chunk_data
            )

        return final_chunks

    # =====================================================
    # CHUNK A SINGLE RECORD
    # =====================================================

    def chunk_record(
        self,
        text,
        source_type,
        dataset_name,
        record_id
    ):
        """
        Generic chunking for an already-combined text record.

        Used as a fallback and retained for compatibility.
        """

        chunks = self.create_token_chunks(
            text
        )

        return self.create_chunk_metadata(
            chunks,
            source_type,
            dataset_name,
            record_id
        )

    # =====================================================
    # HALUEVAL FIELD EXTRACTION
    # =====================================================

    def _get_halueval_fields(self, record):
        """
        Extract retrieval-relevant fields from a HaluEval record.

        Dataset IDs and hallucination annotations are kept
        outside the embedding text. Only actual dataset
        content is used for retrieval.
        """

        fields = []

        # -------------------------------------------------
        # Fields that are NOT part of retrieval text:
        #
        # ID
        # hallucination
        # hallucination_spans
        #
        # These remain as metadata.
        # -------------------------------------------------

        excluded_fields = {
            "ID",
            "hallucination",
            "hallucination_spans"
        }

        # -------------------------------------------------
        # Preserve the original dataset field order.
        # -------------------------------------------------

        for key, value in record.items():

            if key in excluded_fields:
                continue

            if value is None:
                continue

            # -------------------------------------------------
            # Convert list values into readable text.
            # -------------------------------------------------

            if isinstance(value, list):

                value = "\n".join(
                    str(item)
                    for item in value
                    if str(item).strip()
                )

            else:

                value = str(value)

            value = value.strip()

            if value:

                fields.append(
                    (key, value)
                )

        return fields

    # =====================================================
    # HALUEVAL PROCESSING
    # =====================================================

    def chunk_halueval(self, datasets):
        """
        Create structure-aware chunks from all HaluEval datasets.
        """

        all_chunked_datasets = {}

        print("\nHaluEval Chunking")
        print("-" * 40)

        for dataset_name, dataset in datasets.items():

            dataset_chunks = []

            for record_id, record in enumerate(
                dataset
            ):

                fields = (
                    self._get_halueval_fields(
                        record
                    )
                )

                chunks = (
                    self.create_structured_chunks(
                        fields
                    )
                )

                extra_metadata = {}

                if "hallucination" in record:

                    extra_metadata["hallucination"] = (
                        record["hallucination"]
                    )

                if "hallucination_spans" in record:

                    extra_metadata["hallucination_spans"] = (
                        record["hallucination_spans"]
                    )

                chunks = (
                    self.create_chunk_metadata(
                        chunks,
                        "HaluEval",
                        dataset_name,
                        record_id,
                        extra_metadata
                    )
                )

                dataset_chunks.extend(
                    chunks
                )

            all_chunked_datasets[
                dataset_name
            ] = dataset_chunks

            print(
                f"\n{dataset_name}"
            )

            print(
                f"Original Records : "
                f"{len(dataset)}"
            )

            print(
                f"Generated Chunks : "
                f"{len(dataset_chunks)}"
            )

        return all_chunked_datasets

    # =====================================================
    # TRUTHFULQA PROCESSING
    # =====================================================

    def chunk_truthfulqa(self, dataframe):
        """
        Create structure-aware chunks from TruthfulQA.
        """

        all_chunks = []

        text_columns = [
            "Question",
            "Best Answer",
            "Best Incorrect Answer",
            "Correct Answers",
            "Incorrect Answers"
        ]

        print("\nTruthfulQA Chunking")
        print("-" * 40)

        for record_id, row in dataframe.iterrows():

            fields = []

            for column in text_columns:

                if column not in dataframe.columns:
                    continue

                value = row[column]

                if pd.isna(value):
                    continue

                value = str(value).strip()

                if value:

                    fields.append(
                        (
                            column,
                            value
                        )
                    )

            chunks = (
                self.create_structured_chunks(
                    fields
                )
            )

            chunks = (
                self.create_chunk_metadata(
                    chunks,
                    "TruthfulQA",
                    "TruthfulQA",
                    record_id
                )
            )

            all_chunks.extend(
                chunks
            )

        print(
            f"Original Records : "
            f"{len(dataframe)}"
        )

        print(
            f"Generated Chunks : "
            f"{len(all_chunks)}"
        )

        return all_chunks

    # =====================================================
    # SAVE HALUEVAL CHUNKS
    # =====================================================

    def save_halueval(self, datasets):
        """
        Save HaluEval chunks as JSON files.
        """

        print("\nSaving HaluEval Chunks...")

        for dataset_name, chunks in datasets.items():

            output_file = (
                self.output_path
                / f"{dataset_name}_chunked.json"
            )

            with open(
                output_file,
                "w",
                encoding="utf-8"
            ) as file:

                json.dump(
                    chunks,
                    file,
                    indent=4,
                    ensure_ascii=False
                )

            print(
                f"✓ Saved: {output_file}"
            )

    # =====================================================
    # SAVE TRUTHFULQA CHUNKS
    # =====================================================

    def save_truthfulqa(self, chunks):
        """
        Save TruthfulQA chunks as JSON.
        """

        output_file = (
            self.output_path
            / "TruthfulQA_chunked.json"
        )

        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                chunks,
                file,
                indent=4,
                ensure_ascii=False
            )

        print(
            f"✓ Saved: {output_file}"
        )

    # =====================================================
    # CHUNK STATISTICS
    # =====================================================

    def generate_chunk_statistics(
        self,
        halueval_chunks,
        truthfulqa_chunks
    ):
        """
        Generate basic chunk statistics.
        """

        print("\nChunk Statistics")
        print("-" * 40)

        # -------------------------------------------------
        # HaluEval Statistics
        # -------------------------------------------------

        for dataset_name, chunks in (
            halueval_chunks.items()
        ):

            if not chunks:
                continue

            token_counts = [
                chunk["token_count"]
                for chunk in chunks
            ]

            print(
                f"\n{dataset_name}"
            )

            print(
                f"Total Chunks : "
                f"{len(chunks)}"
            )

            print(
                f"Average Tokens : "
                f"{sum(token_counts) / len(token_counts):.2f}"
            )

            print(
                f"Minimum Tokens : "
                f"{min(token_counts)}"
            )

            print(
                f"Maximum Tokens : "
                f"{max(token_counts)}"
            )

        # -------------------------------------------------
        # TruthfulQA Statistics
        # -------------------------------------------------

        if truthfulqa_chunks:

            token_counts = [
                chunk["token_count"]
                for chunk in truthfulqa_chunks
            ]

            print("\nTruthfulQA")

            print(
                f"Total Chunks : "
                f"{len(truthfulqa_chunks)}"
            )

            print(
                f"Average Tokens : "
                f"{sum(token_counts) / len(token_counts):.2f}"
            )

            print(
                f"Minimum Tokens : "
                f"{min(token_counts)}"
            )

            print(
                f"Maximum Tokens : "
                f"{max(token_counts)}"
            )

    # =====================================================
    # COMPLETE CHUNKING PIPELINE
    # =====================================================

    def generate_chunked_dataset(
        self,
        normalized_halueval,
        normalized_truthfulqa
    ):
        """
        Complete chunking pipeline.

        Returns
        -------
        tuple
            HaluEval chunks and TruthfulQA chunks.
        """

        print("\n")
        print("=" * 70)
        print("Starting Dataset Chunking...")
        print("=" * 70)

        # -------------------------------------------------
        # HaluEval
        # -------------------------------------------------

        halueval_chunks = (
            self.chunk_halueval(
                normalized_halueval
            )
        )

        # -------------------------------------------------
        # TruthfulQA
        # -------------------------------------------------

        truthfulqa_chunks = (
            self.chunk_truthfulqa(
                normalized_truthfulqa
            )
        )

        # -------------------------------------------------
        # Statistics
        # -------------------------------------------------

        self.generate_chunk_statistics(
            halueval_chunks,
            truthfulqa_chunks
        )

        # -------------------------------------------------
        # Save
        # -------------------------------------------------

        print("\n")

        self.save_halueval(
            halueval_chunks
        )

        self.save_truthfulqa(
            truthfulqa_chunks
        )

        print("\n")
        print("=" * 70)
        print(
            "Dataset Chunking Completed Successfully."
        )
        print("=" * 70)

        return (
            halueval_chunks,
            truthfulqa_chunks
        )