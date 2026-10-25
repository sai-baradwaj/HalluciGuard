"""
HalluciGuard Batch Embedding Processor

Processes Module 2 chunks through Module 3.

Pipeline:

Module 2 Chunks
        ↓
DatasetEmbedder
        ↓
Embedding Validation
        ↓
Checkpoint / Resume
        ↓
Final Validation
        ↓
JSON + NumPy Export

Features:
- Full dataset processing
- GPU batching through DatasetEmbedder
- Checkpoint/resume support
- Stale output detection
- Checkpoint validation
- Duplicate checkpoint protection
- Final count verification
- 384-dimensional embedding verification
"""

import json
import os

import numpy as np

from chunking.chunk_utils import ChunkUtils

from embeddings.embedding_model import EmbeddingModel
from embeddings.embedder import DatasetEmbedder
from embeddings.embedding_utils import EmbeddingUtils
from embeddings.exporter import EmbeddingExporter


class BatchEmbeddingProcessor:

    # =====================================================
    # DATASETS
    # =====================================================

    DATASETS = [
        "dialogue_data",
        "general_data",
        "qa_data",
        "summarization_data",
        "TruthfulQA"
    ]

    # =====================================================
    # TEST CONFIGURATION
    # =====================================================

    # FALSE = REAL FULL DATASET RUN
    TEST_MODE = False

    # Only used when TEST_MODE = True
    TEST_CHUNKS_PER_DATASET = 100

    # =====================================================
    # CHECKPOINT CONFIGURATION
    # =====================================================

    # 320 embeddings = 10 embedding batches
    CHECKPOINT_INTERVAL = 320

    CHECKPOINT_ENABLED = True

    # =====================================================
    # EMBEDDING CONFIGURATION
    # =====================================================

    EXPECTED_EMBEDDING_DIMENSION = 384

    # =====================================================
    # OUTPUT DIRECTORIES
    # =====================================================

    FINAL_OUTPUT_DIRECTORY = (
        "outputs/embeddings/full"
    )

    CHECKPOINT_DIRECTORY = (
        "outputs/embeddings/checkpoints"
    )

    # =====================================================
    # CONSTRUCTOR
    # =====================================================

    def __init__(self):

        print("\n")
        print("=" * 70)
        print("HalluciGuard Batch Embedding Processor")
        print("=" * 70)

        # -------------------------------------------------
        # Create directories
        # -------------------------------------------------

        os.makedirs(
            self.FINAL_OUTPUT_DIRECTORY,
            exist_ok=True
        )

        os.makedirs(
            self.CHECKPOINT_DIRECTORY,
            exist_ok=True
        )

        print(
            f"\nFinal Output Directory : "
            f"{self.FINAL_OUTPUT_DIRECTORY}"
        )

        print(
            f"Checkpoint Directory : "
            f"{self.CHECKPOINT_DIRECTORY}"
        )

        # -------------------------------------------------
        # Module 2
        # -------------------------------------------------

        self.chunk_utils = ChunkUtils()

        # -------------------------------------------------
        # Embedding Model
        # -------------------------------------------------

        self.embedding_model = EmbeddingModel()

        # -------------------------------------------------
        # Embedder
        # -------------------------------------------------

        self.embedder = DatasetEmbedder(
            self.embedding_model
        )

        # -------------------------------------------------
        # Embedding Utilities
        # -------------------------------------------------

        self.embedding_utils = EmbeddingUtils()

        # -------------------------------------------------
        # Exporter
        # -------------------------------------------------

        self.exporter = EmbeddingExporter(
            output_directory=
            self.FINAL_OUTPUT_DIRECTORY
        )

    # =====================================================
    # PATH HELPERS
    # =====================================================

    def get_checkpoint_path(
        self,
        dataset_name
    ):

        return os.path.join(
            self.CHECKPOINT_DIRECTORY,
            f"{dataset_name}.jsonl"
        )

    # -----------------------------------------------------

    def get_final_json_path(
        self,
        dataset_name
    ):

        return os.path.join(
            self.FINAL_OUTPUT_DIRECTORY,
            f"{dataset_name}.json"
        )

    # -----------------------------------------------------

    def get_final_numpy_path(
        self,
        dataset_name
    ):

        return os.path.join(
            self.FINAL_OUTPUT_DIRECTORY,
            f"{dataset_name}.npy"
        )

    # =====================================================
    # DELETE FINAL OUTPUT
    # =====================================================

    def delete_final_output(
        self,
        dataset_name
    ):

        json_path = (
            self.get_final_json_path(
                dataset_name
            )
        )

        numpy_path = (
            self.get_final_numpy_path(
                dataset_name
            )
        )

        deleted = False

        if os.path.exists(json_path):

            os.remove(
                json_path
            )

            deleted = True

            print(
                f"Removed stale JSON : "
                f"{json_path}"
            )

        if os.path.exists(numpy_path):

            os.remove(
                numpy_path
            )

            deleted = True

            print(
                f"Removed stale NumPy : "
                f"{numpy_path}"
            )

        return deleted

    # =====================================================
    # LOAD JSON FILE
    # =====================================================

    def load_json_file(
        self,
        path
    ):

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    # =====================================================
    # CHECK FINAL OUTPUT
    # =====================================================

    def validate_final_output(
        self,
        dataset_name,
        chunks
    ):
        """
        Determines whether an existing final output
        is genuinely complete.

        Checks:

        1. JSON exists
        2. NumPy exists
        3. JSON record count matches chunk count
        4. NumPy row count matches chunk count
        5. NumPy dimension is 384
        6. Chunk IDs are preserved and ordered
        7. Embedding records contain valid embeddings
        """

        json_path = (
            self.get_final_json_path(
                dataset_name
            )
        )

        numpy_path = (
            self.get_final_numpy_path(
                dataset_name
            )
        )

        # -------------------------------------------------
        # Both files must exist
        # -------------------------------------------------

        if not os.path.exists(
            json_path
        ):

            return False

        if not os.path.exists(
            numpy_path
        ):

            return False

        try:

            # -------------------------------------------------
            # Load JSON
            # -------------------------------------------------

            records = (
                self.load_json_file(
                    json_path
                )
            )

            # -------------------------------------------------
            # Load NumPy
            # -------------------------------------------------

            embeddings = np.load(
                numpy_path,
                mmap_mode="r"
            )

            # -------------------------------------------------
            # Count check
            # -------------------------------------------------

            expected_count = len(
                chunks
            )

            if len(records) != expected_count:

                print(
                    f"\n⚠ Existing JSON is incomplete."
                )

                print(
                    f"Expected Records : "
                    f"{expected_count:,}"
                )

                print(
                    f"Found Records : "
                    f"{len(records):,}"
                )

                return False

            # -------------------------------------------------
            # NumPy shape check
            # -------------------------------------------------

            expected_shape = (
                expected_count,
                self.EXPECTED_EMBEDDING_DIMENSION
            )

            if embeddings.shape != expected_shape:

                print(
                    f"\n⚠ Existing NumPy file "
                    f"has incorrect shape."
                )

                print(
                    f"Expected Shape : "
                    f"{expected_shape}"
                )

                print(
                    f"Found Shape : "
                    f"{embeddings.shape}"
                )

                return False

            # -------------------------------------------------
            # Chunk ID check
            # -------------------------------------------------

            for index, chunk in enumerate(
                chunks
            ):

                if index >= len(records):

                    return False

                expected_id = (
                    chunk.get(
                        "chunk_id"
                    )
                )

                actual_id = (
                    records[index].get(
                        "chunk_id"
                    )
                )

                if (
                    expected_id
                    != actual_id
                ):

                    print(
                        f"\n⚠ Existing JSON "
                        f"chunk ID mismatch."
                    )

                    print(
                        f"Position : {index}"
                    )

                    print(
                        f"Expected : "
                        f"{expected_id}"
                    )

                    print(
                        f"Found : "
                        f"{actual_id}"
                    )

                    return False

            # -------------------------------------------------
            # Embedding record validation
            # -------------------------------------------------

            validation = (
                self.embedding_utils
                .validate_records(
                    records
                )
            )

            if not validation[
                "is_valid"
            ]:

                print(
                    "\n⚠ Existing embeddings "
                    "failed validation."
                )

                return False

            # -------------------------------------------------
            # Everything passed
            # -------------------------------------------------

            return True

        except Exception as error:

            print(
                f"\n⚠ Could not validate "
                f"existing output:"
            )

            print(
                f"  {error}"
            )

            return False

    # =====================================================
    # LOAD CHECKPOINT
    # =====================================================

    def load_checkpoint(
        self,
        dataset_name,
        chunks
    ):

        if not self.CHECKPOINT_ENABLED:

            return []

        checkpoint_path = (
            self.get_checkpoint_path(
                dataset_name
            )
        )

        if not os.path.exists(
            checkpoint_path
        ):

            return []

        print(
            f"\nCheckpoint found:"
            f"\n{checkpoint_path}"
        )

        records = []

        try:

            with open(
                checkpoint_path,
                "r",
                encoding="utf-8"
            ) as file:

                for line in file:

                    line = line.strip()

                    if not line:
                        continue

                    records.append(
                        json.loads(
                            line
                        )
                    )

        except Exception as error:

            print(
                "\n⚠ Checkpoint could not "
                "be read."
            )

            print(
                f"Reason : {error}"
            )

            print(
                "Removing invalid checkpoint."
            )

            self.remove_checkpoint(
                dataset_name
            )

            return []

        # =================================================
        # CHECKPOINT COUNT
        # =================================================

        if len(records) > len(chunks):

            print(
                "\n⚠ Checkpoint contains "
                "more records than the dataset."
            )

            print(
                "Removing invalid checkpoint."
            )

            self.remove_checkpoint(
                dataset_name
            )

            return []

        # =================================================
        # CHECKPOINT RECORD VALIDATION
        # =================================================

        validation = (
            self.embedding_utils
            .validate_records(
                records
            )
        )

        if not validation[
            "is_valid"
        ]:

            print(
                "\n⚠ Checkpoint embeddings "
                "failed validation."
            )

            print(
                "Removing invalid checkpoint."
            )

            self.remove_checkpoint(
                dataset_name
            )

            return []

        # =================================================
        # CHECKPOINT ID VALIDATION
        # =================================================

        valid_checkpoint = True

        for index, record in enumerate(
            records
        ):

            expected_id = (
                chunks[index].get(
                    "chunk_id"
                )
            )

            actual_id = (
                record.get(
                    "chunk_id"
                )
            )

            if (
                expected_id
                != actual_id
            ):

                print(
                    "\n⚠ Checkpoint does not "
                    "match current Module 2 chunks."
                )

                print(
                    f"Position : {index}"
                )

                print(
                    f"Expected : "
                    f"{expected_id}"
                )

                print(
                    f"Found : "
                    f"{actual_id}"
                )

                valid_checkpoint = False

                break

        if not valid_checkpoint:

            print(
                "Removing invalid checkpoint."
            )

            self.remove_checkpoint(
                dataset_name
            )

            return []

        print(
            f"Checkpoint Records : "
            f"{len(records):,}"
        )

        print(
            "✓ Checkpoint validated"
        )

        return records

    # =====================================================
    # APPEND CHECKPOINT
    # =====================================================

    def append_checkpoint(
        self,
        dataset_name,
        records
    ):

        if not self.CHECKPOINT_ENABLED:

            return

        if not records:

            return

        checkpoint_path = (
            self.get_checkpoint_path(
                dataset_name
            )
        )

        with open(
            checkpoint_path,
            "a",
            encoding="utf-8"
        ) as file:

            for record in records:

                file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False
                    )
                    + "\n"
                )

    # =====================================================
    # REMOVE CHECKPOINT
    # =====================================================

    def remove_checkpoint(
        self,
        dataset_name
    ):

        checkpoint_path = (
            self.get_checkpoint_path(
                dataset_name
            )
        )

        if os.path.exists(
            checkpoint_path
        ):

            os.remove(
                checkpoint_path
            )

    # =====================================================
    # PROCESS ONE DATASET
    # =====================================================

    def process_dataset(
        self,
        dataset_name
    ):

        print("\n")
        print("=" * 70)

        print(
            f"PROCESSING DATASET : "
            f"{dataset_name}"
        )

        print("=" * 70)

        # =================================================
        # STEP 1 : LOAD MODULE 2 CHUNKS
        # =================================================

        print(
            "\nLoading saved Module 2 chunks..."
        )

        chunks = (
            self.chunk_utils.load_dataset(
                dataset_name
            )
        )

        original_count = len(
            chunks
        )

        print(
            f"Total Saved Chunks : "
            f"{original_count:,}"
        )

        # =================================================
        # STEP 2 : TEST MODE
        # =================================================

        if self.TEST_MODE:

            chunks = chunks[
                :self.TEST_CHUNKS_PER_DATASET
            ]

            print(
                "\n⚠ TEST MODE ENABLED"
            )

            print(
                f"Chunks Selected : "
                f"{len(chunks):,}"
            )

        # =================================================
        # STEP 3 : VALIDATE MODULE 2 CHUNKS
        # =================================================

        print(
            "\nValidating chunks..."
        )

        validation = (
            self.chunk_utils.validate_chunks(
                chunks
            )
        )

        print(
            f"Valid Chunks : "
            f"{validation['valid_chunks']:,}"
        )

        print(
            f"Invalid Chunks : "
            f"{validation['invalid_chunks']:,}"
        )

        if not validation[
            "is_valid"
        ]:

            raise ValueError(
                f"Chunk validation failed "
                f"for {dataset_name}."
            )

        print(
            "✓ Chunk Validation Passed"
        )

        # =================================================
        # STEP 4 : CHECK EXISTING FINAL OUTPUT
        # =================================================

        if not self.TEST_MODE:

            json_path = (
                self.get_final_json_path(
                    dataset_name
                )
            )

            numpy_path = (
                self.get_final_numpy_path(
                    dataset_name
                )
            )

            either_output_exists = (
                os.path.exists(json_path)
                or
                os.path.exists(numpy_path)
            )

            if either_output_exists:

                print(
                    "\nChecking existing "
                    "final output..."
                )

                if self.validate_final_output(
                    dataset_name,
                    chunks
                ):

                    print(
                        "\n✓ Existing output "
                        "is COMPLETE and VALID."
                    )

                    print(
                        "Skipping dataset."
                    )

                    return {
                        "dataset":
                            dataset_name,

                        "status":
                            "already_completed",

                        "chunks":
                            len(chunks),

                        "embeddings":
                            len(chunks)
                    }

                # -------------------------------------------------
                # Existing files are stale/incomplete
                # -------------------------------------------------

                print(
                    "\n⚠ Existing output is "
                    "incomplete or invalid."
                )

                print(
                    "Removing stale output "
                    "before rebuilding."
                )

                self.delete_final_output(
                    dataset_name
                )

        # =================================================
        # STEP 5 : LOAD CHECKPOINT
        # =================================================

        checkpoint_records = (
            self.load_checkpoint(
                dataset_name,
                chunks
            )
        )

        completed_count = len(
            checkpoint_records
        )

        # =================================================
        # STEP 6 : DETERMINE REMAINING CHUNKS
        # =================================================

        remaining_chunks = (
            chunks[
                completed_count:
            ]
        )

        print(
            f"\nAlready Processed : "
            f"{completed_count:,}"
        )

        print(
            f"Remaining Chunks : "
            f"{len(remaining_chunks):,}"
        )

        # =================================================
        # STEP 7 : RECORD STORAGE
        # =================================================

        all_records = (
            checkpoint_records.copy()
        )

        # Number of records already written
        # to the checkpoint file.

        checkpointed_count = (
            completed_count
        )

        # =================================================
        # STEP 8 : EMBEDDING GENERATION
        # =================================================

        if remaining_chunks:

            print("\n")
            print("=" * 70)

            print(
                "Starting Embedding Generation"
            )

            print("=" * 70)

            print(
                f"\nEmbedding Batch Size : "
                f"{self.embedder.BATCH_SIZE}"
            )

            processing_batch_size = (
                self.embedder.BATCH_SIZE
            )

            processed_this_run = 0

            # -------------------------------------------------
            # Process batches
            # -------------------------------------------------

            for start in range(
                0,
                len(remaining_chunks),
                processing_batch_size
            ):

                end = min(
                    start
                    + processing_batch_size,
                    len(remaining_chunks)
                )

                current_chunks = (
                    remaining_chunks[
                        start:end
                    ]
                )

                # ---------------------------------------------
                # Generate embeddings
                # ---------------------------------------------

                batch_records = (
                    self.embedder.embed_chunks(
                        current_chunks,
                        show_progress=False
                    )
                )

                # ---------------------------------------------
                # Count verification
                # ---------------------------------------------

                if (
                    len(batch_records)
                    != len(current_chunks)
                ):

                    raise ValueError(
                        "Embedding count mismatch "
                        f"in batch {start}:{end} "
                        f"for {dataset_name}."
                    )

                # ---------------------------------------------
                # Embedding validation
                # ---------------------------------------------

                batch_validation = (
                    self.embedding_utils
                    .validate_records(
                        batch_records
                    )
                )

                if not batch_validation[
                    "is_valid"
                ]:

                    raise ValueError(
                        "Embedding validation "
                        "failed for batch "
                        f"{start}:{end} "
                        f"in {dataset_name}."
                    )

                # ---------------------------------------------
                # Add records
                # ---------------------------------------------

                all_records.extend(
                    batch_records
                )

                processed_this_run += (
                    len(batch_records)
                )

                total_completed = (
                    completed_count
                    + processed_this_run
                )

                # ---------------------------------------------
                # Progress
                # ---------------------------------------------

                print(
                    f"Processed "
                    f"{total_completed:,}/"
                    f"{len(chunks):,} chunks"
                )

                # ---------------------------------------------
                # CHECKPOINT
                # ---------------------------------------------
                #
                # We checkpoint only records that have not
                # previously been written.
                #
                # Example:
                #
                # checkpointed_count = 320
                # total_completed    = 640
                #
                # save all_records[320:640]
                #
                # No duplicates.
                #
                # ---------------------------------------------

                while (
                    self.CHECKPOINT_ENABLED
                    and
                    (
                        total_completed
                        - checkpointed_count
                    )
                    >= self.CHECKPOINT_INTERVAL
                ):

                    checkpoint_end = (
                        checkpointed_count
                        + self.CHECKPOINT_INTERVAL
                    )

                    records_to_checkpoint = (
                        all_records[
                            checkpointed_count:
                            checkpoint_end
                        ]
                    )

                    self.append_checkpoint(
                        dataset_name,
                        records_to_checkpoint
                    )

                    checkpointed_count = (
                        checkpoint_end
                    )

                    print(
                        f"✓ Checkpoint saved at "
                        f"{checkpointed_count:,} "
                        f"records"
                    )

        # =================================================
        # STEP 9 : FINAL CHECKPOINT
        # =================================================

        total_records = len(
            all_records
        )

        if (
            self.CHECKPOINT_ENABLED
            and
            checkpointed_count
            < total_records
        ):

            records_to_checkpoint = (
                all_records[
                    checkpointed_count:
                    total_records
                ]
            )

            self.append_checkpoint(
                dataset_name,
                records_to_checkpoint
            )

            checkpointed_count = (
                total_records
            )

            print(
                f"✓ Final checkpoint saved at "
                f"{checkpointed_count:,} records"
            )

        # =================================================
        # STEP 10 : FINAL COUNT CHECK
        # =================================================

        if (
            len(all_records)
            != len(chunks)
        ):

            raise ValueError(
                f"Final embedding count mismatch "
                f"for {dataset_name}."
            )

        # =================================================
        # STEP 11 : FINAL EMBEDDING VALIDATION
        # =================================================

        print("\n")
        print("=" * 70)

        print(
            "Final Dataset Validation"
        )

        print("=" * 70)

        final_validation = (
            self.embedding_utils
            .validate_records(
                all_records
            )
        )

        print(
            f"\nTotal Records : "
            f"{final_validation['total_records']:,}"
        )

        print(
            f"Valid Records : "
            f"{final_validation['valid_records']:,}"
        )

        print(
            f"Invalid Records : "
            f"{final_validation['invalid_records']:,}"
        )

        if not final_validation[
            "is_valid"
        ]:

            raise ValueError(
                f"Final embedding validation "
                f"failed for {dataset_name}."
            )

        print(
            "\n✓ Embedding Validation Passed"
        )

        print(
            "\n✓ Chunk Count = "
            "Embedding Count"
        )

        # =================================================
        # STEP 12 : FINAL EXPORT
        # =================================================

        print("\n")
        print("=" * 70)

        print(
            "Exporting Final Dataset Embeddings"
        )

        print("=" * 70)

        exported_files = (
            self.exporter.export_embeddings(
                all_records,
                dataset_name
            )
        )

        # =================================================
        # STEP 13 : VERIFY FINAL OUTPUT
        # =================================================

        print(
            "\nVerifying exported files..."
        )

        if not self.validate_final_output(
            dataset_name,
            chunks
        ):

            raise RuntimeError(
                f"Final exported output "
                f"verification failed "
                f"for {dataset_name}."
            )

        print(
            "✓ Final Export Verified"
        )

        # =================================================
        # STEP 14 : REMOVE CHECKPOINT
        # =================================================

        self.remove_checkpoint(
            dataset_name
        )

        print(
            "\n✓ Checkpoint removed."
        )

        # =================================================
        # DATASET SUMMARY
        # =================================================

        print("\n")
        print("-" * 70)

        print(
            f"Dataset Completed : "
            f"{dataset_name}"
        )

        print(
            f"Chunks Processed : "
            f"{len(chunks):,}"
        )

        print(
            f"Embeddings Generated : "
            f"{len(all_records):,}"
        )

        print(
            f"JSON : "
            f"{exported_files['json']}"
        )

        print(
            f"NumPy : "
            f"{exported_files['numpy']}"
        )

        print("-" * 70)

        return {
            "dataset":
                dataset_name,

            "status":
                "completed",

            "chunks":
                len(chunks),

            "embeddings":
                len(all_records),

            "files":
                exported_files
        }

    # =====================================================
    # PROCESS ALL DATASETS
    # =====================================================

    def process_all(self):

        print("\n")
        print("=" * 70)

        print(
            "STARTING EMBEDDING PIPELINE"
        )

        print("=" * 70)

        # =================================================
        # MODE
        # =================================================

        if self.TEST_MODE:

            print(
                "\n⚠ TEST MODE ENABLED"
            )

            print(
                f"Chunks per dataset : "
                f"{self.TEST_CHUNKS_PER_DATASET}"
            )

            print(
                f"Expected total test chunks : "
                f"{len(self.DATASETS) * self.TEST_CHUNKS_PER_DATASET}"
            )

        else:

            print(
                "\n✓ FULL DATASET MODE ENABLED"
            )

            print(
                "All available Module 2 chunks "
                "will be processed."
            )

        # =================================================
        # PROCESS DATASETS
        # =================================================

        results = []

        for index, dataset_name in enumerate(
            self.DATASETS,
            start=1
        ):

            print("\n")
            print(
                f"DATASET {index}/"
                f"{len(self.DATASETS)}"
            )

            result = (
                self.process_dataset(
                    dataset_name
                )
            )

            results.append(
                result
            )

        # =================================================
        # FINAL PIPELINE SUMMARY
        # =================================================

        print("\n")
        print("=" * 70)

        print(
            "EMBEDDING PIPELINE SUMMARY"
        )

        print("=" * 70)

        total_chunks = 0
        total_embeddings = 0

        print("\n")

        for result in results:

            print(
                f"{result['dataset']:<25}"
                f" : "
                f"{result['embeddings']:,} "
                f"embeddings"
            )

            total_chunks += (
                result["chunks"]
            )

            total_embeddings += (
                result["embeddings"]
            )

        print("-" * 70)

        print(
            f"{'TOTAL':<25}"
            f" : "
            f"{total_embeddings:,} "
            f"embeddings"
        )

        print(
            f"\nTotal Chunks Processed : "
            f"{total_chunks:,}"
        )

        print(
            f"Total Embeddings Generated : "
            f"{total_embeddings:,}"
        )

        # =================================================
        # FINAL COUNT VERIFICATION
        # =================================================

        if (
            total_chunks
            == total_embeddings
        ):

            print(
                "\n✓ Chunk Count = "
                "Embedding Count"
            )

        else:

            print(
                "\n✗ COUNT MISMATCH"
            )

            raise RuntimeError(
                "Final pipeline count mismatch."
            )

        print("\n")
        print("=" * 70)

        print(
            "HalluciGuard Module 3 "
            "Embedding Pipeline Completed"
        )

        print("=" * 70)


# =========================================================
# DIRECT EXECUTION
# =========================================================

if __name__ == "__main__":

    processor = BatchEmbeddingProcessor()

    processor.process_all()