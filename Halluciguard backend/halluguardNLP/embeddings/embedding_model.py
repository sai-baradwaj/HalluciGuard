"""
Embedding Model Module

Responsible for loading and configuring the
embedding model used by HalluciGuard.

Model:
BAAI/bge-small-en-v1.5

Responsibilities:
- Load the pretrained BGE model
- Select GPU or CPU
- Configure maximum input length
- Provide model information

This module does NOT:
- Generate embeddings for datasets
- Save embeddings
- Perform retrieval
- Validate embeddings
"""

import torch
from sentence_transformers import SentenceTransformer


class EmbeddingModel:

    # =====================================================
    # MODEL CONFIGURATION
    # =====================================================

    MODEL_NAME = "BAAI/bge-small-en-v1.5"

    MAX_SEQUENCE_LENGTH = 512

    EMBEDDING_DIMENSION = 384

    # =====================================================
    # CONSTRUCTOR
    # =====================================================

    def __init__(self):

        print("\n")
        print("=" * 70)
        print("Initializing Embedding Model")
        print("=" * 70)

        # -------------------------------------------------
        # Select Device
        # -------------------------------------------------

        if torch.cuda.is_available():

            self.device = "cuda"

            print("\nDevice : CUDA GPU")

            print(
                f"GPU : "
                f"{torch.cuda.get_device_name(0)}"
            )

        else:

            self.device = "cpu"

            print("\nDevice : CPU")

        # -------------------------------------------------
        # Load Model
        # -------------------------------------------------

        print(
            f"\nLoading Embedding Model : "
            f"{self.MODEL_NAME}"
        )

        self.model = SentenceTransformer(
            self.MODEL_NAME,
            device=self.device
        )

        # -------------------------------------------------
        # Configure Maximum Sequence Length
        # -------------------------------------------------

        self.model.max_seq_length = (
            self.MAX_SEQUENCE_LENGTH
        )

        print(
            f"Maximum Sequence Length : "
            f"{self.model.max_seq_length}"
        )

        # -------------------------------------------------
        # Display Embedding Dimension
        # -------------------------------------------------

        print(
            f"Embedding Dimension : "
            f"{self.EMBEDDING_DIMENSION}"
        )

        print(
            "\nEmbedding Model Loaded Successfully."
        )

    # =====================================================
    # GET MODEL
    # =====================================================

    def get_model(self):

        return self.model

    # =====================================================
    # GET DEVICE
    # =====================================================

    def get_device(self):

        return self.device

    # =====================================================
    # GET MODEL NAME
    # =====================================================

    def get_model_name(self):

        return self.MODEL_NAME

    # =====================================================
    # GET EMBEDDING DIMENSION
    # =====================================================

    def get_embedding_dimension(self):

        return self.EMBEDDING_DIMENSION

    # =====================================================
    # MODEL INFORMATION
    # =====================================================

    def get_model_info(self):

        return {

            "model_name":
                self.MODEL_NAME,

            "device":
                self.device,

            "max_sequence_length":
                self.MAX_SEQUENCE_LENGTH,

            "embedding_dimension":
                self.EMBEDDING_DIMENSION

        }

    # =====================================================
    # PRINT MODEL INFORMATION
    # =====================================================

    def print_model_info(self):

        information = (
            self.get_model_info()
        )

        print("\nEmbedding Model Information")
        print("-" * 40)

        print(
            f"Model Name : "
            f"{information['model_name']}"
        )

        print(
            f"Device : "
            f"{information['device']}"
        )

        print(
            f"Maximum Sequence Length : "
            f"{information['max_sequence_length']}"
        )

        print(
            f"Embedding Dimension : "
            f"{information['embedding_dimension']}"
        )


if __name__ == "__main__":

    embedding_model = EmbeddingModel()

    embedding_model.print_model_info()