import numpy as np
import pytest

from retrieval.retrieval_pipeline import RetrievalPipeline


class TestRetrievalPipeline:

    # =====================================================
    # SETUP
    # =====================================================

    def setup_method(self):

        self.pipeline = RetrievalPipeline()

    # =====================================================
    # TEST INITIALIZATION
    # =====================================================

    def test_pipeline_initialization(self):

        assert self.pipeline is not None

        assert (
            len(
                self.pipeline.get_dataset_names()
            )
            == 5
        )

    # =====================================================
    # TEST DATASETS
    # =====================================================

    def test_all_datasets_loaded(self):

        dataset_names = (
            self.pipeline.get_dataset_names()
        )

        expected_datasets = [

            "dialogue_data",

            "general_data",

            "qa_data",

            "summarization_data",

            "TruthfulQA"

        ]

        for dataset_name in expected_datasets:

            assert (
                dataset_name
                in dataset_names
            )

    # =====================================================
    # TEST QUERY EMBEDDING
    # =====================================================

    def test_query_embedding(self):

        query = (
            "What is an example of hallucination?"
        )

        embedding = (
            self.pipeline.embed_query(
                query
            )
        )

        assert isinstance(
            embedding,
            np.ndarray
        )

        assert embedding.shape == (
            384,
        )

        assert embedding.dtype == np.float32

        assert np.all(
            np.isfinite(embedding)
        )

    # =====================================================
    # TEST SINGLE DATASET RETRIEVAL
    # =====================================================

    def test_retrieve_from_dataset(self):

        query = (
            "What is an example of hallucination?"
        )

        results = (
            self.pipeline.retrieve_from_dataset(
                query,
                "general_data",
                top_k=5
            )
        )

        assert isinstance(
            results,
            list
        )

        assert len(results) <= 5

        assert len(results) > 0

    # =====================================================
    # TEST RETRIEVAL OUTPUT STRUCTURE
    # =====================================================

    def test_retrieval_output_structure(self):

        query = (
            "What is an example of hallucination?"
        )

        results = (
            self.pipeline.retrieve(
                query,
                top_k=5
            )
        )

        assert len(results) > 0

        for result in results:

            assert "chunk_uid" in result

            assert "chunk_text" in result

            assert "similarity_score" in result

            assert "rank" in result

            assert "dataset" in result

            assert isinstance(
                result["chunk_text"],
                str
            )

            assert isinstance(
                result["similarity_score"],
                float
            )

    # =====================================================
    # TEST RESULTS ARE SORTED
    # =====================================================

    def test_results_are_sorted(self):

        query = (
            "What is an example of hallucination?"
        )

        results = (
            self.pipeline.retrieve(
                query,
                top_k=5
            )
        )

        scores = [

            result["similarity_score"]

            for result in results

        ]

        assert scores == sorted(
            scores,
            reverse=True
        )

    # =====================================================
    # TEST RESULT RANKS
    # =====================================================

    def test_result_ranks(self):

        query = (
            "What is an example of hallucination?"
        )

        results = (
            self.pipeline.retrieve(
                query,
                top_k=5
            )
        )

        for expected_rank, result in enumerate(
            results,
            start=1
        ):

            assert (
                result["rank"]
                == expected_rank
            )

    # =====================================================
    # TEST TOP-K
    # =====================================================

    def test_top_k(self):

        query = (
            "What is natural language processing?"
        )

        results = (
            self.pipeline.retrieve(
                query,
                top_k=3
            )
        )

        assert len(results) <= 3

    # =====================================================
    # TEST BEST RESULT
    # =====================================================

    def test_best_result(self):

        query = (
            "What is an example of hallucination?"
        )

        best_result = (
            self.pipeline.retrieve_best(
                query
            )
        )

        assert best_result is not None

        assert (
            best_result["rank"]
            == 1
        )

        assert "chunk_text" in (
            best_result
        )

        assert "similarity_score" in (
            best_result
        )

    # =====================================================
    # TEST THRESHOLD RETRIEVAL
    # =====================================================

    def test_threshold_retrieval(self):

        query = (
            "What is an example of hallucination?"
        )

        results = (
            self.pipeline.retrieve_with_threshold(
                query,
                top_k=5,
                min_score=0.5
            )
        )

        for result in results:

            assert (
                result["similarity_score"]
                >= 0.5
            )

    # =====================================================
    # TEST MULTIPLE QUERIES
    # =====================================================

    def test_multiple_queries(self):

        queries = [

            "What is hallucination?",

            "What is artificial intelligence?",

            "What is natural language processing?"

        ]

        for query in queries:

            results = (
                self.pipeline.retrieve(
                    query,
                    top_k=3
                )
            )

            assert isinstance(
                results,
                list
            )

            assert len(results) > 0

            assert len(results) <= 3

    # =====================================================
    # TEST DIFFERENT DATASETS
    # =====================================================

    def test_different_datasets(self):

        query = (
            "What is hallucination?"
        )

        for dataset_name in (
            self.pipeline.get_dataset_names()
        ):

            results = (
                self.pipeline.retrieve_from_dataset(
                    query,
                    dataset_name,
                    top_k=2
                )
            )

            assert isinstance(
                results,
                list
            )

            assert len(results) > 0

    # =====================================================
    # TEST INVALID DATASET
    # =====================================================

    def test_invalid_dataset(self):

        query = (
            "What is hallucination?"
        )

        with pytest.raises(
            ValueError
        ):

            self.pipeline.retrieve_from_dataset(
                query,
                "invalid_dataset",
                top_k=5
            )

    # =====================================================
    # TEST INVALID TOP-K
    # =====================================================

    def test_invalid_top_k(self):

        query = (
            "What is hallucination?"
        )

        with pytest.raises(
            ValueError
        ):

            self.pipeline.retrieve(
                query,
                top_k=0
            )

    # =====================================================
    # TEST REAL RETRIEVAL RELEVANCE
    # =====================================================

    def test_hallucination_query_returns_relevant_text(
        self
    ):

        query = (
            "What is an example of hallucination?"
        )

        results = (
            self.pipeline.retrieve(
                query,
                top_k=5
            )
        )

        assert len(results) > 0

        # At least one retrieved chunk should
        # contain the word "hallucination".

        contains_hallucination = any(

            "hallucination"
            in result["chunk_text"].lower()

            for result in results

        )

        assert contains_hallucination