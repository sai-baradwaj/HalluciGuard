from chunking.chunk_utils import ChunkUtils


def main():

    print("=" * 70)
    print("HALLUCIGUARD CHUNK COUNT")
    print("=" * 70)

    chunk_utils = ChunkUtils()

    datasets = [
        "dialogue_data",
        "general_data",
        "qa_data",
        "summarization_data",
        "TruthfulQA"
    ]

    total_chunks = 0

    print("\n")

    for dataset_name in datasets:

        chunks = chunk_utils.load_dataset(
            dataset_name
        )

        count = len(chunks)

        total_chunks += count

        print(
            f"{dataset_name:<25} : {count:,} chunks"
        )

    print("\n" + "-" * 70)

    print(
        f"{'TOTAL':<25} : "
        f"{total_chunks:,} chunks"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()