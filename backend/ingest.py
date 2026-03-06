from dotenv import load_dotenv

from ingestion import ingest_file

load_dotenv()


def ingest_data(file_path: str) -> None:
    result = ingest_file(file_path)
    print(
        f"Ingestion complete! source={result['source']} chunks={result['chunks']} index={result['index']}"
    )


if __name__ == "__main__":
    ingest_data("data.txt")
