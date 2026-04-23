import os
from dotenv import load_dotenv
from ingestion import ingest_file
from main import ensure_pinecone_index_exists

def main():
    load_dotenv()
    print("Checking/Creating Pinecone index...")
    try:
        ensure_pinecone_index_exists()
        print("Index ready.")
    except Exception as e:
        print(f"Failed to ensure index exists: {e}")
        return

    kb_path = r"C:\AI ChatBot\visit-to-lead\backend\Knowledge base\DIGICoCo_Knowledge_Base.txt"
    print(f"Starting ingestion for {kb_path}...")
    try:
        result = ingest_file(kb_path)
        print("Ingestion successful!")
        print(f"Result: {result}")
    except Exception as e:
        print(f"Ingestion failed: {e}")

if __name__ == "__main__":
    main()
