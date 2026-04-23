import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_openai import AzureOpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _get_embedding_model() -> str:
    return _get_required_env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")


def _get_azure_openai_endpoint() -> str:
    return _get_required_env("AZURE_OPENAI_ENDPOINT")


def _get_azure_openai_api_key() -> str:
    return _get_required_env("AZURE_OPENAI_API_KEY")


def _get_azure_openai_api_version() -> str:
    return os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")


def _load_documents(file_path: str):
    extension = Path(file_path).suffix.lower()

    if extension == ".pdf":
        return PyPDFLoader(file_path).load()

    if extension in {".txt", ".md", ".csv", ".log"}:
        return TextLoader(file_path, encoding="utf-8").load()

    raise ValueError("Unsupported file type. Allowed: .pdf, .txt, .md, .csv, .log")


def ingest_file(file_path: str, source_name: str | None = None) -> dict:
    documents = _load_documents(file_path)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    resolved_source = source_name or Path(file_path).name
    for chunk in chunks:
        chunk.metadata = {**chunk.metadata, "source": resolved_source}

    embedding_deployment = _get_embedding_model()
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=_get_azure_openai_endpoint(),
        api_key=_get_azure_openai_api_key(),
        openai_api_version=_get_azure_openai_api_version(),
        azure_deployment=embedding_deployment,
        model=embedding_deployment,
    )
    index_name = _get_required_env("PINECONE_INDEX_NAME")

    # Upload in batches to avoid connection timeouts
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        if i == 0:
             vectorstore = PineconeVectorStore.from_documents(batch, embeddings, index_name=index_name)
        else:
             vectorstore.add_documents(batch)

    return {
        "source": resolved_source,
        "chunks": len(chunks),
        "index": index_name,
    }

