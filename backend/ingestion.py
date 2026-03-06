import os
from functools import lru_cache
from pathlib import Path

import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _get_embedding_model() -> str:
    configured_model = os.getenv("GOOGLE_EMBEDDING_MODEL")
    if configured_model:
        return configured_model

    return _detect_embedding_model()


@lru_cache(maxsize=1)
def _detect_embedding_model() -> str:
    genai.configure(api_key=_get_required_env("GOOGLE_API_KEY"))

    available_models: list[str] = []
    for model in genai.list_models():
        methods = set(getattr(model, "supported_generation_methods", []) or [])
        if "embedContent" in methods:
            model_name = getattr(model, "name", "")
            if model_name:
                available_models.append(model_name)

    if not available_models:
        raise ValueError(
            "No Google embedding models are available for this API key. "
            "Set GOOGLE_EMBEDDING_MODEL explicitly to a supported model."
        )

    preferred_suffixes = ["text-embedding-004", "embedding-001", "text-embedding-005"]
    for suffix in preferred_suffixes:
        for model_name in available_models:
            if model_name.endswith(suffix):
                return model_name

    return available_models[0]


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

    embeddings = GoogleGenerativeAIEmbeddings(model=_get_embedding_model())
    index_name = _get_required_env("PINECONE_INDEX_NAME")

    PineconeVectorStore.from_documents(chunks, embeddings, index_name=index_name)

    return {
        "source": resolved_source,
        "chunks": len(chunks),
        "index": index_name,
    }
