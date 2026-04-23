import json
import os
import uuid
from datetime import datetime, timezone
from io import BytesIO
from typing import Any

import azure.functions as func
from langchain_openai import AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from pypdf import PdfReader

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)


# Optional CORS override for deployed function apps.
ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "*")


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


def _get_file_namespace() -> str:
    return os.getenv("PINECONE_FILE_NAMESPACE", "uploaded-pdf-files")


def _get_manifest_namespace() -> str:
    return os.getenv("PINECONE_MANIFEST_NAMESPACE", "__file_manifest__")


def _json_response(payload: dict[str, Any], status_code: int = 200) -> func.HttpResponse:
    return func.HttpResponse(
        body=json.dumps(payload),
        status_code=status_code,
        mimetype="application/json",
        headers={
            "Access-Control-Allow-Origin": ALLOWED_ORIGIN,
            "Access-Control-Allow-Methods": "GET,POST,PUT,DELETE,OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type,x-file-name",
        },
    )


def _options_response() -> func.HttpResponse:
    return func.HttpResponse(
        status_code=204,
        headers={
            "Access-Control-Allow-Origin": ALLOWED_ORIGIN,
            "Access-Control-Allow-Methods": "GET,POST,PUT,DELETE,OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type,x-file-name",
        },
    )


def _create_embeddings_client() -> AzureOpenAIEmbeddings:
    deployment = _get_embedding_model()
    return AzureOpenAIEmbeddings(
        azure_endpoint=_get_azure_openai_endpoint(),
        api_key=_get_azure_openai_api_key(),
        openai_api_version=_get_azure_openai_api_version(),
        azure_deployment=deployment,
        model=deployment,
    )


def _get_pinecone_index():
    pinecone = Pinecone(api_key=_get_required_env("PINECONE_API_KEY"))
    return pinecone.Index(_get_required_env("PINECONE_INDEX_NAME"))


def _extract_pdf_text(file_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(file_bytes))
    pages_text: list[str] = []
    for page in reader.pages:
        pages_text.append((page.extract_text() or "").strip())
    return "\n\n".join(text for text in pages_text if text)


def _chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=120)
    chunks = [chunk.strip() for chunk in splitter.split_text(text)]
    return [chunk for chunk in chunks if chunk]


def _coerce_filename(request: func.HttpRequest, explicit_name: str | None = None) -> str:
    file_name = (explicit_name or "").strip()
    if not file_name:
        file_name = (request.params.get("file_name") or "").strip()
    if not file_name:
        file_name = (request.headers.get("x-file-name") or "").strip()
    if not file_name:
        raise ValueError("Missing file name. Provide query param file_name or x-file-name header.")
    if not file_name.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are supported.")
    return file_name


def _read_pdf_bytes(request: func.HttpRequest) -> bytes:
    body = request.get_body() or b""
    if not body:
        raise ValueError("Request body is empty. Send the PDF bytes in the body.")
    return body


def _upsert_file(index, embeddings: AzureOpenAIEmbeddings, file_id: str, file_name: str, pdf_bytes: bytes) -> dict[str, Any]:
    text = _extract_pdf_text(pdf_bytes)
    if not text.strip():
        raise ValueError("Could not extract readable text from PDF.")

    chunks = _chunk_text(text)
    if not chunks:
        raise ValueError("No valid text chunks produced from PDF.")

    vectors = embeddings.embed_documents(chunks)
    timestamp = datetime.now(timezone.utc).isoformat()

    chunk_records = []
    for idx, vector in enumerate(vectors):
        chunk_records.append(
            {
                "id": f"{file_id}::chunk::{idx}",
                "values": vector,
                "metadata": {
                    "file_id": file_id,
                    "file_name": file_name,
                    "chunk_index": idx,
                    "uploaded_at": timestamp,
                    "source": file_name,
                },
            }
        )

    index.upsert(vectors=chunk_records, namespace=_get_file_namespace())

    manifest_vector = embeddings.embed_query(f"manifest:{file_id}:{file_name}:{timestamp}")
    manifest_record = {
        "id": file_id,
        "values": manifest_vector,
        "metadata": {
            "file_id": file_id,
            "file_name": file_name,
            "chunk_count": len(chunk_records),
            "uploaded_at": timestamp,
        },
    }
    index.upsert(vectors=[manifest_record], namespace=_get_manifest_namespace())

    return {
        "file_id": file_id,
        "file_name": file_name,
        "chunk_count": len(chunk_records),
        "uploaded_at": timestamp,
    }


def _list_index_ids(index, namespace: str, prefix: str | None = None) -> list[str]:
    found: list[str] = []

    try:
        iterator = index.list(namespace=namespace, prefix=prefix)
        for page in iterator:
            if isinstance(page, list):
                found.extend(page)
            elif isinstance(page, dict):
                ids = page.get("ids") or []
                if isinstance(ids, list):
                    found.extend([item for item in ids if isinstance(item, str)])
    except Exception:
        return []

    # Keep stable unique ids.
    unique: dict[str, None] = {}
    for item in found:
        unique[item] = None
    return list(unique.keys())


def _batch_fetch_vectors(index, namespace: str, ids: list[str]) -> dict[str, Any]:
    if not ids:
        return {}

    vectors: dict[str, Any] = {}
    batch_size = 200
    for start in range(0, len(ids), batch_size):
        batch = ids[start : start + batch_size]
        fetched = index.fetch(ids=batch, namespace=namespace)
        page_vectors = fetched.get("vectors") if isinstance(fetched, dict) else getattr(fetched, "vectors", {})
        if isinstance(page_vectors, dict):
            vectors.update(page_vectors)

    return vectors


def _get_file_record(index, file_id: str) -> dict[str, Any] | None:
    fetched = index.fetch(ids=[file_id], namespace=_get_manifest_namespace())
    vectors = fetched.get("vectors") if isinstance(fetched, dict) else getattr(fetched, "vectors", {})
    if not isinstance(vectors, dict) or file_id not in vectors:
        return None

    record = vectors[file_id]
    metadata = record.get("metadata") if isinstance(record, dict) else getattr(record, "metadata", {})
    return metadata if isinstance(metadata, dict) else None


def _delete_file_chunks(index, file_id: str) -> None:
    ids = _list_index_ids(index, _get_file_namespace(), prefix=f"{file_id}::chunk::")
    if ids:
        index.delete(ids=ids, namespace=_get_file_namespace())


@app.route(route="files", methods=["GET", "POST", "OPTIONS"])
def files(req: func.HttpRequest) -> func.HttpResponse:
    if req.method == "OPTIONS":
        return _options_response()

    try:
        index = _get_pinecone_index()
        embeddings = _create_embeddings_client()

        if req.method == "GET":
            ids = _list_index_ids(index, _get_manifest_namespace())
            vectors = _batch_fetch_vectors(index, _get_manifest_namespace(), ids)

            items: list[dict[str, Any]] = []
            for record_id, record in vectors.items():
                metadata = record.get("metadata") if isinstance(record, dict) else getattr(record, "metadata", {})
                if not isinstance(metadata, dict):
                    continue
                items.append(
                    {
                        "file_id": metadata.get("file_id", record_id),
                        "file_name": metadata.get("file_name", ""),
                        "chunk_count": metadata.get("chunk_count", 0),
                        "uploaded_at": metadata.get("uploaded_at", ""),
                    }
                )

            items.sort(key=lambda item: str(item.get("uploaded_at", "")), reverse=True)
            return _json_response({"files": items})

        file_name = _coerce_filename(req)
        pdf_bytes = _read_pdf_bytes(req)
        file_id = uuid.uuid4().hex
        result = _upsert_file(index, embeddings, file_id, file_name, pdf_bytes)

        return _json_response({"message": "File uploaded and indexed.", "file": result}, status_code=201)

    except ValueError as error:
        return _json_response({"error": str(error)}, status_code=400)
    except Exception as error:
        return _json_response({"error": f"Server error: {type(error).__name__}: {error}"}, status_code=500)


@app.route(route="files/{file_id}", methods=["PUT", "DELETE", "OPTIONS"])
def file_item(req: func.HttpRequest) -> func.HttpResponse:
    if req.method == "OPTIONS":
        return _options_response()

    file_id = (req.route_params.get("file_id") or "").strip()
    if not file_id:
        return _json_response({"error": "Missing file_id in route."}, status_code=400)

    try:
        index = _get_pinecone_index()
        existing = _get_file_record(index, file_id)
        if not existing:
            return _json_response({"error": "File not found."}, status_code=404)

        if req.method == "DELETE":
            _delete_file_chunks(index, file_id)
            index.delete(ids=[file_id], namespace=_get_manifest_namespace())
            return _json_response({"message": "File deleted successfully.", "file_id": file_id})

        embeddings = _create_embeddings_client()
        file_name = _coerce_filename(req, explicit_name=str(existing.get("file_name", "")).strip())
        # Allow changing file name on update if caller sends one.
        override_name = (req.params.get("file_name") or req.headers.get("x-file-name") or "").strip()
        if override_name:
            if not override_name.lower().endswith(".pdf"):
                raise ValueError("Only PDF files are supported.")
            file_name = override_name

        pdf_bytes = _read_pdf_bytes(req)

        _delete_file_chunks(index, file_id)
        result = _upsert_file(index, embeddings, file_id, file_name, pdf_bytes)

        return _json_response({"message": "File updated and re-indexed.", "file": result})

    except ValueError as error:
        return _json_response({"error": str(error)}, status_code=400)
    except Exception as error:
        return _json_response({"error": f"Server error: {type(error).__name__}: {error}"}, status_code=500)
