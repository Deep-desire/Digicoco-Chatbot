import os
import re
import shutil
import uuid
import logging
from datetime import datetime, timezone
from collections import deque
from functools import lru_cache
from pathlib import Path
from threading import Lock
from time import sleep
from time import time

import edge_tts
import google.generativeai as genai
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from groq import Groq
from ingestion import ingest_file
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from pinecone.core.client.exceptions import NotFoundException

load_dotenv()
logger = logging.getLogger(__name__)

app = FastAPI(title="Hybrid Voice + Text RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-User-Query", "X-Bot-Reply"],
)

SUPPORTED_INGEST_EXTENSIONS = {".pdf", ".txt", ".md", ".csv", ".log"}

SERVICE_SUMMARY = (
    "Desire Infoweb provides Microsoft-focused IT services including SharePoint, "
    "Power Apps, Power Automate, Power BI, Office 365, Teams, Dynamics 365, Azure, "
    ".NET, migration, automation, and AI/chatbot solutions."
)

AI_SUMMARY = (
    "Desire Infoweb AI services include Azure OpenAI-based solutions, Teams chatbots, "
    "Copilot-aligned workflows, intelligent automation, and document-grounded chatbot implementations."
)

AI_PROJECTS_SUMMARY = (
    "Some AI project examples from Desire Infoweb include: "
    "(1) a Microsoft Teams chatbot integrated with ChatGPT, and "
    "(2) a document-grounded chatbot using SharePoint/Azure Blob as data sources "
    "to provide responses based on uploaded files."
)

BUDGET_SUMMARY = (
    "Budget depends on scope, integrations, data volume, and deployment model. "
    "For an AI chatbot, we usually start with a discovery session and then share a tailored estimate "
    "with timeline and milestones. If you share your use case, channels (website/Teams/WhatsApp), "
    "and expected users, we can provide a more accurate proposal."
)

DOTNET_SUMMARY = (
    "Desire Infoweb .NET services include custom enterprise application development, "
    "secure and scalable backend systems, workflow and approval systems, and modernization of existing applications."
)

CHATBOT_IMPLEMENTATION_SUMMARY = (
    "For a typical business chatbot project, we usually deliver: "
    "discovery and requirements, data ingestion from documents/web/SharePoint, "
    "RAG-based answer engine, website or Teams chat interface, optional voice support, "
    "testing, and production deployment."
)

CHATBOT_DATA_SOURCE_SUMMARY = (
    "Yes, chatbot data can come from SharePoint. We commonly use SharePoint libraries/sites, "
    "Azure Blob storage, PDFs, Word/Excel files, and website content as knowledge sources. "
    "Then we index that content so answers are grounded in your business data."
)

INDUSTRY_SUMMARY = (
    "Desire Infoweb serves industries such as education, retail/e-commerce, finance, "
    "real estate, travel, healthcare, and logistics/distribution."
)

_conversation_lock = Lock()
_conversation_store: dict[str, deque[tuple[str, str]]] = {}
_lead_lock = Lock()
_lead_store: dict[str, dict[str, str]] = {}
_graph_token_lock = Lock()
_graph_token: dict[str, float | str] = {"access_token": "", "expires_at": 0.0}


def _get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _sanitize_header_value(value: str) -> str:
    normalized = value.replace("\r", " ").replace("\n", " ").strip()
    return normalized.encode("latin1", "ignore").decode("latin1")


def _normalize_user_query(query: str) -> str:
    normalized = query.strip()
    replacements = {
        "serivce": "service",
        "serivces": "services",
        "qhat": "what",
        "wht": "what",
        "u": "you",
    }
    words = [replacements.get(token.lower(), token) for token in normalized.split()]
    return " ".join(words)


def _normalize_session_id(session_id: str | None) -> str:
    value = (session_id or "").strip()
    if not value:
        return "default"
    return re.sub(r"[^a-zA-Z0-9_-]", "", value)[:64] or "default"


def _normalize_lead_email(email: str | None) -> str:
    value = (email or "").strip().lower()
    if not value:
        return ""
    if not re.fullmatch(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", value):
        raise HTTPException(status_code=400, detail="Invalid email format")
    return value


def _normalize_lead_name(name: str | None) -> str:
    value = (name or "").strip()
    return re.sub(r"\s+", " ", value)[:120]


def _resolve_lead_identity(session_id: str, email: str | None, name: str | None) -> tuple[str, str]:
    normalized_email = _normalize_lead_email(email)
    normalized_name = _normalize_lead_name(name)

    with _lead_lock:
        if session_id not in _lead_store:
            _lead_store[session_id] = {
                "email": "",
                "name": "",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

        if normalized_email:
            _lead_store[session_id]["email"] = normalized_email
        if normalized_name:
            _lead_store[session_id]["name"] = normalized_name

        return _lead_store[session_id]["email"], _lead_store[session_id]["name"]


def _build_conversation_transcript(session_id: str) -> str:
    with _conversation_lock:
        history = list(_conversation_store.get(session_id, []))

    lines: list[str] = []
    for user_text, assistant_text in history:
        lines.append(f"User: {user_text}")
        lines.append(f"Assistant: {assistant_text}")

    return "\n".join(lines)


def _is_sharepoint_sync_enabled() -> bool:
    return os.getenv("ENABLE_SHAREPOINT_SYNC", "false").lower() == "true"


def _is_sharepoint_always_insert_enabled() -> bool:
    return os.getenv("SHAREPOINT_ALWAYS_INSERT", "true").lower() == "true"


def _get_sharepoint_field_names() -> dict[str, str]:
    return {
        "title": os.getenv("SHAREPOINT_FIELD_TITLE", "Title").strip() or "Title",
        "name": os.getenv("SHAREPOINT_FIELD_NAME", "Name").strip() or "Name",
        "email": os.getenv("SHAREPOINT_FIELD_EMAIL", "email").strip() or "email",
        "conversation": os.getenv("SHAREPOINT_FIELD_CONVERSATION", "Conversation").strip() or "Conversation",
    }


def _get_graph_token() -> str:
    if not _is_sharepoint_sync_enabled():
        return ""

    with _graph_token_lock:
        token_value = str(_graph_token.get("access_token", ""))
        expires_at = float(_graph_token.get("expires_at", 0.0) or 0.0)
        if token_value and expires_at - 60 > time():
            return token_value

    tenant_id = _get_required_env("SHAREPOINT_TENANT_ID")
    client_id = _get_required_env("SHAREPOINT_CLIENT_ID")
    client_secret = _get_required_env("SHAREPOINT_CLIENT_SECRET")
    token_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"

    response = httpx.post(
        token_url,
        data={
            "client_id": client_id,
            "client_secret": client_secret,
            "grant_type": "client_credentials",
            "scope": "https://graph.microsoft.com/.default",
        },
        timeout=15.0,
    )
    response.raise_for_status()
    payload = response.json()
    access_token = payload.get("access_token")
    if not access_token:
        raise ValueError("Failed to obtain Microsoft Graph access token.")

    expires_in = float(payload.get("expires_in", 3600))
    with _graph_token_lock:
        _graph_token["access_token"] = access_token
        _graph_token["expires_at"] = time() + expires_in

    return access_token


def _graph_request(method: str, url: str, **kwargs) -> dict:
    token = _get_graph_token()
    if not token:
        raise ValueError("SharePoint sync is disabled or missing credentials.")

    headers = dict(kwargs.pop("headers", {}))
    headers["Authorization"] = f"Bearer {token}"
    headers["Accept"] = "application/json"

    response = httpx.request(method, url, headers=headers, timeout=20.0, **kwargs)
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as error:
        detail = ""
        try:
            payload = response.json()
            detail = str(payload.get("error", payload))
        except Exception:
            detail = response.text[:500]
        raise ValueError(
            f"Microsoft Graph request failed ({response.status_code}) for {url}. Detail: {detail}"
        ) from error
    if response.status_code == 204:
        return {}
    return response.json()


def _build_sharepoint_fields(lead_name: str, lead_email: str, transcript: str) -> dict[str, str]:
    field_names = _get_sharepoint_field_names()
    title_value = lead_name or lead_email
    return {
        field_names["title"]: title_value,
        field_names["name"]: lead_name,
        field_names["email"]: lead_email,
        field_names["conversation"]: transcript,
    }


def _find_sharepoint_item_id(site_id: str, list_id: str, email_field: str, email_value: str) -> str | None:
    escaped_email = email_value.replace("'", "''")
    url = (
        "https://graph.microsoft.com/v1.0"
        f"/sites/{site_id}/lists/{list_id}/items"
        f"?$expand=fields&$filter=fields/{email_field} eq '{escaped_email}'"
    )
    payload = _graph_request("GET", url)
    items = payload.get("value", [])
    if not items:
        return None
    return str(items[0].get("id") or "") or None


def _upsert_sharepoint_lead(session_id: str) -> None:
    if not _is_sharepoint_sync_enabled():
        return

    with _lead_lock:
        lead = dict(_lead_store.get(session_id, {}))

    if not lead:
        return

    lead_email = lead.get("email", "").strip()
    lead_name = lead.get("name", "").strip()
    if not lead_email or not lead_name:
        return

    transcript = _build_conversation_transcript(session_id)
    if not transcript:
        return

    site_id = _get_required_env("SHAREPOINT_SITE_ID")
    list_id = _get_required_env("SHAREPOINT_LIST_ID")
    fields_payload = _build_sharepoint_fields(lead_name, lead_email, transcript)

    if _is_sharepoint_always_insert_enabled():
        create_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/lists/{list_id}/items"
        _graph_request("POST", create_url, json={"fields": fields_payload})
        return

    field_names = _get_sharepoint_field_names()
    item_id = _find_sharepoint_item_id(site_id, list_id, field_names["email"], lead_email)

    if item_id:
        update_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/lists/{list_id}/items/{item_id}/fields"
        _graph_request("PATCH", update_url, json=fields_payload)
        return

    create_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/lists/{list_id}/items"
    _graph_request("POST", create_url, json={"fields": fields_payload})


async def _sync_sharepoint_lead_safely(session_id: str) -> None:
    try:
        _upsert_sharepoint_lead(session_id)
    except Exception as error:
        logger.warning("SharePoint sync skipped/failed for session %s: %s", session_id, error)


def _direct_company_answer(query: str) -> str | None:
    q = query.lower().strip()
    compact = re.sub(r"[^a-z0-9\s]", "", q)

    if re.fullmatch(r"(hi|hello|hey|hii|hiii|good morning|good afternoon|good evening)", compact):
        return (
            "Hello! Welcome to Desire Infoweb. "
            f"{SERVICE_SUMMARY} "
            "Tell me your requirement and I can suggest the best service approach."
        )

    if any(keyword in compact for keyword in ["what service", "services", "what do you do", "what you do", "what do you provide", "offer"]):
        return (
            "We provide end-to-end Microsoft technology services: "
            "SharePoint and intranet solutions, Power Platform (Power Apps/Automate), "
            "Power BI analytics, Office 365 and Teams implementation, Dynamics 365, Azure, .NET development, "
            "migration, governance, and AI/chatbot solutions."
        )

    if any(keyword in compact for keyword in ["budget", "cost", "pricing", "price", "estimate", "quotation", "quote"]):
        return BUDGET_SUMMARY

    if any(keyword in compact for keyword in ["build ai chatbot", "want to build ai chatbot", "ai chatbot project", "chatbot project"]):
        return (
            "Great choice. We can build an AI chatbot for your website or Microsoft Teams with your business data as context. "
            "Typical scope includes discovery, data ingestion (PDF/web/SharePoint), prompt tuning, voice/text support, testing, and deployment. "
            "If you share your goal and preferred channel, I can suggest the best implementation approach."
        )

    if any(
        keyword in compact
        for keyword in [
            "ever done",
            "done this type",
            "this type of project",
            "done similar",
            "have done",
            "previous chatbot",
            "chatbot past project",
        ]
    ):
        return AI_PROJECTS_SUMMARY

    if any(keyword in compact for keyword in ["normal chatbot", "just chatbot", "simple chatbot", "basic chatbot"]):
        return CHATBOT_IMPLEMENTATION_SUMMARY

    if any(
        keyword in compact
        for keyword in [
            "sharepoint",
            "data source",
            "where data came",
            "data came from",
            "chatbot where data",
            "data from sharepoint",
        ]
    ) and "chatbot" in compact:
        return CHATBOT_DATA_SOURCE_SUMMARY

    if any(keyword in compact for keyword in ["past project", "case study", "ai project", "previous ai", "what this company ai"]):
        return AI_PROJECTS_SUMMARY

    if any(keyword in compact for keyword in [".net", "dotnet", "net service", "what about net"]):
        return DOTNET_SUMMARY

    if any(keyword in compact for keyword in ["industry", "industries", "domain", "sector"]):
        return INDUSTRY_SUMMARY

    if any(keyword in compact for keyword in [" ai", "ai ", "chatbot", "openai", "copilot", "machine learning", "automation"]):
        return AI_SUMMARY

    return None


def _get_embedding_model() -> str:
    configured_model = os.getenv("GOOGLE_EMBEDDING_MODEL")
    if configured_model:
        return configured_model

    return _detect_embedding_model()


def _get_chat_model() -> str:
    configured_model = os.getenv("GOOGLE_CHAT_MODEL")
    if configured_model:
        return configured_model

    return _detect_chat_model()


def _get_transcription_model() -> str:
    return os.getenv("GROQ_TRANSCRIPTION_MODEL", "whisper-large-v3")


def _get_tts_voice() -> str:
    return os.getenv("EDGE_TTS_VOICE", "en-US-AriaNeural")


def _get_max_output_tokens() -> int:
    return int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "4024"))


def _get_memory_turns() -> int:
    return int(os.getenv("CONVERSATION_MEMORY_TURNS", "6"))


def _build_model_input(session_id: str, current_query: str) -> str:
    with _conversation_lock:
        history = list(_conversation_store.get(session_id, []))

    if not history:
        return current_query

    history_lines: list[str] = []
    for user_text, assistant_text in history:
        history_lines.append(f"User: {user_text}")
        history_lines.append(f"Assistant: {assistant_text}")

    return (
        "Conversation history:\n"
        + "\n".join(history_lines)
        + "\n\nCurrent user question:\n"
        + current_query
    )


def _save_conversation_turn(session_id: str, user_text: str, assistant_text: str) -> None:
    with _conversation_lock:
        if session_id not in _conversation_store:
            _conversation_store[session_id] = deque(maxlen=_get_memory_turns())
        _conversation_store[session_id].append((user_text, assistant_text))


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


@lru_cache(maxsize=1)
def _detect_chat_model() -> str:
    genai.configure(api_key=_get_required_env("GOOGLE_API_KEY"))

    available_models: list[str] = []
    for model in genai.list_models():
        methods = set(getattr(model, "supported_generation_methods", []) or [])
        if "generateContent" in methods:
            model_name = getattr(model, "name", "")
            if model_name:
                available_models.append(model_name)

    if not available_models:
        raise ValueError(
            "No Google chat models are available for this API key. "
            "Set GOOGLE_CHAT_MODEL explicitly to a supported model."
        )

    preferred_suffixes = ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro", "gemini-pro"]
    for suffix in preferred_suffixes:
        for model_name in available_models:
            if model_name.endswith(suffix):
                return model_name

    return available_models[0]


def ensure_pinecone_index_exists() -> None:
    api_key = _get_required_env("PINECONE_API_KEY")
    index_name = _get_required_env("PINECONE_INDEX_NAME")
    auto_create = os.getenv("AUTO_CREATE_PINECONE_INDEX", "false").lower() == "true"

    pinecone_client = Pinecone(api_key=api_key)

    try:
        pinecone_client.describe_index(index_name)
        return
    except NotFoundException as error:
        if not auto_create:
            raise ValueError(
                f"Pinecone index '{index_name}' was not found. "
                "Create it manually or set AUTO_CREATE_PINECONE_INDEX=true."
            ) from error

    dimension = int(os.getenv("PINECONE_DIMENSION", "768"))
    metric = os.getenv("PINECONE_METRIC", "cosine")
    cloud = os.getenv("PINECONE_CLOUD", "aws")
    region = os.getenv("PINECONE_REGION", "us-east-1")

    pinecone_client.create_index(
        name=index_name,
        dimension=dimension,
        metric=metric,
        spec=ServerlessSpec(cloud=cloud, region=region),
    )

    for _ in range(30):
        description = pinecone_client.describe_index(index_name)
        status = description.status

        ready = status.get("ready") if isinstance(status, dict) else getattr(status, "ready", False)
        if ready:
            return

        sleep(2)

    raise ValueError(
        f"Pinecone index '{index_name}' was created but is not ready yet. Try again shortly."
    )


system_prompt = (
    "You are Desire Infoweb's professional virtual assistant for an IT services company. "
    "Answer the user's exact question directly and clearly using only company context. "
    "Do not start with generic filler like 'Would you like to know more?'. "
    "If the user asks about services, provide concrete service categories first. "
    "If the user asks about AI, explain Desire Infoweb AI offerings specifically. "
    "If the user asks about budget/cost, explain that pricing depends on scope and ask for key requirements. "
    "If the user asks about previous projects, provide relevant examples from available context. "
    "For follow-up questions, continue in context and avoid repeating generic summaries. "
    "If you do not know, say that clearly and offer to connect the user with the team. "
    "Keep answers business-focused, friendly, and practical. Prefer complete answers (around 3-8 sentences) when useful.\n\n"
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])


@lru_cache(maxsize=1)
def get_groq_client() -> Groq:
    return Groq(api_key=_get_required_env("GROQ_API_KEY"))


@lru_cache(maxsize=1)
def get_rag_chain():
    ensure_pinecone_index_exists()

    llm = ChatGoogleGenerativeAI(
        model=_get_chat_model(),
        temperature=0.1,
        max_output_tokens=_get_max_output_tokens(),
        convert_system_message_to_human=True,
    )
    embeddings = GoogleGenerativeAIEmbeddings(model=_get_embedding_model())
    vectorstore = PineconeVectorStore(
        index_name=_get_required_env("PINECONE_INDEX_NAME"),
        embedding=embeddings,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/api/chat/text")
async def text_chat(
    query: str = Form(...),
    session_id: str | None = Form(default=None),
    lead_email: str | None = Form(default=None),
    lead_name: str | None = Form(default=None),
) -> dict:
    try:
        normalized_query = _normalize_user_query(query)

        effective_session_id = _normalize_session_id(session_id)
        current_lead_email, current_lead_name = _resolve_lead_identity(
            effective_session_id,
            lead_email,
            lead_name,
        )

        model_input = _build_model_input(effective_session_id, normalized_query)

        rag_chain = get_rag_chain()
        response = rag_chain.invoke({"input": model_input})
        answer = response["answer"]
        _save_conversation_turn(effective_session_id, normalized_query, answer)
        await _sync_sharepoint_lead_safely(effective_session_id)

        return {
            "reply": answer,
            "session_id": effective_session_id,
            "lead": {
                "email": current_lead_email,
                "name": current_lead_name,
            },
        }
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "RAG backend is not ready. Verify GOOGLE_API_KEY, PINECONE_API_KEY, "
                "PINECONE_INDEX_NAME, and ensure the Pinecone index exists."
            ),
        ) from error


@app.post("/api/ingest/upload")
async def ingest_upload(
    file: UploadFile = File(...),
    x_ingest_key: str | None = Header(default=None),
) -> dict:
    configured_ingest_key = os.getenv("INGEST_API_KEY")
    if configured_ingest_key and x_ingest_key != configured_ingest_key:
        raise HTTPException(status_code=401, detail="Invalid ingestion API key")

    original_name = file.filename or "upload"
    extension = Path(original_name).suffix.lower()
    if extension not in SUPPORTED_INGEST_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Allowed: .pdf, .txt, .md, .csv, .log",
        )

    temp_file_path = f"ingest_{uuid.uuid4()}_{original_name}"

    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = ingest_file(temp_file_path, source_name=original_name)
        return {
            "status": "success",
            "message": "File ingested successfully",
            **result,
        }
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.post("/api/chat/voice")
async def voice_chat(
    audio: UploadFile = File(...),
    x_session_id: str | None = Header(default=None),
    x_lead_email: str | None = Header(default=None),
    x_lead_name: str | None = Header(default=None),
) -> Response:
    input_filename = audio.filename or "recording.webm"

    try:
        groq_client = get_groq_client()
        audio_bytes = await audio.read()

        transcription_model = _get_transcription_model()
        try:
            transcription = groq_client.audio.transcriptions.create(
                file=(input_filename, audio_bytes),
                model=transcription_model,
                prompt="The user is asking a question.",
                response_format="json",
            )
        except Exception as primary_error:
            should_retry_with_turbo = (
                "GROQ_TRANSCRIPTION_MODEL" not in os.environ
                and transcription_model != "whisper-large-v3-turbo"
            )

            if not should_retry_with_turbo:
                raise

            logger.warning(
                "Primary transcription model failed (%s). Retrying with whisper-large-v3-turbo.",
                primary_error,
            )
            transcription = groq_client.audio.transcriptions.create(
                file=(input_filename, audio_bytes),
                model="whisper-large-v3-turbo",
                prompt="The user is asking a question.",
                response_format="json",
            )

        user_text = _normalize_user_query((transcription.text or "").strip())
        if not user_text:
            raise HTTPException(status_code=400, detail="Could not transcribe user audio.")

        effective_session_id = _normalize_session_id(x_session_id)
        _resolve_lead_identity(
            effective_session_id,
            x_lead_email,
            x_lead_name,
        )

        model_input = _build_model_input(effective_session_id, user_text)

        rag_chain = get_rag_chain()
        response = rag_chain.invoke({"input": model_input})
        bot_reply_text = response["answer"]
        _save_conversation_turn(effective_session_id, user_text, bot_reply_text)
        await _sync_sharepoint_lead_safely(effective_session_id)

        communicate = edge_tts.Communicate(bot_reply_text, _get_tts_voice())
        output_audio_bytes = bytearray()
        async for chunk in communicate.stream():
            if chunk.get("type") == "audio":
                output_audio_bytes.extend(chunk.get("data", b""))

        return Response(
            content=bytes(output_audio_bytes),
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": "inline; filename=reply.mp3",
                "X-User-Query": _sanitize_header_value(user_text),
                "X-Bot-Reply": _sanitize_header_value(bot_reply_text),
            },
        )
    except HTTPException:
        raise
    except Exception as error:
        logger.exception("Voice pipeline failed")
        raise HTTPException(
            status_code=500,
            detail=(
                "Voice pipeline failed: "
                f"{type(error).__name__}: {error}"
            ),
        ) from error
    finally:
        await audio.close()
