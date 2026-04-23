import os
import re
import shutil
import uuid
import logging
import json
from datetime import datetime, timezone
from collections import deque
from functools import lru_cache
from pathlib import Path
from threading import Lock
from time import sleep
from time import time
from typing import AsyncGenerator, Iterable
from urllib.parse import quote

import edge_tts
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from groq import Groq
from ingestion import ingest_file
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from openai import AsyncAzureOpenAI, AzureOpenAI

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
    expose_headers=[
        "X-Session-Id",
        "X-User-Query",
        "X-Bot-Reply",
        "X-User-Query-Encoded",
        "X-Bot-Reply-Encoded",
    ],
)

SUPPORTED_INGEST_EXTENSIONS = {".pdf", ".txt", ".md", ".csv", ".log"}

SERVICE_SUMMARY = (
    "DIGICoCo provides Microsoft-focused IT services including SharePoint, "
    "Power Apps, Power Automate, Power BI, Office 365, Teams, Dynamics 365, Azure, "
    ".NET, migration, automation, and AI/chatbot solutions."
)

AI_SUMMARY = (
    "DIGICoCo AI services include Azure OpenAI-based solutions, Teams chatbots, "
    "Copilot-aligned workflows, intelligent automation, and document-grounded chatbot implementations."
)

AI_PROJECTS_SUMMARY = (
    "Some AI project examples from DIGICoCo include: "
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
    "DIGICoCo .NET services include custom enterprise application development, "
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
    "DIGICoCo serves industries such as education, retail/e-commerce, finance, "
    "real estate, travel, healthcare, and logistics/distribution."
)

_conversation_lock = Lock()
_conversation_store: dict[str, deque[tuple[str, str]]] = {}



def _get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _sanitize_header_value(value: str, *, max_chars: int = 700) -> str:
    normalized = value.replace("\r", " ").replace("\n", " ").strip()
    normalized = normalized[:max_chars]
    return normalized.encode("latin1", "ignore").decode("latin1")


def _encode_header_value(value: str, *, max_chars: int = 2500) -> str:
    normalized = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    normalized = normalized[:max_chars]
    return quote(normalized, safe="")


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


def _resolve_lead_identity(session_id: str) -> None:
    pass


def _build_conversation_transcript(session_id: str) -> str:
    with _conversation_lock:
        history = list(_conversation_store.get(session_id, []))

    lines: list[str] = []
    for user_text, assistant_text in history:
        lines.append(f"User: {user_text}")
        lines.append(f"Assistant: {assistant_text}")

    return "\n".join(lines)


def _get_last_conversation_turn(session_id: str) -> tuple[str, str] | None:
    with _conversation_lock:
        history = _conversation_store.get(session_id)
        if not history:
            return None
        return history[-1]


# SharePoint logic removed

def _direct_company_answer(query: str) -> str | None:
    q = query.lower().strip()
    compact = re.sub(r"[^a-z0-9\s]", "", q)

    if re.fullmatch(r"(hi|hello|hey|hii|hiii|good morning|good afternoon|good evening)", compact):
        return (
            "Hello! Welcome to DIGICoCo. "
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
    return _get_required_env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")


def _get_chat_model() -> str:
    return _get_required_env("AZURE_OPENAI_CHAT_DEPLOYMENT")


def _get_azure_openai_endpoint() -> str:
    return _get_required_env("AZURE_OPENAI_ENDPOINT")


def _get_azure_openai_api_key() -> str:
    return _get_required_env("AZURE_OPENAI_API_KEY")


def _get_azure_openai_api_version() -> str:
    return os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")


def _get_transcription_model() -> str:
    return os.getenv("GROQ_TRANSCRIPTION_MODEL", "whisper-large-v3")


def _get_tts_voice() -> str:
    return os.getenv("EDGE_TTS_VOICE", "en-US-AriaNeural")


def _get_max_output_tokens() -> int:
    requested_raw = os.getenv("LLM_MAX_OUTPUT_TOKENS", "1200")
    model_cap_raw = os.getenv("AZURE_OPENAI_MAX_COMPLETION_TOKENS", "16384")

    try:
        requested_tokens = int(requested_raw)
    except ValueError:
        requested_tokens = 1200

    try:
        model_cap = int(model_cap_raw)
    except ValueError:
        model_cap = 16384

    bounded_tokens = max(64, min(requested_tokens, model_cap))
    if bounded_tokens != requested_tokens:
        logger.warning(
            "LLM_MAX_OUTPUT_TOKENS=%s exceeds allowed range; using %s instead.",
            requested_tokens,
            bounded_tokens,
        )

    return bounded_tokens


def _get_llm_temperature() -> float:
    return float(os.getenv("AZURE_OPENAI_TEMPERATURE", "0.1"))


def _get_embedding_similarity_threshold() -> float:
    raw_value = os.getenv("EMBEDDING_SIMILARITY_THRESHOLD", "0.45")
    try:
        threshold = float(raw_value)
    except ValueError:
        threshold = 0.45
    return max(0.0, min(threshold, 1.0))


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


def _sse_event(event: str, data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def _save_conversation_turn(session_id: str, user_text: str, assistant_text: str) -> None:
    with _conversation_lock:
        if session_id not in _conversation_store:
            _conversation_store[session_id] = deque(maxlen=_get_memory_turns())
        _conversation_store[session_id].append((user_text, assistant_text))


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

    dimension = int(os.getenv("PINECONE_DIMENSION", "1536"))
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
    "You are DIGICoCo's professional virtual assistant for an IT services company. "
    "Answer the user's exact question directly and clearly using only company context. "
    "Do not start with generic filler like 'Would you like to know more?'. "
    "Always return the final answer in valid GitHub-flavored Markdown (GFM). "
    "Use clean Markdown structure with short paragraphs and bullet points when useful. "
    "Do not output raw HTML. Do not output JSON unless the user explicitly asks for JSON. "
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
def get_azure_openai_client() -> AzureOpenAI:
    return AzureOpenAI(
        api_version=_get_azure_openai_api_version(),
        azure_endpoint=_get_azure_openai_endpoint(),
        api_key=_get_azure_openai_api_key(),
    )


@lru_cache(maxsize=1)
def get_async_azure_openai_client() -> AsyncAzureOpenAI:
    return AsyncAzureOpenAI(
        api_version=_get_azure_openai_api_version(),
        azure_endpoint=_get_azure_openai_endpoint(),
        api_key=_get_azure_openai_api_key(),
    )



@lru_cache(maxsize=1)
def get_groq_client() -> Groq:
    return Groq(api_key=_get_required_env("GROQ_API_KEY"))


@lru_cache(maxsize=1)
def get_vectorstore() -> PineconeVectorStore:
    ensure_pinecone_index_exists()
    embedding_deployment = _get_embedding_model()
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=_get_azure_openai_endpoint(),
        api_key=_get_azure_openai_api_key(),
        openai_api_version=_get_azure_openai_api_version(),
        azure_deployment=embedding_deployment,
        model=embedding_deployment,
    )
    vectorstore = PineconeVectorStore(
        index_name=_get_required_env("PINECONE_INDEX_NAME"),
        embedding=embeddings,
    )
    return vectorstore


@lru_cache(maxsize=1)
def get_rag_chain():
    llm = AzureChatOpenAI(
        azure_endpoint=_get_azure_openai_endpoint(),
        api_key=_get_azure_openai_api_key(),
        openai_api_version=_get_azure_openai_api_version(),
        azure_deployment=_get_chat_model(),
        temperature=_get_llm_temperature(),
        max_tokens=_get_max_output_tokens(),
    )
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)


@lru_cache(maxsize=1)
def get_retriever():
    return get_vectorstore().as_retriever(search_kwargs={"k": 5})


def _retrieve_context_and_score(query: str) -> tuple[str, float]:
    try:
        matches = get_vectorstore().similarity_search_with_relevance_scores(query, k=5)
    except Exception as retriever_error:
        logger.warning("Retriever lookup failed for score-based retrieval (%s).", retriever_error)
        return "", 0.0

    context_chunks: list[str] = []
    top_score = 0.0


    for document, score in matches:
        try:
            score_value = float(score)
        except (TypeError, ValueError):
            score_value = 0.0

        if score_value > top_score:
            top_score = score_value

        page_content = str(getattr(document, "page_content", "") or "").strip()
        if not page_content:
            continue

        context_chunks.append(page_content[:2000])
        if len(context_chunks) >= 5:
            break

    return "\n\n".join(context_chunks), top_score


def _should_use_embedding_context(retrieved_context: str, top_score: float) -> bool:
    if not retrieved_context:
        return False
    return top_score >= _get_embedding_similarity_threshold()


def _build_retrieved_context(query: str) -> str:
    context, _ = _retrieve_context_and_score(query)
    return context


async def _generate_completion_with_context(model_input: str, retrieved_context: str) -> str:
    client = get_async_azure_openai_client()
    completion = await client.chat.completions.create(
        model=_get_chat_model(),
        messages=[
            {"role": "system", "content": system_prompt.replace("{context}", retrieved_context)},
            {"role": "user", "content": model_input},
        ],
        temperature=_get_llm_temperature(),
        max_tokens=_get_max_output_tokens(),
        stream=False,
    )


    if not completion.choices:
        raise ValueError("Completion response returned no choices.")

    message = completion.choices[0].message
    answer = str(getattr(message, "content", "") or "").strip()
    if not answer:
        raise ValueError("Completion response returned an empty answer.")

    return answer


async def _stream_answer_tokens(model_input: str, normalized_query: str) -> AsyncGenerator[str, None]:
    retrieved_context, top_score = _retrieve_context_and_score(normalized_query)
    use_embedding_context = _should_use_embedding_context(retrieved_context, top_score)

    if not use_embedding_context:
        direct_answer = _direct_company_answer(normalized_query)
        if direct_answer:
            yield direct_answer
            return

    try:
        client = get_async_azure_openai_client()
        completion_stream = await client.chat.completions.create(
            model=_get_chat_model(),
            messages=[
                {"role": "system", "content": system_prompt.replace("{context}", retrieved_context)},
                {"role": "user", "content": model_input},
            ],
            temperature=_get_llm_temperature(),
            max_tokens=_get_max_output_tokens(),
            stream=True,
        )

        has_streamed_content = False
        async for chunk in completion_stream:
            if not chunk.choices:
                continue


            delta = chunk.choices[0].delta
            token = getattr(delta, "content", None) if delta else None
            if not token:
                continue

            has_streamed_content = True
            yield token

        if not has_streamed_content:
            raise ValueError("Streaming response returned no content.")
    except Exception as stream_error:
        logger.warning(
            "Streaming generation failed (%s). Falling back to non-streaming answer.",
            stream_error,
        )
        fallback_answer = await _generate_answer(
            model_input,
            normalized_query,
            retrieved_context=retrieved_context,
            top_score=top_score,
        )
        if fallback_answer:
            yield fallback_answer



async def _generate_answer(
    model_input: str,
    normalized_query: str,
    retrieved_context: str | None = None,
    top_score: float | None = None,
) -> str:

    if retrieved_context is None or top_score is None:
        retrieved_context, top_score = _retrieve_context_and_score(normalized_query)

    use_embedding_context = _should_use_embedding_context(retrieved_context, top_score)
    direct_answer = None if use_embedding_context else _direct_company_answer(normalized_query)

    if use_embedding_context:
        try:
            return await _generate_completion_with_context(model_input, retrieved_context)
        except Exception as completion_error:

            logger.warning(
                "Context-grounded completion failed (%s). Falling back to RAG chain.",
                completion_error,
            )

    try:
        rag_chain = get_rag_chain()
        response = rag_chain.invoke({"input": model_input})
        answer = str(response.get("answer", "")).strip()
        if answer:
            return answer
        raise ValueError("Azure OpenAI RAG chain returned an empty answer.")
    except Exception as rag_error:
        if direct_answer:
            logger.warning(
                "RAG generation failed (%s). Returning direct fallback answer.",
                rag_error,
            )
            return direct_answer
        raise


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/api/chat/text")
async def text_chat(
    query: str = Form(...),
    session_id: str | None = Form(default=None),
) -> dict:
    try:
        normalized_query = _normalize_user_query(query)
        effective_session_id = _normalize_session_id(session_id)

        model_input = _build_model_input(effective_session_id, normalized_query)
        answer = await _generate_answer(model_input, normalized_query)
        _save_conversation_turn(effective_session_id, normalized_query, answer)


        return {
            "reply": answer,
            "session_id": effective_session_id,
        }
    except Exception as error:
        logger.exception("Text chat pipeline failed")
        raise HTTPException(
            status_code=500,
            detail=(
                "Answer generation failed. Verify AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, "
                "AZURE_OPENAI_CHAT_DEPLOYMENT, AZURE_OPENAI_EMBEDDING_DEPLOYMENT, and Pinecone settings."
            ),
        ) from error


@app.post("/api/chat/text/stream")
async def text_chat_stream(
    query: str = Form(...),
    session_id: str | None = Form(default=None),
) -> StreamingResponse:
    normalized_query = _normalize_user_query(query)
    effective_session_id = _normalize_session_id(session_id)
    model_input = _build_model_input(effective_session_id, normalized_query)

    async def event_generator() -> AsyncGenerator[str, None]:
        answer_parts: list[str] = []

        try:
            async for token in _stream_answer_tokens(model_input, normalized_query):
                if not token:

                    continue
                answer_parts.append(token)
                yield _sse_event("token", {"token": token})

            final_answer = "".join(answer_parts).strip()
            if not final_answer:
                final_answer = await _generate_answer(model_input, normalized_query)
                if final_answer:
                    yield _sse_event("token", {"token": final_answer})


            _save_conversation_turn(effective_session_id, normalized_query, final_answer)

            yield _sse_event(
                "done",
                {
                    "reply": final_answer,
                    "session_id": effective_session_id,
                },
            )
        except Exception as error:
            logger.exception("Text chat streaming pipeline failed")
            yield _sse_event(
                "error",
                {
                    "message": (
                        "Answer generation failed. Verify AZURE_OPENAI_ENDPOINT, "
                        "AZURE_OPENAI_API_KEY, AZURE_OPENAI_CHAT_DEPLOYMENT, "
                        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT, and Pinecone settings. "
                        f"Error: {type(error).__name__}: {str(error)}"
                    ),
                    "error_type": type(error).__name__,

                },
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


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
        model_input = _build_model_input(effective_session_id, user_text)
        bot_reply_text = await _generate_answer(model_input, user_text)
        _save_conversation_turn(effective_session_id, user_text, bot_reply_text)


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
                "X-Session-Id": effective_session_id,
                "X-User-Query": _sanitize_header_value(user_text),
                "X-Bot-Reply": _sanitize_header_value(bot_reply_text),
                "X-User-Query-Encoded": _encode_header_value(user_text),
                "X-Bot-Reply-Encoded": _encode_header_value(bot_reply_text),
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


@app.get("/api/chat/last")
async def get_last_chat_turn(session_id: str) -> dict:
    effective_session_id = _normalize_session_id(session_id)
    last_turn = _get_last_conversation_turn(effective_session_id)
    if not last_turn:
        raise HTTPException(status_code=404, detail="No conversation found for session_id")

    user_text, bot_reply_text = last_turn

    return {
        "session_id": effective_session_id,
        "user_query": user_text,
        "reply": bot_reply_text,
    }


