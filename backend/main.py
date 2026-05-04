import os
import re
import shutil
import uuid
import logging
import json
import asyncio
from datetime import datetime, timezone
from collections import deque
from functools import lru_cache
from pathlib import Path
from threading import Lock
from threading import Thread
from time import sleep
from time import time
from typing import Any, AsyncGenerator, Iterable
from urllib.parse import quote

import edge_tts
from dotenv import load_dotenv
from fastapi import Body, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from groq import Groq

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ingestion import ingest_file
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from openai import AsyncAzureOpenAI, AzureOpenAI
from supabase import create_client
import pandas as pd
import io
import smtplib
from email.message import EmailMessage
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import timedelta

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from pinecone.core.client.exceptions import NotFoundException

try:
    import av
    import numpy as np
    from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
except Exception:  # pragma: no cover - optional runtime dependency
    av = None
    np = None
    RTCPeerConnection = None
    RTCSessionDescription = None
    VideoStreamTrack = object

load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

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
_pipeline_log_lock = Lock()
_pipeline_logs: deque[dict[str, Any]] = deque(maxlen=500)
_digicoco_kb_ready = False
_digicoco_kb_ingest_attempted = False

_employees_seed: list[dict[str, Any]] = [
    {"id": 1, "name": "Aarav Sharma", "active": True},
    {"id": 2, "name": "Neha Patel", "active": True},
    {"id": 3, "name": "Rohan Mehta", "active": False},
]

_cameras_seed: list[dict[str, Any]] = [
    {
        "id": 1,
        "camera_id": "cam1",
        "name": "Main Gate Cam",
        "location": "Unknown",
        "source": "",
        "status": "online",
        "fps": 0.0,
        "people": 0,
    },
    {
        "id": 2,
        "camera_id": "cam2",
        "name": "Lobby Cam",
        "location": "Unknown",
        "source": "",
        "status": "online",
        "fps": 0.0,
        "people": 0,
    },
    {
        "id": 3,
        "camera_id": "cam3",
        "name": "Parking Cam",
        "location": "Unknown",
        "source": "",
        "status": "offline",
        "fps": 0.0,
        "people": 0,
    },
]
_camera_lock = Lock()
EMPLOYEE_FACES_DIR = Path("data") / "employee_faces"
_webrtc_connections: dict[str, set[Any]] = {}


def _next_camera_id() -> int:
    if not _cameras_seed:
        return 1
    return max(int(camera.get("id", 0)) for camera in _cameras_seed) + 1


def _find_camera_index(camera_id: int) -> int:
    for index, camera in enumerate(_cameras_seed):
        if int(camera.get("id", -1)) == camera_id:
            return index
    return -1


def _find_camera_index_by_ref(camera_ref: str) -> int:
    normalized_ref = str(camera_ref).strip().lower()
    for index, camera in enumerate(_cameras_seed):
        if str(camera.get("id", "")).strip().lower() == normalized_ref:
            return index
        if str(camera.get("camera_id", "")).strip().lower() == normalized_ref:
            return index
    return -1


def _find_camera(camera_ref: str) -> dict[str, Any] | None:
    normalized_ref = str(camera_ref).strip().lower()
    for camera in _cameras_seed:
        if str(camera.get("id", "")).lower() == normalized_ref:
            return camera
        if str(camera.get("camera_id", "")).lower() == normalized_ref:
            return camera
        if str(camera.get("name", "")).strip().lower() == normalized_ref:
            return camera
    return None


def _next_employee_id() -> int:
    if not _employees_seed:
        return 1
    return max(int(employee.get("id", 0)) for employee in _employees_seed) + 1


class _SyntheticCameraTrack(VideoStreamTrack):
    def __init__(self, camera_label: str) -> None:
        super().__init__()
        self._camera_label = camera_label
        self._start_time = time()

    async def recv(self):  # type: ignore[override]
        pts, time_base = await self.next_timestamp()
        frame_age = int(time() - self._start_time)
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(canvas, format="bgr24")
        frame.pts = pts
        frame.time_base = time_base
        metadata = {
            "camera": self._camera_label,
            "uptime_seconds": frame_age,
            "status": "live-synthetic",
        }
        frame.metadata.update({k: str(v) for k, v in metadata.items()})
        return frame

KNOWLEDGE_BASE_FILE = Path("Knowledge base") / "DIGICoCo_Knowledge_Base.txt"
KNOWLEDGE_BASE_SOURCE_NAME = KNOWLEDGE_BASE_FILE.name
KB_CHUNK_SIZE_CHARS = 1800
KB_CHUNK_OVERLAP_CHARS = 300
OUT_OF_CONTEXT_FALLBACK = (
    "Apologise, I might not help you answering this question.\n\n"
    "I can help with DIGICoCo-related topics such as:\n"
    "- AI and chatbot services\n"
    "- Microsoft technology solutions\n"
    "- Project scope, implementation approach, and support options\n\n"
    "I don’t have that exact information right now, but I can connect you with the DIGICoCo team who will be able to help you quickly.\n"
    "You can reach them at:\n"
    "📧 info@digicoco.co.za\n"
    "📞 +27 11 656 8528"
)



def _get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


# --- Supabase client initialization ---
_supabase_client = None
def _init_supabase_client() -> None:
    global _supabase_client
    url = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
    # Prefer service role key for server-side writes (bypasses RLS). Falls back to publishable key (read-only / limited).
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY")

    if not url or not key:
        logger.warning("Supabase credentials not configured (NEXT_PUBLIC_SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY). Writes will be disabled.")
        _supabase_client = None
        return

    try:
        _supabase_client = create_client(url, key)
        if os.getenv("SUPABASE_SERVICE_ROLE_KEY"):
            logger.info("Supabase service-role key detected: server-side writes enabled.")
        else:
            logger.warning(
                "Supabase service-role key not found. Using publishable key — server-side writes may fail due to Row Level Security (RLS)."
            )
    except Exception as e:
        logger.exception("Failed to initialize Supabase client: %s", e)
        _supabase_client = None

_init_supabase_client()

def _supabase_upsert_user(name: str | None, email: str | None) -> None:
    if not _supabase_client or not email:
        return
    try:
        payload = {"email": email.strip(), "name": (name or "").strip()}
        # upsert by email
        _supabase_client.table("users").upsert(payload).execute()
    except Exception as e:
        if "Expecting value" not in str(e):
            logger.exception("Failed to upsert user into Supabase")


def _supabase_insert_message(name: str | None, email: str | None, session_id: str, role: str, content: str, ts: datetime) -> None:
    if not _supabase_client:
        return
    try:
        payload = {
            "session_id": session_id,
            "name": (name or "").strip(),
            "email": (email or "").strip(),
            "role": role,
            "content": content,
            "timestamp": ts.replace(tzinfo=timezone.utc).isoformat(),
        }
        _supabase_client.table("messages").insert(payload).execute()
    except Exception as e:
        if "Expecting value" not in str(e):
            logger.exception("Failed to insert message into Supabase")


def _supabase_upsert_session(session_id: str, name: str | None, email: str | None) -> None:
    """Create or update a session row mapping session_id to user (requires service role key or RLS policy)."""
    if not _supabase_client or not session_id or not email:
        return
    try:
        payload = {"session_id": session_id, "email": email.strip(), "name": (name or "").strip(), "last_seen_at": datetime.utcnow().isoformat()}
        _supabase_client.table("sessions").upsert(payload).execute()
    except Exception as e:
        if "Expecting value" not in str(e):
            logger.exception("Failed to upsert session into Supabase")


def _supabase_get_session(session_id: str) -> dict | None:
    if not _supabase_client or not session_id:
        return None
    try:
        resp = _supabase_client.table("sessions").select("*").filter("session_id", "eq", session_id).limit(1).execute()
        data = resp.data or []
        if not data:
            return None
        return data[0]
    except Exception:
        logger.exception("Failed to query session from Supabase")
        return None


def _generate_daily_report_and_send() -> None:
    """Query Supabase for full conversation history, build an Excel file and email it every 10 minutes."""
    global _supabase_client
    
    # Re-initialize Supabase client if not available (handles thread context issues)
    if not _supabase_client:
        logger.info("Reinitializing Supabase client in scheduler thread...")
        _init_supabase_client()
    
    if not _supabase_client:
        logger.warning("Skipping report: Supabase not configured.")
        return

    # determine recipient
    recipient = os.getenv("DAILY_REPORT_RECIPIENT") or os.getenv("REPORT_EMAIL") or os.getenv("ADMIN_EMAIL")
    if not recipient:
        logger.warning("No report recipient configured. Set DAILY_REPORT_RECIPIENT or REPORT_EMAIL or ADMIN_EMAIL in .env")
        return

    now = datetime.utcnow()
    max_retries = 3
    retry_delay = 1
    rows = []

    # Retry logic for querying messages (handles transient network issues)
    for attempt in range(max_retries):
        try:
            logger.info(f"Querying Supabase for messages (attempt {attempt + 1}/{max_retries})...")
            resp = _supabase_client.table("messages").select("*").order("timestamp", desc=False).execute()
            rows = resp.data or []
            logger.info(f"Successfully retrieved {len(rows)} messages from Supabase")
            break  # Success, exit retry loop
        except Exception as e:
            logger.warning(f"Failed to query messages (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                logger.exception("Max retries reached for querying messages")
                rows = []

    if not rows:
        logger.info("No messages found in Supabase; sending empty report with headers.")

    # Build human-readable conversation per user (group by email)
    data_by_user: dict[str, dict] = {}
    for row in rows:
        email = (row.get("email") or "").strip()
        name = (row.get("name") or "").strip()
        session = row.get("session_id") or ""
        role = row.get("role") or ""
        content = row.get("content") or ""
        ts = row.get("timestamp") or ""

        key = f"{email}:{session}"
        entry = data_by_user.setdefault(key, {"name": name, "email": email, "session_id": session, "messages": []})
        entry["messages"].append({"role": role, "content": content, "timestamp": ts})

    # Flatten into rows for Excel: name, email, session_id, conversation
    excel_rows = []
    for key, entry in data_by_user.items():
        conv_lines = []
        # sort by timestamp
        msgs = sorted(entry["messages"], key=lambda m: m.get("timestamp"))
        for m in msgs:
            label = "User" if m.get("role") == "user" else "Assistant"
            conv_lines.append(f"{label}: {m.get('content')}")
        excel_rows.append({"name": entry.get("name"), "email": entry.get("email"), "session_id": entry.get("session_id"), "conversation": "\n".join(conv_lines)})

    df = pd.DataFrame(excel_rows, columns=["name", "email", "session_id", "conversation"])
    output = io.BytesIO()
    df.to_excel(output, index=False, engine="openpyxl")
    output.seek(0)

    # Send email with attachment via SMTP if configured
    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT", "0") or 0)
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")

    if not smtp_host or not smtp_port or not smtp_user or not smtp_password:
        logger.warning("SMTP not fully configured; cannot send daily report email. Configure SMTP_HOST/SMTP_PORT/SMTP_USER/SMTP_PASSWORD in .env")
        return

    try:
        msg = EmailMessage()
        msg["Subject"] = f"Chat Conversations Report (Daily) - {now.strftime('%Y-%m-%d %H:%M:%S')} UTC"
        msg["From"] = smtp_user
        msg["To"] = recipient
        msg.set_content("Attached is the latest chatbot conversation report in human-readable format (Name, Email, Session ID, Conversation).")
        msg.add_attachment(output.read(), maintype="application", subtype="vnd.openxmlformats-officedocument.spreadsheetml.sheet", filename=f"conversations_{now.strftime('%Y%m%d_%H%M%S')}.xlsx")

        if smtp_port == 465:
            server = smtplib.SMTP_SSL(smtp_host, smtp_port)
        else:
            server = smtplib.SMTP(smtp_host, smtp_port)
            server.ehlo()
            server.starttls()
            server.ehlo()

        server.login(smtp_user, smtp_password)
        server.send_message(msg)
        server.quit()
        logger.info("Daily report emailed to %s", recipient)
    except Exception:
        logger.exception("Failed to send daily report email")


# Start Supabase client and scheduler
_scheduler: BackgroundScheduler | None = None

def _start_daily_report_scheduler() -> None:
    global _scheduler
    _init_supabase_client()
    # If already running, skip
    if _scheduler:
        return
    _scheduler = BackgroundScheduler()
    # run every 24 hours
    _scheduler.add_job(
        _generate_daily_report_and_send,
        'interval',
        hours=24,
        id='chat_report_24_hours',
        replace_existing=True,
        coalesce=True,
        max_instances=1,
    )
    _scheduler.start()
    logger.info("Report scheduler started: every 24 hours")


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


def _append_pipeline_log(
    *,
    session_id: str,
    user_query: str,
    ai_search: dict[str, Any] | None = None,
    ai_response: str | None = None,
    outcome: str,
    error: str | None = None,
) -> None:
    with _pipeline_log_lock:
        _pipeline_logs.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "session_id": session_id,
                "user_query": user_query,
                "ai_search": ai_search or {},
                "ai_response": ai_response or "",
                "outcome": outcome,
                "error": error or "",
            }
        )


def _get_recent_user_prompts(limit: int = 10) -> list[str]:
    safe_limit = max(1, min(limit, 100))
    with _pipeline_log_lock:
        recent_logs = list(_pipeline_logs)[-safe_limit:]
    return [entry["user_query"] for entry in recent_logs]


def _get_last_conversation_turn(session_id: str) -> tuple[str, str] | None:
    with _conversation_lock:
        history = _conversation_store.get(session_id)
        if not history:
            return None
        return history[-1]


def _is_digicoco_source(source_value: str | None) -> bool:
    if not source_value:
        return False
    return Path(str(source_value)).name.lower() == KNOWLEDGE_BASE_SOURCE_NAME.lower()


@lru_cache(maxsize=1)
def _load_digicoco_kb_text() -> str:
    kb_path = Path(__file__).parent / KNOWLEDGE_BASE_FILE
    try:
        return kb_path.read_text(encoding="utf-8")
    except Exception as read_error:
        logger.warning("Unable to read DIGICoCo knowledge-base text file: %s", read_error)
        return ""


def _build_kb_chunks() -> list[str]:
    kb_text = _load_digicoco_kb_text().strip()
    if not kb_text:
        return []

    chunks: list[str] = []
    start = 0
    text_len = len(kb_text)
    while start < text_len:
        end = min(start + KB_CHUNK_SIZE_CHARS, text_len)
        chunk = kb_text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_len:
            break
        start = max(0, end - KB_CHUNK_OVERLAP_CHARS)
    return chunks


@lru_cache(maxsize=1)
def _get_kb_chunks() -> list[str]:
    return _build_kb_chunks()


def _simple_tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2}


def _retrieve_local_kb_context(query: str, *, limit: int = 5) -> tuple[str, float]:
    query_tokens = _simple_tokens(query)
    if not query_tokens:
        return "", 0.0

    scored_chunks: list[tuple[float, str]] = []
    for chunk in _get_kb_chunks():
        chunk_tokens = _simple_tokens(chunk)
        if not chunk_tokens:
            continue
        overlap = query_tokens.intersection(chunk_tokens)
        if not overlap:
            continue

        # Reward direct token overlap while keeping score in [0, 1].
        score = len(overlap) / max(len(query_tokens), 1)
        scored_chunks.append((score, chunk))

    if not scored_chunks:
        return "", 0.0

    scored_chunks.sort(key=lambda item: item[0], reverse=True)
    selected = [chunk for _, chunk in scored_chunks[: max(1, min(limit, 8))]]
    top_score = float(scored_chunks[0][0])
    return "\n\n".join(selected), top_score


def _ensure_digicoco_knowledge_base_ingested() -> None:
    global _digicoco_kb_ready, _digicoco_kb_ingest_attempted
    if _digicoco_kb_ready:
        return

    # Do not retry expensive ingestion on every user request if it already failed once.
    if _digicoco_kb_ingest_attempted:
        return
    _digicoco_kb_ingest_attempted = True

    if os.getenv("AUTO_INGEST_DIGICOCO_KB", "true").lower() != "true":
        return

    kb_path = Path(__file__).parent / KNOWLEDGE_BASE_FILE
    if not kb_path.exists():
        logger.warning("DIGICoCo knowledge-base file not found at %s", kb_path)
        return

    try:
        ingest_file(str(kb_path), source_name=KNOWLEDGE_BASE_SOURCE_NAME)
        _digicoco_kb_ready = True
        logger.info("DIGICoCo knowledge-base ingested for RAG: %s", kb_path)
    except Exception as ingest_error:
        logger.warning("DIGICoCo knowledge-base ingestion failed: %s", ingest_error)


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


def _get_out_of_context_threshold() -> float:
    raw_value = os.getenv("OUT_OF_CONTEXT_THRESHOLD", "0.08")
    try:
        threshold = float(raw_value)
    except ValueError:
        threshold = 0.08
    return max(0.0, min(threshold, 1.0))


def _is_out_of_context(retrieved_context: str, top_score: float) -> bool:
    if not retrieved_context.strip():
        return True
    return top_score < _get_out_of_context_threshold()


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
    "Answer the user's exact question directly and clearly using ONLY the provided DIGICoCo knowledge-base context. "
    "Do not start with generic filler like 'Would you like to know more?'. "
    "Always return the final answer in valid GitHub-flavored Markdown (GFM). "
    "Use clean Markdown structure with short paragraphs and bullet points when useful. "
    "Do not output raw HTML. Do not output JSON unless the user explicitly asks for JSON. "
    "If the user asks about services, provide concrete service categories first. "
    "If the user asks about AI, explain DIGICoCo AI offerings specifically. "
    "If the user asks about budget/cost, explain that pricing depends on scope and ask for key requirements. "
    "If the user asks about previous projects, provide relevant examples only if they are present in context. "
    "For follow-up questions, continue in context and avoid repeating generic summaries. "
    "If the answer is not present in context, clearly say the information is not available in DIGICoCo knowledge base. "
    "Do not use external knowledge or assumptions. "
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
    pinecone_context = ""
    pinecone_score = 0.0

    try:
        matches = get_vectorstore().similarity_search_with_relevance_scores(
            query,
            k=5,
            filter={"source": KNOWLEDGE_BASE_SOURCE_NAME},
        )
    except Exception as retriever_error:
        logger.warning("Retriever lookup failed for score-based retrieval (%s).", retriever_error)
        matches = []

    context_chunks: list[str] = []
    top_score = 0.0


    for document, score in matches:
        source_name = str(getattr(document, "metadata", {}).get("source", "") or "")
        if not _is_digicoco_source(source_name):
            continue

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

    pinecone_context = "\n\n".join(context_chunks)
    pinecone_score = top_score
    if pinecone_context:
        return pinecone_context, pinecone_score

    local_context, local_score = _retrieve_local_kb_context(query, limit=5)
    return local_context, local_score


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
    async for token in _stream_answer_tokens_with_context(
        model_input=model_input,
        normalized_query=normalized_query,
        retrieved_context=retrieved_context,
        top_score=top_score,
    ):
        yield token


async def _stream_answer_tokens_with_context(
    model_input: str,
    normalized_query: str,
    retrieved_context: str,
    top_score: float,
) -> AsyncGenerator[str, None]:
    if _is_out_of_context(retrieved_context, top_score):
        yield OUT_OF_CONTEXT_FALLBACK
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

    if _is_out_of_context(retrieved_context, top_score):
        return OUT_OF_CONTEXT_FALLBACK

    try:
        return await _generate_completion_with_context(model_input, retrieved_context)
    except Exception as completion_error:
        logger.warning("Context-grounded completion failed (%s).", completion_error)
        raise


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/api/employees/")
async def list_employees(active_only: bool = Query(default=False)) -> list[dict[str, Any]]:
    if active_only:
        return [employee for employee in _employees_seed if employee.get("active")]
    return _employees_seed


@app.post("/api/employees/register")
async def register_employee(
    name: str = Form(...),
    employee_code: str = Form(default=""),
    department: str = Form(default=""),
    photos: list[UploadFile] = File(...),
) -> dict[str, Any]:
    if len(photos) < 5:
        raise HTTPException(status_code=400, detail="At least 5 photos are required for registration.")

    employee_id = _next_employee_id()
    normalized_code = (employee_code or f"EMP{employee_id:04d}").strip()
    employee_dir = EMPLOYEE_FACES_DIR / normalized_code
    employee_dir.mkdir(parents=True, exist_ok=True)

    saved_photos = 0
    for index, upload in enumerate(photos, start=1):
        raw = await upload.read()
        if not raw:
            continue
        extension = Path(upload.filename or "").suffix.lower()
        if extension not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        output_path = employee_dir / f"{index:02d}{extension}"
        output_path.write_bytes(raw)
        saved_photos += 1
        await upload.close()

    if saved_photos < 5:
        raise HTTPException(
            status_code=400,
            detail="At least 5 valid image files (.jpg/.jpeg/.png/.webp) are required.",
        )

    employee_record: dict[str, Any] = {
        "id": employee_id,
        "name": name.strip(),
        "employee_code": normalized_code,
        "department": department.strip() or "Unknown",
        "active": True,
        "face_samples": saved_photos,
    }
    _employees_seed.append(employee_record)
    return {"status": "success", "employee": employee_record}


@app.get("/api/cameras/")
async def list_cameras() -> list[dict[str, Any]]:
    return _cameras_seed


@app.post("/api/cameras/")
async def create_camera(payload: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
    with _camera_lock:
        new_camera = dict(payload)
        new_camera["id"] = _next_camera_id()
        new_camera.setdefault("camera_id", f"cam{new_camera['id']}")
        new_camera.setdefault("name", f"CAM_{new_camera['id']:03d}")
        new_camera.setdefault("location", "Unknown")
        if "source" not in new_camera:
            new_camera["source"] = new_camera.get("stream_source", "")
        new_camera.setdefault("fps", 0.0)
        new_camera.setdefault("people", 0)
        new_camera.setdefault("status", "offline")
        _cameras_seed.append(new_camera)
        return new_camera


@app.put("/api/cameras/{camera_ref}")
async def update_camera(camera_ref: str, payload: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
    with _camera_lock:
        camera_index = _find_camera_index_by_ref(camera_ref)
        if camera_index < 0:
            raise HTTPException(status_code=404, detail="Camera not found")

        updated_camera = dict(_cameras_seed[camera_index])
        for key, value in payload.items():
            if key != "id":
                updated_camera[key] = value
        _cameras_seed[camera_index] = updated_camera
        return updated_camera


@app.delete("/api/cameras/{camera_ref}")
async def delete_camera(camera_ref: str) -> dict[str, Any]:
    with _camera_lock:
        camera_index = _find_camera_index_by_ref(camera_ref)
        if camera_index < 0:
            raise HTTPException(status_code=404, detail="Camera not found")
        deleted_camera = _cameras_seed.pop(camera_index)
        return {"status": "success", "deleted": deleted_camera}


@app.post("/api/webrtc/{camera_ref}/offer")
async def webrtc_offer(camera_ref: str, payload: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
    camera = _find_camera(camera_ref)
    if camera is None:
        raise HTTPException(status_code=404, detail="Camera not found")

    if not (RTCPeerConnection and RTCSessionDescription and av and np):
        raise HTTPException(
            status_code=500,
            detail=(
                "WebRTC runtime dependencies are missing. Install aiortc, av, and numpy "
                "in backend virtualenv to enable live feed streaming."
            ),
        )

    offer_sdp = str(payload.get("sdp", ""))
    offer_type = str(payload.get("type", "offer"))
    if not offer_sdp:
        raise HTTPException(status_code=400, detail="Missing SDP offer")
    if offer_type != "offer":
        raise HTTPException(status_code=400, detail="Invalid SDP type; expected 'offer'")

    peer = RTCPeerConnection()
    camera_id = str(camera.get("camera_id", camera_ref))
    _webrtc_connections.setdefault(camera_id, set()).add(peer)

    @peer.on("connectionstatechange")
    async def _on_connection_state_change() -> None:
        if peer.connectionState in {"failed", "closed", "disconnected"}:
            await peer.close()
            _webrtc_connections.get(camera_id, set()).discard(peer)

    peer.addTrack(_SyntheticCameraTrack(camera_label=camera_id))
    await peer.setRemoteDescription(RTCSessionDescription(sdp=offer_sdp, type=offer_type))
    answer = await peer.createAnswer()
    await peer.setLocalDescription(answer)

    # Give ICE gathering a short window so localDescription has candidates.
    for _ in range(25):
        if peer.iceGatheringState == "complete":
            break
        await asyncio.sleep(0.05)

    local = peer.localDescription
    if local is None:
        raise HTTPException(status_code=500, detail="Failed to create WebRTC answer")

    return {
        "type": local.type,
        "sdp": local.sdp,
        "camera_id": camera_id,
        "mode": "synthetic-video",
    }


@app.on_event("startup")
async def _startup_ingestion() -> None:
    run_on_startup = os.getenv("AUTO_INGEST_DIGICOCO_KB_ON_STARTUP", "false").lower() == "true"
    if not run_on_startup:
        logger.info("Skipping DIGICoCo knowledge-base ingestion at startup for faster boot.")
        # still start the daily report scheduler even if ingestion is skipped
        try:
            _start_daily_report_scheduler()
        except Exception:
            logger.exception("Failed to start daily report scheduler")
        return

    def _ingest_in_background() -> None:
        try:
            _ensure_digicoco_knowledge_base_ingested()
        except Exception as startup_ingest_error:
            logger.warning("Background startup ingestion failed: %s", startup_ingest_error)

    Thread(target=_ingest_in_background, daemon=True).start()
    try:
        _start_daily_report_scheduler()
    except Exception:
        logger.exception("Failed to start daily report scheduler")


@app.post("/api/chat/text")
async def text_chat(
    query: str = Form(...),
    session_id: str | None = Form(default=None),
    x_user_name: str | None = Header(default=None),
    x_user_email: str | None = Header(default=None),
) -> dict:
    try:
        normalized_query = _normalize_user_query(query)
        effective_session_id = _normalize_session_id(session_id)

        model_input = _build_model_input(effective_session_id, normalized_query)
        retrieved_context, top_score = _retrieve_context_and_score(normalized_query)
        answer = await _generate_answer(
            model_input,
            normalized_query,
            retrieved_context=retrieved_context,
            top_score=top_score,
        )
        _save_conversation_turn(effective_session_id, normalized_query, answer)
        # persist user and assistant messages to Supabase (if configured)
        try:
            # Ensure user and session rows exist, then persist messages
            _supabase_upsert_user(name=x_user_name, email=x_user_email)
            _supabase_upsert_session(session_id=effective_session_id, name=x_user_name, email=x_user_email)
            _supabase_insert_message(x_user_name, x_user_email, effective_session_id, "user", normalized_query, datetime.utcnow())
            _supabase_insert_message(x_user_name, x_user_email, effective_session_id, "assistant", answer, datetime.utcnow())
        except Exception:
            logger.exception("Failed to persist conversation to Supabase")
        _append_pipeline_log(
            session_id=effective_session_id,
            user_query=normalized_query,
            ai_search={
                "knowledge_source": KNOWLEDGE_BASE_SOURCE_NAME,
                "top_score": top_score,
                "context_found": bool(retrieved_context),
            },
            ai_response=answer,
            outcome="success",
        )


        return {
            "reply": answer,
            "session_id": effective_session_id,
        }
    except Exception as error:
        _append_pipeline_log(
            session_id=_normalize_session_id(session_id),
            user_query=_normalize_user_query(query),
            ai_search={"knowledge_source": KNOWLEDGE_BASE_SOURCE_NAME},
            outcome="error",
            error=f"{type(error).__name__}: {str(error)}",
        )
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
    x_user_name: str | None = Header(default=None),
    x_user_email: str | None = Header(default=None),
) -> StreamingResponse:
    normalized_query = _normalize_user_query(query)
    effective_session_id = _normalize_session_id(session_id)
    model_input = _build_model_input(effective_session_id, normalized_query)

    async def event_generator() -> AsyncGenerator[str, None]:
        answer_parts: list[str] = []
        retrieved_context = ""
        top_score = 0.0

        try:
            retrieved_context, top_score = _retrieve_context_and_score(normalized_query)
            async for token in _stream_answer_tokens_with_context(
                model_input=model_input,
                normalized_query=normalized_query,
                retrieved_context=retrieved_context,
                top_score=top_score,
            ):
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
            # persist user and assistant messages to Supabase (if configured)
            try:
                _supabase_upsert_user(x_user_name, x_user_email)
                _supabase_upsert_user(name=x_user_name, email=x_user_email)
                _supabase_upsert_session(session_id=effective_session_id, name=x_user_name, email=x_user_email)
                _supabase_insert_message(x_user_name, x_user_email, effective_session_id, "user", normalized_query, datetime.utcnow())
                _supabase_insert_message(x_user_name, x_user_email, effective_session_id, "assistant", final_answer, datetime.utcnow())
            except Exception:
                logger.exception("Failed to persist streamed conversation to Supabase")
            _append_pipeline_log(
                session_id=effective_session_id,
                user_query=normalized_query,
                ai_search={
                    "knowledge_source": KNOWLEDGE_BASE_SOURCE_NAME,
                    "top_score": top_score,
                    "context_found": bool(retrieved_context),
                },
                ai_response=final_answer,
                outcome="success",
            )

            yield _sse_event(
                "done",
                {
                    "reply": final_answer,
                    "session_id": effective_session_id,
                },
            )
        except Exception as error:
            _append_pipeline_log(
                session_id=effective_session_id,
                user_query=normalized_query,
                ai_search={"knowledge_source": KNOWLEDGE_BASE_SOURCE_NAME},
                outcome="error",
                error=f"{type(error).__name__}: {str(error)}",
            )
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


@app.post('/api/session/register')
async def register_session(payload: dict = Body(default_factory=dict)) -> dict:
    """Register or update a session. Expects JSON: {"email": "..", "name": "..", "session_id": "optional"}.
    Returns: {"session_id": "..."}
    """
    email = (payload.get('email') or '').strip()
    name = (payload.get('name') or '').strip()
    session_id = (payload.get('session_id') or '').strip() or f"s_{uuid.uuid4().hex}"

    if not email:
        raise HTTPException(status_code=400, detail='Missing email')

    try:
        logger.info("Session register called: session_id=%s email=%s supabase_client_set=%s", session_id, email, bool(_supabase_client))
        _supabase_upsert_user(name=name, email=email)
        _supabase_upsert_session(session_id=session_id, name=name, email=email)
        logger.info("Session register completed: session_id=%s", session_id)
        return {"session_id": session_id}
    except Exception as error:
        logger.exception('Session register failed')
        raise HTTPException(status_code=500, detail='Failed to register session') from error


@app.get('/api/session/{session_id}')
async def get_session(session_id: str) -> dict:
    session_id = (session_id or '').strip()
    if not session_id:
        raise HTTPException(status_code=400, detail='Missing session_id')
    session = _supabase_get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail='Session not found')
    return {
        'session_id': session.get('session_id'),
        'email': session.get('email'),
        'name': session.get('name'),
        'last_seen_at': session.get('last_seen_at'),
    }


@app.post("/api/chat/voice")
async def voice_chat(
    audio: UploadFile = File(...),
    x_session_id: str | None = Header(default=None),
    x_user_name: str | None = Header(default=None),
    x_user_email: str | None = Header(default=None),
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
        retrieved_context, top_score = _retrieve_context_and_score(user_text)
        bot_reply_text = await _generate_answer(
            model_input,
            user_text,
            retrieved_context=retrieved_context,
            top_score=top_score,
        )
        _save_conversation_turn(effective_session_id, user_text, bot_reply_text)
        try:
            _supabase_upsert_user(x_user_name, x_user_email)
            _supabase_upsert_user(name=x_user_name, email=x_user_email)
            _supabase_upsert_session(session_id=effective_session_id, name=x_user_name, email=x_user_email)
            _supabase_insert_message(x_user_name, x_user_email, effective_session_id, "user", user_text, datetime.utcnow())
            _supabase_insert_message(x_user_name, x_user_email, effective_session_id, "assistant", bot_reply_text, datetime.utcnow())
        except Exception:
            logger.exception("Failed to persist voice conversation to Supabase")
        _append_pipeline_log(
            session_id=effective_session_id,
            user_query=user_text,
            ai_search={
                "knowledge_source": KNOWLEDGE_BASE_SOURCE_NAME,
                "top_score": top_score,
                "context_found": bool(retrieved_context),
            },
            ai_response=bot_reply_text,
            outcome="success",
        )


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
        _append_pipeline_log(
            session_id=_normalize_session_id(x_session_id),
            user_query=input_filename,
            ai_search={"knowledge_source": KNOWLEDGE_BASE_SOURCE_NAME},
            outcome="error",
            error=f"{type(error).__name__}: {str(error)}",
        )
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


@app.get("/api/chat/logs")
async def get_chat_logs(limit: int = 10) -> dict:
    safe_limit = max(1, min(limit, 100))
    with _pipeline_log_lock:
        logs = list(_pipeline_logs)[-safe_limit:]

    return {
        "knowledge_source": KNOWLEDGE_BASE_SOURCE_NAME,
        "recent_user_prompts": _get_recent_user_prompts(limit=10),
        "logs": logs,
    }


