# Setup and Build Guide

## 1) Prerequisites

- Python 3.11+
- Node.js 18+ and npm
- API keys:
  - Google API key
  - Pinecone API key
  - Groq API key
- A Pinecone serverless index configured as:
  - Name: `chatbot-rag` (or your custom value)
  - Dimension: `768`
  - Metric: `cosine`

## 2) Backend Setup (Windows PowerShell)

```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Set values in `backend/.env`:

```env
GOOGLE_API_KEY="your_google_api_key_here"
GOOGLE_EMBEDDING_MODEL=""
GOOGLE_CHAT_MODEL=""
PINECONE_API_KEY="your_pinecone_api_key_here"
GROQ_API_KEY="your_groq_api_key_here"
GROQ_TRANSCRIPTION_MODEL="whisper-large-v3"
EDGE_TTS_VOICE="en-US-AriaNeural"
PINECONE_INDEX_NAME="chatbot-rag"
AUTO_CREATE_PINECONE_INDEX="false"
PINECONE_CLOUD="aws"
PINECONE_REGION="us-east-1"
PINECONE_DIMENSION="768"
PINECONE_METRIC="cosine"
ENABLE_SHAREPOINT_SYNC="false"
SHAREPOINT_TENANT_ID="your_tenant_id"
SHAREPOINT_CLIENT_ID="your_app_client_id"
SHAREPOINT_CLIENT_SECRET="your_app_client_secret"
SHAREPOINT_SITE_ID="your_site_id"
SHAREPOINT_LIST_ID="your_list_id"
SHAREPOINT_FIELD_TITLE="Title"
SHAREPOINT_FIELD_NAME="Name"
SHAREPOINT_FIELD_EMAIL="email"
SHAREPOINT_FIELD_CONVERSATION="Conversation"
```

`GOOGLE_EMBEDDING_MODEL` is optional. If empty, the backend auto-detects a supported embedding model from your Google account.
`GOOGLE_CHAT_MODEL` is optional. If empty, the backend auto-detects a supported chat model from your Google account.

### SharePoint list sync (optional)

If you want leads to be stored in a SharePoint list, set `ENABLE_SHAREPOINT_SYNC="true"` and configure the Microsoft Graph app credentials.

Required setup (app-only auth):

1. Create an app registration in Microsoft Entra ID.
2. Add application permission `Sites.ReadWrite.All` for Microsoft Graph and grant admin consent.
3. Create a client secret for the app.
4. Get the SharePoint site ID and list ID (Graph Explorer or SharePoint API).
5. Ensure your list has fields matching the internal names you set in the env vars (defaults: `Title`, `Name`, `email`, `Conversation`).

### Ingest your knowledge base

1. Put your source content into `backend/data.txt`
2. Run:

```powershell
python ingest.py
```

### Ingest a PDF or text file via API upload

1. Optionally set `INGEST_API_KEY` in `backend/.env`
2. Start backend server
3. Upload file:

```powershell
curl -X POST "http://localhost:8000/api/ingest/upload" ^
  -H "X-Ingest-Key: your_ingest_key_if_set" ^
  -F "file=@C:\path\to\your-document.pdf"
```

Supported extensions: `.pdf`, `.txt`, `.md`, `.csv`, `.log`

### Run backend locally

```powershell
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Health check:

- `http://localhost:8000/health`

## 3) Frontend Setup (Windows PowerShell)

```powershell
cd frontend
npm install
Copy-Item .env.example .env
```

Set `frontend/.env`:

```env
VITE_API_BASE_URL="http://localhost:8000"
```

Run frontend:

```powershell
npm run dev
```

## 4) Build Commands

### Frontend production build

```powershell
cd frontend
npm run build
```

Output directory:

- `frontend/dist/`

### Backend syntax check (optional)

```powershell
cd backend
python -m py_compile main.py ingest.py
```

## 5) Quick Local Smoke Test

1. Start backend (`:8000`)
2. Start frontend (Vite URL shown in terminal)
3. Open the frontend URL
4. Send a text query
5. Hold mic button and test voice request

If voice fails, verify browser microphone permissions and backend env keys.

## 6) Troubleshooting Common Startup Errors

### Error: `Client.__init__() got an unexpected keyword argument 'proxies'`

Cause: incompatible `httpx` version with `groq` SDK.

Fix:

```powershell
cd backend
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

This project pins `httpx==0.27.2` to avoid this issue.

### Error: `pinecone ... NotFoundException: Resource chatbot-rag not found`

Cause: the Pinecone index in `PINECONE_INDEX_NAME` does not exist in your account/project.

Fix:

- Create the index in Pinecone (dimension `768`, metric `cosine`), or
- Update `PINECONE_INDEX_NAME` in `backend/.env` to an existing index name.

Alternative auto-fix:

- Set `AUTO_CREATE_PINECONE_INDEX="true"`
- Ensure `PINECONE_CLOUD` and `PINECONE_REGION` are correct for your Pinecone environment
- Restart backend

### Error: `models/text-embedding-004 is not found`

Cause: your Google API key/project does not expose that embedding model in the currently used API path.

Fix:

- Set `GOOGLE_EMBEDDING_MODEL` in `backend/.env` to a model available in your account, or leave it empty for auto-detection
- Re-run ingestion (`python ingest.py`) so vectors are generated with the same embedding model

### Voice endpoint returns 500

If the response detail mentions Groq model issues:

- Set `GROQ_TRANSCRIPTION_MODEL="whisper-large-v3-turbo"` in `backend/.env`

If the response detail mentions Edge TTS voice issues:

- Set `EDGE_TTS_VOICE` to another valid voice (for example `en-US-JennyNeural`)
