# Setup and Build Guide

## 1) Prerequisites

- Python 3.11+
- Node.js 18+ and npm
- API keys:
  - Azure OpenAI API key
  - Pinecone API key
  - Groq API key (voice transcription)
- A Pinecone serverless index configured as:
  - Name: `chatbot-rag` (or your custom value)
  - Dimension: `1536` (for `text-embedding-3-small`)
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
AZURE_OPENAI_ENDPOINT="https://thomaschat.cognitiveservices.azure.com/"
AZURE_OPENAI_API_KEY="your_azure_openai_api_key_here"
AZURE_OPENAI_API_VERSION="2024-12-01-preview"
AZURE_OPENAI_CHAT_DEPLOYMENT="gpt-4o"
AZURE_OPENAI_EMBEDDING_DEPLOYMENT="text-embedding-3-small"
AZURE_OPENAI_TEMPERATURE="0.1"
LLM_MAX_OUTPUT_TOKENS="1200"
AZURE_OPENAI_MAX_COMPLETION_TOKENS="16384"
PINECONE_API_KEY="your_pinecone_api_key_here"
GROQ_API_KEY="your_groq_api_key_here"
GROQ_TRANSCRIPTION_MODEL="whisper-large-v3"
EDGE_TTS_VOICE="en-US-AriaNeural"
PINECONE_INDEX_NAME="chatbot-rag"
AUTO_CREATE_PINECONE_INDEX="false"
PINECONE_CLOUD="aws"
PINECONE_REGION="us-east-1"
PINECONE_DIMENSION="1536"
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

### Error: dependency import/version issues

Cause: your virtual environment is stale after dependency changes.

Fix:

```powershell
cd backend
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Error: `pinecone ... NotFoundException: Resource chatbot-rag not found`

Cause: the Pinecone index in `PINECONE_INDEX_NAME` does not exist in your account/project.

Fix:

- Create the index in Pinecone with dimension matching your embedding deployment (for example `1536` for `text-embedding-3-small`), or
- Update `PINECONE_INDEX_NAME` in `backend/.env` to an existing index name.

Alternative auto-fix:

- Set `AUTO_CREATE_PINECONE_INDEX="true"`
- Ensure `PINECONE_CLOUD` and `PINECONE_REGION` are correct for your Pinecone environment
- Restart backend

### Error: Azure OpenAI deployment not found (404/400)

Cause: deployment name in env does not match your Azure OpenAI deployment name.

Fix:

- Verify `AZURE_OPENAI_CHAT_DEPLOYMENT` and `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`
- Ensure the deployments exist in your Azure OpenAI resource
- Re-run ingestion (`python ingest.py`) so vectors are generated with the configured embedding deployment

### Voice endpoint returns 500

If transcription fails, verify `GROQ_API_KEY` and `GROQ_TRANSCRIPTION_MODEL`.

If the response detail mentions Edge TTS voice issues:

- Set `EDGE_TTS_VOICE` to another valid voice (for example `en-US-JennyNeural`)
