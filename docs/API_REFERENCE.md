# API Reference

Base URL examples:

- Local: `http://localhost:8000`
- Production: `https://your-backend-domain`

## GET /health

Returns service health.

### Response

```json
{
  "status": "ok"
}
```

## POST /api/chat/text

Text-in, text-out endpoint.

### Request

- Content type: `multipart/form-data`
- Field: `query` (string)
- Field: `session_id` (string, optional but recommended)
- Field: `lead_email` (string, optional)
- Field: `lead_name` (string, optional)

### Example (curl)

```bash
curl -X POST "http://localhost:8000/api/chat/text" \
  -F "query=What does your service do?" \
  -F "session_id=abc123" \
  -F "lead_email=user@example.com" \
  -F "lead_name=John Doe"
```

### Response

```json
{
  "reply": "...",
  "session_id": "abc123",
  "lead": {
    "email": "user@example.com",
    "name": "John Doe"
  }
}
```

## POST /api/chat/voice

Voice-in, audio-out endpoint.

### Request

- Content type: `multipart/form-data`
- Field: `audio` (file, for example `.webm`)
- Optional headers:
  - `X-Session-Id`
  - `X-Lead-Email`
  - `X-Lead-Name`

### Example (curl)

```bash
curl -X POST "http://localhost:8000/api/chat/voice" \
  -H "X-Session-Id: abc123" \
  -H "X-Lead-Email: user@example.com" \
  -H "X-Lead-Name: John Doe" \
  -F "audio=@recording.webm" \
  --output reply.mp3
```

### Response

- Body: MP3 audio stream (`audio/mpeg`)
- Headers:
  - `X-User-Query`: transcribed user text
  - `X-Bot-Reply`: generated assistant response text

## POST /api/ingest/upload

Uploads and ingests a PDF/text-like file into Pinecone.

### Request

- Content type: `multipart/form-data`
- Field: `file` (file)
- Optional header: `X-Ingest-Key` (required only if `INGEST_API_KEY` is configured)

### Example (curl)

```bash
curl -X POST "http://localhost:8000/api/ingest/upload" \
  -H "X-Ingest-Key: your_ingest_key_if_set" \
  -F "file=@./knowledge.pdf"
```

### Response

```json
{
  "status": "success",
  "message": "File ingested successfully",
  "source": "knowledge.pdf",
  "chunks": 42,
  "index": "chatbot-rag"
}
```

## SharePoint List Lead Sync

When `ENABLE_SHAREPOINT_SYNC=true`, the backend automatically upserts lead data after each chat turn into a SharePoint list via Microsoft Graph.

Required env vars:

- `SHAREPOINT_TENANT_ID`
- `SHAREPOINT_CLIENT_ID`
- `SHAREPOINT_CLIENT_SECRET`
- `SHAREPOINT_SITE_ID`
- `SHAREPOINT_LIST_ID`

Optional behavior toggle:

- `SHAREPOINT_ALWAYS_INSERT` (default `true`; when `true`, always creates a new list item and skips email lookup)

Field mapping (internal names):

- `SHAREPOINT_FIELD_TITLE` (default `Title`)
- `SHAREPOINT_FIELD_NAME` (default `Name`)
- `SHAREPOINT_FIELD_EMAIL` (default `email`)
- `SHAREPOINT_FIELD_CONVERSATION` (default `Conversation`)

Behavior:

- Reuses existing item when the same email already exists.
- Updates full transcript in the conversation field on every turn.
