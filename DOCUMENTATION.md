# Ultron — AI Assistant

A self-hosted, ChatGPT-style AI assistant built with Flask + Vanilla JavaScript. Runs entirely on localhost as a single-user application demonstrating RAG, long-term memory, multi-provider AI fallback, vision, document analysis, and tool use.

**Recent Updates (v2.1):**
- **Source isolation**: RAG now strictly uses only the current message's attached documents
- **Namespace isolation**: Knowledge base documents separated from chat documents via namespace tagging
- **Upload progress bar**: Real-time progress indicator for file uploads
- **Send blocking**: Prevents message send while uploads are in progress
- **Image routing**: Automatically routes images to vision-capable providers (Gemini/Cyfuture)
- **Enhanced PDF extraction**: PyMuPDF → pypdf → OCR (tesseract) cascade
- **Web search fix**: Fixed max_results type conversion bug
- **Tool Manager**: New unified routing layer for multi-tool chaining and model selection

---

## Table of Contents

- [Overview](#overview)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [File Structure](#file-structure)
- [AI Providers](#ai-providers)
- [Tool System](#tool-system)
- [RAG Pipeline](#rag-pipeline)
- [Memory System](#memory-system)
- [Streaming (SSE)](#streaming-sse)
- [Database Schema](#database-schema)
- [Frontend](#frontend)
- [Settings](#settings)
- [Voice Features](#voice-features)
- [Chat Management](#chat-management)
- [Security](#security)
- [Deployment](#deployment)

---

## Overview

Ultron is a full-stack AI assistant that connects to three LLM providers (Cyfuture, Groq, Gemini) with automatic fallback. It supports:

- **Multi-provider AI** with automatic fallback on failure
- **RAG** — upload PDFs/DOCX/TXT and ask questions about them
- **Long-term memory** — remembers user facts across sessions
- **Tools** — time, weather, calendar, web search, image generation
- **Vision** — analyze images via Cyfuture/Gemini vision models
- **Voice** — text-to-speech (Edge neural TTS) and speech-to-text (Web Speech API / Groq Whisper)
- **Streaming** — real-time token-by-token responses via Server-Sent Events

**Entry Points:**
```
python app.py          # Flask dev server (http://127.0.0.1:5000)
python serve.py        # Waitress production server (8 threads)
```

---

## Tech Stack

### Backend (Python 3.10+)

| Library | Purpose |
|---------|---------|
| Flask | Web framework, routes, SSE streaming |
| Waitress | Production WSGI server |
| Groq SDK | Groq AI provider |
| google-generativeai | Gemini AI provider |
| requests | HTTP calls (Cyfuture/Groq/Gemini REST streaming, weather, search) |
| sentence-transformers | Embedding model (all-MiniLM-L6-v2) |
| scikit-learn | TF-IDF fallback embedder |
| qdrant-client | Vector database (embedded LocalMode) |
| ddgs | Keyless web search (DuckDuckGo, renamed from duckduckgo-search) |
| PyMuPDF / pypdf | PDF text extraction |
| python-docx | DOCX text extraction |
| Pillow | Image processing |
| pytesseract | OCR for scanned PDFs (requires tesseract-ocr system package) |
| edge-tts | Microsoft neural TTS (free) |
| python-dotenv | Environment variable loading |

### Frontend (Vanilla JS — zero frameworks)

| Library | Purpose |
|---------|---------|
| marked.js | Markdown to HTML rendering |
| highlight.js | Code syntax highlighting |
| DOMPurify | HTML sanitization (XSS prevention) |

### External Services (Keyless / Free)

| Service | Purpose |
|---------|---------|
| Hugging Face Inference API | Image generation (FLUX.1-dev / FLUX.1-schnell) |
| Open-Meteo | Weather + geocoding |
| DuckDuckGo | Web search |
| Wikipedia API | Knowledge fallback |
| ip-api.com | IP geolocation |

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Browser (SPA)                      │
│  Vanilla JS · marked.js · highlight.js · DOMPurify  │
│  HTTP + SSE (Server-Sent Events)                    │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│               Flask App (app.py)                     │
│  Routes · SSE Streaming · Background Services       │
├──────────┬───────────┬──────────┬───────────────────┤
│ database │  models/  │  tools/  │      rag/         │
│  db.py   │providers.py│ tool_manager│  core.py      │
│ (SQLite) │ (3 AI     │ (8 tools)│  (Qdrant/numpy)  │
│          │  providers)│          │                  │
├──────────┴───────────┴──────────┴───────────────────┤
│  memory/manager.py  │  utils/security.py, ocr.py   │
└─────────────────────────────────────────────────────┘
```

### Data Flow (Single Chat Message)

1. User types message + optional file attachments
2. Frontend POSTs to `/api/chat` with `{message, chat_id, files}`
3. Server creates user message in SQLite, auto-names chat if first message
4. Auto-learn facts from user text (if enabled) via `memory.extract_facts()`
5. Builds system prompt = PERSONA + long-term memory + RAG context + OCR text
6. Detects tool intent (time/weather/calendar/web/imagegen) via regex
7. If tool detected: executes tool, feeds result to LLM (or streams directly)
8. Streams tokens via SSE to browser through `providers.stream_with_fallback()`
9. Server saves assistant message + sources to SQLite
10. Browser renders markdown in real-time with syntax highlighting

---

## File Structure

```
Ultron/
├── app.py                    # Main Flask app (992 lines)
├── serve.py                  # Waitress production entry (42 lines)
├── requirements.txt          # Python dependencies
├── .env                      # API keys (gitignored)
│
├── database/
│   ├── db.py                 # SQLite schema + CRUD (242 lines)
│   └── ultron.db             # Runtime database (auto-created)
│
├── models/
│   └── providers.py          # AI provider classes + fallback (335 lines)
│
├── tools/
│   ├── tool_manager.py       # Intent detection + dispatch + model routing
│   ├── time_tool.py          # Current time with timezone
│   ├── weather.py            # Weather via Open-Meteo
│   ├── calendar_tool.py      # Calendar grids + date math
│   ├── websearch.py          # Web search + scraping
│   ├── imagegen.py           # Image generation
│   └── geo.py                # Geocoding + IP geolocation
│
├── rag/
│   └── core.py               # Extraction + chunking + embeddings + vector store
│
├── memory/
│   └── manager.py            # Long-term memory CRUD (131 lines)
│
├── utils/
│   ├── security.py           # Upload validation + hashing (36 lines)
│   ├── ocr.py                # Tesseract OCR wrapper (32 lines)
│   └── logger.py             # Rotating file + console logging (58 lines)
│
├── templates/
│   └── index.html            # Single-page HTML (393 lines)
│
├── static/
│   ├── css/style.css         # Glassmorphism UI (715 lines)
│   ├── js/app.js             # Frontend controller (1864 lines)
│   └── images/ultron-icon.png
│
├── scripts/                  # Windows deployment scripts
│   ├── backend.ps1           # PowerShell controller
│   ├── start_backend.bat     # Start launcher
│   ├── stop_backend.bat      # Stop launcher
│   └── ...                   # Auto-start service scripts
│
├── data/
│   ├── qdrant/               # Qdrant embedded storage
│   └── kb_inbox/             # Auto-ingest drop zone
│
├── uploads/                  # User-uploaded files
└── logs/                     # Rotating log files
```

---

## AI Providers

### Supported Providers

| Provider | Default Model | Vision Support | API Key Setting |
|----------|---------------|----------------|-----------------|
| Groq | llama-3.1-8b-instant (vision: llama-3.2-90b-vision-instruct) | Yes | `groq_key` / `GROQ_API_KEY` |
| Gemini | gemini-1.5-flash | Yes | `gemini_key` / `GEMINI_API_KEY` |

### Fallback Mechanism

When `provider` is set to `"auto"`:

1. Try providers in order: Groq → Gemini
2. If images are attached, use a vision-capable provider (Gemini preferred)
3. On failure: log error, yield `fallback` event, try next provider
4. If all fail: yield `error` event

```
User → Groq (fail) → Gemini (success) → Stream response
              ↓
        fallback event
```

### Message Format Conversion

Each provider has its own message format:
- **Chat-completions style (Groq)**: `{"role": "user", "content": [{"type": "text", "text": "..."}]}`
- **Gemini**: `{"role": "user", "parts": [{"text": "..."}, {"inline_data": {...}}]}`

Images are converted to base64 `data:` URIs for vision models.

---

## Tool System

### Tool Manager (New in v2.1)

The `ToolManager` class (`tools/tool_manager.py`) provides a unified routing layer:

- **Intent classification** — Pattern-based (extensible to LLM-based)
- **Multi-tool chaining** — Primary tool + up to 2 secondary tools
- **Smart model selection** — Per task type: fast/reasoning/coding/vision
- **Vision-aware routing** — Filters to vision-capable models when images present
- **Combined context** — Merges results from multiple tools for LLM

### Detection (Regex-Based)

Tools are detected by matching user text against regex patterns. No LLM call needed for routing.

| Priority | Tool | Example Phrases |
|----------|------|-----------------|
| 1 | Time | "what time is it", "time in London" |
| 2 | Weather | "weather", "temperature", "is it raining" |
| 3 | Calendar | "what day", "calendar", "days until Christmas" |
| 4 | Image Gen | "generate an image", "draw a picture" |
| 5 | Web Search | "search the web", "latest news", "who is" |

### Tool Types

**Direct Tools** (result streamed directly, no LLM involved):
- `time` — Current time with timezone resolution
- `weather` — Current weather via Open-Meteo
- `calendar` — Date math, weekday lookup, month grids
- `imagegen` — Image generation via Hugging Face (FLUX.1-dev / FLUX.1-schnell)

**Context Tools** (result fed to LLM as extra context):
- `websearch` — DuckDuckGo search + page scraping

### Image Generation Flow

1. User: "generate an image of a sunset"
2. `imagegen.py` detects intent, extracts prompt
3. Primary: Calls Hugging Face Inference API (FLUX.1-dev via fal-ai)
4. Fallback: Calls Hugging Face Inference API (FLUX.1-schnell via together)
5. Saves PNG to `uploads/<uuid>.png`
6. Returns `{"image_url": "/uploads/<uuid>.png", "direct": "[Image generated: sunset]\n\n![sunset](/uploads/<uuid>.png)"}`
7. SSE streams image URL to frontend
8. Frontend creates `<img>` element and appends to chat

### Web Search Flow

1. User: "what's the latest news about AI"
2. `websearch.py` queries DuckDuckGo (with multiple backend fallbacks)
3. Returns top snippets + URLs
4. Optionally fetches full page content (limited to 5000 chars)
5. Result fed to LLM as system context
6. LLM synthesizes answer with citations

---

## RAG Pipeline

### Document Processing

**Supported formats**: PDF, DOCX, TXT, MD, CSV

| Format | Extraction Method |
|--------|-------------------|
| PDF | PyMuPDF (primary, with "blocks" mode for complex layouts) → pypdf (fallback, `extraction_mode="layout"`) → OCR via tesseract (scanned PDFs) |
| DOCX | python-docx (paragraphs + tables) |
| TXT/MD | stdlib open() with UTF-8/latin-1 |
| CSV | csv.reader → Markdown table rendering |

**Chunking**:
- Chunk size: 1200 characters
- Overlap: 200 characters
- Prefers cutting at newlines, then spaces
- PDFs: chunks are page-tagged (never span pages)

**Deduplication**: Files are SHA-256 hashed. Duplicate uploads reuse existing document ID.

### Embedding (Two-Tier)

| Tier | Model | Dimensions | When Used |
|------|-------|------------|-----------|
| 1 (preferred) | all-MiniLM-L6-v2 (sentence-transformers) | 384 | Default |
| 2 (fallback) | HashingVectorizer (scikit-learn TF-IDF) | 512 | If torch unavailable |

### Vector Store (Two-Backend)

| Backend | Storage | When Used |
|---------|---------|-----------|
| Qdrant (primary) | `data/qdrant/` (embedded LocalMode) | Default |
| Local numpy | In-memory + SQLite `embeddings` table | If Qdrant unavailable |

**Namespace Support**: Both backends support namespaces for multi-tenant isolation:
- **Chat documents**: Empty namespace (`""`)
- **Knowledge base documents**: Named namespace (e.g., `"healthcare"`)
- Searches filter by namespace to prevent cross-contamination

**Key operations**:
- `add_document()` — Embed chunks, store with UUID5 point IDs, optional namespace
- `search()` — Cosine similarity, min score 0.2, top-k=4, optional namespace filter
- `delete_document()` — Remove all chunks for a document

### RAG in Chat (Source Isolation)

1. Collect document IDs **only from current message's attachments** (not chat history)
2. Search vector store with user's query (k=4), filtered to those doc_ids
3. Also search global knowledge base namespace (if configured)
4. Deduplicate, keep top 4 by score
5. Format context with source labels (filename + page numbers)
5. Inject into system prompt
6. **Sources shown to user**: Only the current message's documents (not web search or knowledge base)

### Source Separation

- **RAG sources** → shown in UI as source chips (current message's docs only)
- **Web search sources** → used for LLM context only, NOT shown as source chips
- **Knowledge base sources** → used for LLM context only, NOT shown as source chips

---

## Memory System

### Long-Term Memory

Stored in SQLite `memory` table with `content` and `category` fields.

**Auto-extraction** (when `memory_auto` enabled):

| Pattern | Category | Example |
|---------|----------|---------|
| `my name is X` | name | "My name is Alex" |
| `I work at X` | work | "I work at Google" |
| `I am learning X` | learning | "I am learning Python" |
| `I like/love X` | preference | "I love coffee" |
| `I live in X` | location | "I live in Mumbai" |
| `remember that X` | custom | "Remember that my birthday is March 15" |

Facts are deduplicated by lowercase content, limited to 80-200 chars.

**System prompt injection**: Up to 40 memories formatted as bullet points:
```
Long-term memory about the user:
- Name: Alex
- Works at: Google
- Learning: Python
```

### Short-Term Memory

All messages stored in `messages` table per chat. Full conversation history replayed into every LLM call.

---

## Streaming (SSE)

### Event Types

| Event | Data | Purpose |
|-------|------|---------|
| `token` | `{type: "token", content: "..."}` | Incremental text chunk |
| `meta` | `{type: "meta", provider: "...", model: "..."}` | Provider/model info |
| `fallback` | `{type: "fallback", from: "...", to: "..."}` | Provider switch |
| `tool` | `{type: "tool", message: "...", kind: "...", image_url: "..."}` | Tool status |
| `sources` | `{type: "sources", sources: [...]}` | RAG source documents |
| `error` | `{type: "error", message: "..."}` | Error message |
| `done` | `{type: "done"}` | Stream complete |

### Backend Flow

```
1. Create user message in DB
2. Build system prompt (persona + memory + RAG + OCR)
3. Detect + run tools
4. For direct tools → stream tool result as single token
5. For context tools → append context, stream from provider
6. Use tail buffer (120 chars) to avoid tiny chunks
7. On completion → save assistant message to DB
8. Yield sources + done events
```

### Frontend Flow

```
1. POST to /api/chat
2. Open fetch() with Accept: text/event-stream
3. Read response body as ReadableStream
4. Split on \n\n, parse each data: line as JSON
5. Process events:
   - token → accumulate text, re-render markdown
   - tool → update badge, insert image
   - sources → render source chips
6. On finish → reload chat from server to sync
```

---

## Database Schema

**Engine**: SQLite with WAL journal mode, thread-safe via `threading.Lock()`

### Tables

#### `chats`
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| title | TEXT | Chat title (auto-named from first message) |
| created_at | TEXT | Creation timestamp |
| updated_at | TEXT | Last activity timestamp |

#### `messages`
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| chat_id | INTEGER FK | References chats.id |
| role | TEXT | 'user' or 'assistant' |
| content | TEXT | Message text (image markdown stripped for assistant) |
| files | TEXT | JSON array of attachment metadata |
| sources | TEXT | JSON array of RAG source documents |
| model | TEXT | Provider/model that answered |
| status | TEXT | 'complete' or 'stopped' |
| created_at | TEXT | Creation timestamp |

#### `memory`
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| content | TEXT | Fact text |
| category | TEXT | 'name', 'work', 'preference', etc. |
| created_at | TEXT | Creation timestamp |

#### `documents`
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| filename | TEXT | Original filename |
| kind | TEXT | 'doc' or 'image' |
| path | TEXT | Stored path in uploads/ |
| doc_hash | TEXT | SHA-256 for dedup |
| filetype | TEXT | .pdf, .docx, etc. |
| chunks | INTEGER | Number of chunks created |

#### `settings`
| Column | Type | Description |
|--------|------|-------------|
| key | TEXT PK | Setting name |
| value | TEXT | Setting value (all stored as text) |

### Default Settings

| Key | Default | Description |
|-----|---------|-------------|
| provider | "auto" | AI provider selection (auto/groq/gemini) |
| temperature | "0.7" | Generation temperature (0-2) |
| max_tokens | "1024" | Max tokens per response (64-8192) |
| rag_enabled | "true" | Enable RAG (document retrieval) |
| memory_enabled | "true" | Enable long-term memory |
| memory_auto | "true" | Auto-learn facts from conversation |
| tools_enabled | "true" | Enable tool detection (time/weather/calendar/web/imagegen) |
| default_city | "" | Default city for weather |
| kb_namespace | "healthcare" | Knowledge base namespace (empty = disabled) |
| web_results | "4" | Number of web search results per query (1-6) |

---

## Frontend

### Design System

- **Theme**: Dark/Light via CSS custom properties (`data-theme` attribute)
- **Style**: Glassmorphism with `backdrop-filter: blur()`, translucent panels
- **Color palette**: Red/orange accent (`#ff2d2d`, `#c1121f`, `#ff6a00`)
- **Layout**: Flexbox sidebar + main area, max-width 820px for messages
- **Responsive**: Breakpoints at 1024px (tablet) and 640px (mobile)

### Upload Progress Bar (v2.1)

When files are uploaded via the attach button:
- **Pending state** — Attachment chip appears with 0% progress
- **Progress bar** — Green bar at bottom of chip, updates in real-time (XHR upload events)
- **Percentage text** — Shows 0-100% next to filename
- **Complete** — Progress bar fills, chip opacity normalizes
- **Send blocking** — Send button disabled while any upload is pending (shows toast error)

### State Management

Simple global `state` object (no framework):

```javascript
const state = {
  chatId,           // Current chat ID
  providers,        // Available providers
  settings,         // All settings
  attachments,      // Pending file uploads
  messages,         // Rendered messages
  streaming,        // Is currently streaming
  abortController,  // For cancelling streams
  currentProvider,  // Active provider
  theme,            // "dark" or "light"
  voiceSend,        // Voice mode active
};
```

### Markdown Rendering Pipeline

1. `marked.parse(text)` with `breaks: true`
2. `DOMPurify.sanitize(raw)` for XSS prevention
3. `hljs.highlightElement(block)` on all `<pre><code>` blocks

### Key UI Components

- **Sidebar**: Chat list, search, settings/memory/KB buttons
- **Main area**: Messages, composer with attachment/voice buttons
- **Modals**: Settings (5 tabs), Memory, Knowledge Base, Confirm
- **Voice Mode**: Full-screen overlay with animated orb + waveform

---

## Settings

### Tab 1: Model
- Default provider dropdown (Auto/Cyfuture/Groq/Gemini)
- Model name text input for each provider
- Cyfuture base URL input

### Tab 2: Generation
- Temperature slider (0 to 2, default 0.7)
- Max tokens input (64 to 8192, default 1024)

### Tab 3: Features
- Toggle: Theme, RAG, Memory, Auto-learn, Tools
- Text: Default city for weather
- Number: Web results per search (1-6)
- Text: Knowledge base namespace
- Select: TTS voice

### Tab 4: Data
- Export chats (downloads `ultron-backup.json`)
- Import chats (uploads JSON file)

---

## Voice Features

### Text-to-Speech (TTS)

**Two engines:**

| Engine | Quality | Notes |
|--------|---------|-------|
| Edge Neural TTS | High (neural) | Free, no API key, deep male voices |
| Web Speech API | Medium (robotic) | Browser fallback |

Edge TTS voices: Christopher, Guy, Eric, Roger, Steffan, Ryan, Thomas

### Speech-to-Text (STT)

**Two approaches:**

| Method | Notes |
|--------|-------|
| Web Speech API | Preferred, browser-native |
| MediaRecorder + Groq Whisper | Fallback, records audio → uploads → Groq API |

### Voice Mode (Full-Screen)

- Continuous conversation with automatic turn-taking
- SpeechRecognition in continuous mode
- Waveform visualization via Canvas + Web Audio API
- Mute/interrupt controls
- Automatic listen → think → speak → listen cycle

---

## Chat Management

### API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/chats` | List all chats |
| POST | `/api/chats` | Create new chat |
| GET | `/api/chats/<id>` | Get chat + messages |
| PATCH | `/api/chats/<id>` | Rename chat |
| DELETE | `/api/chats/<id>` | Delete chat + messages + uploaded files |
| GET | `/api/search?q=` | Search chats |
| POST | `/api/chat` | Send message (SSE streaming) |
| POST | `/api/upload` | Upload file |
| GET | `/api/settings` | Get all settings |
| PUT | `/api/settings` | Update settings |
| GET | `/api/memory` | List memories |
| POST | `/api/memory` | Add memory |
| PUT | `/api/memory/<id>` | Update memory |
| DELETE | `/api/memory/<id>` | Delete memory |
| GET | `/api/tts?text=` | Text-to-speech audio |
| POST | `/api/voice/transcribe` | Speech-to-text |
| GET | `/api/export` | Export all data |
| POST | `/api/import` | Import data |

### Chat Lifecycle

1. **Creation**: First message creates chat, auto-names from first 6 words
2. **Loading**: Sidebar shows title + last message preview
3. **Renaming**: Prompt dialog → PATCH request
4. **Deletion**: Confirmation modal → DELETE request → removes chat, messages, and all associated files from `uploads/`
5. **Search**: Debounced LIKE search across titles + message content
6. **Regenerate**: Deletes messages after last user message, re-sends

---

## Security

### What Exists
- Upload validation (size limit 200MB, extension whitelist, magic bytes)
- Filename sanitization (strips special characters)
- Path traversal prevention (`os.path.basename()` for file serving)
- HTML sanitization via DOMPurify (XSS prevention)
- API keys in `.env` file (gitignored)
- PBKDF2 password hashing utility (available but not actively used)

### What Does NOT Exist
- No authentication/login system
- No rate limiting
- No CORS restrictions (localhost only)
- No CSRF protection
- No API key encryption at rest

> Note: This is acceptable for a single-user localhost application.

---

## Deployment

### Development
```bash
pip install -r requirements.txt
python app.py
# → http://127.0.0.1:5000
```

### Production
```bash
python serve.py
# Waitress: 8 threads, 120s timeout
```

### Windows Auto-Start
```bash
scripts/install_backend_service.bat    # Installs to Windows Startup
scripts/uninstall_backend_service.bat  # Removes from Startup
scripts/start_backend.bat              # Manual start
scripts/stop_backend.bat               # Manual stop
scripts/backend_status.bat             # Check status
```

### Environment Variables (.env)
```env
GROQ_API_KEY=gsk_...
GEMINI_API_KEY=AIza...
```

---

## Key Design Decisions

1. **No frontend framework** — Vanilla JS for zero build step complexity
2. **No auth** — Single-user localhost, simplicity over security
3. **Lazy imports** — AI SDKs imported inside functions so app starts even if packages missing
4. **Two-tier everything** — Sentence-transformers → TF-IDF, Qdrant → numpy, Edge TTS → Web Speech, FLUX.1-dev → FLUX.1-schnell
5. **Regex-based tool detection** — Simple, fast, no LLM call needed for routing
6. **Image markdown in DB** — Saved for display, stripped before sending to LLM
7. **Single worker process** — `use_reloader=False` keeps in-memory caches consistent
8. **Windows-native deployment** — PowerShell + VBScript for auto-start without admin
9. **File cleanup on chat deletion** — Deletes physical files from `uploads/`, document records, and vector store chunks
10. **Source isolation** — RAG strictly uses only current message's attachments (not chat history or knowledge base)
11. **Namespace isolation** — Knowledge base documents tagged with namespace, filtered during search
12. **Image routing** — Automatically routes images to Cyfuture/Gemini (vision-capable)

---

## Logging

Three log handlers:
- **Console**: INFO level
- **File**: DEBUG level, `logs/ultron.log`, 5MB rotation, 3 backups
- **Error file**: ERROR level, `logs/ultron.err.log`, 2MB rotation, 2 backups

---

---

## Changelog

### v2.1 (2026-08-19)
- **Source isolation**: RAG now strictly uses only current message's attached documents
- **Namespace isolation**: Knowledge base documents tagged with namespace, filtered during search
- **Upload progress bar**: Real-time progress indicator for file uploads
- **Send blocking**: Prevents message send while uploads are in progress
- **Image routing**: Automatically routes images to vision-capable providers (Gemini/Cyfuture)
- **Enhanced PDF extraction**: PyMuPDF → pypdf → OCR (tesseract) cascade
- **Web search fix**: Fixed max_results type conversion bug
- **Tool Manager**: New unified routing layer for multi-tool chaining and model selection
- **Local vector store namespace support**: Added namespace column and filtering to SQLite backend

### v2.0
- Initial documentation

*Generated documentation for Ultron AI Assistant.*
