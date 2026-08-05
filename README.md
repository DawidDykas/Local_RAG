# Local RAG

A fully local Retrieval-Augmented Generation system that ingests documents from object storage, generates vector embeddings, and answers queries using local LLMs — all without cloud dependencies.

## Architecture

```
┌──────────┐    webhook     ┌───────────┐    task     ┌─────────────┐
│  MinIO   │ ──────────────▸│  FastAPI   │ ──────────▸│   Celery    │
│ (S3)    │                │  (API)     │            │  (Worker)   │
└────┬─────┘                └─────┬─────┘            └──────┬──────┘
     │                            │                          │
     │ file upload                │ POST /query              │ process file
     │                            ▼                          ▼
     │                    ┌──────────────┐         ┌─────────────────┐
     │                    │   Ollama     │         │  Document       │
     │                    │  (LLM +     │         │  Pipeline       │
     │                    │  Embeddings) │         │  (load→chunk→   │
     │                    └──────┬───────┘         │   embed→store)  │
     │                           │                 └────────┬────────┘
     │                           │                          │
     │                           ▼                          ▼
     │                    ┌──────────────┐         ┌─────────────────┐
     └───────────────────▸│   Qdrant     │◂────────│                 │
                          │  (Vectors)   │         └─────────────────┘
                          └──────────────┘
```

## Tech Stack

| Component    | Technology                          | Purpose                          |
| ------------ | ----------------------------------- | -------------------------------- |
| API          | FastAPI + Uvicorn                   | REST endpoints                   |
| Task Queue   | Celery + Redis                      | Async document processing        |
| Storage      | MinIO (S3-compatible)               | Document file storage            |
| LLM          | Ollama (gemma3:4b)                  | Query planning & answer generation |
| Embeddings   | Ollama (nomic-embed-text)           | Text vectorization (768d)        |
| Vector DB    | Qdrant                              | Vector storage & similarity search |
| RAG Framework| LlamaIndex                          | Indexing & retrieval pipeline    |
| CI/CD        | GitHub Actions + Semantic Release   | Automated versioning & releases  |

## Features

- **Automatic document ingestion** — upload a file to MinIO and it's automatically processed (text extraction → chunking → embedding → vector storage)
- **Multi-format support** — PDF, DOCX, CSV, HTML, TXT
- **Multi-query RAG** — rewrites user queries into 5 search plans for better retrieval coverage
- **Automatic cleanup** — deleting a file from MinIO removes its embeddings from Qdrant
- **Fully local** — no external API keys or cloud services required (runs on GPU)
- **Dockerized** — single `docker compose up` to run everything

## Prerequisites

- Docker & Docker Compose
- NVIDIA GPU + NVIDIA Container Toolkit (for Ollama GPU acceleration)
- ~8 GB RAM (for LLM + embeddings)

## Quick Start

1. Clone the repository:

```bash
git clone https://github.com/DawidDykas/Local_RAG.git
cd Local_RAG
```

2. Create a `.env` file in `backend/`:

```env
OLLAMA_SERVER_PORT=11434
REDIS_PORT=6379
FASTAPI_PORT=8000
MINIO_SERVER_PORT=9000
MINIO_CONSOLE_PORT=9001
QDRANT_PORT=6333
QDRANT_QRPC_PORT=6334
```

3. Start all services:

```bash
cd backend
docker compose up --build
```

4. Wait for the `ollama-init` container to finish pulling models, then access:

| Service      | URL                          |
| ------------ | ---------------------------- |
| FastAPI      | http://localhost:8000         |
| MinIO Console| http://localhost:9001         |
| Qdrant UI    | http://localhost:6333/dashboard |

## API Reference

### Query (RAG)

```
POST /ollama-event/query
```

**Request:**

```json
{
  "text": "What is the main topic of the uploaded documents?"
}
```

**Response:**

```json
{
  "text": "Based on the uploaded documents, the main topic is...",
  "tokens_used": null,
  "model_name": "ollama"
}
```

### MinIO Webhook (internal)

```
POST /minio-event
```

Receives MinIO webhook events for file create/delete operations. Handled automatically by the system.

## Project Structure

```
Local_RAG/
├── backend/
│   ├── app/
│   │   ├── api/                    # FastAPI routes & schemas
│   │   │   ├── main.py             # App entrypoint
│   │   │   ├── routers/            # Route handlers
│   │   │   └── schemas/            # Pydantic models
│   │   ├── core/                   # Config & logging
│   │   │   ├── global_config.py    # Settings (MinIO, Celery, FastAPI)
│   │   │   └── logger_config.py    # Rotating file + console logger
│   │   ├── infrastructure/         # External service clients
│   │   │   ├── celery/             # Celery app config
│   │   │   ├── minio/              # S3/Boto3 client
│   │   │   ├── ollama/             # Ollama HTTP client
│   │   │   └── vector_db/          # Qdrant client
│   │   ├── loaders/                # Document parsers
│   │   │   └── Load_module/        # PDF, DOCX, CSV, HTML, TXT loaders
│   │   ├── services/               # Business logic
│   │   │   ├── embeddingServices.py    # Document ingestion pipeline
│   │   │   ├── qdrantServices.py       # Vector CRUD operations
│   │   │   ├── ragServices.py          # RAG query pipeline
│   │   │   └── storage_events/         # MinIO event handlers
│   │   ├── workers/                # Celery task definitions
│   │   └── tests/                  # Unit & API tests
│   ├── docker-compose.yml          # Service orchestration
│   └── Dockerfile
├── .github/
│   └── workflows/                  # CI/CD pipelines
├── CHANGELOG.md
└── pyproject.toml                  # Root project config
```

## How It Works

### Document Ingestion Pipeline

1. User uploads a file to MinIO (via console or API)
2. MinIO sends a webhook event to FastAPI (`POST /minio-event`)
3. FastAPI dispatches a Celery task asynchronously
4. The Celery worker:
   - Downloads the file from MinIO
   - Detects file type and selects the appropriate loader
   - Extracts text content
   - Splits text into chunks (500 chars, 50 overlap)
   - Generates embeddings via Ollama (`nomic-embed-text`)
   - Stores vectors + metadata in Qdrant

### Query Pipeline

1. User sends a question to `POST /ollama-event/query`
2. The RAG planner (Ollama `gemma3:4b`) generates 5 search queries
3. Each query retrieves top-5 similar chunks from Qdrant
4. Results are deduplicated and ranked by score
5. Top-K context is passed to the LLM for final answer generation

## Development

### Install dependencies

```bash
cd backend
pip install uv
uv sync
```

### Run tests

```bash
cd backend
pytest app/tests/ -v
```

### Code formatting

```bash
black . --line-length 100
ruff check .
```

## Environment Variables

| Variable            | Default              | Description                  |
| ------------------- | -------------------- | ---------------------------- |
| `OLLAMA_SERVER_PORT`| `11434`              | Ollama API port              |
| `REDIS_PORT`        | `6379`               | Redis broker port            |
| `FASTAPI_PORT`      | `8000`               | FastAPI exposed port         |
| `MINIO_SERVER_PORT` | `9000`               | MinIO S3 API port            |
| `MINIO_CONSOLE_PORT`| `9001`               | MinIO web console port       |
| `QDRANT_PORT`       | `6333`               | Qdrant REST API port         |
| `QDRANT_QRPC_PORT`  | `6334`               | Qdrant gRPC port             |

## License

MIT
