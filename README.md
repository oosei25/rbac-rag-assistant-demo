# RBAC-RAG Assistant

RBAC-RAG Assistant is a local-first retrieval-augmented generation application for internal knowledge bases. It combines role-based access control, document retrieval, source citations, and local LLM inference to demonstrate how an organization can expose private documents through a controlled assistant experience.

The stack uses Streamlit for the user interface, FastAPI for the backend API, Qdrant for vector search, sentence-transformers for local embeddings, LangGraph for the optional graph workflow, and Ollama for local chat completion.

<p align="center">
  <img src="docs/screenshot-home.png" alt="RBAC-RAG Assistant home screen" width="750">
</p>

## Highlights

- Role-scoped document retrieval across Engineering, Finance, HR, Marketing, General, and executive access levels.
- RAG and LangGraph execution modes with prompt-aligned structured citations.
- Local LLM runtime through Ollama, with no hosted LLM required by default.
- Qdrant-backed semantic search with optional Chroma support.
- API-backed document explorer with server-side authorization.
- Admin reindex action for Engineering and C-level users.
- Session-local usage analytics for demo and evaluation workflows.
- Evaluation fixtures for correctness and RBAC leakage checks.

## Architecture

```mermaid
flowchart LR
  subgraph UI["User Interface"]
    Streamlit["Streamlit app"]
  end

  subgraph API["FastAPI Backend"]
    Auth["Basic auth and RBAC"]
    Rag["RAG service"]
    Graph["LangGraph workflow"]
    Indexer["Indexer service"]
    Explorer["Authorized document API"]
  end

  subgraph Data["Knowledge Base"]
    Docs["resources/data/<department>"]
    Qdrant["Qdrant vector store"]
  end

  subgraph LLM["Local Model Runtime"]
    Ollama["Ollama - qwen2.5:3b-instruct"]
  end

  Streamlit -->|"HTTP + Basic Auth"| Auth
  Auth --> Rag
  Auth --> Graph
  Auth --> Explorer
  Docs --> Indexer
  Docs --> Explorer
  Indexer --> Qdrant
  Rag --> Qdrant
  Graph --> Qdrant
  Rag --> Ollama
  Graph --> Ollama
  Rag -->|"answer + structured citations"| Streamlit
  Graph -->|"answer + structured citations"| Streamlit
  Explorer -->|"authorized documents only"| Streamlit
```

## Repository Layout

```text
.
|-- app/
|   |-- graph/              # LangGraph RAG workflow
|   |-- services/           # Auth, indexing, retrieval, generation helpers
|   |-- schemas/            # Pydantic models
|   |-- utils/              # File reading and chunking utilities
|   |-- main.py             # FastAPI application
|   `-- policy.py           # Role-to-department access policy
|-- docs/                   # Screenshots
|-- evals/                  # Evaluation cases
|-- pages/                  # Streamlit multipage views
|-- resources/data/         # Sample department documents
|-- scripts/                # CLI utilities
|-- tests/                  # RBAC and evaluation tests
|-- Home.py                 # Streamlit entrypoint
|-- docker-compose.yml      # Qdrant, Ollama, API, and web services
`-- requirements*.txt       # Runtime and development dependencies
```

## Quick Start

### Prerequisites

- Docker Desktop
- Python 3.10 or newer for local development
- At least several GB of free disk space for model and embedding dependencies

### 1. Configure Environment

```bash
cp .env.example .env
```

The defaults use:

- `VECTOR_DB=qdrant`
- `EMBED_BACKEND=local`
- `ST_MODEL=sentence-transformers/all-MiniLM-L6-v2`
- `OLLAMA_MODEL=qwen2.5:3b-instruct`

### 2. Start Infrastructure

```bash
docker compose up -d qdrant ollama
docker compose exec ollama ollama pull qwen2.5:3b-instruct
```

### 3. Run the Full Docker Stack

```bash
docker compose up --build api web
```

Open the application at:

- Streamlit UI: <http://localhost:8501>
- FastAPI backend: <http://localhost:8000>
- API health check: <http://localhost:8000/healthz>

### 4. Local Development Mode

For faster iteration, run Qdrant and Ollama in Docker and run the API/UI from the local virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.dev.txt

docker compose up -d qdrant ollama
docker compose exec ollama ollama pull qwen2.5:3b-instruct
```

Start the API. `AUTO_INDEX=1` is the default and rebuilds the index before the API becomes ready:

```bash
PYTHONPATH=. \
VECTOR_DB=qdrant \
QDRANT_URL=http://localhost:6333 \
OLLAMA_HOST=http://localhost:11434 \
DATA_DIR=resources/data \
AUTO_INDEX=1 \
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Start the UI in a second terminal:

```bash
PYTHONPATH=. \
API_URL=http://127.0.0.1:8000 \
streamlit run Home.py --server.port=8501 --server.address=127.0.0.1
```

## Demo Users

The project ships with local demo users for role-based testing:

| Username | Password | Role |
| --- | --- | --- |
| `Peter` | `pete123` | Engineering |
| `Mariam` | `mariampass123` | Marketing |
| `Natasha` | `hrpass123` | HR |
| `Sam` | `financepass` | Finance |
| `Cathy` | `cathyceo` | C-level |
| `Emma` | `password` | Employee |

These credentials are for local demonstration only. Replace them with `BASIC_USERS_JSON` or a production identity provider before using this pattern beyond a demo environment.

## Application Views

### Chat

The main chat screen supports both RAG and Graph execution modes, displays generated answers, and shows the authorized excerpt behind each numbered citation. Each citation includes its prompt ID, stable document ID, safe relative path, title, department, section, retrieval score, and snippet.

<p align="center">
  <img src="docs/screenshot-answer.png" alt="RAG answer with citations" width="750">
</p>

### Document Explorer

The document explorer calls authenticated FastAPI endpoints. The API filters documents by role before reading or returning their content; the Streamlit container does not contain the knowledge-base files.

<p align="center">
  <img src="docs/screenshot-docs.png" alt="Document explorer" width="750">
</p>

### Admin Tools

Engineering and C-level users can trigger reindexing from the UI. The page also shows API health and runtime model/vector-store settings.

### Usage Analytics

The analytics page tracks session-local request counts, engine usage, request status, latency, answer length, and citation count. It is intended for local demos and lightweight validation, not durable production reporting.

## API Endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/healthz` | Service and index-readiness status |
| `GET` | `/version` | Runtime model and vector database settings |
| `GET` | `/login` | Validate Basic Auth credentials |
| `GET` | `/documents` | List documents authorized for the signed-in role |
| `GET` | `/documents/{document_id}` | Read one authorized document |
| `POST` | `/chat/rag` | Single-turn RAG answer generation |
| `POST` | `/chat/graph` | LangGraph answer generation with thread memory |
| `POST` | `/admin/reindex` | Rebuild vector index from `resources/data` |

Example RAG request:

```bash
curl -u Peter:pete123 \
  -H "Content-Type: application/json" \
  -d '{"message":"What are the key components of the engineering architecture?"}' \
  http://localhost:8000/chat/rag
```

## Data Model and Access Control

Documents are organized by department:

```text
resources/data/
|-- engineering/
|-- finance/
|-- general/
|-- hr/
`-- marketing/
```

The access policy is defined in `app/policy.py`. Each known role maps to an allowed set of departments; unknown roles map to no access. Retrieval applies the role filter before content can reach the model and rechecks returned metadata defensively. Intent inference can narrow retrieval to an allowed department, but it never grants access or rejects a request.

The retrieval order is:

```text
query -> embedding -> RBAC-filtered vector search -> vector threshold
      -> lexical threshold -> diversification -> optional reranking
      -> generation -> citation validation
```

Diversification permits up to three relevant chunks from one document so a multi-section answer does not lose necessary context.

## Configuration

Most runtime behavior is controlled through `.env`.

| Variable | Purpose |
| --- | --- |
| `VECTOR_DB` | Vector backend: `qdrant` or `chroma` |
| `QDRANT_URL` | Qdrant endpoint used by the API |
| `DATA_DIR` | Document root used during indexing |
| `EMBED_BACKEND` | Embedding provider: `local` or `openai` |
| `ST_MODEL` | Local sentence-transformer model |
| `OLLAMA_HOST` | Ollama server URL |
| `OLLAMA_MODEL` | Chat model used for answer generation |
| `AUTO_INDEX` | Rebuild the index on API startup; defaults to `1` |
| `INTENT_NARROWING` | Let inferred intent narrow retrieval within authorized departments |
| `QDRANT_SCORE_MIN` | Minimum Qdrant similarity score |
| `CHROMA_DIST_MAX` | Maximum Chroma cosine distance |
| `LEXICAL_MIN` | Minimum query-token overlap after vector retrieval |
| `MAX_CHUNKS_PER_DOCUMENT` | Diversification cap, constrained to `2`–`3` |
| `RERANK_CE` | Enable optional cross-encoder reranking |
| `PASSAGE_SELECTION` | Enable optional LLM passage selection |

See `.env.example` for the full set of supported settings.

## Testing

Install development dependencies:

```bash
pip install -r requirements.dev.txt
```

Run the test suite:

```bash
pytest -q
```

The core suite uses deterministic model and index fakes. It covers authorization, document access, citation ordering and consistency, relevance thresholds, no unauthorized model context, intent behavior, diversification, denial consistency, Basic Auth, reindex permissions, and empty/populated health states without requiring Ollama, an embedding download, or a prebuilt index.

## Operational Notes

- The first local run may download the embedding model from Hugging Face and the chat model from Ollama.
- Docker Compose persists Qdrant, Ollama, and Hugging Face cache data in named volumes.
- The checked-in documents are synthetic sample data for demonstration and evaluation.
- Demo authentication uses plaintext configured credentials with constant-time comparisons. It should still be replaced by a production identity provider outside a local demo.
- For production use, add durable auth, encrypted secrets management, persistent analytics, monitoring, and document-level ACLs.

## Roadmap

- Document-level access control.
- Durable analytics and audit logging.
- CI workflow for tests and linting.
- Optional hosted LLM and embedding providers.
- Additional evaluation coverage for retrieval quality and access-boundary behavior.
