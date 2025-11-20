# RBAC-RAG Assistant

> A retrieval-augmented assistant with **role-based access control (RBAC)** for internal knowledge bases.  
> Streamlit UI · FastAPI backend · Qdrant vector DB · Ollama-hosted LLM

<p align="center">
  <img src="docs/screenshot-home.png" alt="Ask the Knowledge Base screen" width="750">
</p>

---

## 1. What is this?

RBAC-RAG Assistant is a small end-to-end demo of how to build a **secure internal chatbot**:

- Answers questions strictly from your documents (no internet access).
- Enforces **department-level permissions** (e.g., Engineering vs HR vs Marketing).
- Supports both:
  - **RAG** – single-shot retrieval + generation.
  - **Graph (LangGraph)** – multi-step pipeline with conversation memory.

### Design goals

This project is intentionally:

- **Realistic enough for learners**  
  It shows a complete RBAC-aware RAG stack: Streamlit UI, FastAPI backend, vector DB (Qdrant/Chroma), LLM via Ollama, indexing pipeline, and role-based access controls + evals.

- **Small enough to run on a laptop**  
  Everything is containerized and tuned so it can run on a single developer machine (or a tiny VM) with Docker, without requiring paid cloud services or massive GPUs.

---

## 2. Screenshots

**Login + Role selection**

<p align="center">
  <img src="docs/screenshot-login.png" alt="Login screen with quick-pick demo users and roles" width="750">
</p>

**RAG Answer view**

<p align="center">
  <img src="docs/screenshot-answer.png" alt="Answer with citations and engine selector" width="750">
</p>

**Graph (LangGraph) Denial**

<p align="center">
  <img src="docs/screenshot-denial.png" alt="Usage analytics and admin reindex controls" width="750">
</p>

**Document Explorer**

<p align="center">
  <img src="docs/screenshot-docs.png" alt="Document explorer by department and file" width="750">
</p>

---

## 3. Features

- 🧑‍💼 **Role-based answering**
  - Users authenticate with a pre-configured role (Engineering, Finance, HR, Marketing, General).
  - The assistant only retrieves from documents they are allowed to see.

- 📚 **Private retrieval-augmented generation (RAG)**
  - Markdown documents stored under `resources/data/<department>/`.
  - Vector search via **Qdrant** (or Chroma, depending on config).
  - Answers always include **source file paths**.

- 🧠 **Two execution engines**
  - **RAG** – fast, stateless, single-turn answers.
  - **Graph** – LangGraph-based multi-step pipeline with retrieval, reranking, passage selection, and validation.

- 🔍 **LLM-assisted passage selection (optional)**
  - Model picks the top-K passages from retrieved hits, reducing context size.

- 🛡️ **Safety & RBAC intent detection**
  - Simple heuristic to infer which department(s) a question is about.
  - Hard or soft deny depending on config (`RBAC_INTENT` / `RBAC_INTENT_SOFT`).

- 🗂️ **Document Explorer**
  - Browse indexed documents and their department tags.
  - Helpful for **answer verification** during demos.

- 📊 **Usage analytics (lightweight)**
  - Track basic stats (e.g., number of questions, engine used, answer length).

- 🛠 **Admin tools**
  - Trigger **re-indexing** from the UI after adding or editing documents.
  - View which vector backend and model are currently in use.

---

## 4. High-level system architecture (Streamlit + API + LLM + Vector DB)

```mermaid
flowchart LR
  subgraph User["User browser"]
    UI["Streamlit UI / RBAC chat app"]
  end

  subgraph Backend["Backend API (FastAPI)"]
    AUTH["Auth & RBAC (basic auth, roles)"]
    RAG["RAG service (RagService class)"]
    GRAPH["Graph pipeline (LangGraph)"]
  end

  subgraph DataPlane["Knowledge base"]
    VDB["Vector DB (Qdrant or Chroma)"]
    STORE["Documents on disk (resources/data)"]
  end

  subgraph LLMPlane["LLM runtime"]
    OLLAMA["Ollama host (qwen2.5:3b-instruct)"]
  end

  %% Edges
  UI -->|"HTTP JSON /chat/rag or /chat/graph"| AUTH
  AUTH -->|"role, username"| RAG
  AUTH --> GRAPH

  RAG -->|"embed query"| VDB
  GRAPH -->|"retrieve chunks"| VDB

  STORE -->|"offline reindex /admin/reindex"| VDB

  RAG -->|"build prompt + context snippets"| OLLAMA
  GRAPH -->|"chat with context"| OLLAMA
  OLLAMA -->|"answer + citations"| RAG
  OLLAMA --> GRAPH

  RAG -->|"answer + sources"| UI
  GRAPH -->|"answer + sources"| UI
```

## 4.1  Inside the RAG pipeline

```mermaid
flowchart TD
  Q["User question (Streamlit UI)"]
  API["FastAPI /chat/rag endpoint"]
  RAG["RagService.generate(query, role)"]

  Q --> API --> RAG

  subgraph Retrieval["Retrieval phase"]
    REWRITE["Rewrite query (rewrite_query)"]
    EMBED["Embed query (indexer.embed_one)"]
    RBACFILT["RBAC filter (allowed_departments + intent)"]
    SEARCH1["Vector search strict (TOP_K)"]
    SEARCH2["Vector search relaxed (wider K)"]
    DEDUP["Deduplicate + diversify (diversify_by_path)"]
    LEX["Lexical rerank (lexical_rerank)"]
    CE["Cross-encoder rerank (optional)"]
  end

  subgraph Generation["Generation & validation"]
    PROMPT["Build prompt (build_prompt)"]
    LLM["Ollama chat (qwen2.5:3b-instruct)"]
    SAN["Sanitize answer (sanitize_answer)"]
    FALLBACK["Keyword slice fallback (optional)"]
    VALID["Validate answer (validate_answer)"]
  end

  RAG --> REWRITE --> EMBED --> RBACFILT --> SEARCH1
  SEARCH1 --> SEARCH2 --> DEDUP --> LEX --> CE
  CE --> PROMPT --> LLM --> SAN --> VALID

  VALID -->|ok| RESP["Return answer + sources (back to API)"]
  VALID -->|fail & USE_KEYWORD_FALLBACK| FALLBACK --> RESP
  VALID -->|fail only| DENY["Deny message: 'I don’t have enough information\nin your accessible documents.'"]

  RESP --> API --> Q
  DENY --> API
```

---

## 5. Roadmap / Ideas

- ✅ Initial RAG + LangGraph pipeline with RBAC.
- ✅ Document explorer & admin reindex.
- ⏳ Better analytics (per-user, per-department).
- ⏳ More granular RBAC (document-level ACLs).
- ⏳ Optional hosted LLM fallback (OpenAI / HF Inference) with cost controls.
- ⏳ CI pipeline (tests + linting) via GitHub Actions.

---

## Demo

> If you’d like a live walkthrough or a short video demo, reach out:
> [goofosuosei@gmail.com](mailto:goofosuosei@gmail.com)