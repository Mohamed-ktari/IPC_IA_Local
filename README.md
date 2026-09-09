# IPC IA Local

An on-premise generative AI platform built for **Ingénierie Pilotage Coordination (IPC)**, a French engineering firm specialized in the refurbishment of occupied social housing and asbestos removal project management.

The platform runs entirely on local infrastructure — no data ever leaves the company network — and assists non-technical staff (architects, engineers, cost estimators, assistants) with document analysis, technical report drafting, and general-purpose AI chat, all while enforcing GDPR compliance by design.

## Why local?

IPC handles highly sensitive documents on a daily basis: asbestos diagnostics, social housing occupant data, and financial bid information. Sending this data to a third-party cloud AI API would risk it being retained or reused for model training outside the company's control — a direct conflict with GDPR and with the trust IPC's clients place in it. This project was built to remove that trade-off entirely: full AI capability, zero external data exposure.

## What it does

- **Technical memo generation** — extracts the expected structure from a *règlement de consultation* (RC), lets the user validate/edit it as a Word document, then fills each section using retrieval-augmented generation (RAG) over past memos and the current project's programme.
- **Document Q&A** — multi-turn conversational search over any ingested document, with query rewriting for follow-up questions and per-answer relevance scoring.
- **General chat agent** — web search (self-hosted, via SearXNG), code generation, and free-form document analysis through a familiar chat UI (OpenWebUI).
- **Automatic section classification** — a lightweight cascade (dictionary match → embedding similarity → LLM fallback) decides, per memo section, whether to search past memos, the current project's programme, or skip retrieval entirely.
- **Hybrid document retrieval** — combines semantic search (embeddings via ChromaDB) with BM25 keyword search, tuned toward keyword matching since most source documents are table-heavy technical reports.

## Architecture

```
                    ┌─────────────┐      ┌─────────────┐
                    │  OpenWebUI  │      │   SearXNG   │
                    │  (chat UI)  │      │ (web search)│
                    └──────┬──────┘      └──────┬──────┘
                           │                    │
        ┌──────────────────┴────────────────────┘
        │
┌───────▼────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│    Backend      │──▶│   Ollama    │   │  ChromaDB   │   │ PostgreSQL  │
│   (FastAPI)     │   │ (LLM infer.)│   │  (vectors)  │   │ (users/audit)│
└───────┬─────────┘   └─────────────┘   └─────────────┘   └─────────────┘
        │
        ▼
┌─────────────┐
│    Redis     │
│ (queue/cache)│
└─────────────┘
```

All services are containerized (Docker Compose) and communicate over an isolated internal network; only the reverse proxy (Caddy) and the chat/search UIs are exposed.

**LLM stack:** `qwen2.5:14b-instruct-q4_K_M` for inference, `bge-m3` for embeddings, `docling` for document parsing — sized to fit ~13GB of VRAM on a single RTX 5080 (16GB), supporting 6 concurrent users.

## Tech stack

- **Backend:** FastAPI, Redis (RQ for async jobs), PostgreSQL, ChromaDB
- **LLM runtime:** Ollama
- **Retrieval:** hybrid search (dense embeddings + BM25), Docling for parsing (PDF/DOCX/Excel, OCR fallback for scans)
- **Frontend:** OpenWebUI (chat) + custom web app (document/project management, memo generation workflow)
- **Infra:** Docker Compose, Caddy (reverse proxy), NVIDIA GPU passthrough
- **Security:** full-disk encryption (LUKS) with remote unlock over Dropbear, SSH key-based access only, GET-only egress for external lookups, per-action audit logging

## Getting started

### Prerequisites

- Docker and Docker Compose
- An NVIDIA GPU with drivers and the NVIDIA Container Toolkit installed (required for Ollama's GPU acceleration)
- At least 16GB of VRAM to run the default model configuration (`qwen2.5:14b-instruct-q4_K_M`)
- Python 3 (only needed for local backend development outside Docker)

### Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/Mohamed-ktari/IPC_IA_Local.git
   cd IPC_IA_Local
   ```

2. **Configure environment variables**

   Copy the example file and fill in your own values (database credentials, SearXNG secret, etc.):

   ```bash
   cp .env.example .env
   ```

3. **Start all services**

   ```bash
   make docker-up
   ```

   This starts every service defined in `docker-compose.yml` (Ollama,
   ChromaDB, Redis, PostgreSQL, backend, OpenWebUI, SearXNG, Caddy).

4. **Pull the required models** (first run only)

   ```bash
   make ollama-pull
   ```

   Pulls `qwen2.5:14b-instruct-q4_K_M` (inference) and `bge-m3` (embeddings) inside the Ollama container.

5. **Check that everything is running**

   ```bash
   make docker-ps      # list running containers
   make health         # check backend + dependent services status
   ```

6. **Access the services**

   | Service | URL |
   |---|---|
   | Backend API (FastAPI docs) | `http://localhost:8000/docs` |
   | OpenWebUI (chat) | `http://localhost:3000` |
   | ChromaDB | `http://localhost:8001` |

   The reverse proxy (Caddy) exposes the unified entry point on port `80` for production-style access.

### Local backend development (outside Docker)

For faster iteration on the backend itself, without rebuilding the Docker image on every change:

```bash
make setup    # create a virtualenv and install Python dependencies
make dev      # run the backend with hot reload (http://localhost:8000)
```

Note that `ollama`, `chromadb`, `redis`, and `postgres` still need to be
running (via `make docker-up`) for the backend to work — only the
`backend` service itself is bypassed here.

### Background jobs

Memo generation runs asynchronously through an RQ worker (see
[Génération du contenu section par section](#) in the technical report).
To process the generation queue:

```bash
make worker
```

### All available commands

```bash
make help
```

Prints every target defined in the `Makefile`, including:

| Command | Description |
|---|---|
| `make setup` | Create virtualenv and install dependencies |
| `make dev` | Run backend locally with hot reload |
| `make start` | Run backend locally in production mode |
| `make docker-up` / `make docker-down` | Start / stop all containerized services |
| `make docker-logs` | Follow logs from all containers |
| `make docker-ps` | List running containers |
| `make ollama-pull` | Pull required LLM and embedding models |
| `make ollama-list` | List models available in the Ollama container |
| `make worker` | Start the RQ worker for memo generation jobs |
| `make health` | Check backend and dependent services health |
| `make freeze` | Save current Python dependencies to `requirements.txt` |

## Project context

This project was developed as part of a 3-month internship, starting from a feasibility study and user requirements specification (in `docs/`), through infrastructure procurement and setup, to the development of the agents described above. Not every use case from the original requirements was completed — BIM/CAD format support, diagram generation, and technical photo analysis remain open for future work.

## Status

Actively evolving. This README describes the architecture as deployed at the end of the internship; some components referenced in the codebase (e.g. diagram/vision agents) are early-stage and not yet production-ready.

## License

Internal project — not licensed for external use.
