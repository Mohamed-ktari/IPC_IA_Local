# Makefile
# Run from the project root: architecture-ai/
# Usage: make <command>

# ──────────────────────────────────────────
# VARIABLES
# ──────────────────────────────────────────
BACKEND_DIR = backend
VENV        = $(BACKEND_DIR)/venv
PYTHON      = $(VENV)/bin/python
PIP         = $(VENV)/bin/pip
UVICORN     = $(VENV)/bin/uvicorn

# ──────────────────────────────────────────
# SETUP
# ──────────────────────────────────────────

install:                          ## Install all Python dependencies
	cd $(BACKEND_DIR) && ../$(PIP) install -r requirements.txt

venv:                             ## Create virtual environment
	python3 -m venv $(VENV)

docling-init:                     ## Download Docling models (run once after install)
	cd $(BACKEND_DIR) && ../$(PYTHON) -c "from docling.document_converter import DocumentConverter; print('Docling ready')"

setup: venv install               ## Full setup from scratch (venv + install)

# ──────────────────────────────────────────
# RUN
# ──────────────────────────────────────────

dev:                              ## Start backend in development mode (hot reload)
	cd $(BACKEND_DIR) && ../$(UVICORN) app.main:app --reload --port 8000

start:                            ## Start backend in production mode
	cd $(BACKEND_DIR) && ../$(UVICORN) app.main:app --port 8000 --workers 2

# ──────────────────────────────────────────
# DOCKER
# ──────────────────────────────────────────

docker-up:                        ## Start all services (ollama, chromadb, redis)
	docker compose up -d

docker-down:                      ## Stop all services
	docker compose down

docker-logs:                      ## Follow logs from all containers
	docker compose logs -f

docker-ps:                        ## Show running containers
	docker compose ps

ollama-pull:                      ## Pull the model inside the Ollama container
	docker exec -it ollama ollama pull mistral:7b-instruct-q4_K_M
	docker exec -it ollama ollama pull nomic-embed-text

ollama-list:                      ## List models available in Ollama container
	docker exec -it ollama ollama list

# ──────────────────────────────────────────
# HEALTH
# ──────────────────────────────────────────

health:                           ## Check all services health
	curl -s http://localhost:8000/health | python3 -m json.tool

# ──────────────────────────────────────────
# DEPENDENCIES
# ──────────────────────────────────────────

freeze:                           ## Save current pip packages to requirements.txt
	cd $(BACKEND_DIR) && ../$(PIP) freeze > requirements.txt

# ──────────────────────────────────────────
# UTILS
# ──────────────────────────────────────────

help:                             ## Show all available commands
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | \
	awk 'BEGIN {FS = ":.*##"}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

.PHONY: install venv setup dev start docker-up docker-down \
        docker-logs docker-ps ollama-pull ollama-list \
        health freeze help