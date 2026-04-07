# RAG Production App

This project is a small retrieval pipeline built with FastAPI, Inngest, Azure OpenAI embeddings, and Qdrant. It ingests PDF files, chunks them into text, creates embeddings, and stores those vectors in a local Qdrant collection named `docs`.

## What This Project Uses

- Python 3.11+
- `uv` for dependency management and running the app
- Docker Desktop or another Docker runtime for local Qdrant
- Azure OpenAI for embeddings
- Qdrant as the vector database

Core Python dependencies are defined in `pyproject.toml` and include:

- `fastapi`
- `inngest`
- `langchain-openai`
- `llama-index-core`
- `llama-index-readers-file`
- `python-dotenv`
- `qdrant-client`
- `uvicorn`

## External Services

### 1. Azure OpenAI

This app requires Azure OpenAI to generate embeddings before documents are stored in Qdrant.

Required environment variables:

- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`

Optional but commonly included for future chat or answer-generation flows:

- `AZURE_OPENAI_DEPLOYMENT`

### Preferred Models

The current code expects an embedding deployment that returns 3072-dimensional vectors because `QdrantStorage` creates the collection with `dim=3072`.

Recommended embedding deployment:

- `text-embedding-3-large`

If you switch to a different embedding model, the vector dimension in `vector_db.py` must match the model output dimension.

For a future answer-generation model, a lightweight deployment such as `gpt-5.4-nano` is a reasonable default, but that chat deployment is not used by the current code path yet.

### 2. Qdrant

The app stores vectors in Qdrant and currently connects to:

- `http://localhost:6333`

The repository already contains a local `qdrant_storage/` directory for persisted Qdrant data.

## Get Started

### 1. Install prerequisites

Install the following on your machine:

- Python 3.11 or newer
- `uv`: https://docs.astral.sh/uv/
- Docker Desktop: https://www.docker.com/products/docker-desktop/

### 2. Install Python dependencies

From the project root:

```bash
uv sync
```

### 3. Create a `.env` file

Add the Azure OpenAI settings your environment needs:

```env
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=your-chat-model-deployment
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-large
QDRANT_URL=http://localhost:6333
```

Do not commit real secrets to source control.

### 4. Start Qdrant

Run Qdrant locally with Docker:

```bash
docker run -d --name qdrantRagDb -p 6333:6333 -v "$PWD/qdrant_storage:/qdrant/storage" qdrant/qdrant
```

If a container with that name already exists, remove it first or use a different container name:

```bash
docker rm -f qdrantRagDb
docker run -d --name qdrantRagDb -p 6333:6333 -v "$PWD/qdrant_storage:/qdrant/storage" qdrant/qdrant
```

### 5. Start the app

Use the same command currently used in this repo:

```bash
uv run uvicorn main:app --reload
```

Without auto-reload:

```bash
uv run uvicorn main:app
```

By default the app will be available at:

- `http://127.0.0.1:8000`

## How the App Works

- `main.py` wires FastAPI and Inngest together.
- `data_loader.py` loads PDF files, chunks them with `SentenceSplitter`, and generates embeddings with Azure OpenAI.
- `vector_db.py` ensures the `docs` collection exists in Qdrant and upserts vectors there.

The implemented function listens for the Inngest event:

- `rag/ingest_pdf`

That flow:

1. Reads a PDF path from the event payload.
2. Extracts and chunks the document text.
3. Embeds each chunk with Azure OpenAI.
4. Stores the vectors and metadata in Qdrant.

## Run and Test Commands

### Start the project

```bash
uv run uvicorn main:app --reload
```

### Dependency install

```bash
uv sync
```

### Qdrant local startup

```bash
docker run -d --name qdrantRagDb -p 6333:6333 -v "$PWD/qdrant_storage:/qdrant/storage" qdrant/qdrant
```

### Current test status

There is no automated test suite checked into this repository yet. No `tests/` directory or `pytest` configuration is present today.

### Recommended smoke tests

Verify the app imports and starts:

```bash
uv run python -c "import main; print('app import ok')"
uv run uvicorn main:app --reload
```

Verify Qdrant is reachable:

```bash
curl http://localhost:6333/collections
```

## Notes and Limitations

- The vector collection is created with dimension `3072`, so the embedding deployment must be compatible.
- `QDRANT_URL` is a useful environment variable to keep in `.env`, but the current `QdrantStorage` implementation defaults directly to `http://localhost:6333`.
- The repository includes `streamlit` as a dependency, but there is no Streamlit entrypoint in the current codebase.
- The repository defines RAG result models for search and answer responses, but the current implemented workflow is document ingestion.
