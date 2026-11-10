# DocLens

A local chatbot that answers from your PDFs with page citations, and chats normally when the documents don't cover the question.

**Stack:** Ollama (`qwen2.5:3b`) · ChromaDB · sentence-transformers (embeddings + cross-encoder reranker) · Streamlit · Docker Compose

![Architecture](docs/architecture.drawio.svg)

## Run

```bash
./scripts/download_data.sh        # or put your own PDFs in data/
docker compose up -d
docker compose exec ollama ollama pull qwen2.5:3b
docker compose exec app python ingest.py
```

Open http://localhost:8501. CLI: `docker compose exec app python query.py "your question"`. Retrieval eval: `docker compose exec app python eval.py`.
