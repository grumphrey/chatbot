# Local RAG CLI

A minimal retrieval-augmented generation (RAG) application written in Python. It uses [Ollama](https://ollama.com/) to run the embedding and language models locally, and [ChromaDB](https://www.trychroma.com/) as a persistent vector database.

The CLI provides two commands:

- `ingest` reads a text file, splits it into chunks, creates embeddings, and stores them in ChromaDB.
- `ask` retrieves the most relevant chunks and uses them as context for a locally generated answer.

## Requirements

- Python 3.14 or later
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/)

Download the models used by the application:

```bash
ollama pull nomic-embed-text
ollama pull qwen3:4b
```

## Setup

Install the Python dependencies:

```bash
uv sync
```

Make sure Ollama is running before using the CLI.

## Usage

Ingest a text file:

```bash
uv run chatbot ingest data/hello-world.txt
```

Ask a question about the ingested content:

```bash
uv run chatbot ask "What is this document about?"
```

The vector database is stored locally in `chroma_db/`. By default, the application retrieves the three closest chunks, generates a response with `qwen3:4b`, and creates embeddings with `nomic-embed-text`.
