import sys
import ollama
from pathlib import Path
import chromadb

EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen3:4b"
DB_PATH = "chroma_db"
TOP_K = 3

client = chromadb.PersistentClient(path=DB_PATH)
# cosine distance is the usual choice for text embeddings
collection = client.get_or_create_collection("docs", metadata={"hnsw:space": "cosine"})


def main() -> None:
    if len(sys.argv) < 2:
        print("wrong input. try again!")
        return

    cmd = sys.argv[1]
    if cmd == "ingest" and len(sys.argv) == 3:
        ingest(sys.argv[2])
    elif cmd == "ask" and len(sys.argv) > 2:
        query = " ".join(sys.argv[2:])
        ask(query)
    else:
        print("wrong input. try again!")


def greet():
    print("hello world!")


"""
Read the input file, separate into chunks, embed the chunks, then store embeddings into a vector database.

Chunking: start with naive method, using periods to separate chunks
"""


def ingest(path: str):
    if not path:
        return

    # read file content
    file_path = Path(path)
    if not file_path.is_file():
        print(f"The file '{path}' does not exist.")
        return
    content = file_path.read_text(encoding="utf-8")
    print(content)

    # chunk the file content
    chunks = [c.strip() for c in content.strip().split(".") if c]
    print(chunks)

    # embed chunks
    embeddings = ollama.embed(model=EMBED_MODEL, input=chunks)["embeddings"]
    print("embeddings created")

    # store embeddings in vector DB
    collection.upsert(
        ids=[f"{path}-{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings,
        metadatas=[{"source": path, "chunk": i} for i in range(len(chunks))],
    )
    print(f"Indexed {path} ({len(chunks)} chunks)")


def ask(query: str):
    # embed query with same embedding model used for ingesting documents
    print("query:", query)
    q_vec = ollama.embed(model=EMBED_MODEL, input=query)["embeddings"]

    # run semantic search to retrieve nearest chunks to query
    res = collection.query(query_embeddings=q_vec, n_results=TOP_K)
    docs = res["documents"][0]
    sources = [m["source"] for m in res["metadatas"][0]]
    distances = res["distances"][0]

    print("\n--- Retrieved chunks: ---")
    for d, s, dist in zip(docs, sources, distances):
        print(f"[{s}] distance={dist:.3f}: {d[:100].replace(chr(10), ' ')}...")
    print("------------------------\n")

    # augment by putting chunks into prompt
    context = "\n\n---\n\n".join(docs)
    prompt = (
        # "Answer the question using only the context below:"
        # "If the context doesn't contain the answer, tell the user that you don't know."
        # "Answer the query naturally without hinting to the user that you have context.\n\n"
        "You are a helpful assistant. Answer the question using the provided context. "
        "Quote or paraphrase the relevant parts. Only say you don't know if the "
        "context clearly has nothing related to the question."
        f"Context:\n{context}\n\nQuestion: {query}"
    )

    # generate response (streamed)
    stream = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    for part in stream:
        print(part["message"]["content"], end="", flush=True)
    print("\n")
