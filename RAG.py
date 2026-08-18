#!/usr/bin/env python3
"""
Run the complete project:

    pip install -r requirements.txt
    python RAG.py

The script will:
1. Load configuration from .env or environment variables.
2. Select OpenAI when OPENAI_API_KEY is configured, otherwise Ollama.
3. Load every .txt file from data/.
4. Chunk and embed the documents.
5. Persist the vectors in ChromaDB.
6. Run the examples defined at the bottom of this file.

Useful commands:

    python RAG.py
    python RAG.py --question "What is Tagore's current role?"
    python RAG.py --rebuild
    python RAG.py --status
    python RAG.py --interactive
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    import chromadb
    from openai import OpenAI
except ImportError as exc:
    print(
        "Missing Python packages. Install them with:\n"
        "python -m pip install -U openai chromadb python-dotenv tiktoken",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
VECTOR_STORE_DIR = ROOT / ".rag_store"
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

if load_dotenv is not None:
    load_dotenv(ROOT / ".env")

RAG_PROVIDER = os.getenv("RAG_PROVIDER", "ollama").strip().lower()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2")
OLLAMA_EMBEDDING_MODEL = os.getenv(
    "OLLAMA_EMBEDDING_MODEL",
    "nomic-embed-text",
)

if RAG_PROVIDER == "ollama":
    BASE_URL = OLLAMA_BASE_URL
    API_KEY = "ollama"
    CHAT_MODEL = OLLAMA_CHAT_MODEL
    EMBEDDING_MODEL = OLLAMA_EMBEDDING_MODEL
elif RAG_PROVIDER == "openai":
    BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    API_KEY = os.getenv("OPENAI_API_KEY")
    CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    EMBEDDING_MODEL = os.getenv(
        "OPENAI_EMBEDDING_MODEL",
        "text-embedding-3-small",
    )
    if not API_KEY:
        raise RuntimeError(
            "RAG_PROVIDER=openai requires OPENAI_API_KEY in the environment."
        )
else:
    raise ValueError("RAG_PROVIDER must be either 'ollama' or 'openai'.")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

TEMPERATURE = 0.0
TOP_K = 4
MIN_RELEVANCE_SCORE = 0.20
CHUNK_SIZE_WORDS = 80
CHUNK_OVERLAP_WORDS = 20

ABSTENTION = "I don't know based on the available documents."

SYSTEM_PROMPT = f"""You answer questions using only the provided document context.
Rules:
- Do not use outside knowledge, memory, or assumptions.
- If the context does not support the answer, reply exactly: {ABSTENTION}
- If supported, be concise and factual.
- Cite supporting files in square brackets, for example [03-professional.txt].
- Do not reveal or infer sensitive information that is not explicitly present.
"""


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

try:
    import tiktoken

    TOKEN_ENCODING = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(TOKEN_ENCODING.encode(text))

except ImportError:

    def count_tokens(text: str) -> int:
        return max(1, len(text.split())) if text.strip() else 0


# ---------------------------------------------------------------------------
# Provider calls
# ---------------------------------------------------------------------------

def check_connection() -> None:
    try:
        models = client.models.list()
        names = [model.id for model in models.data[:10]]
        print("Connection OK.")
        print("Visible models:", names if names else "(no model list returned)")
    except Exception as exc:
        if RAG_PROVIDER == "ollama":
            raise RuntimeError(
                "Could not reach Ollama. Run 'ollama serve', verify "
                f"OLLAMA_BASE_URL, and pull '{CHAT_MODEL}' and "
                f"'{EMBEDDING_MODEL}'. Original error: {exc}"
            ) from exc
        raise RuntimeError(
            f"Could not reach the configured model provider: {exc}"
        ) from exc


def chat(
    messages: list[dict[str, str]],
    temperature: float = TEMPERATURE,
) -> tuple[str, dict[str, int | None] | None]:
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=temperature,
    )
    content = response.choices[0].message.content or ""

    usage = getattr(response, "usage", None)
    usage_dict = None
    if usage is not None:
        usage_dict = {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }

    return content.strip(), usage_dict


def embed_texts(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    if not texts:
        return []

    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )
        vectors.extend(item.embedding for item in response.data)

    if len(vectors) != len(texts):
        raise RuntimeError(
            "The embedding provider returned an unexpected number of vectors."
        )

    return vectors


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Cosine similarity requires equal vector dimensions.")

    dot = sum(x * y for x, y in zip(a, b))
    length_a = sum(x * x for x in a) ** 0.5
    length_b = sum(x * x for x in b) ** 0.5

    if length_a == 0 or length_b == 0:
        return 0.0

    return dot / (length_a * length_b)


# ---------------------------------------------------------------------------
# Load and chunk documents
# ---------------------------------------------------------------------------

def load_documents(data_dir: Path = DATA_DIR) -> list[dict[str, str]]:
    paths = sorted(data_dir.glob("*.txt"))
    if not paths:
        raise FileNotFoundError(
            f"No .txt files found in {data_dir.resolve()}"
        )

    documents = []
    for path in paths:
        text = path.read_text(encoding="utf-8").strip()
        if text:
            documents.append(
                {
                    "source": path.name,
                    "text": text,
                    "sha256": hashlib.sha256(
                        text.encode("utf-8")
                    ).hexdigest(),
                }
            )

    if not documents:
        raise ValueError(f"The .txt files in {data_dir.resolve()} are empty.")

    return documents


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE_WORDS,
    overlap: int = CHUNK_OVERLAP_WORDS,
) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    words = text.split()
    if not words:
        return []

    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(words):
            break

    return chunks


def build_chunks(
    documents: list[dict[str, str]],
) -> list[dict[str, str | int]]:
    all_chunks = []

    for document in documents:
        for index, text in enumerate(chunk_text(document["text"])):
            all_chunks.append(
                {
                    "source": document["source"],
                    "document_sha256": document["sha256"],
                    "chunk_index": index,
                    "text": text,
                    "token_count": count_tokens(text),
                }
            )

    return all_chunks


# ---------------------------------------------------------------------------
# Chroma vector index
# ---------------------------------------------------------------------------

def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_")[:40]


INDEX_SIGNATURE = hashlib.sha256(
    (
        f"{RAG_PROVIDER}|{EMBEDDING_MODEL}|"
        f"{CHUNK_SIZE_WORDS}|{CHUNK_OVERLAP_WORDS}"
    ).encode("utf-8")
).hexdigest()[:12]

COLLECTION_NAME = (
    f"rag_{safe_name(RAG_PROVIDER)}_"
    f"{safe_name(EMBEDDING_MODEL)}_{INDEX_SIGNATURE}"
)[:63]

chroma_client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={
        "hnsw:space": "cosine",
        "embedding_model": EMBEDDING_MODEL,
    },
)


def chunk_id(chunk: dict[str, str | int]) -> str:
    raw = (
        f"{chunk['source']}|{chunk['document_sha256']}|"
        f"{chunk['chunk_index']}|{chunk['text']}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_index(
    chunks: list[dict[str, str | int]],
) -> dict[str, str | int]:
    if not chunks:
        raise ValueError("Cannot index an empty chunk list.")

    vectors = embed_texts([str(chunk["text"]) for chunk in chunks])
    ids = [chunk_id(chunk) for chunk in chunks]
    metadatas = [
        {
            "source": str(chunk["source"]),
            "document_sha256": str(chunk["document_sha256"]),
            "chunk_index": int(chunk["chunk_index"]),
            "token_count": int(chunk["token_count"]),
        }
        for chunk in chunks
    ]

    collection.upsert(
        ids=ids,
        documents=[str(chunk["text"]) for chunk in chunks],
        embeddings=vectors,
        metadatas=metadatas,
    )

    existing_ids = set(collection.get()["ids"])
    stale_ids = existing_ids - set(ids)
    if stale_ids:
        collection.delete(ids=list(stale_ids))

    return {
        "collection": COLLECTION_NAME,
        "indexed_chunks": len(ids),
        "deleted_stale_chunks": len(stale_ids),
        "total_chunks": collection.count(),
        "embedding_model": EMBEDDING_MODEL,
    }


# ---------------------------------------------------------------------------
# Retrieval and answer generation
# ---------------------------------------------------------------------------

def retrieve(
    question: str,
    k: int = TOP_K,
    min_score: float = MIN_RELEVANCE_SCORE,
) -> list[dict[str, str | float | int]]:
    question = question.strip()
    if not question:
        raise ValueError("question must not be empty")
    if collection.count() == 0:
        raise RuntimeError("The vector collection is empty. Run build_index().")

    query_vector = embed_texts([question])[0]
    result = collection.query(
        query_embeddings=[query_vector],
        n_results=min(k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    matches = []
    for text, metadata, distance in zip(
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        score = 1.0 - float(distance)
        if score >= min_score:
            matches.append(
                {
                    "text": text,
                    "source": metadata["source"],
                    "chunk_index": metadata["chunk_index"],
                    "score": score,
                    "distance": float(distance),
                }
            )

    return matches


def show_retrieval(
    question: str,
    k: int = TOP_K,
    min_score: float = MIN_RELEVANCE_SCORE,
) -> list[dict[str, str | float | int]]:
    matches = retrieve(question, k=k, min_score=min_score)
    print(f"Question: {question}")
    print(f"Accepted matches: {len(matches)}")

    for rank, match in enumerate(matches, 1):
        print(
            f"\n{rank}. score={float(match['score']):.3f} "
            f"[{match['source']} :: chunk {match['chunk_index']}]\n"
            f"   {match['text']}"
        )

    return matches


def build_context(
    matches: list[dict[str, str | float | int]],
) -> str:
    return "\n\n".join(
        (
            f"[source: {match['source']} | "
            f"chunk: {match['chunk_index']} | "
            f"score: {float(match['score']):.3f}]\n"
            f"{match['text']}"
        )
        for match in matches
    )


def ask(
    question: str,
    k: int = TOP_K,
    min_score: float = MIN_RELEVANCE_SCORE,
) -> dict[str, object]:
    matches = retrieve(question, k=k, min_score=min_score)

    if not matches:
        return {
            "question": question,
            "answer": ABSTENTION,
            "sources": [],
            "matches": [],
            "usage": None,
        }

    prompt = f"""Use the context below to answer the question.

Context:
{build_context(matches)}

Question: {question}

If the context does not contain the answer, reply exactly: {ABSTENTION}"""

    answer, usage = chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=TEMPERATURE,
    )

    return {
        "question": question,
        "answer": answer or ABSTENTION,
        "sources": sorted({str(match["source"]) for match in matches}),
        "matches": matches,
        "usage": usage,
    }


def print_answer(result: dict[str, object]) -> None:
    print(result["answer"])
    sources = result.get("sources", [])
    print("Sources:", ", ".join(sources) if sources else "none")

    usage = result.get("usage")
    if usage:
        print("Usage:", usage)


def run_smoke_tests() -> None:
    cases = [
        ("What is my current professional role?", "03-professional.txt"),
        ("Where did I complete my BTech?", "02-education.txt"),
        ("What is my GitHub profile?", "04-social.txt"),
    ]

    failures = []
    for question, expected_source in cases:
        sources = {str(match["source"]) for match in retrieve(question)}
        if expected_source not in sources:
            failures.append(
                {
                    "question": question,
                    "expected": expected_source,
                    "got": sorted(sources),
                }
            )

    if failures:
        raise AssertionError(f"Retrieval smoke tests failed: {failures}")

    print(f"Retrieval smoke tests passed: {len(cases)}")


# ---------------------------------------------------------------------------
# Command-line runner
# ---------------------------------------------------------------------------

EXAMPLE_QUESTIONS = [
    "What is my current professional role?",
    "Where did I complete my BTech, and what did I study?",
    "Which professional experiences are listed?",
    "How can someone find me on GitHub?",
    "What is my educational background?",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the local Ollama RAG pipeline over data/*.txt."
    )
    parser.add_argument(
        "--question",
        "-q",
        help="Ask one question after building the index.",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Start an interactive question loop.",
    )
    parser.add_argument(
        "--skip-examples",
        action="store_true",
        help="Build the index without running the example questions.",
    )
    parser.add_argument(
        "--show-retrieval",
        action="store_true",
        help="Print retrieved chunks for the question before answering.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run retrieval smoke tests after building the index.",
    )
    return parser.parse_args()


def run_question(question: str, show_matches: bool = False) -> None:
    if show_matches:
        show_retrieval(question)
    print_answer(ask(question))


def main() -> None:
    args = parse_args()

    print(f"Provider: {RAG_PROVIDER}")
    print(f"Chat model: {CHAT_MODEL}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Vector store: {VECTOR_STORE_DIR}")
    print()

    check_connection()

    documents = load_documents()
    print(f"Loaded {len(documents)} text files.")
    for document in documents:
        print(
            f"- {document['source']}: "
            f"{len(document['text']):,} characters, "
            f"{count_tokens(document['text']):,} tokens"
        )

    chunks = build_chunks(documents)
    print(f"Created {len(chunks)} chunks.")
    print()

    print("Building vector index...")
    print(build_index(chunks))
    print()

    if args.smoke_test:
        run_smoke_tests()
        print()

    if args.question:
        run_question(args.question, show_matches=args.show_retrieval)
        return

    if args.interactive:
        print("Interactive mode. Type 'exit' or 'quit' to stop.")
        while True:
            try:
                question = input("\nQuestion: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if question.lower() in {"exit", "quit"}:
                break
            if question:
                run_question(question, show_matches=args.show_retrieval)
        return

    if not args.skip_examples:
        for question in EXAMPLE_QUESTIONS:
            print("=" * 80)
            print(f"Question: {question}")
            run_question(question, show_matches=args.show_retrieval)


if __name__ == "__main__":
    main()