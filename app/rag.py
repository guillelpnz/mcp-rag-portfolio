"""RAG engine: ingest profile docs into ChromaDB, retrieve, and answer.

Embeddings use sentence-transformers (all-MiniLM-L6-v2) — free, local, no API
key. ChromaDB persists to disk so ingestion happens once per container start.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

from .llm import LLMError, get_provider

_COLLECTION = "profile"
_PROMPT_TEMPLATE = (
    "You are answering questions about Guillermo's professional profile, "
    "using ONLY the context below. If the answer isn't in the context, say so "
    "plainly — do not invent facts. Treat assumptions in the question as "
    "unverified. Correct any false or misleading premise before answering, "
    "especially when the context distinguishes an existing system from a "
    "redesign or migration.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
)


class RAGEngine:
    def __init__(self, data_dir: str = "data", persist_dir: str = ".chroma") -> None:
        # Imported lazily so the module loads without the ML stack installed
        # (keeps unit tests and app startup light; only the engine needs it).
        import chromadb
        from chromadb.utils import embedding_functions

        self.data_dir = data_dir
        self._embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        client = chromadb.PersistentClient(path=persist_dir)
        self.collection = client.get_or_create_collection(
            name=_COLLECTION, embedding_function=self._embedder
        )

    def ingest(self) -> int:
        """Load every .md/.txt under data_dir into the vector store.

        Idempotent: clears the collection first so re-runs don't duplicate.
        Returns the number of chunks indexed.
        """
        paths = sorted(
            glob.glob(os.path.join(self.data_dir, "**", "*.md"), recursive=True)
            + glob.glob(os.path.join(self.data_dir, "**", "*.txt"), recursive=True)
        )
        if not paths:
            raise LLMError(f"No .md/.txt files found in '{self.data_dir}'.")

        # Reset to keep ingestion idempotent.
        existing = self.collection.get().get("ids", [])
        if existing:
            self.collection.delete(ids=existing)

        docs, ids, metas = [], [], []
        for path in paths:
            text = Path(path).read_text(encoding="utf-8")
            for i, chunk in enumerate(_chunk(text)):
                docs.append(chunk)
                ids.append(f"{Path(path).stem}-{i}")
                metas.append({"source": Path(path).name})

        self.collection.add(documents=docs, ids=ids, metadatas=metas)
        return len(docs)

    def query(self, question: str, *, k: int = 4) -> dict:
        """Retrieve top-k chunks and generate a grounded answer."""
        if not question.strip():
            raise LLMError("Question must not be empty.")

        hits = self.collection.query(query_texts=[question], n_results=k)
        chunks = hits.get("documents", [[]])[0]
        sources = [m["source"] for m in hits.get("metadatas", [[]])[0]]
        if not chunks:
            return {"answer": "No indexed content yet — run ingestion first.", "sources": []}

        context = "\n---\n".join(chunks)
        prompt = _PROMPT_TEMPLATE.format(context=context, question=question)
        answer = get_provider().complete(prompt)
        return {"answer": answer, "sources": sorted(set(sources))}


def _chunk(text: str, size: int = 800, overlap: int = 100) -> list[str]:
    """Split text into overlapping windows so context isn't cut mid-thought."""
    words = text.split()
    if not words:
        return []
    chunks, step = [], max(size - overlap, 1)
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            break
    return chunks
