"""Codebase indexer with pluggable vector store backend.

Supports two backends (auto-selected based on availability):
- FAISS: default, works on all Python versions including 3.14+
- Chroma: requires Python 3.11-3.13 (chromadb uses pydantic v1)

The index is stored in .coco_index/ in the working directory.
FAISS index is saved as index.faiss + index.pkl.
Chroma index uses its own internal format.
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Callable, Generator, Optional

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config.settings import IndexConfig


def _build_vectorstore_backend(
    persist_dir: Path,
    collection_name: str,
    embeddings: Embeddings,
) -> tuple[str, Optional[VectorStore]]:
    """Try to load an existing index. Returns (backend_name, vectorstore_or_None)."""
    # Try FAISS first (works on all Python versions)
    faiss_path = persist_dir / "index.faiss"
    if faiss_path.exists():
        try:
            from langchain_community.vectorstores import FAISS  # noqa: PLC0415
            vs = FAISS.load_local(
                str(persist_dir),
                embeddings,
                allow_dangerous_deserialization=True,
            )
            return "faiss", vs
        except Exception:
            pass

    # Try Chroma if available (Python 3.11–3.13 only)
    chroma_dir = persist_dir / "chroma"
    if chroma_dir.exists():
        try:
            import chromadb  # noqa: PLC0415
            from langchain_chroma import Chroma  # noqa: PLC0415
            client = chromadb.PersistentClient(path=str(chroma_dir))
            vs = Chroma(
                client=client,
                collection_name=collection_name,
                embedding_function=embeddings,
            )
            return "chroma", vs
        except Exception:
            pass

    return "none", None


class CodebaseIndexer:
    """Manages codebase indexing and semantic search.

    Auto-selects FAISS or Chroma as the vector store backend based on
    what's available and what index already exists on disk.
    """

    def __init__(
        self,
        config: IndexConfig,
        embeddings: Embeddings,
        working_dir: Optional[Path] = None,
    ):
        self.config = config
        self.embeddings = embeddings
        self.working_dir = working_dir or Path.cwd()
        self._persist_dir = self.working_dir / config.persist_dir
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._backend, self._vectorstore = _build_vectorstore_backend(
            self._persist_dir, config.collection_name, embeddings
        )

    def index(
        self,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> dict[str, int]:
        """Index the codebase. Returns stats: files_processed, chunks_created."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            add_start_index=True,
        )

        files = list(self._iter_files())
        total = len(files)
        all_docs: list[Document] = []

        for i, file_path in enumerate(files):
            if progress_callback:
                progress_callback(i + 1, total, str(file_path))
            try:
                loader = TextLoader(
                    str(file_path),
                    encoding="utf-8",
                    autodetect_encoding=True,
                )
                raw_docs = loader.load()
                chunks = splitter.split_documents(raw_docs)
                all_docs.extend(chunks)
            except Exception:
                continue  # Skip unreadable files

        if all_docs:
            self._vectorstore = self._build_new_index(all_docs)

        return {"files_processed": total, "chunks_created": len(all_docs)}

    def _build_new_index(self, docs: list[Document]) -> VectorStore:
        """Build a new vector store from documents. Prefers FAISS."""
        # Try FAISS first
        try:
            from langchain_community.vectorstores import FAISS  # noqa: PLC0415
            vs = FAISS.from_documents(docs, self.embeddings)
            vs.save_local(str(self._persist_dir))
            self._backend = "faiss"
            return vs
        except ImportError:
            pass

        # Fall back to Chroma
        import chromadb  # noqa: PLC0415
        from langchain_chroma import Chroma  # noqa: PLC0415
        chroma_dir = self._persist_dir / "chroma"
        chroma_dir.mkdir(exist_ok=True)
        client = chromadb.PersistentClient(path=str(chroma_dir))
        try:
            client.delete_collection(self.config.collection_name)
        except Exception:
            pass
        vs = Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings,
            client=client,
            collection_name=self.config.collection_name,
        )
        self._backend = "chroma"
        return vs

    def search(self, query: str, n_results: int = 5) -> str:
        """Search the index and return formatted results."""
        if self._vectorstore is None:
            return "Codebase not indexed. Run /index first."
        try:
            results = self._vectorstore.similarity_search_with_score(query, k=n_results)
        except Exception as e:
            return f"Index search error: {e}"

        if not results:
            return "No relevant code found."

        lines = []
        for doc, score in results:
            source = doc.metadata.get("source", "unknown")
            start = doc.metadata.get("start_index", "?")
            try:
                source = str(Path(source).relative_to(self.working_dir))
            except ValueError:
                pass
            lines.append(f"--- {source} (offset {start}, relevance {1 - score:.2f}) ---")
            lines.append(doc.page_content[:600])
            lines.append("")
        return "\n".join(lines)

    def get_stats(self) -> str:
        """Return a human-readable summary of the index."""
        if self._vectorstore is None:
            return "Index not found. Run /index to build it."
        try:
            # FAISS docstore
            if self._backend == "faiss":
                count = len(self._vectorstore.docstore._dict)
                rel = self._persist_dir.relative_to(Path.cwd()) if self._persist_dir.is_relative_to(Path.cwd()) else self._persist_dir
                return f"Indexed: {count:,} chunks | Backend: FAISS | Path: {rel}"
            # Chroma
            count = self._vectorstore._collection.count()
            rel = self._persist_dir.relative_to(Path.cwd()) if self._persist_dir.is_relative_to(Path.cwd()) else self._persist_dir
            return f"Indexed: {count:,} chunks | Backend: Chroma | Path: {rel}"
        except Exception:
            return "Index exists but stats unavailable."

    def is_indexed(self) -> bool:
        return self._vectorstore is not None

    def _iter_files(self) -> Generator[Path, None, None]:
        """Walk working_dir yielding files matching include/exclude config."""
        include_exts = set(self.config.include_extensions)
        exclude_dirs = set(self.config.exclude_dirs)

        for root, dirs, files in os.walk(self.working_dir):
            dirs[:] = [
                d for d in dirs
                if d not in exclude_dirs and not d.startswith(".")
            ]
            for fname in files:
                p = Path(root) / fname
                if p.suffix in include_exts:
                    yield p
