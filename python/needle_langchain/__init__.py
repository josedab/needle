"""needle-langchain: LangChain vector store integration for Needle."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type

try:
    import needle  # type: ignore[import-untyped]
except ImportError:
    needle = None  # type: ignore[assignment]


@dataclass
class NeedleVectorStoreConfig:
    """Configuration for NeedleVectorStore."""

    collection_name: str = "langchain"
    dimensions: int = 384
    distance: str = "cosine"
    content_key: str = "page_content"
    metadata_key: str = "metadata"


class Document:
    """LangChain-compatible document."""

    def __init__(self, page_content: str = "", metadata: Optional[Dict[str, Any]] = None):
        self.page_content = page_content
        self.metadata = metadata or {}


class NeedleVectorStore:
    """LangChain-compatible vector store backed by Needle.

    Usage::

        from needle_langchain import NeedleVectorStore

        store = NeedleVectorStore(collection_name="docs", dimensions=384)
        store.add_texts(["Hello world", "Goodbye world"], metadatas=[{"k": "v"}])
        results = store.similarity_search_with_score(query_vector=[0.1]*384, k=5)
    """

    def __init__(
        self,
        collection_name: str = "langchain",
        dimensions: int = 384,
        distance: str = "cosine",
        content_key: str = "page_content",
        **kwargs: Any,
    ) -> None:
        self._config = NeedleVectorStoreConfig(
            collection_name=collection_name,
            dimensions=dimensions,
            distance=distance,
            content_key=content_key,
        )
        self._vectors: Dict[str, Tuple[List[float], Dict[str, Any]]] = {}
        self._next_id = 0

    # ── Core LangChain interface ─────────────────────────────────────────

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add texts (pre-embedded) to the store. Returns assigned IDs."""
        added_ids: List[str] = []
        for i, text in enumerate(texts):
            doc_id = ids[i] if ids and i < len(ids) else f"doc_{self._next_id}"
            self._next_id += 1
            meta = metadatas[i] if metadatas and i < len(metadatas) else {}
            meta[self._config.content_key] = text
            self._vectors[doc_id] = ([], meta)
            added_ids.append(doc_id)
        return added_ids

    def add_vectors(
        self,
        vectors: List[List[float]],
        documents: List[Document],
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add pre-computed vectors with associated documents."""
        added_ids: List[str] = []
        for i, (vec, doc) in enumerate(zip(vectors, documents)):
            doc_id = ids[i] if ids and i < len(ids) else f"doc_{self._next_id}"
            self._next_id += 1
            meta = dict(doc.metadata)
            meta[self._config.content_key] = doc.page_content
            self._vectors[doc_id] = (vec, meta)
            added_ids.append(doc_id)
        return added_ids

    def similarity_search_with_score(
        self,
        query_vector: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return top-k documents with similarity scores."""
        results: List[Tuple[str, float]] = []
        for doc_id, (vec, meta) in self._vectors.items():
            if not vec:
                continue
            if filter and not self._matches_filter(meta, filter):
                continue
            dist = self._cosine_distance(query_vector, vec)
            results.append((doc_id, dist))

        results.sort(key=lambda x: x[1])
        output: List[Tuple[Document, float]] = []
        for doc_id, dist in results[:k]:
            _, meta = self._vectors[doc_id]
            content = meta.get(self._config.content_key, "")
            doc = Document(page_content=content, metadata={
                k: v for k, v in meta.items() if k != self._config.content_key
            })
            output.append((doc, dist))
        return output

    def similarity_search(
        self,
        query_vector: List[float],
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:
        """Return top-k similar documents."""
        return [doc for doc, _ in self.similarity_search_with_score(query_vector, k, **kwargs)]

    def delete(self, ids: List[str], **kwargs: Any) -> bool:
        """Delete documents by ID."""
        deleted = False
        for doc_id in ids:
            if doc_id in self._vectors:
                del self._vectors[doc_id]
                deleted = True
        return deleted

    def get_by_id(self, doc_id: str) -> Optional[Document]:
        """Retrieve a document by ID."""
        if doc_id not in self._vectors:
            return None
        _, meta = self._vectors[doc_id]
        content = meta.get(self._config.content_key, "")
        return Document(page_content=content, metadata={
            k: v for k, v in meta.items() if k != self._config.content_key
        })

    @property
    def count(self) -> int:
        return len(self._vectors)

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _cosine_distance(a: List[float], b: List[float]) -> float:
        import math
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 1.0
        return 1.0 - dot / (norm_a * norm_b)

    @staticmethod
    def _matches_filter(meta: Dict[str, Any], filter: Dict[str, Any]) -> bool:
        for key, value in filter.items():
            if isinstance(value, dict):
                if "$eq" in value and meta.get(key) != value["$eq"]:
                    return False
                if "$ne" in value and meta.get(key) == value["$ne"]:
                    return False
            elif meta.get(key) != value:
                return False
        return True


__all__ = [
    "Document",
    "NeedleVectorStore",
    "NeedleVectorStoreConfig",
]
