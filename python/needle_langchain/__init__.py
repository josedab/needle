"""needle-langchain: LangChain vector store integration for Needle."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Type

try:
    import needle  # type: ignore[import-untyped]
except ImportError:
    needle = None  # type: ignore[assignment]


# Protocol for embedding functions (compatible with LangChain Embeddings interface)
class EmbeddingFunction:
    """Protocol for embedding functions. Implement embed_documents and embed_query."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> List[float]:
        raise NotImplementedError


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
        embedding_function: Optional[Any] = None,
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
        self._embedding_function = embedding_function

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Any,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        collection_name: str = "langchain",
        **kwargs: Any,
    ) -> "NeedleVectorStore":
        """Create a NeedleVectorStore from texts with an embedding function.

        This is the standard LangChain pattern for creating a vector store.

        Args:
            texts: List of text documents to embed and store.
            embedding: Embedding function with embed_documents() method.
            metadatas: Optional list of metadata dicts for each text.
            collection_name: Name for the collection.
        """
        embeddings = embedding.embed_documents(texts)
        dimensions = len(embeddings[0]) if embeddings else 384
        store = cls(
            collection_name=collection_name,
            dimensions=dimensions,
            embedding_function=embedding,
            **kwargs,
        )
        documents = [
            Document(page_content=text, metadata=meta or {})
            for text, meta in zip(texts, metadatas or [{}] * len(texts))
        ]
        store.add_vectors(embeddings, documents)
        return store

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embedding: Any,
        collection_name: str = "langchain",
        **kwargs: Any,
    ) -> "NeedleVectorStore":
        """Create a NeedleVectorStore from documents with an embedding function."""
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return cls.from_texts(
            texts, embedding, metadatas=metadatas,
            collection_name=collection_name, **kwargs,
        )

    def similarity_search_by_text(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:
        """Search by text query (requires embedding_function)."""
        if self._embedding_function is None:
            raise ValueError("embedding_function required for text search. Use from_texts() or pass embedding_function in constructor.")
        query_vector = self._embedding_function.embed_query(query)
        return self.similarity_search(query_vector, k, **kwargs)

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

    def max_marginal_relevance_search(
        self,
        query_vector: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Maximal Marginal Relevance search for diversity.

        Args:
            query_vector: Query embedding vector.
            k: Number of results to return.
            fetch_k: Number of candidates to fetch before MMR re-ranking.
            lambda_mult: Balance between relevance (1.0) and diversity (0.0).
            filter: Optional metadata filter.
        """
        candidates = self.similarity_search_with_score(query_vector, fetch_k, filter=filter)
        if not candidates:
            return []

        selected: List[int] = []
        candidate_vecs = [
            self._vectors[self._get_doc_id(doc)][0] for doc, _ in candidates
        ]

        for _ in range(min(k, len(candidates))):
            best_idx = -1
            best_score = float("-inf")
            for i, (doc, dist) in enumerate(candidates):
                if i in selected:
                    continue
                relevance = 1.0 - dist
                max_sim = 0.0
                for j in selected:
                    sim = 1.0 - self._cosine_distance(candidate_vecs[i], candidate_vecs[j])
                    max_sim = max(max_sim, sim)
                mmr_score = lambda_mult * relevance - (1.0 - lambda_mult) * max_sim
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
            if best_idx >= 0:
                selected.append(best_idx)

        return [candidates[i][0] for i in selected]

    # ── Async interface ─────────────────────────────────────────────────

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Async version of add_texts."""
        return self.add_texts(texts, metadatas=metadatas, ids=ids, **kwargs)

    async def asimilarity_search(
        self,
        query_vector: List[float],
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:
        """Async version of similarity_search."""
        return self.similarity_search(query_vector, k, **kwargs)

    async def asimilarity_search_with_score(
        self,
        query_vector: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Async version of similarity_search_with_score."""
        return self.similarity_search_with_score(query_vector, k, filter=filter, **kwargs)

    async def amax_marginal_relevance_search(
        self,
        query_vector: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        """Async version of max_marginal_relevance_search."""
        return self.max_marginal_relevance_search(
            query_vector, k, fetch_k=fetch_k, lambda_mult=lambda_mult, **kwargs
        )

    async def adelete(self, ids: List[str], **kwargs: Any) -> bool:
        """Async version of delete."""
        return self.delete(ids, **kwargs)

    def _get_doc_id(self, doc: Document) -> str:
        """Get the stored ID for a document (reverse lookup)."""
        for doc_id, (_, meta) in self._vectors.items():
            content = meta.get(self._config.content_key, "")
            if content == doc.page_content:
                return doc_id
        return ""

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
    "EmbeddingFunction",
    "NeedleVectorStore",
    "NeedleVectorStoreConfig",
]
