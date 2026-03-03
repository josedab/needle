"""needle-llamaindex: LlamaIndex vector store integration for Needle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import uuid


class TextNode:
    """LlamaIndex-compatible text node."""

    def __init__(
        self,
        text: str = "",
        id_: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
    ):
        self.text = text
        self.id_ = id_ or str(uuid.uuid4())
        self.metadata = metadata or {}
        self.embedding = embedding


class NodeWithScore:
    """A node paired with its relevance score."""

    def __init__(self, node: TextNode, score: float = 0.0):
        self.node = node
        self.score = score


@dataclass
class NeedleIndexConfig:
    """Configuration for NeedleVectorStoreIndex."""

    collection_name: str = "llamaindex"
    dimensions: int = 384
    distance: str = "cosine"


class NeedleVectorStoreIndex:
    """LlamaIndex-compatible vector store backed by Needle.

    Usage::

        from needle_llamaindex import NeedleVectorStoreIndex, TextNode

        index = NeedleVectorStoreIndex(dimensions=384)
        nodes = [TextNode(text="Hello", embedding=[0.1]*384)]
        index.add(nodes)
        results = index.query([0.1]*384, similarity_top_k=5)
    """

    def __init__(
        self,
        collection_name: str = "llamaindex",
        dimensions: int = 384,
        distance: str = "cosine",
        embed_model: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        self._config = NeedleIndexConfig(
            collection_name=collection_name,
            dimensions=dimensions,
            distance=distance,
        )
        self._nodes: Dict[str, TextNode] = {}
        self._embed_model = embed_model

    @classmethod
    def from_nodes(
        cls,
        nodes: List[TextNode],
        collection_name: str = "llamaindex",
        **kwargs: Any,
    ) -> "NeedleVectorStoreIndex":
        """Create an index from pre-embedded nodes."""
        dims = 384
        if nodes and nodes[0].embedding:
            dims = len(nodes[0].embedding)
        index = cls(collection_name=collection_name, dimensions=dims, **kwargs)
        index.add(nodes)
        return index

    def add(self, nodes: List[TextNode], **kwargs: Any) -> List[str]:
        """Add nodes to the index. Returns node IDs."""
        ids: List[str] = []
        for node in nodes:
            self._nodes[node.id_] = node
            ids.append(node.id_)
        return ids

    def delete(self, ref_doc_id: str, **kwargs: Any) -> None:
        """Delete a node by ID."""
        self._nodes.pop(ref_doc_id, None)

    def query(
        self,
        query_embedding: List[float],
        similarity_top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[NodeWithScore]:
        """Query the index with a vector embedding."""
        scored: List[Tuple[str, float]] = []

        for node_id, node in self._nodes.items():
            if node.embedding is None:
                continue
            if filters and not self._matches_filters(node.metadata, filters):
                continue
            dist = self._cosine_distance(query_embedding, node.embedding)
            score = 1.0 - dist  # convert distance to similarity
            scored.append((node_id, score))

        scored.sort(key=lambda x: -x[1])  # highest score first
        results: List[NodeWithScore] = []
        for node_id, score in scored[:similarity_top_k]:
            results.append(NodeWithScore(node=self._nodes[node_id], score=score))
        return results

    def get_by_id(self, node_id: str) -> Optional[TextNode]:
        """Get a node by ID."""
        return self._nodes.get(node_id)

    def query_with_mmr(
        self,
        query_embedding: List[float],
        similarity_top_k: int = 10,
        mmr_threshold: float = 0.5,
        fetch_k: int = 20,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[NodeWithScore]:
        """Query with Maximal Marginal Relevance for diverse results.

        Args:
            query_embedding: Query vector.
            similarity_top_k: Final number of results.
            mmr_threshold: Balance relevance (1.0) vs diversity (0.0).
            fetch_k: Candidates to fetch before MMR re-ranking.
            filters: Optional metadata filters.
        """
        candidates = self.query(query_embedding, fetch_k, filters=filters)
        if not candidates:
            return []

        selected: List[int] = []
        for _ in range(min(similarity_top_k, len(candidates))):
            best_idx = -1
            best_score = float("-inf")
            for i, nws in enumerate(candidates):
                if i in selected or nws.node.embedding is None:
                    continue
                relevance = nws.score
                max_sim = 0.0
                for j in selected:
                    other = candidates[j]
                    if other.node.embedding is not None:
                        sim = 1.0 - self._cosine_distance(nws.node.embedding, other.node.embedding)
                        max_sim = max(max_sim, sim)
                mmr_score = mmr_threshold * relevance - (1.0 - mmr_threshold) * max_sim
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
            if best_idx >= 0:
                selected.append(best_idx)

        return [candidates[i] for i in selected]

    # ── Async interface ─────────────────────────────────────────────────

    async def aadd(self, nodes: List[TextNode], **kwargs: Any) -> List[str]:
        """Async version of add."""
        return self.add(nodes, **kwargs)

    async def adelete(self, ref_doc_id: str, **kwargs: Any) -> None:
        """Async version of delete."""
        self.delete(ref_doc_id, **kwargs)

    async def aquery(
        self,
        query_embedding: List[float],
        similarity_top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[NodeWithScore]:
        """Async version of query."""
        return self.query(query_embedding, similarity_top_k, filters=filters, **kwargs)

    async def aquery_with_mmr(
        self,
        query_embedding: List[float],
        similarity_top_k: int = 10,
        mmr_threshold: float = 0.5,
        fetch_k: int = 20,
        **kwargs: Any,
    ) -> List[NodeWithScore]:
        """Async version of query_with_mmr."""
        return self.query_with_mmr(
            query_embedding, similarity_top_k, mmr_threshold=mmr_threshold,
            fetch_k=fetch_k, **kwargs,
        )

    @property
    def count(self) -> int:
        return len(self._nodes)

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
    def _matches_filters(meta: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        for key, value in filters.items():
            if meta.get(key) != value:
                return False
        return True


class DocumentChunker:
    """Simple document chunker for LlamaIndex integration."""

    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, doc_id: Optional[str] = None) -> List[TextNode]:
        """Split text into overlapping chunks as TextNodes."""
        nodes: List[TextNode] = []
        start = 0
        idx = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]
            node_id = f"{doc_id or 'doc'}_{idx}" if doc_id else str(uuid.uuid4())
            nodes.append(TextNode(
                text=chunk_text,
                id_=node_id,
                metadata={"source": doc_id or "unknown", "chunk_index": idx},
            ))
            start += self.chunk_size - self.overlap
            idx += 1
        return nodes


__all__ = [
    "DocumentChunker",
    "NeedleIndexConfig",
    "NeedleVectorStoreIndex",
    "NodeWithScore",
    "TextNode",
]
