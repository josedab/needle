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
        **kwargs: Any,
    ) -> None:
        self._config = NeedleIndexConfig(
            collection_name=collection_name,
            dimensions=dimensions,
            distance=distance,
        )
        self._nodes: Dict[str, TextNode] = {}

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
