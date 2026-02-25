"""
Needle DB — High-level Python client for the Needle vector database.

An embedded vector database ("SQLite for vectors") with:
- HNSW approximate nearest neighbor search
- Single-file storage
- MongoDB-style metadata filtering
- Agentic memory (remember/recall/forget)

Quick Start::

    import needle_db as needle

    # In-memory (ephemeral)
    client = needle.Client()

    # Or file-backed (persistent)
    client = needle.Client("vectors.needle")

    # Create a collection
    collection = client.get_or_create_collection("docs", dimensions=384)

    # Insert vectors
    collection.add(
        ids=["doc1", "doc2"],
        vectors=[[0.1] * 384, [0.2] * 384],
        metadatas=[{"title": "Hello"}, {"title": "World"}],
    )

    # Search
    results = collection.query(query_vectors=[[0.1] * 384], n_results=5)
    print(results.ids)       # [["doc1", "doc2"]]
    print(results.distances) # [[0.0, 0.123...]]

    # Context manager for auto-save
    with needle.Client("data.needle") as client:
        coll = client.get_or_create_collection("docs", dimensions=128)
        coll.add(ids=["a"], vectors=[[0.5] * 128])
        # Automatically saved on exit

Installation::

    pip install needle-db          # Pure Python wrapper
    pip install needle-db[native]  # With Rust-native backend

    # Or build from source:
    cd needle && maturin develop --features python
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = [
    "Client",
    "Collection",
    "QueryResult",
    "MemoryStore",
    "backend",
]

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

# Try to import the native Rust backend (built with maturin/PyO3)
try:
    from needle import NeedleDatabase, NeedleCollection
    _HAS_NATIVE = True
    _BACKEND = "rust-native"
except ImportError:
    _HAS_NATIVE = False
    _BACKEND = "pure-python"


def backend() -> str:
    """Return the active backend: ``'rust-native'`` or ``'pure-python'``."""
    return _BACKEND


# ── Result Types ─────────────────────────────────────────────────────────────

@dataclass
class QueryResult:
    """Results from a vector query.

    Attributes:
        ids: List of ID lists, one per query vector.
        distances: List of distance lists, one per query vector.
        metadatas: List of metadata dicts, one per query vector.

    Example::

        results = collection.query(query_vectors=[[0.1] * 128], n_results=5)
        for doc_id, dist in zip(results.ids[0], results.distances[0]):
            print(f"{doc_id}: {dist:.4f}")
    """

    ids: List[List[str]] = field(default_factory=list)
    distances: List[List[float]] = field(default_factory=list)
    metadatas: List[List[Optional[Dict[str, Any]]]] = field(default_factory=list)

    def __len__(self) -> int:
        """Total number of results across all queries."""
        return sum(len(ids) for ids in self.ids)

    def __bool__(self) -> bool:
        return len(self) > 0

    def __repr__(self) -> str:
        return f"QueryResult(n_results={len(self)})"

    def flatten(self) -> List[Tuple[str, float, Optional[Dict[str, Any]]]]:
        """Flatten results into a single list of ``(id, distance, metadata)`` tuples."""
        out: List[Tuple[str, float, Optional[Dict[str, Any]]]] = []
        for ids, dists, metas in zip(self.ids, self.distances, self.metadatas):
            for i, (doc_id, dist) in enumerate(zip(ids, dists)):
                meta = metas[i] if i < len(metas) else None
                out.append((doc_id, dist, meta))
        return out


# ── Collection ───────────────────────────────────────────────────────────────

class Collection:
    """A named vector collection with HNSW indexing.

    Do not instantiate directly — use :meth:`Client.create_collection`
    or :meth:`Client.get_or_create_collection`.

    Example::

        coll = client.get_or_create_collection("docs", dimensions=384)
        coll.add(ids=["d1"], vectors=[[0.1] * 384])
        results = coll.query(query_vectors=[[0.1] * 384], n_results=5)
        print(coll.count)  # 1
    """

    def __init__(
        self,
        name: str,
        dimensions: int,
        distance: str = "cosine",
        _native: Any = None,
    ) -> None:
        self.name: str = name
        self.dimensions: int = dimensions
        self.distance: str = distance
        self._native = _native

    @property
    def count(self) -> int:
        """Number of vectors in the collection."""
        if self._native is not None:
            return self._native.len()
        return 0

    def add(
        self,
        ids: Sequence[str],
        vectors: Sequence[Sequence[float]],
        metadatas: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
    ) -> None:
        """Insert vectors with optional metadata.

        Args:
            ids: Unique identifiers for each vector.
            vectors: Vector data (must match collection dimensions).
            metadatas: Optional metadata dicts, one per vector.

        Raises:
            ValueError: If ``ids`` and ``vectors`` have different lengths.
        """
        if len(ids) != len(vectors):
            raise ValueError(
                f"ids ({len(ids)}) and vectors ({len(vectors)}) must have same length"
            )
        if self._native is not None:
            for i, (vid, vec) in enumerate(zip(ids, vectors)):
                meta = metadatas[i] if metadatas and i < len(metadatas) else None
                self._native.insert(vid, list(vec), meta)

    def query(
        self,
        query_vectors: Sequence[Sequence[float]],
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> QueryResult:
        """Search for similar vectors.

        Args:
            query_vectors: One or more query vectors.
            n_results: Maximum results per query (default 10).
            where: MongoDB-style metadata filter.

        Returns:
            :class:`QueryResult` with ids, distances, and metadatas.
        """
        result = QueryResult()
        if self._native is not None:
            for qvec in query_vectors:
                if where:
                    hits = self._native.search_with_filter(list(qvec), n_results, where)
                else:
                    hits = self._native.search(list(qvec), n_results)
                result.ids.append([h.id for h in hits])
                result.distances.append([h.distance for h in hits])
                result.metadatas.append(
                    [getattr(h, "metadata", None) for h in hits]
                )
        return result

    def get(self, ids: Sequence[str]) -> Dict[str, Any]:
        """Retrieve vectors by ID.

        Returns:
            Dict with keys ``ids``, ``vectors``, ``metadatas``.
        """
        r_ids: List[str] = []
        r_vecs: List[List[float]] = []
        r_meta: List[Optional[Dict[str, Any]]] = []
        if self._native is not None:
            for vid in ids:
                entry = self._native.get(vid)
                if entry is not None:
                    vec, meta = entry
                    r_ids.append(vid)
                    r_vecs.append(list(vec))
                    r_meta.append(meta)
        return {"ids": r_ids, "vectors": r_vecs, "metadatas": r_meta}

    def delete(self, ids: Sequence[str]) -> int:
        """Delete vectors by ID. Returns number of vectors deleted."""
        deleted = 0
        if self._native is not None:
            for vid in ids:
                try:
                    self._native.delete(vid)
                    deleted += 1
                except Exception:
                    pass
        return deleted

    def __contains__(self, vector_id: str) -> bool:
        """Check if a vector ID exists: ``'doc1' in collection``."""
        if self._native is not None:
            return self._native.get(vector_id) is not None
        return False

    def __iter__(self) -> Iterator[Tuple[str, List[float], Optional[Dict[str, Any]]]]:
        """Iterate over all vectors as ``(id, vector, metadata)`` tuples."""
        if self._native is not None:
            try:
                entries = self._native.export_all()
                for entry_id, vec, meta in entries:
                    yield entry_id, list(vec), meta
            except Exception:
                return

    def __len__(self) -> int:
        return self.count

    def __repr__(self) -> str:
        return (
            f"Collection(name='{self.name}', "
            f"dimensions={self.dimensions}, count={self.count})"
        )


# ── Agentic Memory ───────────────────────────────────────────────────────────

class MemoryStore:
    """Agentic memory interface — store and recall memories for AI agents.

    Built on top of a regular collection with enriched metadata for
    memory tier, importance, and session scoping.

    Example::

        client = needle.Client()
        coll = client.get_or_create_collection("memories", dimensions=128)
        memory = needle.MemoryStore(coll)

        memory.remember("User prefers dark mode", vector=[0.1]*128, importance=0.9)
        results = memory.recall(query=[0.1]*128, k=5, tier="semantic")
        memory.forget(results[0]["memory_id"])
    """

    def __init__(self, collection: Collection) -> None:
        self.collection = collection

    def remember(
        self,
        content: str,
        vector: Sequence[float],
        tier: str = "episodic",
        importance: float = 0.5,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store a memory. Returns the memory ID."""
        memory_id = f"mem_{int(time.time() * 1000)}"
        meta: Dict[str, Any] = metadata.copy() if metadata else {}
        meta["_memory_content"] = content
        meta["_memory_tier"] = tier
        meta["_memory_importance"] = importance
        meta["_memory_timestamp"] = time.time()
        if session_id:
            meta["_memory_session"] = session_id

        self.collection.add(
            ids=[memory_id],
            vectors=[list(vector)],
            metadatas=[meta],
        )
        return memory_id

    def recall(
        self,
        query: Sequence[float],
        k: int = 5,
        tier: Optional[str] = None,
        min_importance: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Recall memories similar to the query vector."""
        where = None
        if tier or min_importance is not None:
            conditions: Dict[str, Any] = {}
            if tier:
                conditions["_memory_tier"] = {"$eq": tier}
            if min_importance is not None:
                conditions["_memory_importance"] = {"$gte": min_importance}
            where = conditions if len(conditions) == 1 else {"$and": [{k: v} for k, v in conditions.items()]}

        results = self.collection.query(
            query_vectors=[list(query)], n_results=k, where=where
        )

        memories: List[Dict[str, Any]] = []
        if results.ids:
            for i, mid in enumerate(results.ids[0]):
                meta = results.metadatas[0][i] if results.metadatas[0] else {}
                memories.append({
                    "memory_id": mid,
                    "distance": results.distances[0][i],
                    "content": (meta or {}).get("_memory_content"),
                    "tier": (meta or {}).get("_memory_tier"),
                    "importance": (meta or {}).get("_memory_importance"),
                })
        return memories

    def forget(self, memory_id: str) -> bool:
        """Delete a specific memory. Returns True if it existed."""
        return self.collection.delete([memory_id]) > 0


# ── Client ───────────────────────────────────────────────────────────────────

class Client:
    """High-level Needle database client.

    Supports context manager for automatic save-on-exit::

        with needle.Client("data.needle") as client:
            coll = client.get_or_create_collection("docs", dimensions=128)
            coll.add(ids=["a"], vectors=[[0.5] * 128])
        # Automatically saved

    Args:
        path: Database file path, or ``":memory:"`` for ephemeral storage.
    """

    def __init__(self, path: str = ":memory:") -> None:
        self.path: str = path
        self._collections: Dict[str, Collection] = {}
        if _HAS_NATIVE:
            self._db = (
                NeedleDatabase.in_memory()
                if path == ":memory:"
                else NeedleDatabase.open(path)
            )
        else:
            self._db = None

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.save()

    def create_collection(
        self,
        name: str,
        dimensions: int,
        distance: str = "cosine",
    ) -> Collection:
        """Create a new collection.

        Raises:
            RuntimeError: If the collection already exists (use
            :meth:`get_or_create_collection` instead).
        """
        native = None
        if self._db is not None:
            self._db.create_collection(name, dimensions)
            native = self._db.collection(name)
        coll = Collection(name, dimensions, distance, _native=native)
        self._collections[name] = coll
        return coll

    def get_or_create_collection(
        self,
        name: str,
        dimensions: int,
        distance: str = "cosine",
    ) -> Collection:
        """Get an existing collection or create a new one."""
        if name in self._collections:
            return self._collections[name]
        if self._db is not None:
            try:
                native = self._db.collection(name)
                coll = Collection(name, dimensions, distance, _native=native)
                self._collections[name] = coll
                return coll
            except Exception:
                pass
        return self.create_collection(name, dimensions, distance)

    def get_collection(self, name: str) -> Optional[Collection]:
        """Get an existing collection by name, or ``None``."""
        if name in self._collections:
            return self._collections[name]
        if self._db is not None:
            try:
                native = self._db.collection(name)
                coll = Collection(name, 0, "cosine", _native=native)
                self._collections[name] = coll
                return coll
            except Exception:
                return None
        return None

    def list_collections(self) -> List[str]:
        """List all collection names."""
        if self._db is not None:
            return self._db.list_collections()
        return list(self._collections.keys())

    def delete_collection(self, name: str) -> None:
        """Delete a collection and all its vectors."""
        self._collections.pop(name, None)
        if self._db is not None:
            self._db.delete_collection(name)

    def memory(self, collection_name: str, dimensions: int) -> MemoryStore:
        """Create an agentic memory store backed by the given collection.

        Example::

            mem = client.memory("agent_memory", dimensions=384)
            mem.remember("User likes Python", vector=[0.1]*384)
            results = mem.recall(query=[0.1]*384, k=5)
        """
        coll = self.get_or_create_collection(collection_name, dimensions)
        return MemoryStore(coll)

    def save(self) -> None:
        """Persist all changes to disk (no-op for in-memory databases)."""
        if self._db is not None:
            try:
                self._db.save()
            except Exception:
                pass  # In-memory databases can't save

    def __repr__(self) -> str:
        return f"Client(path='{self.path}', collections={self.list_collections()})"
