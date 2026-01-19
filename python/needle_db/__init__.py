"""
Needle DB — High-level Python client for the Needle vector database.

Provides a ChromaDB-style API for vector operations:

    >>> import needle_db as needle
    >>> client = needle.Client("vectors.needle")
    >>> collection = client.get_or_create_collection("docs", dimensions=384)
    >>> collection.add(ids=["doc1"], vectors=[[0.1] * 384], metadatas=[{"title": "Hello"}])
    >>> results = collection.query(query_vectors=[[0.1] * 384], n_results=5)

Installation:
    pip install needle-db          # Pure Python wrapper
    pip install needle-db[native]  # With Rust-native backend (requires maturin build)

    # Or build from source with native backend:
    cd needle && maturin develop --features python
"""

from __future__ import annotations

__version__ = "0.1.0"

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Try to import the native Rust backend (built with maturin/PyO3)
try:
    from needle import NeedleDatabase, NeedleCollection
    _HAS_NATIVE = True
    _BACKEND = "rust-native"
except ImportError:
    _HAS_NATIVE = False
    _BACKEND = "pure-python"


def backend() -> str:
    """Return the active backend: 'rust-native' or 'pure-python'."""
    return _BACKEND


@dataclass
class QueryResult:
    """Results from a vector query."""
    ids: List[List[str]] = field(default_factory=list)
    distances: List[List[float]] = field(default_factory=list)
    metadatas: List[List[Optional[Dict[str, Any]]]] = field(default_factory=list)

    def __repr__(self) -> str:
        n = sum(len(ids) for ids in self.ids)
        return f"QueryResult(n_results={n})"


class Collection:
    """High-level collection interface with ChromaDB-style API."""

    def __init__(self, name: str, dimensions: int, distance: str = "cosine",
                 _native=None):
        self.name = name
        self.dimensions = dimensions
        self.distance = distance
        self._native = _native

    @property
    def count(self) -> int:
        if self._native is not None:
            return self._native.len()
        return 0

    def add(self, ids: List[str], vectors: List[List[float]],
            metadatas: Optional[List[Optional[Dict[str, Any]]]] = None) -> None:
        """Insert vectors with optional metadata."""
        if len(ids) != len(vectors):
            raise ValueError(
                f"ids ({len(ids)}) and vectors ({len(vectors)}) must have same length"
            )
        if self._native is not None:
            for i, (vid, vec) in enumerate(zip(ids, vectors)):
                meta = metadatas[i] if metadatas and i < len(metadatas) else None
                self._native.insert(vid, vec, meta)

    def query(self, query_vectors: List[List[float]], n_results: int = 10,
              where: Optional[Dict[str, Any]] = None) -> QueryResult:
        """Search for similar vectors."""
        result = QueryResult()
        if self._native is not None:
            for qvec in query_vectors:
                if where:
                    hits = self._native.search_with_filter(qvec, n_results, where)
                else:
                    hits = self._native.search(qvec, n_results)
                result.ids.append([h.id for h in hits])
                result.distances.append([h.distance for h in hits])
                result.metadatas.append(
                    [getattr(h, "metadata", None) for h in hits]
                )
        return result

    def get(self, ids: List[str]) -> Dict[str, Any]:
        """Retrieve vectors by ID."""
        r_ids, r_vecs, r_meta = [], [], []
        if self._native is not None:
            for vid in ids:
                entry = self._native.get(vid)
                if entry is not None:
                    vec, meta = entry
                    r_ids.append(vid)
                    r_vecs.append(list(vec))
                    r_meta.append(meta)
        return {"ids": r_ids, "vectors": r_vecs, "metadatas": r_meta}

    def delete(self, ids: List[str]) -> None:
        """Delete vectors by ID."""
        if self._native is not None:
            for vid in ids:
                self._native.delete(vid)

    def __repr__(self) -> str:
        return (f"Collection(name='{self.name}', "
                f"dimensions={self.dimensions}, count={self.count})")


class Client:
    """High-level Needle database client.

    Example::

        client = Client("vectors.needle")
        coll = client.get_or_create_collection("docs", dimensions=384)
        coll.add(ids=["d1"], vectors=[[0.1]*384])
        results = coll.query(query_vectors=[[0.1]*384], n_results=5)
    """

    def __init__(self, path: str = ":memory:"):
        self.path = path
        self._collections: Dict[str, Collection] = {}
        if _HAS_NATIVE:
            self._db = (NeedleDatabase.in_memory() if path == ":memory:"
                        else NeedleDatabase.open(path))
        else:
            self._db = None

    def create_collection(self, name: str, dimensions: int,
                          distance: str = "cosine") -> Collection:
        """Create a new collection."""
        native = None
        if self._db is not None:
            self._db.create_collection(name, dimensions)
            native = self._db.collection(name)
        coll = Collection(name, dimensions, distance, _native=native)
        self._collections[name] = coll
        return coll

    def get_or_create_collection(self, name: str, dimensions: int,
                                 distance: str = "cosine") -> Collection:
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

    def list_collections(self) -> List[str]:
        if self._db is not None:
            return self._db.list_collections()
        return list(self._collections.keys())

    def delete_collection(self, name: str) -> None:
        self._collections.pop(name, None)
        if self._db is not None:
            self._db.delete_collection(name)

    def save(self) -> None:
        if self._db is not None:
            self._db.save()

    def __repr__(self) -> str:
        return f"Client(path='{self.path}', collections={self.list_collections()})"
