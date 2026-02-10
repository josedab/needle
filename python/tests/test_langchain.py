"""Tests for needle-langchain integration."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from needle_langchain import NeedleVectorStore, Document


def test_add_and_search():
    store = NeedleVectorStore(dimensions=4)
    vec_a = [1.0, 0.0, 0.0, 0.0]
    vec_b = [0.0, 1.0, 0.0, 0.0]

    ids = store.add_vectors(
        vectors=[vec_a, vec_b],
        documents=[
            Document(page_content="hello", metadata={"k": "a"}),
            Document(page_content="world", metadata={"k": "b"}),
        ],
        ids=["a", "b"],
    )
    assert ids == ["a", "b"]
    assert store.count == 2

    results = store.similarity_search_with_score(vec_a, k=2)
    assert len(results) == 2
    assert results[0][0].page_content == "hello"
    assert results[0][1] < results[1][1]  # closer first


def test_delete():
    store = NeedleVectorStore(dimensions=4)
    store.add_vectors(
        vectors=[[1.0, 0.0, 0.0, 0.0]],
        documents=[Document(page_content="x")],
        ids=["x"],
    )
    assert store.count == 1
    store.delete(["x"])
    assert store.count == 0


def test_get_by_id():
    store = NeedleVectorStore(dimensions=4)
    store.add_vectors(
        vectors=[[0.5, 0.5, 0.0, 0.0]],
        documents=[Document(page_content="test", metadata={"a": 1})],
        ids=["t1"],
    )
    doc = store.get_by_id("t1")
    assert doc is not None
    assert doc.page_content == "test"
    assert store.get_by_id("missing") is None


def test_filter():
    store = NeedleVectorStore(dimensions=4)
    store.add_vectors(
        vectors=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        documents=[
            Document(page_content="a", metadata={"color": "red"}),
            Document(page_content="b", metadata={"color": "blue"}),
        ],
        ids=["r", "b"],
    )
    results = store.similarity_search_with_score(
        [1.0, 0.0, 0.0, 0.0], k=2, filter={"color": "red"}
    )
    assert len(results) == 1
    assert results[0][0].metadata["color"] == "red"


if __name__ == "__main__":
    test_add_and_search()
    test_delete()
    test_get_by_id()
    test_filter()
    print("All langchain tests passed!")
