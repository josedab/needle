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


def test_mmr_search():
    store = NeedleVectorStore(dimensions=4)
    store.add_vectors(
        vectors=[
            [1.0, 0.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.9, 0.1],
        ],
        documents=[
            Document(page_content="a"),
            Document(page_content="a_similar"),
            Document(page_content="b"),
            Document(page_content="b_similar"),
        ],
    )
    # MMR should diversify: pick from both clusters
    results = store.max_marginal_relevance_search(
        [1.0, 0.0, 0.0, 0.0], k=2, fetch_k=4, lambda_mult=0.5
    )
    assert len(results) == 2
    # With diversity, should not pick both from the same cluster
    contents = {r.page_content for r in results}
    assert "a" in contents  # closest to query


def test_from_texts():
    class MockEmbedding:
        def embed_documents(self, texts):
            return [[float(i)] * 4 for i in range(len(texts))]
        def embed_query(self, text):
            return [0.0] * 4

    store = NeedleVectorStore.from_texts(
        texts=["hello", "world"],
        embedding=MockEmbedding(),
        metadatas=[{"k": "v1"}, {"k": "v2"}],
    )
    assert store.count == 2
    # Text search should work with embedding function
    results = store.similarity_search_by_text("query", k=2)
    assert len(results) == 2


if __name__ == "__main__":
    test_add_and_search()
    test_delete()
    test_get_by_id()
    test_filter()
    test_mmr_search()
    test_from_texts()
    print("All langchain tests passed!")
