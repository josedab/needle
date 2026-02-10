"""Tests for needle-llamaindex integration."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from needle_llamaindex import NeedleVectorStoreIndex, TextNode, DocumentChunker


def test_add_and_query():
    index = NeedleVectorStoreIndex(dimensions=4)
    nodes = [
        TextNode(text="hello", id_="n1", embedding=[1.0, 0.0, 0.0, 0.0]),
        TextNode(text="world", id_="n2", embedding=[0.0, 1.0, 0.0, 0.0]),
    ]
    ids = index.add(nodes)
    assert ids == ["n1", "n2"]
    assert index.count == 2

    results = index.query([1.0, 0.0, 0.0, 0.0], similarity_top_k=2)
    assert len(results) == 2
    assert results[0].node.text == "hello"
    assert results[0].score > results[1].score


def test_delete():
    index = NeedleVectorStoreIndex(dimensions=4)
    index.add([TextNode(text="x", id_="x1", embedding=[1.0, 0.0, 0.0, 0.0])])
    assert index.count == 1
    index.delete("x1")
    assert index.count == 0


def test_get_by_id():
    index = NeedleVectorStoreIndex(dimensions=4)
    index.add([TextNode(text="test", id_="t1", embedding=[0.5, 0.5, 0.0, 0.0])])
    node = index.get_by_id("t1")
    assert node is not None
    assert node.text == "test"
    assert index.get_by_id("missing") is None


def test_document_chunker():
    chunker = DocumentChunker(chunk_size=20, overlap=5)
    text = "This is a long document that should be split into multiple chunks."
    nodes = chunker.chunk(text, doc_id="doc1")
    assert len(nodes) > 1
    assert all(n.metadata["source"] == "doc1" for n in nodes)
    assert nodes[0].text == text[:20]


def test_query_with_filter():
    index = NeedleVectorStoreIndex(dimensions=4)
    index.add([
        TextNode(text="a", id_="n1", embedding=[1.0, 0.0, 0.0, 0.0], metadata={"category": "x"}),
        TextNode(text="b", id_="n2", embedding=[0.0, 1.0, 0.0, 0.0], metadata={"category": "y"}),
    ])
    results = index.query(
        [1.0, 0.0, 0.0, 0.0],
        similarity_top_k=2,
        filters={"category": "x"},
    )
    assert len(results) == 1
    assert results[0].node.metadata["category"] == "x"


if __name__ == "__main__":
    test_add_and_query()
    test_delete()
    test_get_by_id()
    test_document_chunker()
    test_query_with_filter()
    print("All llamaindex tests passed!")
