"""Tests for needle-langchain server-backed integration.

Uses a lightweight mock HTTP server to verify the HTTP client works
correctly without needing a real Needle server.
"""

import json
import sys
import os
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from needle_langchain import NeedleVectorStore, NeedleHttpError, Document


# ── Mock Needle Server ────────────────────────────────────────────────────────

class _MockNeedleHandler(BaseHTTPRequestHandler):
    """Minimal mock of the Needle REST API."""

    # Shared state across requests
    collections: dict = {}
    vectors: dict = {}  # {collection: {id: {vector, metadata}}}

    def log_message(self, *_args):
        pass  # suppress request logs

    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0))
        if length:
            return json.loads(self.rfile.read(length))
        return {}

    def _send_json(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, code, message):
        self._send_json(code, {"error": message})

    def do_POST(self):
        data = self._read_json()

        if self.path == "/v1/collections":
            name = data.get("name", "")
            self.collections[name] = data
            self.vectors.setdefault(name, {})
            self._send_json(201, {"name": name, "dimensions": data.get("dimensions", 0)})
            return

        parts = self.path.split("/")
        # POST /v1/collections/{name}/vectors/batch
        if len(parts) == 6 and parts[4] == "vectors" and parts[5] == "batch":
            col = parts[3]
            self.vectors.setdefault(col, {})
            vecs = data.get("vectors", [])
            for v in vecs:
                vid = v.get("id", "")
                self.vectors[col][vid] = {
                    "vector": v.get("vector", []),
                    "metadata": v.get("metadata", {}),
                }
            self._send_json(200, {"inserted": len(vecs)})
            return

        # POST /v1/collections/{name}/vectors
        if len(parts) == 5 and parts[4] == "vectors":
            col = parts[3]
            if col not in self.vectors:
                self._send_error(404, f"Collection '{col}' not found")
                return
            vid = data.get("id", "")
            self.vectors[col][vid] = {
                "vector": data.get("vector", []),
                "metadata": data.get("metadata", {}),
            }
            self._send_json(201, {"id": vid})
            return

        # POST /v1/collections/{name}/search
        if len(parts) == 5 and parts[4] == "search":
            col = parts[3]
            if col not in self.vectors:
                self._send_error(404, f"Collection '{col}' not found")
                return
            k = data.get("k", 10)
            results = []
            for vid, vdata in list(self.vectors[col].items())[:k]:
                results.append({
                    "id": vid,
                    "distance": 0.1,
                    "metadata": vdata.get("metadata", {}),
                })
            self._send_json(200, {"results": results})
            return

        self._send_error(404, "Not found")

    def do_GET(self):
        parts = self.path.split("/")
        # GET /v1/collections/{name}/vectors/{id}
        if len(parts) == 6 and parts[4] == "vectors":
            col, vid = parts[3], parts[5]
            if col in self.vectors and vid in self.vectors[col]:
                vdata = self.vectors[col][vid]
                self._send_json(200, {
                    "id": vid,
                    "vector": vdata["vector"],
                    "metadata": vdata["metadata"],
                })
            else:
                self._send_error(404, "Vector not found")
            return
        self._send_error(404, "Not found")

    def do_DELETE(self):
        parts = self.path.split("/")
        # DELETE /v1/collections/{name}/vectors/{id}
        if len(parts) == 6 and parts[4] == "vectors":
            col, vid = parts[3], parts[5]
            if col in self.vectors and vid in self.vectors[col]:
                del self.vectors[col][vid]
                self._send_json(200, {})
            else:
                self._send_error(404, "Vector not found")
            return
        self._send_error(404, "Not found")


def _start_mock_server():
    """Start mock server on a random port, return (url, server)."""
    # Reset state
    _MockNeedleHandler.collections = {}
    _MockNeedleHandler.vectors = {}

    server = HTTPServer(("127.0.0.1", 0), _MockNeedleHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return f"http://127.0.0.1:{port}", server


# ── Tests ────────────────────────────────────────────────────────────────────

def test_server_backed_insert_and_search():
    url, server = _start_mock_server()
    try:
        store = NeedleVectorStore(
            collection_name="test_col",
            dimensions=4,
            server_url=url,
        )
        store.add_vectors(
            vectors=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            documents=[
                Document(page_content="hello", metadata={"k": "a"}),
                Document(page_content="world", metadata={"k": "b"}),
            ],
            ids=["a", "b"],
        )
        results = store.similarity_search_with_score([1.0, 0.0, 0.0, 0.0], k=2)
        assert len(results) == 2
        assert results[0][0].page_content in ("hello", "world")
    finally:
        server.shutdown()


def test_server_backed_delete():
    url, server = _start_mock_server()
    try:
        store = NeedleVectorStore(
            collection_name="test_col",
            dimensions=4,
            server_url=url,
        )
        store.add_vectors(
            vectors=[[1.0, 0.0, 0.0, 0.0]],
            documents=[Document(page_content="x")],
            ids=["x"],
        )
        assert store.delete(["x"])
        assert not store.delete(["nonexistent"])
    finally:
        server.shutdown()


def test_server_backed_get_by_id():
    url, server = _start_mock_server()
    try:
        store = NeedleVectorStore(
            collection_name="test_col",
            dimensions=4,
            server_url=url,
        )
        store.add_vectors(
            vectors=[[0.5, 0.5, 0.0, 0.0]],
            documents=[Document(page_content="test", metadata={"a": 1})],
            ids=["t1"],
        )
        doc = store.get_by_id("t1")
        assert doc is not None
        assert doc.page_content == "test"
        assert store.get_by_id("missing") is None
    finally:
        server.shutdown()


def test_server_backed_api_key_header():
    """Verify that api_key is sent as X-API-Key header."""
    received_headers = {}

    class _HeaderCapture(_MockNeedleHandler):
        def do_POST(self):
            received_headers.update(dict(self.headers))
            super().do_POST()

    _HeaderCapture.collections = {}
    _HeaderCapture.vectors = {}

    server = HTTPServer(("127.0.0.1", 0), _HeaderCapture)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        store = NeedleVectorStore(
            collection_name="secured",
            dimensions=4,
            server_url=f"http://127.0.0.1:{port}",
            api_key="test-key-123",
        )
        # The init creates the collection, which sends a POST with the key
        assert received_headers.get("X-Api-Key") == "test-key-123" or \
               received_headers.get("X-API-Key") == "test-key-123"
    finally:
        server.shutdown()


def test_http_error_handling():
    """Verify NeedleHttpError is raised for server errors."""
    class _ErrorHandler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            body = json.dumps({"error": "Something broke"}).encode()
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = HTTPServer(("127.0.0.1", 0), _ErrorHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        raised = False
        try:
            NeedleVectorStore(
                collection_name="fail",
                dimensions=4,
                server_url=f"http://127.0.0.1:{port}",
            )
        except NeedleHttpError as exc:
            raised = True
            assert exc.status_code == 500
            assert "Something broke" in exc.body
        assert raised, "Expected NeedleHttpError to be raised"
    finally:
        server.shutdown()


def test_connection_refused():
    """Verify NeedleHttpError is raised when server is unreachable."""
    raised = False
    try:
        NeedleVectorStore(
            collection_name="unreachable",
            dimensions=4,
            server_url="http://127.0.0.1:1",  # port 1 — nothing listening
        )
    except NeedleHttpError as exc:
        raised = True
        assert exc.status_code == 503
        assert "Connection failed" in exc.body or "refused" in exc.body.lower()
    assert raised, "Expected NeedleHttpError for connection refused"


def test_from_texts_with_server():
    """Verify from_texts passes server_url through."""
    url, server = _start_mock_server()
    try:
        class MockEmbed:
            def embed_documents(self, texts):
                return [[0.1] * 4 for _ in texts]
            def embed_query(self, text):
                return [0.1] * 4

        store = NeedleVectorStore.from_texts(
            texts=["alpha", "beta"],
            embedding=MockEmbed(),
            collection_name="from_texts_test",
            server_url=url,
        )
        results = store.similarity_search([0.1] * 4, k=2)
        assert len(results) == 2
    finally:
        server.shutdown()


if __name__ == "__main__":
    test_server_backed_insert_and_search()
    test_server_backed_delete()
    test_server_backed_get_by_id()
    test_server_backed_api_key_header()
    test_http_error_handling()
    test_connection_refused()
    test_from_texts_with_server()
    print("All langchain server tests passed!")
