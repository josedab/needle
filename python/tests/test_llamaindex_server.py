"""Tests for needle-llamaindex server-backed integration.

Uses a lightweight mock HTTP server to verify the HTTP client works
correctly without needing a real Needle server.
"""

import json
import sys
import os
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from needle_llamaindex import NeedleVectorStoreIndex, NeedleHttpError, TextNode


# ── Mock Needle Server ────────────────────────────────────────────────────────

class _MockNeedleHandler(BaseHTTPRequestHandler):
    """Minimal mock of the Needle REST API."""

    collections: dict = {}
    vectors: dict = {}

    def log_message(self, *_args):
        pass

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
        parts = self.path.split("/")

        if self.path == "/v1/collections":
            name = data.get("name", "")
            self.collections[name] = data
            self.vectors.setdefault(name, {})
            self._send_json(201, {"name": name})
            return

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

        if len(parts) == 5 and parts[4] == "vectors":
            col = parts[3]
            self.vectors.setdefault(col, {})
            vid = data.get("id", "")
            self.vectors[col][vid] = {
                "vector": data.get("vector", []),
                "metadata": data.get("metadata", {}),
            }
            self._send_json(201, {"id": vid})
            return

        if len(parts) == 5 and parts[4] == "search":
            col = parts[3]
            k = data.get("k", 10)
            results = []
            for vid, vdata in list(self.vectors.get(col, {}).items())[:k]:
                results.append({
                    "id": vid,
                    "distance": 0.05,
                    "metadata": vdata.get("metadata", {}),
                })
            self._send_json(200, {"results": results})
            return

        self._send_error(404, "Not found")

    def do_GET(self):
        parts = self.path.split("/")
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
                self._send_error(404, "Not found")
            return
        self._send_error(404, "Not found")

    def do_DELETE(self):
        parts = self.path.split("/")
        if len(parts) == 6 and parts[4] == "vectors":
            col, vid = parts[3], parts[5]
            if col in self.vectors and vid in self.vectors[col]:
                del self.vectors[col][vid]
                self._send_json(200, {})
            else:
                self._send_error(404, "Not found")
            return
        self._send_error(404, "Not found")


def _start_mock_server():
    _MockNeedleHandler.collections = {}
    _MockNeedleHandler.vectors = {}
    server = HTTPServer(("127.0.0.1", 0), _MockNeedleHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return f"http://127.0.0.1:{port}", server


# ── Tests ────────────────────────────────────────────────────────────────────

def test_server_backed_add_and_query():
    url, server = _start_mock_server()
    try:
        index = NeedleVectorStoreIndex(
            collection_name="test_idx",
            dimensions=4,
            server_url=url,
        )
        nodes = [
            TextNode(text="hello", id_="n1", embedding=[1.0, 0.0, 0.0, 0.0]),
            TextNode(text="world", id_="n2", embedding=[0.0, 1.0, 0.0, 0.0]),
        ]
        ids = index.add(nodes)
        assert ids == ["n1", "n2"]

        results = index.query([1.0, 0.0, 0.0, 0.0], similarity_top_k=2)
        assert len(results) == 2
        assert results[0].score > 0  # distance 0.05 → score 0.95
    finally:
        server.shutdown()


def test_server_backed_delete():
    url, server = _start_mock_server()
    try:
        index = NeedleVectorStoreIndex(
            collection_name="test_idx",
            dimensions=4,
            server_url=url,
        )
        index.add([TextNode(text="x", id_="x1", embedding=[1.0, 0.0, 0.0, 0.0])])
        index.delete("x1")
        # Deleting again should not raise (just returns None)
        index.delete("x1")
    finally:
        server.shutdown()


def test_server_backed_get_by_id():
    url, server = _start_mock_server()
    try:
        index = NeedleVectorStoreIndex(
            collection_name="test_idx",
            dimensions=4,
            server_url=url,
        )
        index.add([TextNode(text="doc", id_="d1", embedding=[0.5, 0.5, 0.0, 0.0],
                            metadata={"tag": "test"})])
        node = index.get_by_id("d1")
        assert node is not None
        assert node.text == "doc"
        assert index.get_by_id("missing") is None
    finally:
        server.shutdown()


def test_server_backed_from_nodes():
    url, server = _start_mock_server()
    try:
        nodes = [
            TextNode(text="a", id_="a", embedding=[1.0, 0.0, 0.0, 0.0]),
            TextNode(text="b", id_="b", embedding=[0.0, 1.0, 0.0, 0.0]),
        ]
        index = NeedleVectorStoreIndex.from_nodes(
            nodes, collection_name="from_nodes", server_url=url
        )
        results = index.query([1.0, 0.0, 0.0, 0.0], similarity_top_k=1)
        assert len(results) == 1
    finally:
        server.shutdown()


def test_http_error_propagation():
    """Verify NeedleHttpError is raised for server errors."""

    class _ErrorHandler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            body = json.dumps({"error": "Internal failure"}).encode()
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
            NeedleVectorStoreIndex(
                collection_name="fail",
                dimensions=4,
                server_url=f"http://127.0.0.1:{port}",
            )
        except NeedleHttpError as exc:
            raised = True
            assert exc.status_code == 500
            assert "Internal failure" in exc.body
        assert raised, "Expected NeedleHttpError"
    finally:
        server.shutdown()


def test_connection_refused():
    """Verify NeedleHttpError is raised when server is unreachable."""
    raised = False
    try:
        NeedleVectorStoreIndex(
            collection_name="unreachable",
            dimensions=4,
            server_url="http://127.0.0.1:1",
        )
    except NeedleHttpError as exc:
        raised = True
        assert exc.status_code == 503
        assert "Connection failed" in exc.body or "refused" in exc.body.lower()
    assert raised, "Expected NeedleHttpError for connection refused"


if __name__ == "__main__":
    test_server_backed_add_and_query()
    test_server_backed_delete()
    test_server_backed_get_by_id()
    test_server_backed_from_nodes()
    test_http_error_propagation()
    test_connection_refused()
    print("All llamaindex server tests passed!")
