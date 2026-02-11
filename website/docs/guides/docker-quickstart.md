---
sidebar_position: 7
---

# Docker & HTTP Quickstart

Get a working Needle HTTP server running in minutes — no Rust required.

## Option A: Docker (Fastest)

```bash
# Pull and run
docker run -d --name needle -p 8080:8080 ghcr.io/anthropics/needle:latest

# Verify it's running
curl http://127.0.0.1:8080/health
```

## Option B: Docker Compose

```bash
git clone https://github.com/anthropics/needle.git
cd needle
docker compose --profile demo up -d --build
```

## Option C: Build from Source

```bash
cargo run --features server -- serve -a 127.0.0.1:8080
```

---

## Step-by-Step: Your First Search

### 1. Create a Collection

```bash
curl -X POST http://127.0.0.1:8080/collections \
  -H "Content-Type: application/json" \
  -d '{"name": "docs", "dimensions": 3}'
```

### 2. Insert Vectors

```bash
curl -X POST http://127.0.0.1:8080/collections/docs/vectors \
  -H "Content-Type: application/json" \
  -d '{"id": "doc1", "vector": [0.1, 0.2, 0.3], "metadata": {"title": "Hello Needle"}}'
```

```bash
curl -X POST http://127.0.0.1:8080/collections/docs/vectors \
  -H "Content-Type: application/json" \
  -d '{"id": "doc2", "vector": [0.9, 0.1, 0.0], "metadata": {"title": "Second Doc"}}'
```

### 3. Search

```bash
curl -X POST http://127.0.0.1:8080/collections/docs/search \
  -H "Content-Type: application/json" \
  -d '{"vector": [0.1, 0.2, 0.3], "k": 2}'
```

You'll get results ranked by distance:

```json
{
  "results": [
    {"id": "doc1", "distance": 0.0, "metadata": {"title": "Hello Needle"}},
    {"id": "doc2", "distance": 0.72, "metadata": {"title": "Second Doc"}}
  ]
}
```

### 4. Clean Up

```bash
# Delete individual vectors
curl -X DELETE http://127.0.0.1:8080/collections/docs/vectors/doc1
curl -X DELETE http://127.0.0.1:8080/collections/docs/vectors/doc2

# Delete the collection
curl -X DELETE http://127.0.0.1:8080/collections/docs
```

---

## Full API Reference

Once your server is running, these endpoints are available:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/collections` | List all collections |
| `POST` | `/collections` | Create a collection |
| `GET` | `/collections/:name` | Get collection info |
| `DELETE` | `/collections/:name` | Delete a collection |
| `POST` | `/collections/:name/vectors` | Insert a vector |
| `GET` | `/collections/:name/vectors/:id` | Get vector by ID |
| `DELETE` | `/collections/:name/vectors/:id` | Delete a vector |
| `POST` | `/collections/:name/search` | Search similar vectors |
| `POST` | `/collections/:name/compact` | Compact collection |
| `GET` | `/collections/:name/export` | Export collection |
| `POST` | `/save` | Persist database to disk |

### Search with Filters

```bash
curl -X POST http://127.0.0.1:8080/collections/docs/search \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.1, 0.2, 0.3],
    "k": 10,
    "filter": {"title": {"$eq": "Hello Needle"}}
  }'
```

### Search with Explain

```bash
curl -X POST http://127.0.0.1:8080/collections/docs/search \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.1, 0.2, 0.3],
    "k": 10,
    "explain": true
  }'
```

---

## Next Steps

- [Getting Started](/docs/getting-started) — Full Rust API tutorial
- [HTTP Server Reference](/docs/advanced/http-server) — TLS, auth, and advanced config
- [Production Deployment](/docs/guides/production) — Go live with confidence
