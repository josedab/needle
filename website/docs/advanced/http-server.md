---
sidebar_position: 1
---

# HTTP Server

Needle includes a built-in HTTP REST API server for language-agnostic access to your vector database.

## Starting the Server

### Command Line

```bash
# Start server on default port
needle serve -d vectors.needle

# Custom address and port
needle serve -a 0.0.0.0:8080 -d vectors.needle

# With TLS
needle serve -a 0.0.0.0:8443 -d vectors.needle \
  --tls-cert /path/to/cert.pem \
  --tls-key /path/to/key.pem
```

### Programmatic

```rust
use needle::{Database, server};

#[tokio::main]
async fn main() -> needle::Result<()> {
    let db = Database::open("vectors.needle")?;

    server::run("0.0.0.0:8080", db).await?;

    Ok(())
}
```

## API Endpoints

### Health Check

```bash
GET /health
```

Response:
```json
{
  "status": "healthy",
  "version": "0.1.0"
}
```

### Collections

#### List Collections

```bash
GET /collections
```

Response:
```json
{
  "collections": ["documents", "images"]
}
```

#### Create Collection

```bash
POST /collections
Content-Type: application/json

{
  "name": "documents",
  "dimensions": 384,
  "distance": "cosine",
  "hnsw_m": 16,
  "hnsw_ef_construction": 200
}
```

Response:
```json
{
  "name": "documents",
  "dimensions": 384,
  "distance": "cosine",
  "count": 0
}
```

#### Get Collection Info

```bash
GET /collections/:name
```

Response:
```json
{
  "name": "documents",
  "dimensions": 384,
  "distance": "cosine",
  "count": 10000
}
```

#### Delete Collection

```bash
DELETE /collections/:name
```

Response:
```json
{
  "deleted": true
}
```

### Vectors

#### Insert Vector

```bash
POST /collections/:name/vectors
Content-Type: application/json

{
  "id": "doc1",
  "vector": [0.1, 0.2, 0.3, ...],
  "metadata": {
    "title": "Hello World",
    "category": "greeting"
  },
  "ttl_seconds": 3600
}
```

The `ttl_seconds` field is optional. If provided, the vector will automatically expire after the specified number of seconds.

Response:
```json
{
  "id": "doc1",
  "inserted": true
}
```

#### Batch Insert

```bash
POST /collections/:name/vectors/batch
Content-Type: application/json

{
  "vectors": [
    {
      "id": "doc1",
      "vector": [0.1, 0.2, ...],
      "metadata": {"title": "Doc 1"}
    },
    {
      "id": "doc2",
      "vector": [0.2, 0.3, ...],
      "metadata": {"title": "Doc 2"}
    }
  ]
}
```

Response:
```json
{
  "inserted": 2
}
```

#### Get Vector

```bash
GET /collections/:name/vectors/:id
```

Response:
```json
{
  "id": "doc1",
  "vector": [0.1, 0.2, 0.3, ...],
  "metadata": {
    "title": "Hello World"
  }
}
```

#### Delete Vector

```bash
DELETE /collections/:name/vectors/:id
```

Response:
```json
{
  "deleted": true
}
```

### Search

#### Basic Search

```bash
POST /collections/:name/search
Content-Type: application/json

{
  "vector": [0.1, 0.2, 0.3, ...],
  "k": 10
}
```

Response:
```json
{
  "results": [
    {
      "id": "doc1",
      "distance": 0.123,
      "metadata": {"title": "Hello World"}
    },
    {
      "id": "doc2",
      "distance": 0.456,
      "metadata": {"title": "Hello"}
    }
  ],
  "took_ms": 5
}
```

#### Search with Filter

```bash
POST /collections/:name/search
Content-Type: application/json

{
  "vector": [0.1, 0.2, 0.3, ...],
  "k": 10,
  "filter": {
    "category": "programming",
    "year": {"$gte": 2020}
  }
}
```

#### Search with Distance Override

Override the distance function at query time. Falls back to brute-force search if different from the collection's index.

```bash
POST /collections/:name/search
Content-Type: application/json

{
  "vector": [0.1, 0.2, 0.3, ...],
  "k": 10,
  "distance": "euclidean"
}
```

Available distance functions: `cosine`, `euclidean`, `dot`, `manhattan`

#### Search with Explain

```bash
POST /collections/:name/search
Content-Type: application/json

{
  "vector": [0.1, 0.2, 0.3, ...],
  "k": 10,
  "explain": true
}
```

Response:
```json
{
  "results": [...],
  "took_ms": 5,
  "explain": {
    "nodes_visited": 150,
    "distance_computations": 1234,
    "layers_traversed": 3,
    "index_time_ms": 3,
    "filter_time_ms": 2
  }
}
```

### Database Operations

#### Save Database

```bash
POST /save
```

Response:
```json
{
  "saved": true
}
```

#### Compact Collection

```bash
POST /collections/:name/compact
```

Response:
```json
{
  "compacted": true,
  "reclaimed_bytes": 1024000
}
```

#### Export Collection

```bash
GET /collections/:name/export
```

Returns NDJSON stream:
```
{"id":"doc1","vector":[0.1,...],"metadata":{"title":"Hello"}}
{"id":"doc2","vector":[0.2,...],"metadata":{"title":"World"}}
```

### Aliases

Aliases provide alternative names for collections, useful for blue-green deployments.

#### Create Alias

```bash
POST /aliases
Content-Type: application/json

{
  "alias": "prod",
  "collection": "documents_v2"
}
```

Response:
```json
{
  "alias": "prod",
  "collection": "documents_v2"
}
```

#### List Aliases

```bash
GET /aliases
```

Response:
```json
{
  "aliases": [
    {"alias": "prod", "collection": "documents_v2"},
    {"alias": "staging", "collection": "documents_v1"}
  ]
}
```

#### Update Alias

```bash
PUT /aliases/:alias
Content-Type: application/json

{
  "collection": "documents_v3"
}
```

Response:
```json
{
  "alias": "prod",
  "collection": "documents_v3"
}
```

#### Delete Alias

```bash
DELETE /aliases/:alias
```

Response:
```json
{
  "deleted": true
}
```

#### Using Aliases

Aliases work transparently with all collection endpoints:

```bash
# Query using alias (works exactly like collection name)
POST /collections/prod/search
Content-Type: application/json

{
  "vector": [0.1, 0.2, ...],
  "k": 10
}
```

### TTL / Expiration

Manage automatic vector expiration.

#### Expire Vectors

Remove all expired vectors from a collection:

```bash
POST /collections/:name/expire
```

Response:
```json
{
  "expired_count": 42
}
```

## Client Examples

### cURL

```bash
# Create collection
curl -X POST http://localhost:8080/collections \
  -H "Content-Type: application/json" \
  -d '{"name":"docs","dimensions":384,"distance":"cosine"}'

# Insert vector
curl -X POST http://localhost:8080/collections/docs/vectors \
  -H "Content-Type: application/json" \
  -d '{"id":"doc1","vector":[0.1,0.2,...],"metadata":{"title":"Hello"}}'

# Search
curl -X POST http://localhost:8080/collections/docs/search \
  -H "Content-Type: application/json" \
  -d '{"vector":[0.1,0.2,...],"k":10}'
```

### Python

```python
import requests

BASE_URL = "http://localhost:8080"

# Create collection
requests.post(f"{BASE_URL}/collections", json={
    "name": "documents",
    "dimensions": 384,
    "distance": "cosine"
})

# Insert
requests.post(f"{BASE_URL}/collections/documents/vectors", json={
    "id": "doc1",
    "vector": [0.1] * 384,
    "metadata": {"title": "Hello World"}
})

# Search
response = requests.post(f"{BASE_URL}/collections/documents/search", json={
    "vector": [0.1] * 384,
    "k": 10
})
results = response.json()["results"]
```

### JavaScript

```javascript
const BASE_URL = 'http://localhost:8080';

// Create collection
await fetch(`${BASE_URL}/collections`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    name: 'documents',
    dimensions: 384,
    distance: 'cosine'
  })
});

// Insert
await fetch(`${BASE_URL}/collections/documents/vectors`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    id: 'doc1',
    vector: Array(384).fill(0.1),
    metadata: { title: 'Hello World' }
  })
});

// Search
const response = await fetch(`${BASE_URL}/collections/documents/search`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    vector: Array(384).fill(0.1),
    k: 10
  })
});
const { results } = await response.json();
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NEEDLE_BIND_ADDRESS` | Listen address | `127.0.0.1:8080` |
| `NEEDLE_DATABASE_PATH` | Database file path | Required |
| `NEEDLE_TLS_CERT` | TLS certificate path | None |
| `NEEDLE_TLS_KEY` | TLS key path | None |
| `NEEDLE_LOG_LEVEL` | Log level (debug, info, warn, error) | `info` |
| `NEEDLE_CORS_ORIGINS` | Allowed CORS origins | `*` |

### Programmatic Configuration

```rust
use needle::{Database, server::{ServerConfig, run_with_config}};

let config = ServerConfig {
    bind_address: "0.0.0.0:8080".parse()?,
    tls_config: None,
    cors_origins: vec!["https://example.com".to_string()],
    max_request_size: 10 * 1024 * 1024, // 10MB
    request_timeout: Duration::from_secs(30),
};

let db = Database::open("vectors.needle")?;
run_with_config(config, db).await?;
```

## Authentication

### API Key Authentication

```rust
use axum::{middleware, http::Request, response::Response};

async fn auth_middleware<B>(req: Request<B>, next: Next<B>) -> Response {
    let api_key = req.headers()
        .get("X-API-Key")
        .and_then(|v| v.to_str().ok());

    match api_key {
        Some(key) if key == std::env::var("API_KEY").unwrap() => {
            next.run(req).await
        }
        _ => Response::builder()
            .status(401)
            .body("Unauthorized".into())
            .unwrap()
    }
}
```

### JWT Authentication

```rust
use jsonwebtoken::{decode, DecodingKey, Validation};

async fn jwt_middleware<B>(req: Request<B>, next: Next<B>) -> Response {
    let token = req.headers()
        .get("Authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));

    match token {
        Some(token) => {
            let key = DecodingKey::from_secret(b"secret");
            match decode::<Claims>(token, &key, &Validation::default()) {
                Ok(_) => next.run(req).await,
                Err(_) => Response::builder()
                    .status(401)
                    .body("Invalid token".into())
                    .unwrap()
            }
        }
        None => Response::builder()
            .status(401)
            .body("Missing token".into())
            .unwrap()
    }
}
```

## Rate Limiting

```rust
use tower_governor::{GovernorConfig, GovernorLayer};

let governor_conf = GovernorConfig::default();
let governor_limiter = governor_conf.limiter().clone();

let app = Router::new()
    .route("/search", post(search_handler))
    .layer(GovernorLayer { config: governor_conf });
```

## Monitoring

### Prometheus Metrics

```bash
# Enable metrics endpoint
needle serve -d vectors.needle --metrics-port 9090

# Scrape config for Prometheus
scrape_configs:
  - job_name: 'needle'
    static_configs:
      - targets: ['localhost:9090']
```

Available metrics:
- `needle_http_requests_total` - Total HTTP requests
- `needle_http_request_duration_seconds` - Request duration histogram
- `needle_search_latency_seconds` - Search latency
- `needle_vectors_total` - Total vectors stored

## Next Steps

- [CLI Reference](/docs/advanced/cli)
- [Production Deployment](/docs/guides/production)
- [API Reference](/docs/api-reference)
