# HTTP Quickstart

This guide gets you from zero to a working HTTP search in a few minutes.

## 1) Start the server

```bash
cargo run --features server -- serve -a 127.0.0.1:8080
```

```bash
curl -fsS http://127.0.0.1:8080/health
```

## 2) Create a collection

```bash
curl -fsS -X POST http://127.0.0.1:8080/collections \
  -H "Content-Type: application/json" \
  -d '{"name":"docs","dimensions":3}'
```

## 3) Insert vectors

```bash
curl -fsS -X POST http://127.0.0.1:8080/collections/docs/vectors \
  -H "Content-Type: application/json" \
  -d '{"id":"doc1","vector":[0.1,0.2,0.3],"metadata":{"title":"Hello Needle"}}'
```

```bash
curl -fsS -X POST http://127.0.0.1:8080/collections/docs/vectors \
  -H "Content-Type: application/json" \
  -d '{"id":"doc2","vector":[0.2,0.1,0.0],"metadata":{"title":"Second Doc"}}'
```

## 4) Search

```bash
curl -fsS -X POST http://127.0.0.1:8080/collections/docs/search \
  -H "Content-Type: application/json" \
  -d '{"vector":[0.1,0.2,0.3],"k":2}'
```

## 5) Clean up

```bash
curl -fsS -X DELETE http://127.0.0.1:8080/collections/docs/vectors/doc1
curl -fsS -X DELETE http://127.0.0.1:8080/collections/docs/vectors/doc2
```

```bash
curl -fsS -X DELETE http://127.0.0.1:8080/collections/docs
```

## 6) Explore the OpenAPI spec

```bash
curl -fsS http://127.0.0.1:8080/openapi.json | head -n 20
```
