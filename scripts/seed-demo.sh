#!/bin/sh
# Seed a demo collection into a running Needle server.
# Used by docker-compose (demo profile) and can be run standalone:
#   NEEDLE_URL=http://127.0.0.1:8080 ./scripts/seed-demo.sh

set -e

BASE_URL="${NEEDLE_URL:-http://needle:8080}"

echo "Seeding demo data into ${BASE_URL}..."

# Create demo collection (3 dimensions for simplicity)
curl -fsS -X POST "${BASE_URL}/collections" \
  -H "Content-Type: application/json" \
  -d '{"name":"demo","dimensions":3}' >/dev/null

# Insert sample vectors
curl -fsS -X POST "${BASE_URL}/collections/demo/vectors" \
  -H "Content-Type: application/json" \
  -d '{"id":"doc1","vector":[0.1,0.2,0.3],"metadata":{"title":"Hello Needle","category":"greeting"}}' >/dev/null

curl -fsS -X POST "${BASE_URL}/collections/demo/vectors" \
  -H "Content-Type: application/json" \
  -d '{"id":"doc2","vector":[0.4,0.5,0.6],"metadata":{"title":"Vector Search","category":"tutorial"}}' >/dev/null

curl -fsS -X POST "${BASE_URL}/collections/demo/vectors" \
  -H "Content-Type: application/json" \
  -d '{"id":"doc3","vector":[0.7,0.8,0.9],"metadata":{"title":"HNSW Algorithm","category":"algorithm"}}' >/dev/null

curl -fsS -X POST "${BASE_URL}/collections/demo/vectors" \
  -H "Content-Type: application/json" \
  -d '{"id":"doc4","vector":[0.2,0.3,0.4],"metadata":{"title":"Embeddings 101","category":"tutorial"}}' >/dev/null

curl -fsS -X POST "${BASE_URL}/collections/demo/vectors" \
  -H "Content-Type: application/json" \
  -d '{"id":"doc5","vector":[0.9,0.1,0.2],"metadata":{"title":"Metadata Filtering","category":"feature"}}' >/dev/null

echo "Seeded 5 vectors into 'demo' collection."
echo ""
echo "Try searching:"
echo "  curl -X POST ${BASE_URL}/collections/demo/search \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"vector\":[0.1,0.2,0.3],\"k\":3}'"
