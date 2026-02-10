#!/usr/bin/env bash
# Interactive playground — walk through Needle's HTTP API step by step.
# Starts the server, guides you through create → insert → search → clean up.
#
# Usage: ./scripts/playground.sh
#        NEEDLE_PORT=9090 ./scripts/playground.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${NEEDLE_PORT:-8080}"
ADDR="127.0.0.1:${PORT}"
BASE_URL="http://${ADDR}"
LOG_FILE="$(mktemp -t needle-playground.XXXXXX)"
SERVER_PID=""

cleanup() {
  if [ -n "${SERVER_PID:-}" ]; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

bold()  { printf "\033[1m%s\033[0m" "$1"; }
green() { printf "\033[32m%s\033[0m" "$1"; }
cyan()  { printf "\033[36m%s\033[0m" "$1"; }
dim()   { printf "\033[2m%s\033[0m" "$1"; }

pause() {
  echo ""
  read -r -p "$(dim 'Press Enter to continue...')" < /dev/tty || true
  echo ""
}

# ── Step 0: Build and start server ────────────────────────────────────────────
echo ""
echo "$(bold '🧭 Needle Interactive Playground')"
echo "$(dim '   Walk through the HTTP API step by step.')"
echo ""

if ! command -v cargo >/dev/null 2>&1; then
  echo "cargo is required. Install Rust: https://rustup.rs/"
  exit 1
fi

echo "$(bold 'Step 0:') Building and starting the server..."
cd "$ROOT_DIR"
cargo build --features server --quiet 2>&1

./target/debug/needle serve -a "${ADDR}" >"${LOG_FILE}" 2>&1 &
SERVER_PID=$!

for _ in {1..30}; do
  if curl -fsS "${BASE_URL}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 0.5
done

if ! curl -fsS "${BASE_URL}/health" >/dev/null 2>&1; then
  echo "Server failed to start. Logs: ${LOG_FILE}"
  exit 1
fi

echo "$(green '✓') Server running at $(cyan "${BASE_URL}")"
pause

# ── Step 1: Create a collection ───────────────────────────────────────────────
echo "$(bold 'Step 1:') Create a collection"
echo "$(dim 'Collections hold vectors of a fixed dimension.')"
echo ""
echo "$(cyan '$ curl -X POST '"${BASE_URL}"'/collections \')"
echo "$(cyan '    -H "Content-Type: application/json" \')"
echo "$(cyan '    -d '"'"'{"name":"playground","dimensions":3}'"'"')')"
echo ""

RESULT=$(curl -fsS -X POST "${BASE_URL}/collections" \
  -H "Content-Type: application/json" \
  -d '{"name":"playground","dimensions":3}' 2>&1) || true
echo "Response: ${RESULT}"
pause

# ── Step 2: Insert vectors ────────────────────────────────────────────────────
echo "$(bold 'Step 2:') Insert vectors with metadata"
echo ""

VECTORS=(
  '{"id":"vec1","vector":[0.1,0.2,0.3],"metadata":{"title":"Intro to Needle","category":"tutorial"}}'
  '{"id":"vec2","vector":[0.4,0.5,0.6],"metadata":{"title":"HNSW Deep Dive","category":"algorithm"}}'
  '{"id":"vec3","vector":[0.7,0.8,0.9],"metadata":{"title":"Metadata Filtering","category":"feature"}}'
  '{"id":"vec4","vector":[0.2,0.1,0.4],"metadata":{"title":"Quantization Guide","category":"tutorial"}}'
)

for v in "${VECTORS[@]}"; do
  ID=$(echo "$v" | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)
  curl -fsS -X POST "${BASE_URL}/collections/playground/vectors" \
    -H "Content-Type: application/json" \
    -d "$v" >/dev/null
  echo "$(green '✓') Inserted ${ID}"
done
pause

# ── Step 3: Search ────────────────────────────────────────────────────────────
echo "$(bold 'Step 3:') Search for similar vectors"
echo ""
echo "$(cyan '$ curl -X POST '"${BASE_URL}"'/collections/playground/search \')"
echo "$(cyan '    -H "Content-Type: application/json" \')"
echo "$(cyan '    -d '"'"'{"vector":[0.1,0.2,0.3],"k":3}'"'"')')"
echo ""

RESULTS=$(curl -fsS -X POST "${BASE_URL}/collections/playground/search" \
  -H "Content-Type: application/json" \
  -d '{"vector":[0.1,0.2,0.3],"k":3}')
echo "Results:"
echo "$RESULTS" | python3 -m json.tool 2>/dev/null || echo "$RESULTS"
pause

# ── Step 4: Get a specific vector ─────────────────────────────────────────────
echo "$(bold 'Step 4:') Retrieve a vector by ID"
echo ""
echo "$(cyan "$ curl ${BASE_URL}/collections/playground/vectors/vec2")"
echo ""

RESULT=$(curl -fsS "${BASE_URL}/collections/playground/vectors/vec2" 2>&1) || true
echo "$RESULT" | python3 -m json.tool 2>/dev/null || echo "$RESULT"
pause

# ── Step 5: Delete and clean up ───────────────────────────────────────────────
echo "$(bold 'Step 5:') Clean up"
echo ""

curl -fsS -X DELETE "${BASE_URL}/collections/playground" >/dev/null 2>&1 || true
echo "$(green '✓') Deleted 'playground' collection"

echo ""
echo "$(bold '🎉 Playground complete!')"
echo ""
echo "Next steps:"
echo "  $(cyan 'cargo run --example basic_usage')     — Rust API walkthrough"
echo "  $(cyan 'cargo run --example filtered_search') — Metadata filtering"
echo "  $(cyan 'cat docs/http-quickstart.md')          — Full HTTP API guide"
echo "  $(cyan 'cat docs/rag-quickstart.md')           — RAG with OpenAI embeddings"
echo ""
echo "Server logs: ${LOG_FILE}"
