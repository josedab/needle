#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${NEEDLE_PORT:-8080}"
ADDR="127.0.0.1:${PORT}"
BASE_URL="http://${ADDR}"
LOG_FILE="$(mktemp -t needle-demo.XXXXXX)"

if ! command -v cargo >/dev/null 2>&1; then
  echo "cargo is required to run the demo." >&2
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required to run the demo." >&2
  exit 1
fi

cd "$ROOT_DIR"

cargo build --features server

./target/debug/needle serve -a "${ADDR}" >"${LOG_FILE}" 2>&1 &
SERVER_PID=$!

cleanup() {
  kill "${SERVER_PID}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "Starting Needle server on ${BASE_URL}..."
for _ in {1..30}; do
  if curl -fsS "${BASE_URL}/health" >/dev/null; then
    break
  fi
  sleep 0.5
done

if ! curl -fsS "${BASE_URL}/health" >/dev/null; then
  echo "Server failed to start. Check logs at ${LOG_FILE}." >&2
  echo "Last 50 log lines:" >&2
  tail -n 50 "${LOG_FILE}" >&2 || true
  if lsof -nP -iTCP:"${PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "Port ${PORT} is already in use. Try: NEEDLE_PORT=9090 ./scripts/quickstart.sh" >&2
  fi
  echo "For more detail: RUST_LOG=debug ./scripts/quickstart.sh" >&2
  exit 1
fi

echo "Creating collection..."
curl -fsS -X POST "${BASE_URL}/collections" \
  -H "Content-Type: application/json" \
  -d '{"name":"demo","dimensions":3}' >/dev/null

echo "Inserting vector..."
curl -fsS -X POST "${BASE_URL}/collections/demo/vectors" \
  -H "Content-Type: application/json" \
  -d '{"id":"doc1","vector":[0.1,0.2,0.3],"metadata":{"title":"Hello Needle"}}' >/dev/null

echo "Searching..."
curl -fsS -X POST "${BASE_URL}/collections/demo/search" \
  -H "Content-Type: application/json" \
  -d '{"vector":[0.1,0.2,0.3],"k":3}' | cat

echo
echo "Health check: $(curl -fsS "${BASE_URL}/health")"

echo "Demo complete. Logs: ${LOG_FILE}"
