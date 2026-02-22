#!/usr/bin/env bash
# ann-benchmark.sh — Run reproducible ANN benchmarks for Needle
#
# Usage:
#   ./scripts/ann-benchmark.sh [--vectors 100000] [--dimensions 128] [--queries 1000]
#
# This script benchmarks Needle's core HNSW search performance and outputs
# results in a format compatible with ANN-Benchmarks reporting.

set -euo pipefail

VECTORS="${VECTORS:-100000}"
DIMENSIONS="${DIMENSIONS:-128}"
QUERIES="${QUERIES:-1000}"
K="${K:-10}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --vectors)   VECTORS="$2"; shift 2 ;;
        --dimensions) DIMENSIONS="$2"; shift 2 ;;
        --queries)   QUERIES="$2"; shift 2 ;;
        --k)         K="$2"; shift 2 ;;
        *)           echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "╔═══════════════════════════════════════════════════╗"
echo "║     Needle ANN Benchmark Suite                    ║"
echo "╚═══════════════════════════════════════════════════╝"
echo ""
echo "Configuration:"
echo "  Vectors:    $VECTORS"
echo "  Dimensions: $DIMENSIONS"
echo "  Queries:    $QUERIES"
echo "  K:          $K"
echo ""

# Build release binary
echo "Building release binary..."
cargo build --release --quiet 2>/dev/null

NEEDLE="./target/release/needle"

if [ ! -f "$NEEDLE" ]; then
    echo "Error: Failed to build needle binary"
    exit 1
fi

# Run the built-in benchmark
echo ""
echo "Running benchmark..."
echo "────────────────────────────────────────────────────"
$NEEDLE dev benchmark --count "$VECTORS" --dimensions "$DIMENSIONS" --queries "$QUERIES"
echo "────────────────────────────────────────────────────"

# Also run Criterion benchmarks if available
echo ""
echo "Running Criterion benchmarks (compile check)..."
cargo bench --no-run --quiet 2>/dev/null && echo "  ✓ Criterion benchmarks compile" || echo "  ✗ Criterion benchmarks failed to compile"

echo ""
echo "Done! For detailed Criterion benchmarks, run: cargo bench"
