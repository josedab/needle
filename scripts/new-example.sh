#!/usr/bin/env bash
set -euo pipefail

# Scaffold a new Needle example.
#
# Usage:
#   scripts/new-example.sh <example_name>
#
# Example:
#   scripts/new-example.sh my_search_demo
#
# This creates:
#   1. examples/<example_name>.rs  — example boilerplate using Database::in_memory()

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
EXAMPLES_DIR="$ROOT_DIR/examples"

usage() {
    echo "Usage: $0 <example_name>"
    echo ""
    echo "Scaffold a new example in examples/<example_name>.rs"
    echo ""
    echo "Example:"
    echo "  $0 my_search_demo"
    echo ""
    echo "Existing examples:"
    find "$EXAMPLES_DIR" -name '*.rs' -exec basename {} .rs \; | sort | sed 's/^/  /'
    exit 1
}

if [ $# -ne 1 ]; then
    usage
fi

NAME="$1"

# Validate name (snake_case)
if ! echo "$NAME" | grep -qE '^[a-z][a-z0-9_]*$'; then
    echo "Error: Example name must be snake_case (e.g., my_search_demo)"
    exit 1
fi

EXAMPLE_FILE="$EXAMPLES_DIR/$NAME.rs"

# Check if example already exists
if [ -f "$EXAMPLE_FILE" ]; then
    echo "Error: $EXAMPLE_FILE already exists"
    exit 1
fi

# Convert snake_case to Title Case for display
TITLE=$(echo "$NAME" | sed -E 's/_/ /g; s/\b(\w)/\u\1/g')

cat > "$EXAMPLE_FILE" << RUST
//! ${TITLE} example for Needle vector database
//!
//! Run with: cargo run --example ${NAME}

use needle::Database;

fn main() -> needle::Result<()> {
    println!("=== Needle: ${TITLE} ===\n");

    // Create an in-memory database
    let db = Database::in_memory();

    // Create a collection for 128-dimensional vectors
    db.create_collection("example", 128)?;
    let collection = db.collection("example")?;

    // Insert sample vectors
    let vector: Vec<f32> = (0..128).map(|i| (i as f32 / 128.0).sin()).collect();
    collection.insert("vec_1", &vector, None)?;
    println!("Inserted 1 vector");

    // Search
    let results = collection.search(&vector, 5)?;
    println!("Search returned {} results", results.len());
    for result in &results {
        println!("  ID: {}, Distance: {:.4}", result.id, result.distance);
    }

    println!("\n=== Example Complete ===");
    Ok(())
}
RUST

echo "Created $EXAMPLE_FILE"
echo ""
echo "✓ Example scaffolded successfully!"
echo ""
echo "Next steps:"
echo "  1. Edit $EXAMPLE_FILE to implement your example"
echo "  2. Run with: cargo run --example $NAME"
echo "  3. Test with: cargo build --example $NAME"
