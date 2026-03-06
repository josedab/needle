//! Error handling patterns for Needle vector database
//!
//! Shows how to handle common errors, access error codes, and use recovery hints.
//!
//! Run with: cargo run --example error_handling

use needle::{Database, NeedleError};
use needle::error::Recoverable;

fn main() -> needle::Result<()> {
    println!("=== Needle Error Handling Example ===\n");

    let db = Database::in_memory();
    db.create_collection("docs", 4)?;
    let coll = db.collection("docs")?;
    coll.insert("v1", &[1.0, 0.0, 0.0, 0.0], None)?;

    // ── Pattern 1: Match on specific error types ─────────────────────────
    println!("--- Pattern 1: Matching specific errors ---");

    match db.create_collection("docs", 4) {
        Ok(()) => println!("  Collection created"),
        Err(NeedleError::CollectionAlreadyExists(name)) => {
            println!("  Collection '{}' already exists — using existing one", name);
        }
        Err(e) => return Err(e),
    }

    // ── Pattern 2: Using error codes and recovery hints ──────────────────
    println!("\n--- Pattern 2: Error codes and hints ---");

    let result = coll.insert("v1", &[1.0, 0.0, 0.0, 0.0], None);
    if let Err(ref e) = result {
        let code = e.error_code();
        println!("  Error code: {} ({})", code.code(), code.category());
        println!("  Retryable: {}", e.is_retryable());

        let hints = e.recovery_hints();
        if !hints.is_empty() {
            println!("  Recovery hints:");
            for hint in &hints {
                println!("    → {}", hint);
            }
        }
    }

    // ── Pattern 3: Dimension mismatch ────────────────────────────────────
    println!("\n--- Pattern 3: Dimension mismatch ---");

    match coll.insert("v2", &[1.0, 0.0], None) {
        Ok(()) => println!("  Inserted successfully"),
        Err(NeedleError::DimensionMismatch { expected, got }) => {
            println!("  Dimension mismatch: expected {}, got {}", expected, got);
            println!("  Fix: ensure your embedding model outputs {}-dim vectors", expected);
        }
        Err(e) => println!("  Unexpected error: {}", e),
    }

    // ── Pattern 4: Collection not found ──────────────────────────────────
    println!("\n--- Pattern 4: Collection not found ---");

    match db.collection("nonexistent") {
        Ok(_) => println!("  Found collection"),
        Err(NeedleError::CollectionNotFound(name)) => {
            println!("  Collection '{}' not found", name);
            println!("  Available: {:?}", db.list_collections());
        }
        Err(e) => println!("  Unexpected error: {}", e),
    }

    // ── Pattern 5: Invalid vector values ─────────────────────────────────
    println!("\n--- Pattern 5: Invalid vector values ---");

    match coll.insert("bad", &[1.0, f32::NAN, 0.0, 0.0], None) {
        Ok(()) => println!("  Inserted (unexpected)"),
        Err(NeedleError::InvalidVector(msg)) => {
            println!("  Invalid vector: {}", msg);
        }
        Err(e) => println!("  Unexpected error: {}", e),
    }

    println!("\n=== Done ===");
    Ok(())
}
