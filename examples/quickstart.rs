//! # Needle Quickstart
//!
//! The fastest path from zero to semantic search.
//! Run: `cargo run --example quickstart`

use needle::{Database, Filter};
use serde_json::json;

fn main() -> needle::Result<()> {
    // 1. Create an in-memory database
    let db = Database::in_memory();

    // 2. Create a collection for 4-dimensional vectors
    db.create_collection("docs", 4)?;
    let col = db.collection("docs")?;

    // 3. Insert vectors with metadata
    col.insert("rust", &[0.9, 0.1, 0.0, 0.0], Some(json!({"lang": "rust", "type": "systems"})))?;
    col.insert("python", &[0.1, 0.9, 0.0, 0.0], Some(json!({"lang": "python", "type": "scripting"})))?;
    col.insert("go", &[0.5, 0.3, 0.2, 0.0], Some(json!({"lang": "go", "type": "systems"})))?;

    // 4. Search for the nearest vectors
    let results = col.search(&[0.85, 0.15, 0.0, 0.0], 3)?;
    println!("Top 3 results:");
    for r in &results {
        println!("  {} (distance: {:.4})", r.id, r.distance);
    }

    // 5. Search with metadata filter
    let filter = Filter::eq("type", "systems");
    let filtered = col.search_with_filter(&[0.5, 0.5, 0.0, 0.0], 5, &filter)?;
    println!("\nSystems languages only:");
    for r in &filtered {
        println!("  {} (distance: {:.4})", r.id, r.distance);
    }

    println!("\nDone! {} vectors indexed.", col.len());
    Ok(())
}
