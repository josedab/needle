//! Batch operations example for Needle vector database
//!
//! Demonstrates batch insert, batch search, and batch delete patterns
//! for efficient bulk operations.
//!
//! Run with: cargo run --example batch_operations

use needle::Database;
use serde_json::json;

fn main() -> needle::Result<()> {
    println!("=== Needle Batch Operations Example ===\n");

    let db = Database::in_memory();
    db.create_collection("products", 4)?;
    let coll = db.collection("products")?;

    // ── Batch Insert ─────────────────────────────────────────────────────
    println!("--- Batch Insert ---");

    let products = vec![
        ("prod-001", vec![0.9, 0.1, 0.0, 0.0], json!({"name": "Laptop", "price": 999})),
        ("prod-002", vec![0.8, 0.2, 0.1, 0.0], json!({"name": "Tablet", "price": 499})),
        ("prod-003", vec![0.1, 0.9, 0.0, 0.0], json!({"name": "Headphones", "price": 149})),
        ("prod-004", vec![0.0, 0.1, 0.9, 0.0], json!({"name": "Mouse", "price": 29})),
        ("prod-005", vec![0.0, 0.0, 0.1, 0.9], json!({"name": "Keyboard", "price": 79})),
    ];

    for (id, vector, metadata) in &products {
        coll.insert(*id, vector, Some(metadata.clone()))?;
    }
    println!("Inserted {} products", coll.len());

    // ── Batch Search ─────────────────────────────────────────────────────
    println!("\n--- Batch Search ---");

    let queries = vec![
        vec![0.9, 0.1, 0.0, 0.0],  // Similar to laptop
        vec![0.0, 0.0, 0.5, 0.5],  // Similar to peripherals
    ];

    for (i, query) in queries.iter().enumerate() {
        let results = coll.search(query, 3)?;
        println!("Query {}: top {} results", i + 1, results.len());
        for r in &results {
            println!("  {} (distance: {:.4})", r.id, r.distance);
        }
    }

    // ── Batch Delete ─────────────────────────────────────────────────────
    println!("\n--- Batch Delete ---");

    let ids_to_delete = vec!["prod-003", "prod-004", "nonexistent"];
    let deleted = coll.batch_delete(&ids_to_delete)?;
    println!("Deleted {}/{} requested IDs", deleted, ids_to_delete.len());
    println!("Remaining vectors: {}", coll.len());

    // ── Verify remaining data ────────────────────────────────────────────
    println!("\n--- Remaining Products ---");
    for id in ["prod-001", "prod-002", "prod-005"] {
        if let Some((_vec, meta)) = coll.get(id) {
            println!("  {}: {}", id, meta.map_or("no metadata".into(), |m| m.to_string()));
        }
    }

    println!("\n=== Done ===");
    Ok(())
}
