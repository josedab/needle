//! Basic usage example for Needle vector database
//!
//! Run with: cargo run --example basic_usage

use needle::{Database, Filter};
use serde_json::json;

fn main() -> needle::Result<()> {
    println!("=== Needle Vector Database Basic Example ===\n");

    // Create an in-memory database
    let db = Database::in_memory();

    // Create a collection for 128-dimensional vectors
    db.create_collection("documents", 128)?;
    println!("Created collection 'documents' with 128 dimensions");

    // Get a reference to the collection
    let collection = db.collection("documents")?;

    // Generate some sample vectors
    let vectors: Vec<Vec<f32>> = (0..100)
        .map(|i| {
            (0..128)
                .map(|j| ((i * 128 + j) as f32 / 12800.0).sin())
                .collect()
        })
        .collect();

    // Insert vectors with metadata
    for (i, vector) in vectors.iter().enumerate() {
        let metadata = json!({
            "title": format!("Document {}", i),
            "category": if i % 3 == 0 { "science" } else if i % 3 == 1 { "technology" } else { "engineering" },
            "score": i as f64 / 100.0
        });
        collection.insert(format!("doc_{}", i), vector, Some(metadata))?;
    }
    println!("Inserted 100 vectors with metadata\n");

    // Basic search
    let query = &vectors[42]; // Use one of our vectors as a query
    let results = collection.search(query, 5)?;

    println!("Top 5 nearest neighbors to doc_42:");
    for result in &results {
        println!("  ID: {}, Distance: {:.4}", result.id, result.distance);
    }
    println!();

    // Search with metadata filter
    let filter = Filter::eq("category", "science");
    let filtered_results = collection.search_with_filter(query, 5, &filter)?;

    println!("Top 5 nearest 'science' documents:");
    for result in &filtered_results {
        let category = result
            .metadata
            .as_ref()
            .and_then(|m| m.get("category"))
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        println!(
            "  ID: {}, Distance: {:.4}, Category: {}",
            result.id, result.distance, category
        );
    }
    println!();

    // Get specific vector
    if let Some((vec, meta)) = collection.get("doc_50") {
        println!("Retrieved doc_50:");
        println!("  Vector length: {}", vec.len());
        if let Some(m) = meta {
            println!("  Metadata: {}", m);
        }
    }
    println!();

    // Delete a vector
    let deleted = collection.delete("doc_50")?;
    println!("Deleted doc_50: {}", deleted);
    println!(
        "doc_50 exists after delete: {}",
        collection.get("doc_50").is_some()
    );

    println!("\n=== Example Complete ===");

    println!("\n\u{1f4a1} Next steps:");
    println!("  cargo run --example filtered_search  \u{2014} Search with metadata filters");
    println!("  cargo run --example persistence       \u{2014} Save to disk and reload");
    println!("  See all examples: examples/README.md");

    Ok(())
}
