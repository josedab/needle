//! Persistence example for Needle vector database
//!
//! Run with: cargo run --example persistence

use needle::{CollectionConfig, Database, DistanceFunction};
use serde_json::json;
use std::path::Path;

fn main() -> needle::Result<()> {
    println!("=== Needle Persistence Example ===\n");

    let db_path = "example_vectors.needle";

    // Clean up any existing file
    if Path::new(db_path).exists() {
        std::fs::remove_file(db_path)?;
    }

    // Create and populate a database
    {
        let mut db = Database::open(db_path)?;
        println!("Created database at: {}", db_path);

        // Create a collection with custom configuration
        let config = CollectionConfig::new("embeddings", 64)
            .with_distance(DistanceFunction::Euclidean)
            .with_m(16)
            .with_ef_construction(200);

        db.create_collection_with_config(config)?;
        println!("Created collection 'embeddings' with Euclidean distance");

        let collection = db.collection("embeddings")?;

        // Insert vectors
        for i in 0..50 {
            let vector: Vec<f32> = (0..64).map(|j| (i * 64 + j) as f32 / 3200.0).collect();
            let metadata = json!({"index": i, "type": "sample"});
            collection.insert(format!("vec_{}", i), &vector, Some(metadata))?;
        }
        println!("Inserted 50 vectors");

        // Save to disk
        db.save()?;
        println!("Saved database to disk\n");
    }

    // Reopen and query
    {
        let db = Database::open(db_path)?;
        println!("Reopened database from: {}", db_path);

        let collection = db.collection("embeddings")?;
        println!("Collection size: {} vectors", collection.len());

        // Search
        let query: Vec<f32> = (0..64).map(|j| (25 * 64 + j) as f32 / 3200.0).collect();
        let results = collection.search(&query, 3)?;

        println!("\nSearch results (should find vec_25 as closest):");
        for result in results {
            println!("  ID: {}, Distance: {:.6}", result.id, result.distance);
        }
    }

    // Clean up
    std::fs::remove_file(db_path)?;
    println!("\nCleaned up: {}", db_path);

    println!("\n=== Persistence Example Complete ===");

    println!("\n\u{1f4a1} Next steps:");
    println!("  cargo run --example quantization                          \u{2014} Compress vectors");
    println!("  cargo run --example hybrid_search --features hybrid       \u{2014} Vector + keyword search");
    println!("  cargo run --example rag_chatbot                           \u{2014} Build a RAG pipeline");
    println!("  See all examples: examples/README.md");

    Ok(())
}
