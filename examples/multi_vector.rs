//! Multi-Vector (ColBERT-style) Search Example
//!
//! Demonstrates late-interaction retrieval using multi-vector representations,
//! where documents are represented by multiple token embeddings and scored
//! using MaxSim aggregation.
//!
//! Run with: cargo run --example multi_vector

use needle::multivec::{MultiVector, MultiVectorIndex, MultiVectorConfig};
use needle::DistanceFunction;

fn main() {
    // Create a multi-vector index configuration
    let config = MultiVectorConfig {
        dimensions: 4,
        distance: DistanceFunction::Cosine,
        normalize: true,
        use_centroid: true,
    };
    let mut index = MultiVectorIndex::new(config);

    // Document 1: "machine learning" - 2 token embeddings
    let doc1 = MultiVector::new("doc1", vec![
        vec![0.9, 0.1, 0.0, 0.0],  // "machine"
        vec![0.8, 0.3, 0.1, 0.0],  // "learning"
    ]);

    // Document 2: "deep learning models" - 3 token embeddings
    let doc2 = MultiVector::new("doc2", vec![
        vec![0.7, 0.4, 0.2, 0.0],  // "deep"
        vec![0.8, 0.3, 0.1, 0.0],  // "learning"
        vec![0.5, 0.5, 0.3, 0.0],  // "models"
    ]);

    // Document 3: "computer vision" - 2 token embeddings
    let doc3 = MultiVector::new("doc3", vec![
        vec![0.2, 0.8, 0.1, 0.0],  // "computer"
        vec![0.1, 0.9, 0.2, 0.0],  // "vision"
    ]);

    // Insert documents
    index.insert(doc1).expect("Failed to insert doc1");
    index.insert(doc2).expect("Failed to insert doc2");
    index.insert(doc3).expect("Failed to insert doc3");

    println!("Inserted {} documents", index.len());

    // Query: "learning algorithms" - 2 token embeddings
    let query = vec![
        vec![0.85, 0.25, 0.1, 0.0],  // "learning"
        vec![0.6, 0.4, 0.2, 0.0],    // "algorithms"
    ];

    // Search using MaxSim aggregation
    // For each query token, find max similarity with any document token
    // Then sum across query tokens
    let results = index.search(&query, 3);

    println!("\nSearch Results (MaxSim):");
    for result in &results {
        println!("  {}: score={:.4}, tokens={}", result.id, result.score, result.num_tokens);
    }

    // Get a specific document
    if let Some(doc) = index.get("doc1") {
        println!("\nDocument 'doc1' has {} token vectors", doc.len());
    }

    // Two-stage search (centroid pre-filtering for efficiency)
    let results = index.search_two_stage(&query, 2, 2);
    println!("\nTwo-Stage Search Results:");
    for result in &results {
        println!("  {}: score={:.4}", result.id, result.score);
    }

    // Remove a document
    let removed = index.remove("doc3");
    println!("\nRemoved doc3: {}", removed);
    println!("After removal: {} documents", index.len());
}
