//! Sparse Vector Example
//!
//! Demonstrates sparse vector support for lexical features like TF-IDF,
//! BM25 term weights, and SPLADE embeddings.
//!
//! Run with: cargo run --example sparse_vectors

use needle::sparse::{SparseDistance, SparseIndex, SparseVector};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a sparse index for TF-IDF vectors
    let mut index = SparseIndex::new();

    // Simulate TF-IDF vectors for documents
    // Each index represents a term ID, values are TF-IDF weights
    // Vocabulary: 0=machine, 1=learning, 2=deep, 3=neural, 4=network, 5=computer, 6=vision

    // Document 1: "machine learning"
    let doc1 = SparseVector::new(
        vec![0, 1],     // term indices
        vec![0.8, 0.9], // TF-IDF weights
    );

    // Document 2: "deep learning neural network"
    let doc2 = SparseVector::new(vec![1, 2, 3, 4], vec![0.6, 0.7, 0.8, 0.75]);

    // Document 3: "computer vision neural network"
    let doc3 = SparseVector::new(vec![3, 4, 5, 6], vec![0.65, 0.7, 0.8, 0.85]);

    // Document 4: "deep machine learning"
    let doc4 = SparseVector::new(vec![0, 1, 2], vec![0.7, 0.8, 0.6]);

    // Insert documents with specific IDs
    index.insert_with_id(1, doc1);
    index.insert_with_id(2, doc2);
    index.insert_with_id(3, doc3);
    index.insert_with_id(4, doc4);

    println!("Indexed {} documents", index.len());

    // Query: "machine learning"
    let query = SparseVector::new(vec![0, 1], vec![1.0, 1.0]);

    // Dot product search (faster, good for normalized vectors)
    let results = index.search(&query, 3);
    println!("\nDot Product Search Results:");
    for (id, score) in &results {
        println!("  Document {}: {:.4}", id, score);
    }

    // Cosine similarity search (handles varying document lengths)
    let results = index.search_cosine(&query, 3);
    println!("\nCosine Similarity Search Results:");
    for (id, score) in &results {
        println!("  Document {}: {:.4}", id, score);
    }

    // Create sparse vector from HashMap (useful for term frequency counting)
    let mut term_weights: HashMap<u32, f32> = HashMap::new();
    term_weights.insert(2, 0.9); // "deep"
    term_weights.insert(3, 0.8); // "neural"
    let query2 = SparseVector::from_hashmap(&term_weights);

    let results = index.search(&query2, 3);
    println!("\nSearch for 'deep neural':");
    for (id, score) in &results {
        println!("  Document {}: {:.4}", id, score);
    }

    // Get a specific vector
    if let Some(vec) = index.get(1) {
        println!("\nDocument 1 has {} non-zero terms", vec.len());
        println!("  L2 norm: {:.4}", vec.l2_norm());
    }

    // Remove a document
    let removed = index.remove(3);
    println!("\nRemoved doc 3: {}", removed);
    println!("After removal: {} documents", index.len());

    // Sparse vector operations
    let v1 = SparseVector::new(vec![0, 1, 2], vec![1.0, 2.0, 3.0]);
    let v2 = SparseVector::new(vec![1, 2, 3], vec![1.0, 1.0, 1.0]);

    println!("\nSparse Vector Operations:");
    println!("  v1 dot v2: {:.4}", SparseDistance::dot_product(&v1, &v2));
    println!("  v1 L2 norm: {:.4}", v1.l2_norm());
    println!("  v2 L2 norm: {:.4}", v2.l2_norm());

    Ok(())
}
