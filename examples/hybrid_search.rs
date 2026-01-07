//! Hybrid Search Example
//!
//! Demonstrates combining vector search with BM25 text search
//! using Reciprocal Rank Fusion (RRF) for improved retrieval.
//!
//! Run with: cargo run --example hybrid_search --features hybrid

use needle::{Collection, CollectionConfig, DistanceFunction};
use serde_json::json;

#[cfg(feature = "hybrid")]
use needle::{Bm25Index, reciprocal_rank_fusion, RrfConfig};

fn main() -> needle::Result<()> {
    // Create a collection for semantic search
    let config = CollectionConfig::new("documents", 4)
        .with_distance(DistanceFunction::Cosine);
    let mut collection = Collection::new(config);

    // Sample documents with both text and embeddings
    // In practice, embeddings would come from a model like sentence-transformers
    let documents = vec![
        ("doc1", "Machine learning is transforming industries", vec![0.8, 0.2, 0.1, 0.0]),
        ("doc2", "Deep learning models require large datasets", vec![0.7, 0.3, 0.2, 0.1]),
        ("doc3", "Natural language processing enables chatbots", vec![0.6, 0.5, 0.3, 0.2]),
        ("doc4", "Computer vision detects objects in images", vec![0.2, 0.8, 0.1, 0.0]),
        ("doc5", "Reinforcement learning trains game-playing agents", vec![0.5, 0.4, 0.3, 0.2]),
    ];

    // Insert documents with text as metadata
    for (id, text, embedding) in &documents {
        collection.insert(
            *id,
            embedding,
            Some(json!({ "text": text })),
        )?;
    }

    // Perform vector search
    let query_embedding = vec![0.75, 0.25, 0.15, 0.05]; // "machine learning" embedding
    let vector_results = collection.search(&query_embedding, 5)?;

    println!("Vector Search Results:");
    for result in &vector_results {
        println!("  {}: {:.4}", result.id, result.distance);
    }

    #[cfg(feature = "hybrid")]
    {
        // Create BM25 index for text search
        let mut bm25 = Bm25Index::default();
        for (id, text, _) in &documents {
            bm25.index_document(*id, text);
        }

        // Perform BM25 text search
        let bm25_results = bm25.search("machine learning", 5);
        println!("\nBM25 Search Results:");
        for (id, score) in &bm25_results {
            println!("  {}: {:.4}", id, score);
        }

        // Convert vector results to format expected by RRF
        let vector_for_rrf: Vec<(String, f32)> = vector_results
            .iter()
            .map(|r| (r.id.clone(), r.distance))
            .collect();

        // Fuse results using Reciprocal Rank Fusion
        let config = RrfConfig::default();
        let hybrid_results = reciprocal_rank_fusion(
            &vector_for_rrf,
            &bm25_results,
            &config,
            5,
        );

        println!("\nHybrid Search Results (RRF):");
        for result in &hybrid_results {
            println!("  {}: {:.4}", result.id, result.score);
        }
    }

    #[cfg(not(feature = "hybrid"))]
    {
        println!("\nNote: Enable 'hybrid' feature for BM25 and RRF fusion:");
        println!("  cargo run --example hybrid_search --features hybrid");
    }

    Ok(())
}
