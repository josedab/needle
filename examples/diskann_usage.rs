//! DiskANN Example — On-Disk Index for Large-Scale Search
//!
//! Demonstrates building and querying a DiskANN index that stores
//! vector data on disk, enabling search over datasets larger than RAM.
//!
//! Run with: cargo run --example diskann_usage --features diskann

#[cfg(feature = "diskann")]
fn main() -> needle::Result<()> {
    use needle::indexing::diskann::{DiskAnnConfig, DiskAnnIndex};
    use std::path::Path;

    println!("=== Needle DiskANN Example ===\n");

    let index_path = Path::new("/tmp/needle_diskann_example");

    // Clean up from previous runs
    if index_path.exists() {
        std::fs::remove_dir_all(index_path).ok();
    }

    // Configure DiskANN parameters
    let config = DiskAnnConfig {
        max_degree: 64,
        build_list_size: 100,
        search_list_size: 50,
        ..DiskAnnConfig::default()
    };

    // Create a DiskANN index
    let dimensions = 128;
    let mut index = DiskAnnIndex::create(index_path, dimensions, config)?;
    println!("Created DiskANN index at {:?}", index_path);
    println!("Dimensions: {dimensions}\n");

    // Add vectors
    let num_vectors = 500;
    for i in 0..num_vectors {
        let vector: Vec<f32> = (0..dimensions)
            .map(|j| ((i * dimensions + j) as f32 / 1000.0).sin())
            .collect();
        index.add(&format!("vec_{i}"), &vector)?;
    }
    println!("Added {num_vectors} vectors");

    // Build the index (constructs the Vamana graph)
    index.build()?;
    println!("Index built successfully\n");

    // Search
    let query: Vec<f32> = (0..dimensions).map(|j| (j as f32 / 1000.0).sin()).collect();
    let results = index.search(&query, 5)?;

    println!("Top 5 search results:");
    for (rank, result) in results.iter().enumerate() {
        println!("  #{}: {} (distance: {:.4})", rank + 1, result.id, result.distance);
    }

    // Print stats
    let stats = index.stats();
    println!("\nIndex stats:");
    println!("  Vectors: {}", stats.num_vectors);
    println!("  Dimensions: {}", stats.dimensions);
    println!("  Avg degree: {:.1}", stats.avg_degree);

    // Clean up
    std::fs::remove_dir_all(index_path).ok();
    println!("\nCleanup complete");

    Ok(())
}

#[cfg(not(feature = "diskann"))]
fn main() {
    eprintln!("This example requires the 'diskann' feature.");
    eprintln!("Run with: cargo run --example diskann_usage --features diskann");
}
