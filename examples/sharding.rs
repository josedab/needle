//! Sharding Example
//!
//! Demonstrates horizontal scaling primitives using consistent hashing
//! for shard assignment. This module provides building blocks for
//! distributing vectors across multiple shards.
//!
//! Run with: cargo run --example sharding

use needle::shard::{ShardManager, ShardConfig, ShardId, ShardedCollection};
use needle::{Collection, CollectionConfig, DistanceFunction};
use std::sync::Arc;

fn main() -> needle::Result<()> {
    // Create shard configuration
    let config = ShardConfig::new(4)  // 4 shards
        .with_virtual_nodes(100)       // 100 virtual nodes for distribution
        .with_replication(1);          // Replication factor 1

    // Create the shard manager
    let manager = Arc::new(ShardManager::new(config));

    println!("Created ShardManager with {} shards", manager.num_shards());

    // Create a sharded collection wrapper
    let mut sharded: ShardedCollection<Collection> = ShardedCollection::new(manager.clone());

    // Initialize each shard with its own collection
    for i in 0..4 {
        let shard_id = ShardId::new(i);
        let coll_config = CollectionConfig::new(format!("shard_{}", i), 4)
            .with_distance(DistanceFunction::Cosine);
        sharded.add_shard(shard_id, Collection::new(coll_config));
    }

    // Insert vectors - route to appropriate shard based on ID
    let vectors = vec![
        ("vec1", vec![1.0, 0.0, 0.0, 0.0]),
        ("vec2", vec![0.0, 1.0, 0.0, 0.0]),
        ("vec3", vec![0.0, 0.0, 1.0, 0.0]),
        ("vec4", vec![0.0, 0.0, 0.0, 1.0]),
        ("vec5", vec![0.5, 0.5, 0.0, 0.0]),
        ("vec6", vec![0.0, 0.5, 0.5, 0.0]),
        ("vec7", vec![0.0, 0.0, 0.5, 0.5]),
        ("vec8", vec![0.5, 0.0, 0.0, 0.5]),
    ];

    // Show shard routing
    println!("\nRouting vectors to shards:");
    for (id, _) in &vectors {
        let shard = manager.route_id(id);
        println!("  {} -> {}", id, shard);
    }

    // Insert into appropriate shards
    for (id, vector) in &vectors {
        if let Some(collection) = sharded.get_shard_mut(id) {
            collection.insert(*id, vector, None)?;
        }
    }

    // Show distribution across shards
    println!("\nShard distribution:");
    for (shard_id, collection) in sharded.all_shards() {
        println!("  {}: {} vectors", shard_id, collection.len());
    }

    // Search across all shards and merge results
    let query = vec![0.9, 0.1, 0.0, 0.0];
    let k = 3;

    println!("\nSearching all shards for top {} results:", k);
    let mut all_results = Vec::new();

    for (shard_id, collection) in sharded.all_shards() {
        let results = collection.search(&query, k)?;
        println!("  {} returned {} results", shard_id, results.len());
        for r in results {
            all_results.push((r.id, r.distance));
        }
    }

    // Sort merged results by distance
    all_results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    all_results.truncate(k);

    println!("\nMerged top {} results:", k);
    for (id, dist) in &all_results {
        println!("  {}: {:.4}", id, dist);
    }

    // Get shard stats
    println!("\nShard Manager Stats:");
    let snapshot = manager.stats().snapshot();
    println!("  Total routes: {}", snapshot.routes);
    println!("  Total vectors: {}", snapshot.total_vectors);

    Ok(())
}
