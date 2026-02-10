//! Filtered Search Example
//!
//! Demonstrates metadata filtering using MongoDB-style query syntax
//! to narrow search results based on field values.
//!
//! Run with: cargo run --example filtered_search

use needle::{Collection, CollectionConfig, Filter, DistanceFunction};
use serde_json::json;

fn main() -> needle::Result<()> {
    // Create a collection for product embeddings
    let config = CollectionConfig::new("products", 4)
        .with_distance(DistanceFunction::Cosine);
    let mut collection = Collection::new(config);

    // Insert products with rich metadata
    let products = vec![
        ("prod1", vec![0.9, 0.1, 0.0, 0.0], json!({
            "name": "Laptop Pro",
            "category": "electronics",
            "price": 999.99,
            "in_stock": true,
            "tags": ["computer", "portable"]
        })),
        ("prod2", vec![0.8, 0.2, 0.1, 0.0], json!({
            "name": "Wireless Mouse",
            "category": "electronics",
            "price": 29.99,
            "in_stock": true,
            "tags": ["computer", "accessory"]
        })),
        ("prod3", vec![0.2, 0.8, 0.1, 0.0], json!({
            "name": "Running Shoes",
            "category": "sports",
            "price": 89.99,
            "in_stock": false,
            "tags": ["footwear", "running"]
        })),
        ("prod4", vec![0.1, 0.9, 0.2, 0.0], json!({
            "name": "Yoga Mat",
            "category": "sports",
            "price": 39.99,
            "in_stock": true,
            "tags": ["fitness", "yoga"]
        })),
        ("prod5", vec![0.5, 0.5, 0.3, 0.0], json!({
            "name": "Smart Watch",
            "category": "electronics",
            "price": 299.99,
            "in_stock": true,
            "tags": ["wearable", "fitness"]
        })),
    ];

    for (id, embedding, metadata) in products {
        collection.insert(id, &embedding, Some(metadata))?;
    }

    println!("Inserted {} products\n", collection.len());

    let query = vec![0.85, 0.15, 0.05, 0.0]; // Query for electronics

    // Example 1: Simple equality filter
    println!("=== Electronics only ===");
    let filter = Filter::eq("category", "electronics");
    let results = collection.search_with_filter(&query, 5, &filter)?;
    for r in &results {
        let name = r.metadata.as_ref().unwrap()["name"].as_str().unwrap();
        println!("  {} ({}): {:.4}", r.id, name, r.distance);
    }

    // Example 2: Comparison filter - price under $100
    println!("\n=== Price under $100 ===");
    let filter = Filter::lt("price", 100.0);
    let results = collection.search_with_filter(&query, 5, &filter)?;
    for r in &results {
        let name = r.metadata.as_ref().unwrap()["name"].as_str().unwrap();
        let price = r.metadata.as_ref().unwrap()["price"].as_f64().unwrap();
        println!("  {} - ${:.2}", name, price);
    }

    // Example 3: Boolean filter - in stock only
    println!("\n=== In stock only ===");
    let filter = Filter::eq("in_stock", true);
    let results = collection.search_with_filter(&query, 5, &filter)?;
    for r in &results {
        let name = r.metadata.as_ref().unwrap()["name"].as_str().unwrap();
        println!("  {}", name);
    }

    // Example 4: Combined filters with AND
    println!("\n=== Electronics under $500, in stock ===");
    let filter = Filter::and(vec![
        Filter::eq("category", "electronics"),
        Filter::lt("price", 500.0),
        Filter::eq("in_stock", true),
    ]);
    let results = collection.search_with_filter(&query, 5, &filter)?;
    for r in &results {
        let name = r.metadata.as_ref().unwrap()["name"].as_str().unwrap();
        let price = r.metadata.as_ref().unwrap()["price"].as_f64().unwrap();
        println!("  {} - ${:.2}", name, price);
    }

    // Example 5: OR filter - electronics or sports
    println!("\n=== Electronics OR Sports ===");
    let filter = Filter::or(vec![
        Filter::eq("category", "electronics"),
        Filter::eq("category", "sports"),
    ]);
    let results = collection.search_with_filter(&query, 5, &filter)?;
    for r in &results {
        let name = r.metadata.as_ref().unwrap()["name"].as_str().unwrap();
        let cat = r.metadata.as_ref().unwrap()["category"].as_str().unwrap();
        println!("  {} ({})", name, cat);
    }

    // Example 6: Parse filter from JSON (MongoDB-style)
    println!("\n=== Parsed from JSON: price $30-$100 ===");
    let filter = Filter::parse(&json!({
        "price": { "$gte": 30, "$lte": 100 }
    })).map_err(needle::NeedleError::InvalidInput)?;
    let results = collection.search_with_filter(&query, 5, &filter)?;
    for r in &results {
        let name = r.metadata.as_ref().unwrap()["name"].as_str().unwrap();
        let price = r.metadata.as_ref().unwrap()["price"].as_f64().unwrap();
        println!("  {} - ${:.2}", name, price);
    }

    println!("\n\u{1f4a1} Next steps:");
    println!("  cargo run --example persistence   \u{2014} Save to disk and reload");
    println!("  cargo run --example quantization  \u{2014} Compress vectors to save memory");
    println!("  See all examples: examples/README.md");

    Ok(())
}
