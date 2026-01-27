//! Metrics Example — Prometheus Integration
//!
//! Demonstrates using Needle's Prometheus metrics for observability.
//!
//! Run with: cargo run --example metrics_usage --features metrics

#[cfg(feature = "metrics")]
fn main() -> needle::Result<()> {
    use needle::{metrics, Database};
    use serde_json::json;

    println!("=== Needle Prometheus Metrics Example ===\n");

    // Access the global metrics instance
    let m = metrics::metrics();

    // Create a database and perform some operations
    let db = Database::in_memory();
    db.create_collection("docs", 64)?;
    let collection = db.collection("docs")?;

    // Insert some vectors — metrics are recorded automatically
    for i in 0..50 {
        let vector: Vec<f32> = (0..64).map(|j| ((i * 64 + j) as f32).sin()).collect();
        collection.insert(
            format!("doc_{i}"),
            &vector,
            Some(json!({"index": i})),
        )?;
    }

    // Perform searches
    let query: Vec<f32> = (0..64).map(|j| (j as f32 * 0.1).cos()).collect();
    let _results = collection.search(&query, 5)?;

    // Record a custom operation metric
    m.operations_total
        .with_label_values(&["search", "docs"])
        .inc();

    // Export metrics in Prometheus text format
    let metric_families = prometheus::gather();
    println!("Collected {} metric families\n", metric_families.len());

    for family in &metric_families {
        if family.get_name().starts_with("needle_") {
            println!("  {}: {} metrics", family.get_name(), family.get_metric().len());
        }
    }

    println!("\nMetrics are ready for Prometheus scraping at /metrics endpoint");
    println!("when running in server mode with --features server,metrics");

    Ok(())
}

#[cfg(not(feature = "metrics"))]
fn main() {
    eprintln!("This example requires the 'metrics' feature.");
    eprintln!("Run with: cargo run --example metrics_usage --features metrics");
}
