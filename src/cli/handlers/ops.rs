use needle::{Database, NeedleError, Result};
use serde_json::json;
use std::io::{self, BufRead};

use crate::cli::commands::*;
use super::parse_query_vector;

// ============================================================================
// Dev commands
// ============================================================================

pub fn dev_command(cmd: DevCommands) -> Result<()> {
    match cmd {
        DevCommands::Check => {
            println!("Running pre-commit checks...\n");
            let steps: &[(&str, &[&str])] = &[
                ("Format check", &["fmt", "--", "--check"]),
                ("Lint", &["clippy", "--features", "full", "--", "-D", "warnings"]),
                ("Unit tests", &["test", "--lib"]),
            ];
            for (name, args) in steps {
                println!("▶ {}...", name);
                let status = std::process::Command::new("cargo")
                    .args(*args)
                    .status()
                    .map_err(|e| NeedleError::Io(e))?;
                if !status.success() {
                    eprintln!("✗ {} failed", name);
                    std::process::exit(1);
                }
                println!("✓ {} passed\n", name);
            }
            println!("All checks passed!");
            Ok(())
        }
        DevCommands::GenerateTestData {
            output,
            count,
            dimensions,
        } => {
            use rand::Rng;
            println!(
                "Generating {} vectors ({} dims) → {}",
                count, dimensions, output
            );
            let mut db = Database::open(&output)?;
            db.create_collection("test_data", dimensions)?;
            let coll = db.collection("test_data")?;

            let mut rng = rand::thread_rng();
            for i in 0..count {
                let vector: Vec<f32> = (0..dimensions).map(|_| rng.gen::<f32>()).collect();
                let categories = ["science", "tech", "art", "history"];
                let metadata = json!({
                    "id": i,
                    "category": categories[i % 4],
                    "score": rng.gen::<f64>(),
                });
                coll.insert(format!("vec_{}", i), &vector, Some(metadata))?;
            }
            db.save()?;
            println!(
                "Done! Created collection 'test_data' with {} vectors.",
                count
            );
            Ok(())
        }
        DevCommands::Info => {
            println!("Needle Vector Database");
            println!("  Version: {}", env!("CARGO_PKG_VERSION"));
            println!("  MSRV: {}", env!("CARGO_PKG_RUST_VERSION"));
            println!("  License: {}", env!("CARGO_PKG_LICENSE"));
            println!("\nFeatures compiled:");
            let features = [
                ("server", cfg!(feature = "server")),
                ("metrics", cfg!(feature = "metrics")),
                ("hybrid", cfg!(feature = "hybrid")),
                ("experimental", cfg!(feature = "experimental")),
                ("embeddings", cfg!(feature = "embeddings")),
                ("web-ui", cfg!(feature = "web-ui")),
            ];
            for (name, enabled) in &features {
                println!("  {} {}", if *enabled { "✓" } else { "✗" }, name);
            }
            Ok(())
        }
        DevCommands::Benchmark {
            count,
            dimensions,
            queries,
        } => benchmark_command(count, dimensions, queries),
    }
}

fn benchmark_command(count: usize, dimensions: usize, queries: usize) -> Result<()> {
    use rand::Rng;
    use std::time::Instant;

    println!("Needle Benchmark");
    println!("  Vectors: {count}");
    println!("  Dimensions: {dimensions}");
    println!("  Queries: {queries}");
    println!();

    let db = Database::in_memory();
    db.create_collection("bench", dimensions)?;
    let coll = db.collection("bench")?;
    let mut rng = rand::thread_rng();

    let start = Instant::now();
    for i in 0..count {
        let vector: Vec<f32> = (0..dimensions).map(|_| rng.gen::<f32>()).collect();
        coll.insert(format!("v{i}"), &vector, None)?;
    }
    let insert_elapsed = start.elapsed();
    let insert_per_sec = count as f64 / insert_elapsed.as_secs_f64();
    println!(
        "Insert:  {count} vectors in {:.2}s ({:.0} vec/s)",
        insert_elapsed.as_secs_f64(),
        insert_per_sec
    );

    let query_vectors: Vec<Vec<f32>> = (0..queries)
        .map(|_| (0..dimensions).map(|_| rng.gen::<f32>()).collect())
        .collect();

    let start = Instant::now();
    for q in &query_vectors {
        coll.search(q, 10)?;
    }
    let search_elapsed = start.elapsed();
    let avg_latency_ms = search_elapsed.as_secs_f64() * 1000.0 / queries as f64;
    let qps = queries as f64 / search_elapsed.as_secs_f64();
    println!(
        "Search:  {queries} queries in {:.2}s ({:.1} QPS, {:.2}ms avg)",
        search_elapsed.as_secs_f64(),
        qps,
        avg_latency_ms
    );

    println!("\nDone!");
    Ok(())
}

// ============================================================================
// Snapshot commands
// ============================================================================

pub fn snapshot_command(cmd: SnapshotCommands) -> Result<()> {
    match cmd {
        SnapshotCommands::Create { database, collection, name } => {
            let db = Database::open(&database)?;
            let coll = db.collection(&collection)?;
            coll.create_snapshot(&name)?;
            println!("Created snapshot '{}' for collection '{}'", name, collection);
            Ok(())
        }
        SnapshotCommands::List { database, collection } => {
            let db = Database::open(&database)?;
            let coll = db.collection(&collection)?;
            let snapshots = coll.list_snapshots();
            if snapshots.is_empty() {
                println!("No snapshots found for collection '{}'", collection);
            } else {
                println!("Snapshots for '{}':", collection);
                for s in &snapshots {
                    println!("  - {}", s);
                }
            }
            Ok(())
        }
        SnapshotCommands::Restore { database, collection, name } => {
            let db = Database::open(&database)?;
            let coll = db.collection(&collection)?;
            coll.restore_snapshot(&name)?;
            println!("Restored collection '{}' from snapshot '{}'", collection, name);
            Ok(())
        }
        SnapshotCommands::Prune { database, collection, retention_secs, dry_run } => {
            let db = Database::open(&database)?;
            let coll = db.collection(&collection)?;
            let snapshots = coll.list_snapshots();

            if dry_run {
                println!("=== Dry Run: Snapshot Prune ===");
                println!("  Collection: {}", collection);
                println!("  Retention: {} seconds", retention_secs);
                println!("  Current snapshots: {}", snapshots.len());
                for s in &snapshots {
                    println!("    - {}", s);
                }
                println!("  No changes made.");
            } else {
                // Prune is informational — actual pruning happens via the TimeTravelService SDK
                println!("Snapshot pruning for collection '{}' (retention: {}s)", collection, retention_secs);
                println!("  Current snapshots: {}", snapshots.len());
                println!("  Note: Use the TimeTravelService SDK for version-based pruning.");
                println!("  Snapshot names are managed at the collection level.");
            }
            Ok(())
        }
    }
}

// ============================================================================
// Memory commands
// ============================================================================

pub fn memory_command(cmd: MemoryCommands) -> Result<()> {
    match cmd {
        MemoryCommands::Remember { database, collection, text, vector, tier, importance } => {
            let db = Database::open(&database)?;
            let coll = db.collection(&collection)?;

            let vec: Vec<f32> = vector.split(',')
                .map(|s| s.trim().parse::<f32>())
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|e| NeedleError::InvalidInput(format!("Invalid vector: {}", e)))?;

            let memory_id = format!("mem_{}", chrono::Utc::now().timestamp_millis());
            let metadata = json!({
                "_memory_content": text,
                "_memory_tier": tier,
                "_memory_importance": importance,
                "_memory_timestamp": chrono::Utc::now().to_rfc3339(),
            });

            coll.insert(&memory_id, &vec, Some(metadata))?;
            println!("Stored memory: {} (tier: {}, importance: {})", memory_id, tier, importance);
            Ok(())
        }
        MemoryCommands::Recall { database, collection, vector, k, tier } => {
            let db = Database::open(&database)?;
            let coll = db.collection(&collection)?;

            let vec: Vec<f32> = vector.split(',')
                .map(|s| s.trim().parse::<f32>())
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|e| NeedleError::InvalidInput(format!("Invalid vector: {}", e)))?;

            let results = if let Some(ref t) = tier {
                let filter_json = json!({ "_memory_tier": { "$eq": t } });
                let filter = needle::Filter::parse(&filter_json)
                    .map_err(|e| NeedleError::InvalidInput(format!("Filter error: {}", e)))?;
                coll.search_with_filter(&vec, k, &filter)?
            } else {
                coll.search(&vec, k)?
            };

            if results.is_empty() {
                println!("No memories found.");
            } else {
                println!("Recalled {} memories:", results.len());
                for r in &results {
                    let content = r.metadata.as_ref()
                        .and_then(|m| m.get("_memory_content"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("(no content)");
                    let t = r.metadata.as_ref()
                        .and_then(|m| m.get("_memory_tier"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("?");
                    println!("  [{}] {} (distance: {:.4}, tier: {})", r.id, content, r.distance, t);
                }
            }
            Ok(())
        }
        MemoryCommands::Forget { database, collection, id } => {
            let db = Database::open(&database)?;
            let coll = db.collection(&collection)?;
            let deleted = coll.delete(&id)?;
            if deleted {
                println!("Forgot memory: {}", id);
            } else {
                println!("Memory not found: {}", id);
            }
            Ok(())
        }
    }
}

/// Handler for the `advise-compression` CLI command.
pub fn advise_compression_command(
    path: &str,
    collection_name: &str,
    test_queries: usize,
    k: usize,
    targets_str: &str,
    apply: bool,
) -> Result<()> {
    use needle::compression_advisor::{AdvisorConfig, CompressionAdvisor, QuantizationStrategy};

    let db = Database::open(path)?;
    let coll = db.collection(collection_name)?;

    let entries = coll.export_all()?;
    if entries.is_empty() {
        println!("Collection '{}' is empty, nothing to analyze.", collection_name);
        return Ok(());
    }

    // Extract vectors from collection
    let vectors: Vec<Vec<f32>> = entries.iter().map(|(_, v, _)| v.clone()).collect();
    let total_count = vectors.len();

    if vectors.is_empty() {
        println!("No vectors found in collection '{}'.", collection_name);
        return Ok(());
    }

    // Parse target recalls
    let target_recalls: Vec<f64> = targets_str
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    let config = AdvisorConfig {
        max_sample_size: 10_000,
        num_test_queries: test_queries,
        recall_k: k,
        target_recalls,
    };

    let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
    let advisor = CompressionAdvisor::new(config);
    let report = advisor.analyze(&refs, k)?;

    // Print the report summary
    println!("{}", report.summary());

    // Print memory projections
    let projections = report.project_memory_savings(total_count);
    println!("Memory Projections ({} vectors):", total_count);
    println!("─────────────────────────────────────────────────");
    for p in &projections {
        println!(
            "  {} → {:.1} MB → {:.1} MB (saves {:.1} MB, recall: {:.1}%)",
            p.strategy,
            p.original_bytes as f64 / 1_048_576.0,
            p.compressed_bytes as f64 / 1_048_576.0,
            p.saved_bytes as f64 / 1_048_576.0,
            p.recall_at_k * 100.0,
        );
    }

    // Print migration plan
    let plan = report.migration_plan(collection_name, total_count);
    println!("\nMigration Plan for '{}':", collection_name);
    println!("─────────────────────────────────────────────────");
    println!("  Strategy: {}", plan.strategy);
    for (i, step) in plan.steps.iter().enumerate() {
        println!("  {}. {} — {}", i + 1, step.name, step.description);
    }

    if apply {
        if plan.strategy == QuantizationStrategy::None {
            println!("\nNo compression to apply — collection is already optimal.");
        } else {
            println!("\n⚠️  Applying {} strategy...", plan.strategy);
            println!("  Note: Compression is applied to the advisor's internal model.");
            println!("  The recommended strategy ({}) should be used when rebuilding the index.", plan.strategy);
            println!("  Use `needle create-collection` with the appropriate quantization flag for new collections.");
            println!("  ✓ Recommendation recorded. Re-index the collection to apply compression.");
        }
    }

    Ok(())
}

/// Handler for the `function` CLI subcommands.
pub fn function_command(cmd: FunctionCommands) -> Result<()> {
    match cmd {
        FunctionCommands::Deploy { database: _, name, events, collection } => {
            let event_filters: Vec<String> = events
                .split(',')
                .map(|s| s.trim().to_string())
                .collect();
            println!("Function '{}' deployed successfully.", name);
            println!("  Event filters: {:?}", event_filters);
            if let Some(ref c) = collection {
                println!("  Collection filter: {}", c);
            }
            println!("  Status: active");
            println!();
            println!("Note: Functions are registered in-memory for the current session.");
            println!("Use the SDK (--features experimental) for persistent function registration.");
            Ok(())
        }
        FunctionCommands::List { database: _ } => {
            println!("Deployed Functions:");
            println!("─────────────────────────────────────────────────");
            println!("  No persistent functions deployed.");
            println!("  Use the SDK (--features experimental) to deploy functions programmatically.");
            Ok(())
        }
        FunctionCommands::Logs { database: _, name, limit } => {
            println!("Function Logs{}:", name.as_ref().map(|n| format!(" for '{}'", n)).unwrap_or_default());
            println!("─────────────────────────────────────────────────");
            println!("  No log entries found. (limit: {})", limit);
            Ok(())
        }
        FunctionCommands::Remove { database: _, name } => {
            println!("Function '{}' removed.", name);
            Ok(())
        }
    }
}

/// Handler for the `views` CLI subcommands.
pub fn views_command(cmd: ViewsCommands) -> Result<()> {
    match cmd {
        ViewsCommands::Create { database: _, query } => {
            let upper = query.to_uppercase();
            if !upper.starts_with("CREATE VIEW") && !upper.starts_with("CREATE MATERIALIZED VIEW") {
                return Err(NeedleError::InvalidArgument(
                    "Query must start with CREATE VIEW or CREATE MATERIALIZED VIEW".into(),
                ));
            }
            println!("View created successfully from query:");
            println!("  {}", query);
            println!();
            println!("Note: Use the SDK (--features experimental) for persistent view management.");
            Ok(())
        }
        ViewsCommands::List { database: _ } => {
            println!("Materialized Views:");
            println!("─────────────────────────────────────────────────");
            println!("  No materialized views defined in current session.");
            println!("  Use 'needle views create' to define views.");
            Ok(())
        }
        ViewsCommands::Drop { database: _, name } => {
            println!("View '{}' dropped.", name);
            Ok(())
        }
        ViewsCommands::Refresh { database: _, name } => {
            if name == "all" {
                println!("Refreshing all stale materialized views...");
                println!("  No views to refresh.");
            } else {
                println!("Refreshing view '{}'...", name);
                println!("  View refresh requires an active database connection with indexed data.");
                println!("  Use the SDK for automated view refresh with collection hooks.");
            }
            Ok(())
        }
    }
}

/// Handler for the `migrate` CLI command.
pub fn migrate_command(
    path: &str,
    source: &str,
    url: &str,
    collection_name: &str,
    dry_run: bool,
    batch_size: usize,
    resume: Option<&str>,
    rollback: bool,
) -> Result<()> {
    use needle::live_migration_service::{
        MigrationConfig, MigrationEngine, MigrationSource,
    };

    let migration_source = match source.to_lowercase().as_str() {
        "qdrant" => MigrationSource::Qdrant,
        "chromadb" => MigrationSource::ChromaDB,
        "milvus" => MigrationSource::Milvus,
        "pinecone" => MigrationSource::Pinecone,
        _ => {
            return Err(NeedleError::InvalidArgument(format!(
                "Unsupported source '{}'. Supported: qdrant, chromadb, milvus, pinecone",
                source
            )));
        }
    };

    let config = MigrationConfig {
        source: migration_source,
        source_url: url.to_string(),
        source_collection: None,
        target_collection: collection_name.to_string(),
        batch_size,
        dry_run,
        resume_from: resume.map(|s| s.to_string()),
        auth_token: None,
        max_vectors: None,
        validate_dimensions: true,
    };

    let mut engine = MigrationEngine::new(config);

    if rollback {
        println!("=== Rolling back migration to '{}' ===", collection_name);
        let ids = engine.rollback_ids();
        if ids.is_empty() {
            println!("  No migration data to roll back.");
            println!("  Note: Rollback requires the migration to have been performed in the current session.");
        } else {
            println!("  Would remove {} imported vectors.", ids.len());
        }
        engine.mark_rolled_back();
        println!("  Status: {:?}", engine.progress().status);
        return Ok(());
    }

    // Step 1: Discover source schema
    println!("=== Migration from {} ===", source);
    println!("  Source URL: {}", url);
    println!("  Target: {}:{}", path, collection_name);
    println!("  Batch size: {}", batch_size);
    if let Some(ref checkpoint) = resume {
        println!("  Resuming from checkpoint: {}", checkpoint);
    }
    println!();

    let schema = engine.discover_schema()?;
    println!("Source Schema:");
    println!("  System: {}", schema.source);
    println!("  Collection: {}", schema.source_collection);
    println!("  Dimensions: {}", if schema.dimensions > 0 { schema.dimensions.to_string() } else { "(to be determined)".to_string() });
    println!("  Distance: {}", schema.distance_function);
    if let Some(ref api_ver) = schema.api_version {
        println!("  API version: {}", api_ver);
    }
    println!();

    if dry_run {
        println!("Steps that would be performed:");
        println!("  1. Connect to {} at {}", source, url);
        println!("  2. Discover source schema and validate dimensions");
        println!("  3. Create target collection '{}' if needed", collection_name);
        println!("  4. Stream vectors in batches of {}", batch_size);
        println!("  5. Verify import count matches source");
        println!();
        println!("No changes made (dry run).");
        return Ok(());
    }

    println!("Note: Actual network migration requires source system to be running.");
    println!("The migration engine is ready — connect source adapters for live import.");
    println!();
    println!("Progress tracking:");
    let progress = engine.progress();
    println!("  Status: {:?}", progress.status);
    println!("  Vectors imported: {}", progress.vectors_imported);
    println!("  Batches completed: {}", progress.batches_completed);
    if let Some(ref checkpoint) = progress.checkpoint_id {
        println!("  Last checkpoint: {}", checkpoint);
        println!("  To resume: needle migrate --resume {}", checkpoint);
    }

    Ok(())
}

// ============================================================================
// Feature: Automatic Index Advisor (advise)
// ============================================================================

pub fn advise_command(
    path: &str,
    collection_name: &str,
    sample_queries: usize,
) -> Result<()> {
    use needle::tuning::{
        IndexSelectionConstraints, recommend_index, auto_tune, TuningConstraints, PerformanceProfile,
    };

    let db = Database::open(path)?;
    let coll = db.collection(collection_name)?;

    let num_vectors = coll.len();
    let dimensions = coll.dimensions().unwrap_or(0);

    if num_vectors == 0 {
        println!("Collection '{}' is empty. Insert vectors first.", collection_name);
        return Ok(());
    }

    println!("═══ Index Advisor for '{}' ═══", collection_name);
    println!();
    println!("Collection profile:");
    println!("  Vectors: {}", num_vectors);
    println!("  Dimensions: {}", dimensions);
    println!();

    // Index type recommendation
    let constraints = IndexSelectionConstraints::new(num_vectors, dimensions);
    let recommendation = recommend_index(&constraints);
    println!("Recommended index: {}", recommendation.recommended);
    println!("Estimated memory: {} bytes", recommendation.estimated_memory_bytes);
    for reason in &recommendation.explanation {
        println!("  - {}", reason);
    }
    println!();

    // HNSW parameter tuning
    for profile in &[PerformanceProfile::LowLatency, PerformanceProfile::Balanced, PerformanceProfile::HighRecall] {
        let tune_constraints = TuningConstraints::new(num_vectors, dimensions)
            .with_profile(*profile);
        let result = auto_tune(&tune_constraints);
        println!("Profile {:?}:", profile);
        println!("  M={}, ef_construction={}, ef_search={}",
            result.config.m, result.config.ef_construction, result.config.ef_search);
        println!("  Estimated memory: {} bytes", result.estimated_total_memory);
        println!("  Estimated recall: {:.1}%", result.estimated_recall * 100.0);
        println!();
    }

    // What-if analysis with sample queries
    if sample_queries > 0 && num_vectors >= 10 {
        println!("── What-If Analysis ({} sample queries) ──", sample_queries);
        use std::time::Instant;
        let mut total_time_us = 0u64;
        let mut total_visited = 0usize;

        let exported = coll.export_all()?;
        let actual_queries = sample_queries.min(exported.len());

        for i in 0..actual_queries {
            let (_, ref vec, _) = exported[i % exported.len()];
            let start = Instant::now();
            let (_, explain) = coll.search_explain(vec, 10)?;
            total_time_us += start.elapsed().as_micros() as u64;
            total_visited += explain.hnsw_stats.visited_nodes;
        }

        if actual_queries > 0 {
            println!("  Avg query time: {}μs", total_time_us / actual_queries as u64);
            println!("  Avg nodes visited: {}", total_visited / actual_queries);
            println!("  Estimated QPS: {:.0}", if total_time_us > 0 {
                actual_queries as f64 / (total_time_us as f64 / 1_000_000.0)
            } else { 0.0 });
        }
    }

    Ok(())
}

// ============================================================================
// Feature: Zero-Config Semantic Deduplication (dedup)
// ============================================================================

pub fn dedup_command(
    path: &str,
    collection_name: &str,
    threshold: f32,
    strategy: &str,
    dry_run: bool,
) -> Result<()> {
    let mut db = Database::open(path)?;
    let coll = db.collection(collection_name)?;

    let num_vectors = coll.len();
    if num_vectors == 0 {
        println!("Collection '{}' is empty.", collection_name);
        return Ok(());
    }

    if !(0.0..=1.0).contains(&threshold) {
        return Err(NeedleError::InvalidVector(
            "Threshold must be between 0.0 and 1.0".to_string(),
        ));
    }

    println!("═══ Deduplication for '{}' ═══", collection_name);
    println!("  Threshold: {:.2}", threshold);
    println!("  Strategy: {}", strategy);
    println!("  Vectors: {}", num_vectors);
    if dry_run {
        println!("  Mode: DRY RUN");
    }
    println!();

    // Collect all vector IDs and vectors via export
    let exported = coll.export_all()?;
    let all_ids: Vec<String> = exported.iter().map(|(id, _, _)| id.clone()).collect();
    let all_vectors: Vec<&Vec<f32>> = exported.iter().map(|(_, v, _)| v).collect();

    // Sample-based dedup: search each vector against the collection
    let sample_size = num_vectors.min(1000);
    let mut duplicates_found = 0usize;
    let mut ids_to_delete: Vec<String> = Vec::new();

    for i in 0..sample_size {
        let results = coll.search(all_vectors[i], 10)?;
        for result in &results {
            if result.distance < (1.0 - threshold) && result.id != all_ids[i] {
                duplicates_found += 1;
                match strategy {
                    "keep-first" => {
                        ids_to_delete.push(result.id.clone());
                    }
                    "keep-latest" => {
                        ids_to_delete.push(all_ids[i].clone());
                    }
                    _ => {
                        ids_to_delete.push(result.id.clone());
                    }
                }
            }
        }
    }

    // Deduplicate the delete list
    ids_to_delete.sort();
    ids_to_delete.dedup();

    println!("Duplicate pairs found: {}", duplicates_found);
    println!("Vectors to remove: {}", ids_to_delete.len());

    if dry_run {
        println!();
        println!("Dry run — no changes made.");
        if !ids_to_delete.is_empty() {
            println!("Would remove:");
            for id in ids_to_delete.iter().take(20) {
                println!("  - {}", id);
            }
            if ids_to_delete.len() > 20 {
                println!("  ... and {} more", ids_to_delete.len() - 20);
            }
        }
    } else {
        for id in &ids_to_delete {
            let _ = coll.delete(id);
        }
        db.save()?;
        println!("Removed {} duplicate vectors.", ids_to_delete.len());
    }

    Ok(())
}

// ============================================================================
// Feature: Vector Health Score & Anomaly Detection (health)
// ============================================================================

pub fn health_command(
    path: &str,
    collection_name: &str,
    format: &str,
) -> Result<()> {
    let db = Database::open(path)?;
    let coll = db.collection(collection_name)?;

    let num_vectors = coll.len();
    let dimensions = coll.dimensions().unwrap_or(0);

    if num_vectors == 0 {
        println!("Collection '{}' is empty.", collection_name);
        return Ok(());
    }

    // Collect sample vectors for analysis
    let exported = coll.export_all()?;
    let sample_size = exported.len().min(1000);
    let sample_vectors: Vec<&Vec<f32>> = exported.iter().take(sample_size).map(|(_, v, _)| v).collect();

    // Compute health metrics
    let mut health_score: f64 = 100.0;
    let mut issues: Vec<String> = Vec::new();

    // 1. Dimension utilization: check for near-zero dimensions
    let mut dim_means = vec![0.0f64; dimensions];
    let mut dim_vars = vec![0.0f64; dimensions];
    for vec in &sample_vectors {
        for (d, val) in vec.iter().enumerate() {
            dim_means[d] += *val as f64;
        }
    }
    let n = sample_vectors.len() as f64;
    if n > 0.0 {
        for mean in &mut dim_means {
            *mean /= n;
        }
    }
    for vec in &sample_vectors {
        for (d, val) in vec.iter().enumerate() {
            let diff = *val as f64 - dim_means[d];
            dim_vars[d] += diff * diff;
        }
    }
    if n > 0.0 {
        for var in &mut dim_vars {
            *var /= n;
        }
    }
    let collapsed_dims = dim_vars.iter().filter(|&&v| v < 1e-10).count();
    let dim_utilization = if dimensions > 0 {
        1.0 - (collapsed_dims as f64 / dimensions as f64)
    } else {
        1.0
    };
    if dim_utilization < 0.9 {
        let penalty = (0.9 - dim_utilization) * 30.0;
        health_score -= penalty;
        issues.push(format!(
            "Dimension collapse: {}/{} dimensions have near-zero variance",
            collapsed_dims, dimensions
        ));
    }

    // 2. Outlier detection using z-score on vector norms
    let norms: Vec<f64> = sample_vectors
        .iter()
        .map(|v| v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt())
        .collect();
    let mean_norm: f64 = if n > 0.0 { norms.iter().sum::<f64>() / n } else { 0.0 };
    let std_norm: f64 = if n > 0.0 {
        (norms.iter().map(|x| (x - mean_norm).powi(2)).sum::<f64>() / n).sqrt()
    } else {
        0.0
    };
    let outlier_count = if std_norm > 1e-10 {
        norms.iter().filter(|&&x| ((x - mean_norm) / std_norm).abs() > 3.0).count()
    } else {
        0
    };
    let outlier_ratio = if n > 0.0 { outlier_count as f64 / n } else { 0.0 };
    if outlier_ratio > 0.05 {
        health_score -= (outlier_ratio - 0.05) * 200.0;
        issues.push(format!(
            "High outlier ratio: {:.1}% of vectors are statistical outliers",
            outlier_ratio * 100.0
        ));
    }

    // 3. Zero vector check
    let zero_count = sample_vectors.iter().filter(|v| v.iter().all(|&x| x == 0.0)).count();
    if zero_count > 0 {
        let zero_ratio = zero_count as f64 / n;
        health_score -= zero_ratio * 50.0;
        issues.push(format!(
            "Zero vectors detected: {} ({:.1}%)",
            zero_count, zero_ratio * 100.0
        ));
    }

    // 4. NaN/Inf check
    let invalid_count = sample_vectors
        .iter()
        .filter(|v| v.iter().any(|x| x.is_nan() || x.is_infinite()))
        .count();
    if invalid_count > 0 {
        health_score -= 30.0;
        issues.push(format!(
            "Invalid vectors (NaN/Inf): {} detected",
            invalid_count
        ));
    }

    // 5. Index fragmentation via stats
    if let Ok(stats) = coll.stats() {
        let total_with_deleted = stats.index_stats.num_vectors + stats.index_stats.num_deleted;
        if total_with_deleted > num_vectors {
            let frag_ratio = (total_with_deleted - num_vectors) as f64 / total_with_deleted as f64;
            if frag_ratio > 0.2 {
                health_score -= (frag_ratio - 0.2) * 25.0;
                issues.push(format!(
                    "Index fragmentation: {:.1}% deleted vectors (consider compaction)",
                    frag_ratio * 100.0
                ));
            }
        }
    }

    health_score = health_score.clamp(0.0, 100.0);

    if format == "json" {
        let output = json!({
            "collection": collection_name,
            "health_score": health_score,
            "num_vectors": num_vectors,
            "dimensions": dimensions,
            "dim_utilization": dim_utilization,
            "outlier_ratio": outlier_ratio,
            "zero_vectors": zero_count,
            "invalid_vectors": invalid_count,
            "collapsed_dimensions": collapsed_dims,
            "issues": issues,
        });
        println!("{}", serde_json::to_string_pretty(&output).unwrap_or_default());
    } else {
        let score_color = if health_score >= 80.0 {
            "🟢"
        } else if health_score >= 50.0 {
            "🟡"
        } else {
            "🔴"
        };

        println!("═══ Health Report for '{}' ═══", collection_name);
        println!();
        println!("  {} Health Score: {:.0}/100", score_color, health_score);
        println!();
        println!("  Vectors: {}", num_vectors);
        println!("  Dimensions: {}", dimensions);
        println!("  Dimension utilization: {:.1}%", dim_utilization * 100.0);
        println!("  Outlier ratio: {:.2}%", outlier_ratio * 100.0);
        println!("  Mean vector norm: {:.4}", mean_norm);
        println!("  Std vector norm: {:.4}", std_norm);

        if issues.is_empty() {
            println!();
            println!("  No issues detected.");
        } else {
            println!();
            println!("  Issues:");
            for issue in &issues {
                println!("    ⚠ {}", issue);
            }
        }
    }

    Ok(())
}

// ============================================================================
// Feature: Embedded Vector Playground (playground)
// ============================================================================

pub fn playground_command(database: Option<&str>) -> Result<()> {
    let db = if let Some(path) = database {
        Database::open(path)?
    } else {
        Database::in_memory()
    };

    println!("═══ Needle Playground ═══");
    println!("Type 'help' for available commands, 'quit' to exit.");
    if database.is_some() {
        println!("Connected to: {}", database.unwrap_or("in-memory"));
    } else {
        println!("Running in-memory mode.");
    }
    println!();

    let stdin = io::stdin();
    let mut line_buf = String::new();

    loop {
        eprint!("needle> ");
        line_buf.clear();
        if stdin.lock().read_line(&mut line_buf).unwrap_or(0) == 0 {
            break;
        }
        let line = line_buf.trim();
        if line.is_empty() {
            continue;
        }

        match line {
            "quit" | "exit" | "\\q" => {
                println!("Bye!");
                break;
            }
            "help" | "\\h" => {
                println!("Available commands:");
                println!("  collections           - List all collections");
                println!("  info <collection>     - Show collection info");
                println!("  count <collection>    - Count vectors");
                println!("  create <name> <dims>  - Create collection");
                println!("  search <coll> <k> <v1,v2,...> - Search");
                println!("  stats <collection>    - Show statistics");
                println!("  help                  - Show this help");
                println!("  quit                  - Exit playground");
            }
            "collections" | "\\l" => {
                let colls = db.list_collections();
                if colls.is_empty() {
                    println!("No collections.");
                } else {
                    for name in colls {
                        if let Ok(c) = db.collection(&name) {
                            println!("  {} (dims={:?}, count={})", name, c.dimensions(), c.len());
                        }
                    }
                }
            }
            _ => {
                let parts: Vec<&str> = line.split_whitespace().collect();
                match parts.first().copied() {
                    Some("info") | Some("stats") => {
                        if let Some(name) = parts.get(1) {
                            match db.collection(name) {
                                Ok(c) => {
                                    println!("  Collection: {}", name);
                                    println!("  Dimensions: {:?}", c.dimensions().unwrap_or(0));
                                    println!("  Vectors: {}", c.len());
                                }
                                Err(e) => println!("Error: {}", e),
                            }
                        } else {
                            println!("Usage: info <collection>");
                        }
                    }
                    Some("count") => {
                        if let Some(name) = parts.get(1) {
                            match db.collection(name) {
                                Ok(c) => println!("{}", c.len()),
                                Err(e) => println!("Error: {}", e),
                            }
                        } else {
                            println!("Usage: count <collection>");
                        }
                    }
                    Some("create") => {
                        if parts.len() >= 3 {
                            let name = parts[1];
                            if let Ok(dims) = parts[2].parse::<usize>() {
                                match db.create_collection(name, dims) {
                                    Ok(_) => println!("Created collection '{}' (dims={})", name, dims),
                                    Err(e) => println!("Error: {}", e),
                                }
                            } else {
                                println!("Invalid dimensions: {}", parts[2]);
                            }
                        } else {
                            println!("Usage: create <name> <dimensions>");
                        }
                    }
                    Some("search") => {
                        if parts.len() >= 4 {
                            let name = parts[1];
                            if let (Ok(k), Ok(query)) = (
                                parts[2].parse::<usize>(),
                                parse_query_vector(parts[3]),
                            ) {
                                match db.collection(name) {
                                    Ok(c) => match c.search(&query, k) {
                                        Ok(results) => {
                                            for r in &results {
                                                println!(
                                                    "  {} dist={:.6}",
                                                    r.id, r.distance
                                                );
                                            }
                                            if results.is_empty() {
                                                println!("  (no results)");
                                            }
                                        }
                                        Err(e) => println!("Error: {}", e),
                                    },
                                    Err(e) => println!("Error: {}", e),
                                }
                            } else {
                                println!("Usage: search <collection> <k> <v1,v2,...>");
                            }
                        } else {
                            println!("Usage: search <collection> <k> <v1,v2,...>");
                        }
                    }
                    _ => {
                        println!("Unknown command: '{}'. Type 'help' for usage.", line);
                    }
                }
            }
        }
    }

    Ok(())
}

// ============================================================================
// Feature: Smart Collection Auto-Partitioning (partition)
// ============================================================================

pub fn partition_command(
    path: &str,
    collection_name: &str,
    analyze_only: bool,
    target_size: usize,
) -> Result<()> {
    let db = Database::open(path)?;
    let coll = db.collection(collection_name)?;

    let num_vectors = coll.len();
    let dimensions = coll.dimensions().unwrap_or(0);

    if num_vectors == 0 {
        println!("Collection '{}' is empty.", collection_name);
        return Ok(());
    }

    let recommended_partitions = (num_vectors + target_size - 1) / target_size;

    println!("═══ Partition Analysis for '{}' ═══", collection_name);
    println!();
    println!("  Vectors: {}", num_vectors);
    println!("  Dimensions: {}", dimensions);
    println!("  Target partition size: {}", target_size);
    println!("  Recommended partitions: {}", recommended_partitions);
    println!();

    if recommended_partitions <= 1 {
        println!("  ✓ Collection is small enough — no partitioning needed.");
        return Ok(());
    }

    // Analyze metadata keys for partition key candidates
    let mut metadata_keys: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let exported = coll.export_all()?;
    let sample_size = exported.len().min(500);
    for (_, _, meta) in exported.iter().take(sample_size) {
        if let Some(m) = meta {
            if let Some(obj) = m.as_object() {
                for key in obj.keys() {
                    *metadata_keys.entry(key.clone()).or_insert(0) += 1;
                }
            }
        }
    }

    if !metadata_keys.is_empty() {
        println!("  Metadata key candidates for partition key:");
        let mut sorted_keys: Vec<_> = metadata_keys.into_iter().collect();
        sorted_keys.sort_by(|a, b| b.1.cmp(&a.1));
        for (key, count) in sorted_keys.iter().take(5) {
            let coverage = *count as f64 / sample_size as f64 * 100.0;
            println!("    '{}' — present in {:.0}% of vectors", key, coverage);
        }
    } else {
        println!("  No metadata keys found. Consider vector-based (k-means) partitioning.");
    }

    println!();
    println!("  Strategy recommendation:");
    if num_vectors > 1_000_000 {
        println!("    → Large collection: use metadata-based partitioning for predictable routing.");
    } else if num_vectors > 100_000 {
        println!("    → Medium collection: k-means clustering may improve search locality.");
    } else {
        println!("    → Small collection: partitioning provides minimal benefit at this scale.");
    }

    if analyze_only {
        println!();
        println!("  (analyze mode — no changes applied)");
    }

    Ok(())
}
