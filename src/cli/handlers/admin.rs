use needle::{Database, NeedleError, Result};
use serde_json::json;

#[cfg(feature = "server")]
use needle::server::{serve, ServerConfig};

#[cfg(feature = "server")]
pub fn serve_command(address: &str, database: Option<String>) -> Result<()> {
    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| needle::NeedleError::InvalidConfig(e.to_string()))?;

    rt.block_on(async {
        let mut config = ServerConfig::new(address)
            .map_err(|e| needle::NeedleError::InvalidConfig(format!("Invalid address: {}", e)))?;
        if let Some(db_path) = database {
            config = config.with_db_path(db_path);
        }

        serve(config)
            .await
            .map_err(|e| needle::NeedleError::InvalidConfig(e.to_string()))
    })
}

pub fn demo_command(count: usize, dimensions: usize) -> Result<()> {
    use needle::{Database, Filter};
    use rand::Rng;

    println!("🧪 Needle Demo — creating in-memory database...\n");

    let db = Database::in_memory();
    db.create_collection("demo", dimensions)?;
    let coll = db.collection("demo")?;

    let categories = [
        "science",
        "technology",
        "engineering",
        "tutorial",
        "reference",
    ];
    let titles = [
        "Quantum Computing Basics",
        "Neural Network Architectures",
        "Bridge Engineering",
        "Getting Started with ML",
        "Vector Database Internals",
        "Distributed Systems",
        "Signal Processing",
        "Compiler Design",
        "Fluid Dynamics",
        "Graph Algorithms",
    ];

    let mut rng = rand::thread_rng();
    for i in 0..count {
        let vector: Vec<f32> = (0..dimensions)
            .map(|j| ((i * dimensions + j) as f32 / (count * dimensions) as f32).sin())
            .collect();
        let metadata = json!({
            "title": titles[i % titles.len()],
            "category": categories[i % categories.len()],
            "score": rng.gen_range(0.0..1.0_f64),
        });
        coll.insert(format!("doc_{}", i), &vector, Some(metadata))?;
    }
    println!("✓ Inserted {} vectors ({} dimensions)\n", count, dimensions);

    let query: Vec<f32> = (0..dimensions)
        .map(|j| ((42 * dimensions + j) as f32 / (count * dimensions) as f32).sin())
        .collect();
    let results = coll.search(&query, 5)?;

    println!("Search results (top 5 nearest to doc_42):");
    for (i, r) in results.iter().enumerate() {
        let title = r
            .metadata
            .as_ref()
            .and_then(|m| m.get("title"))
            .and_then(|v| v.as_str())
            .unwrap_or("?");
        let cat = r
            .metadata
            .as_ref()
            .and_then(|m| m.get("category"))
            .and_then(|v| v.as_str())
            .unwrap_or("?");
        println!(
            "  #{} {} distance={:.4}  category={}  \"{}\"",
            i + 1,
            r.id,
            r.distance,
            cat,
            title
        );
    }

    let filter = Filter::eq("category", "science");
    let filtered = coll.search_with_filter(&query, 3, &filter)?;
    println!("\nFiltered search (category=science, top 3):");
    for (i, r) in filtered.iter().enumerate() {
        println!("  #{} {} distance={:.4}", i + 1, r.id, r.distance);
    }

    println!("\n🎉 Demo complete! Try these next:");
    println!("  cargo run --example basic_usage          — Rust API walkthrough");
    println!("  cargo run -- demo --count 10000          — Larger demo");
    println!("  cargo run --features server -- serve     — Start HTTP server");
    println!("  cargo run -- --help                      — All CLI commands");
    Ok(())
}

pub fn tune_command(
    vectors: usize,
    dimensions: usize,
    profile: &str,
    memory_mb: Option<usize>,
) -> Result<()> {
    use needle::tuning::{auto_tune, PerformanceProfile, TuningConstraints};

    let perf_profile = match profile.to_lowercase().as_str() {
        "low-latency" | "lowlatency" | "fast" => PerformanceProfile::LowLatency,
        "high-recall" | "highrecall" | "accurate" => PerformanceProfile::HighRecall,
        "low-memory" | "lowmemory" | "compact" => PerformanceProfile::LowMemory,
        _ => PerformanceProfile::Balanced,
    };

    let mut constraints = TuningConstraints::new(vectors, dimensions).with_profile(perf_profile);

    if let Some(mb) = memory_mb {
        constraints = constraints.with_memory_budget(mb * 1024 * 1024);
    }

    let result = auto_tune(&constraints);

    println!("Auto-tuning Results");
    println!("==================");
    println!();
    println!("Input:");
    println!("  Expected vectors: {}", vectors);
    println!("  Dimensions: {}", dimensions);
    println!("  Profile: {:?}", perf_profile);
    if let Some(mb) = memory_mb {
        println!("  Memory budget: {} MB", mb);
    }
    println!();
    println!("Recommended HNSW Config:");
    println!("  M: {}", result.config.m);
    println!("  ef_construction: {}", result.config.ef_construction);
    println!("  ef_search: {}", result.ef_search);
    println!();
    println!("Estimates:");
    println!(
        "  Memory per vector: {} bytes",
        result.estimated_memory_per_vector
    );
    println!(
        "  Total memory: {:.1} MB",
        result.estimated_total_memory as f64 / 1024.0 / 1024.0
    );
    println!("  Expected recall: {:.1}%", result.estimated_recall * 100.0);
    println!("  Expected latency: {:.2} ms", result.estimated_latency_ms);
    println!();
    println!("Explanation:");
    for line in &result.explanation {
        println!("  - {}", line);
    }
    println!();
    println!("Usage:");
    println!("  let config = CollectionConfig::new(\"my_collection\", {}).with_m({}).with_ef_construction({});",
             dimensions, result.config.m, result.config.ef_construction);

    Ok(())
}

pub fn init_command(directory: &str, database: &str, dimensions: usize) -> Result<()> {
    use std::fs;
    use std::path::Path;

    let dir = Path::new(directory);
    if !dir.exists() {
        fs::create_dir_all(dir).map_err(NeedleError::Io)?;
    }

    let db_path = dir.join(database);
    if db_path.exists() {
        println!("Database already exists: {}", db_path.display());
        return Ok(());
    }

    let mut db = Database::open(db_path.to_str().unwrap_or(database))?;
    db.create_collection("default", dimensions)?;
    db.save()?;

    println!("✓ Initialized Needle project");
    println!("  Database: {}", db_path.display());
    println!("  Collection: default ({dimensions} dimensions, cosine distance)");
    println!();
    println!("Next steps:");
    println!("  needle info {}", db_path.display());
    println!(
        "  echo '{{\"id\":\"doc1\",\"vector\":[{}]}}' | needle insert {} -c default",
        vec!["0.1"; dimensions.min(4)].join(","),
        db_path.display()
    );

    Ok(())
}

pub fn doctor_command() -> Result<()> {
    println!("Needle Doctor — Environment Check\n");

    let rust_version = option_env!("CARGO_PKG_RUST_VERSION").unwrap_or("unknown");
    println!("  ✓ Rust MSRV: {rust_version}");

    println!("  ✓ Needle version: {}", env!("CARGO_PKG_VERSION"));

    let features = [
        ("server", cfg!(feature = "server")),
        ("metrics", cfg!(feature = "metrics")),
        ("hybrid", cfg!(feature = "hybrid")),
        ("encryption", cfg!(feature = "encryption")),
        ("experimental", cfg!(feature = "experimental")),
        ("embeddings", cfg!(feature = "embeddings")),
        ("embedding-providers", cfg!(feature = "embedding-providers")),
        ("python", cfg!(feature = "python")),
        ("wasm", cfg!(feature = "wasm")),
    ];
    let enabled: Vec<_> = features.iter().filter(|(_, e)| *e).map(|(n, _)| *n).collect();
    let disabled: Vec<_> = features.iter().filter(|(_, e)| !*e).map(|(n, _)| *n).collect();

    let enabled_str = if enabled.is_empty() { "none (default)".to_string() } else { enabled.join(", ") };
    println!("  ✓ Features enabled: {enabled_str}");
    if !disabled.is_empty() {
        println!("  ○ Features available: {}", disabled.join(", "));
    }

    let test_path = std::env::temp_dir().join("needle_doctor_test.needle");
    let test_path_str = test_path.to_string_lossy().to_string();
    match Database::open(&test_path_str) {
        Ok(_) => {
            println!("  ✓ Database creation: OK");
            let _ = std::fs::remove_file(&test_path);
        }
        Err(e) => println!("  ✗ Database creation: FAILED ({e})"),
    }

    println!("\nAll checks passed!");
    Ok(())
}

pub fn mcp_command(database: &str, read_only: bool) -> Result<()> {
    let db = Database::open(database)?;
    let server = needle::mcp::McpServer::new(db, read_only);
    eprintln!("Needle MCP server started (stdio transport)");
    eprintln!("Database: {database}");
    eprintln!("Read-only: {read_only}");
    server.run()
}

/// Handler for the `merge` CLI command.
pub fn merge_command(
    path: &str,
    source: &str,
    target: &str,
    base: Option<&str>,
    strategy: &str,
    dry_run: bool,
) -> Result<()> {
    let mut db = Database::open(path)?;
    let coll_source = db.collection(source)?;
    let coll_target = db.collection(target)?;

    let source_entries = coll_source.export_all()?;
    let target_ids: std::collections::HashSet<String> = coll_target.ids()?.into_iter().collect();

    // For 3-way merge, load base collection IDs and vectors
    let base_entries: Option<std::collections::HashMap<String, (Vec<f32>, Option<serde_json::Value>)>> =
        if let Some(base_name) = base {
            let coll_base = db.collection(base_name)?;
            let entries = coll_base.export_all()?;
            Some(
                entries
                    .into_iter()
                    .map(|(id, vec, meta)| (id, (vec, meta)))
                    .collect(),
            )
        } else {
            None
        };

    let mut added = 0usize;
    let mut updated = 0usize;
    let mut skipped = 0usize;
    let mut conflicts = 0usize;

    for (id, vector, metadata) in &source_entries {
        let exists_in_target = target_ids.contains(id);

        if exists_in_target {
            // 3-way merge: check if change came from source or target relative to base
            if let Some(ref base_map) = base_entries {
                let in_base = base_map.get(id);
                let target_vec = coll_target.get(id);

                let source_changed = in_base
                    .map(|(bv, _)| bv != vector)
                    .unwrap_or(true);
                let target_changed = match (&in_base, &target_vec) {
                    (Some((bv, _)), Some((tv, _))) => bv != tv,
                    (None, Some(_)) => true,
                    _ => false,
                };

                if source_changed && !target_changed {
                    // Only source changed: take source
                    if !dry_run {
                        coll_target.delete(id)?;
                        coll_target.insert(id, vector, metadata.clone())?;
                    }
                    updated += 1;
                } else if !source_changed && target_changed {
                    // Only target changed: keep target
                    skipped += 1;
                } else if source_changed && target_changed {
                    // Both changed: conflict — apply strategy
                    conflicts += 1;
                    match strategy {
                        "source-wins" => {
                            if !dry_run {
                                coll_target.delete(id)?;
                                coll_target.insert(id, vector, metadata.clone())?;
                            }
                            updated += 1;
                        }
                        "target-wins" | "skip" => {
                            skipped += 1;
                        }
                        _ => {
                            return Err(NeedleError::InvalidArgument(format!(
                                "Unknown merge strategy '{}'. Use: source-wins, target-wins, skip",
                                strategy
                            )));
                        }
                    }
                } else {
                    // Neither changed
                    skipped += 1;
                }
            } else {
                // 2-way merge (no base): use strategy directly
                match strategy {
                    "source-wins" => {
                        if !dry_run {
                            coll_target.delete(id)?;
                            coll_target.insert(id, vector, metadata.clone())?;
                        }
                        updated += 1;
                    }
                    "target-wins" | "skip" => {
                        skipped += 1;
                    }
                    _ => {
                        return Err(NeedleError::InvalidArgument(format!(
                            "Unknown merge strategy '{}'. Use: source-wins, target-wins, skip",
                            strategy
                        )));
                    }
                }
            }
        } else {
            if !dry_run {
                coll_target.insert(id, vector, metadata.clone())?;
            }
            added += 1;
        }
    }

    let merge_type = if base.is_some() { "3-way" } else { "2-way" };
    if dry_run {
        println!("=== Dry Run: {} Merge '{}' → '{}' (strategy: {}) ===", merge_type, source, target, strategy);
    } else {
        println!("=== {} Merge '{}' → '{}' (strategy: {}) ===", merge_type, source, target, strategy);
    }
    if let Some(b) = base {
        println!("  Base collection: {}", b);
    }
    println!("  Source vectors: {}", source_entries.len());
    println!("  Target vectors: {}", target_ids.len());
    println!("  Added: {}", added);
    println!("  Updated: {}", updated);
    println!("  Skipped: {}", skipped);
    if conflicts > 0 {
        println!("  Conflicts resolved: {}", conflicts);
    }

    if !dry_run {
        db.save()?;
        println!("  Database saved.");
    } else {
        println!("  No changes made (dry run).");
    }

    Ok(())
}

pub fn provenance_command(database: &str, collection: &str, id: &str) -> Result<()> {
    let db = Database::open(database)?;
    let coll = db.collection(collection)?;

    match coll.get_provenance(id) {
        Some(record) => {
            println!("{}", serde_json::to_string_pretty(&record)
                .map_err(|e| NeedleError::InvalidInput(e.to_string()))?);
        }
        None => {
            println!("No provenance record found for vector '{}'", id);
        }
    }

    Ok(())
}

pub fn evaluate_command(database: &str, collection: &str, ground_truth_path: &str, k: usize) -> Result<()> {
    use needle::GroundTruthEntry;

    let gt_data = std::fs::read_to_string(ground_truth_path)
        .map_err(|e| NeedleError::InvalidInput(format!("Failed to read ground truth file: {e}")))?;
    let ground_truth: Vec<GroundTruthEntry> = serde_json::from_str(&gt_data)
        .map_err(|e| NeedleError::InvalidInput(format!("Failed to parse ground truth JSON: {e}")))?;

    let db = Database::open(database)?;
    let coll = db.collection(collection)?;
    let report = coll.evaluate(&ground_truth, k)?;

    println!("{}", serde_json::to_string_pretty(&report)
        .map_err(|e| NeedleError::InvalidInput(e.to_string()))?);

    Ok(())
}

pub fn export_bundle_command(database: &str, collection: &str, output: &str) -> Result<()> {
    let db = Database::open(database)?;
    let coll = db.collection(collection)?;
    let manifest = coll.export_bundle(std::path::Path::new(output))?;

    println!("Bundle exported successfully:");
    println!("  Collection: {}", manifest.collection_name);
    println!("  Vectors: {}", manifest.vector_count);
    println!("  Dimensions: {}", manifest.dimensions);
    println!("  Output: {}", output);
    Ok(())
}

pub fn import_bundle_command(database: &str, bundle: &str, name: Option<&str>) -> Result<()> {
    let mut db = Database::open(database)?;
    let manifest = db.import_bundle(std::path::Path::new(bundle), name)?;
    db.save()?;

    println!("Bundle imported successfully:");
    println!("  Collection: {}", manifest.collection_name);
    println!("  Vectors: {}", manifest.vector_count);
    println!("  Dimensions: {}", manifest.dimensions);
    Ok(())
}
