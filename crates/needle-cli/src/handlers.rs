use needle::{CollectionConfig, Database, DistanceFunction, NeedleError, Result};
use serde_json::json;
use std::io::{self, BufRead};

#[cfg(feature = "server")]
use needle::server::{serve, ServerConfig};

use needle::query_builder::{QueryAnalyzer, VisualQueryBuilder};

pub(crate) fn info_command(path: &str, json_output: bool) -> Result<()> {
    let db = Database::open(path)?;

    if json_output {
        let collections: Vec<serde_json::Value> = db.list_collections().iter().map(|name| {
            let coll = db.collection(name).ok();
            json!({
                "name": name,
                "dimensions": coll.as_ref().map(|c| c.dimensions()),
                "vectors": coll.as_ref().map(|c| c.len()),
            })
        }).collect();
        println!("{}", serde_json::to_string_pretty(&json!({
            "database": path,
            "collection_count": db.list_collections().len(),
            "total_vectors": db.total_vectors(),
            "collections": collections,
        })).unwrap_or_default());
        return Ok(());
    }

    println!("Database: {}", path);
    println!("Collections: {}", db.list_collections().len());
    println!("Total vectors: {}", db.total_vectors());
    println!();

    for name in db.list_collections() {
        let coll = db.collection(&name)?;
        println!("  Collection: {}", name);
        println!("    Dimensions: {:?}", coll.dimensions());
        println!("    Vectors: {}", coll.len());
    }

    Ok(())
}

pub(crate) fn create_command(path: &str) -> Result<()> {
    let _db = Database::open(path)?;
    println!("Created database: {}", path);
    Ok(())
}

pub(crate) fn collections_command(path: &str, json_output: bool) -> Result<()> {
    let db = Database::open(path)?;

    let collections = db.list_collections();
    if json_output {
        let items: Vec<serde_json::Value> = collections.iter().map(|name| {
            let coll = db.collection(name).ok();
            json!({
                "name": name,
                "dimensions": coll.as_ref().map(|c| c.dimensions()),
                "vectors": coll.as_ref().map(|c| c.len()),
            })
        }).collect();
        println!("{}", serde_json::to_string_pretty(&json!({ "collections": items })).unwrap_or_default());
        return Ok(());
    }

    if collections.is_empty() {
        println!("No collections found.");
    } else {
        println!("Collections:");
        for name in collections {
            let coll = db.collection(&name)?;
            println!(
                "  {} (dimensions: {:?}, vectors: {})",
                name,
                coll.dimensions(),
                coll.len()
            );
        }
    }

    Ok(())
}

pub(crate) fn create_collection_command(
    path: &str,
    name: &str,
    dimensions: usize,
    distance: &str,
) -> Result<()> {
    if dimensions == 0 {
        return Err(NeedleError::InvalidConfig(
            "Vector dimensions must be greater than 0".to_string(),
        ));
    }
    let db = Database::open(path)?;

    let dist_fn = match parse_distance(distance) {
        Some(parsed) => parsed,
        None => {
            eprintln!("Unknown distance function: {}. Using cosine.", distance);
            DistanceFunction::Cosine
        }
    };

    let config = CollectionConfig::new(name, dimensions).with_distance(dist_fn);
    db.create_collection_with_config(config)?;
    db.save()?;

    println!(
        "Created collection '{}' with {} dimensions ({} distance)",
        name, dimensions, distance
    );
    Ok(())
}

pub(crate) fn stats_command(path: &str, collection_name: &str, json_output: bool) -> Result<()> {
    let db = Database::open(path)?;
    let coll = db.collection(collection_name)?;

    let active_count = coll.len();
    let deleted_count = coll.deleted_count();
    let total_stored = active_count + deleted_count;

    if json_output {
        let delete_ratio = if total_stored > 0 { deleted_count as f64 / total_stored as f64 } else { 0.0 };
        println!("{}", serde_json::to_string_pretty(&json!({
            "collection": collection_name,
            "dimensions": coll.dimensions(),
            "active_vectors": active_count,
            "deleted_vectors": deleted_count,
            "total_stored": total_stored,
            "empty": coll.is_empty(),
            "deletion_ratio": delete_ratio,
            "needs_compaction": coll.needs_compaction(0.2),
        })).unwrap_or_default());
        return Ok(());
    }

    println!("Collection: {}", collection_name);
    println!("  Dimensions: {:?}", coll.dimensions());
    println!("  Active vectors: {}", active_count);
    println!("  Deleted vectors: {}", deleted_count);
    println!("  Total stored: {}", total_stored);
    println!("  Empty: {}", coll.is_empty());

    if deleted_count > 0 {
        let delete_ratio = deleted_count as f64 / total_stored as f64;
        println!("  Deletion ratio: {:.1}%", delete_ratio * 100.0);
        if coll.needs_compaction(0.2) {
            println!("  Recommendation: Run 'compact' to reclaim space");
        }
    }

    Ok(())
}

pub(crate) fn insert_command(path: &str, collection_name: &str) -> Result<()> {
    let db = Database::open(path)?;
    let coll = db.collection(collection_name)?;

    println!("Reading vectors from stdin (JSON format: {{\"id\": \"...\", \"vector\": [...], \"metadata\": {{...}}}})");
    println!("Press Ctrl+D when done.");

    let stdin = io::stdin();
    let mut count = 0;

    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let value: serde_json::Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("Invalid JSON: {}", e);
                continue;
            }
        };

        let id = value["id"].as_str().unwrap_or("").to_string();
        let vector: Vec<f32> = match value["vector"].as_array() {
            Some(arr) => arr
                .iter()
                .filter_map(|v| v.as_f64().map(|f| f as f32))
                .collect(),
            None => {
                eprintln!("Missing 'vector' field");
                continue;
            }
        };

        let metadata = value.get("metadata").cloned();

        match coll.insert(&id, &vector, metadata) {
            Ok(_) => count += 1,
            Err(e) => eprintln!("Error inserting '{}': {}", id, e),
        }
    }

    db.save()?;
    println!("Inserted {} vectors.", count);
    Ok(())
}

pub(crate) fn search_command(
    path: &str,
    collection_name: &str,
    query_str: &str,
    k: usize,
    explain: bool,
    distance_override: Option<&str>,
    json_output: bool,
) -> Result<()> {
    let db = Database::open(path)?;
    let coll = db.collection(collection_name)?;

    let query = parse_query_vector(query_str)?;

    let distance_fn = distance_override.and_then(|d| match parse_distance(d) {
        Some(parsed) => Some(parsed),
        None => {
            eprintln!(
                "Warning: Unknown distance function '{}', using collection default",
                d
            );
            None
        }
    });

    if explain {
        if distance_override.is_some() {
            eprintln!("Warning: --explain and --distance cannot be combined; ignoring --distance");
        }
        let (results, explain_data) = coll.search_explain(&query, k)?;

        if json_output {
            let result_json: Vec<serde_json::Value> = results.iter().map(|r| {
                json!({"id": r.id, "distance": r.distance, "metadata": r.metadata})
            }).collect();
            println!("{}", serde_json::to_string_pretty(&json!({
                "results": result_json,
                "explain": {
                    "total_time_us": explain_data.total_time_us,
                    "index_time_us": explain_data.index_time_us,
                    "filter_time_us": explain_data.filter_time_us,
                    "enrich_time_us": explain_data.enrich_time_us,
                    "hnsw_stats": {
                        "visited_nodes": explain_data.hnsw_stats.visited_nodes,
                        "layers_traversed": explain_data.hnsw_stats.layers_traversed,
                        "distance_computations": explain_data.hnsw_stats.distance_computations,
                        "traversal_time_us": explain_data.hnsw_stats.traversal_time_us,
                    },
                    "dimensions": explain_data.dimensions,
                    "collection_size": explain_data.collection_size,
                    "requested_k": explain_data.requested_k,
                    "effective_k": explain_data.effective_k,
                    "ef_search": explain_data.ef_search,
                    "distance_function": explain_data.distance_function,
                }
            })).unwrap_or_default());
            return Ok(());
        }

        println!("Search results (k={}):", k);
        for result in &results {
            let meta = result
                .metadata
                .as_ref()
                .map_or_else(|| "null".to_string(), |m| m.to_string());
            println!(
                "  ID: {}, Distance: {:.6}, Metadata: {}",
                result.id, result.distance, meta
            );
        }

        println!();
        println!("Query Profiling:");
        println!("  Total time: {}μs", explain_data.total_time_us);
        println!("  Index traversal: {}μs", explain_data.index_time_us);
        println!("  Filter evaluation: {}μs", explain_data.filter_time_us);
        println!("  Result enrichment: {}μs", explain_data.enrich_time_us);
        println!();
        println!("HNSW Statistics:");
        println!("  Visited nodes: {}", explain_data.hnsw_stats.visited_nodes);
        println!(
            "  Layers traversed: {}",
            explain_data.hnsw_stats.layers_traversed
        );
        println!(
            "  Distance computations: {}",
            explain_data.hnsw_stats.distance_computations
        );
        println!(
            "  Traversal time: {}μs",
            explain_data.hnsw_stats.traversal_time_us
        );
        println!();
        println!("Query Parameters:");
        println!("  Dimensions: {}", explain_data.dimensions);
        println!("  Collection size: {}", explain_data.collection_size);
        println!("  Requested k: {}", explain_data.requested_k);
        println!("  Effective k: {}", explain_data.effective_k);
        println!("  ef_search: {}", explain_data.ef_search);
        println!("  Distance function: {}", explain_data.distance_function);
    } else {
        let results = if let Some(dist) = distance_fn {
            coll.search_with_options(&query, k, Some(dist), None, None, 3)?
        } else {
            coll.search(&query, k)?
        };

        if json_output {
            let result_json: Vec<serde_json::Value> = results.iter().map(|r| {
                json!({"id": r.id, "distance": r.distance, "metadata": r.metadata})
            }).collect();
            println!("{}", serde_json::to_string_pretty(&json!({
                "results": result_json,
                "k": k,
            })).unwrap_or_default());
            return Ok(());
        }

        println!("Search results (k={}):", k);
        if let Some(dist) = distance_fn {
            println!("  (using distance override: {:?})", dist);
        }
        for result in results {
            let meta = result
                .metadata
                .as_ref()
                .map_or_else(|| "null".to_string(), |m| m.to_string());
            println!(
                "  ID: {}, Distance: {:.6}, Metadata: {}",
                result.id, result.distance, meta
            );
        }
    }

    Ok(())
}

pub(crate) fn parse_distance(distance: &str) -> Option<DistanceFunction> {
    match distance.to_lowercase().as_str() {
        "cosine" => Some(DistanceFunction::Cosine),
        "euclidean" | "l2" => Some(DistanceFunction::Euclidean),
        "dot" | "dotproduct" => Some(DistanceFunction::DotProduct),
        "manhattan" | "l1" => Some(DistanceFunction::Manhattan),
        _ => None,
    }
}

pub(crate) fn parse_query_vector(query_str: &str) -> Result<Vec<f32>> {
    let mut values = Vec::new();
    for part in query_str.split(',') {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            continue;
        }
        let value = trimmed.parse::<f32>().map_err(|_| {
            NeedleError::InvalidVector(format!("Invalid float value '{}'", trimmed))
        })?;
        values.push(value);
    }

    if values.is_empty() {
        return Err(NeedleError::InvalidVector(
            "Query vector must contain at least one value".to_string(),
        ));
    }

    Ok(values)
}

pub(crate) fn delete_command(path: &str, collection_name: &str, id: &str) -> Result<()> {
    let db = Database::open(path)?;
    let coll = db.collection(collection_name)?;

    let deleted = coll.delete(id)?;
    db.save()?;

    if deleted {
        println!("Deleted vector '{}'", id);
    } else {
        println!("Vector '{}' not found", id);
    }

    Ok(())
}

pub(crate) fn get_command(path: &str, collection_name: &str, id: &str, json_output: bool) -> Result<()> {
    let db = Database::open(path)?;
    let coll = db.collection(collection_name)?;

    match coll.get(id) {
        Some((vector, metadata)) => {
            let output = json!({
                "id": id,
                "vector": vector,
                "metadata": metadata
            });
            println!("{}", serde_json::to_string_pretty(&output)?);
        }
        None => {
            if json_output {
                println!("{}", serde_json::to_string_pretty(&json!({"error": "not_found", "id": id})).unwrap_or_default());
            } else {
                println!("Vector '{}' not found", id);
            }
        }
    }

    Ok(())
}

pub(crate) fn compact_command(path: &str) -> Result<()> {
    let db = Database::open(path)?;

    let mut total_deleted = 0;
    for name in db.list_collections() {
        let coll = db.collection(&name)?;
        let deleted = coll.compact()?;
        if deleted > 0 {
            println!("  {}: removed {} deleted vectors", name, deleted);
            total_deleted += deleted;
        }
    }

    db.save()?;

    if total_deleted > 0 {
        println!(
            "Compaction complete: removed {} total deleted vectors",
            total_deleted
        );
    } else {
        println!("No deleted vectors to compact");
    }

    Ok(())
}

pub(crate) fn export_command(path: &str, collection_name: &str) -> Result<()> {
    let db = Database::open(path)?;
    let coll = db.collection(collection_name)?;

    let vectors = coll.export_all()?;

    let output = json!({
        "collection": collection_name,
        "dimensions": coll.dimensions(),
        "count": vectors.len(),
        "vectors": vectors.iter().map(|(id, vec, meta)| {
            json!({
                "id": id,
                "vector": vec,
                "metadata": meta
            })
        }).collect::<Vec<_>>()
    });

    println!("{}", serde_json::to_string_pretty(&output)?);

    Ok(())
}

pub(crate) fn import_command(path: &str, collection_name: &str, file_path: &str) -> Result<()> {
    let db = Database::open(path)?;
    let coll = db.collection(collection_name)?;

    // Read from file or stdin
    let content: String = if file_path == "-" {
        let stdin = io::stdin();
        let mut buffer = String::new();
        for line in stdin.lock().lines() {
            buffer.push_str(&line?);
            buffer.push('\n');
        }
        buffer
    } else {
        std::fs::read_to_string(file_path)?
    };

    let data: serde_json::Value = serde_json::from_str(&content)?;

    let vectors = data["vectors"]
        .as_array()
        .ok_or_else(|| needle::NeedleError::InvalidConfig("Missing 'vectors' array".to_string()))?;

    let mut count = 0;
    let mut errors = 0;

    for entry in vectors {
        let id = match entry["id"].as_str() {
            Some(id) => id.to_string(),
            None => {
                eprintln!("Skipping entry without 'id'");
                errors += 1;
                continue;
            }
        };

        let vector: Vec<f32> = match entry["vector"].as_array() {
            Some(arr) => arr
                .iter()
                .filter_map(|v| v.as_f64().map(|f| f as f32))
                .collect(),
            None => {
                eprintln!("Skipping '{}': missing 'vector'", id);
                errors += 1;
                continue;
            }
        };

        let metadata = entry.get("metadata").cloned();

        match coll.insert(&id, &vector, metadata) {
            Ok(_) => count += 1,
            Err(e) => {
                eprintln!("Error inserting '{}': {}", id, e);
                errors += 1;
            }
        }
    }

    db.save()?;

    println!("Imported {} vectors", count);
    if errors > 0 {
        println!("Skipped {} entries due to errors", errors);
    }

    Ok(())
}

pub(crate) fn count_command(path: &str, collection_name: &str, json_output: bool) -> Result<()> {
    let db = Database::open(path)?;
    let coll = db.collection(collection_name)?;

    if json_output {
        println!("{}", serde_json::to_string_pretty(&json!({
            "collection": collection_name,
            "count": coll.len(),
        })).unwrap_or_default());
    } else {
        println!("{}", coll.len());
    }

    Ok(())
}

pub(crate) fn clear_command(path: &str, collection_name: &str, force: bool) -> Result<()> {
    if !force {
        eprint!(
            "Are you sure you want to delete all vectors from '{}'? [y/N] ",
            collection_name
        );
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        if !input.trim().eq_ignore_ascii_case("y") {
            println!("Aborted.");
            return Ok(());
        }
    }

    let db = Database::open(path)?;
    let coll = db.collection(collection_name)?;

    let ids = coll.ids()?;
    let count = ids.len();

    for id in ids {
        coll.delete(&id)?;
    }

    // Compact to reclaim space
    coll.compact()?;

    db.save()?;

    println!("Deleted {} vectors from '{}'", count, collection_name);

    Ok(())
}

#[cfg(feature = "server")]
pub(crate) fn serve_command(address: &str, database: Option<String>) -> Result<()> {
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

pub(crate) fn tune_command(
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

// ============================================================================
// Natural Language Query Command
// ============================================================================

pub(crate) fn query_command(
    path: &str,
    collection_name: &str,
    query_str: &str,
    _k: usize,
    analyze: bool,
) -> Result<()> {
    use needle::query_builder::CollectionProfile;

    let db = Database::open(path)?;
    let coll = db.collection(collection_name)?;

    // Build collection profile from actual collection
    let vector_count = coll.len();
    let dimensions = coll.dimensions().unwrap_or(128);
    let profile = CollectionProfile::new(collection_name, dimensions, vector_count);

    // Build query using natural language interface
    let builder = VisualQueryBuilder::new(profile);
    let analyzer = QueryAnalyzer::new();

    // Parse the natural language query
    let build_result = builder.build(query_str);

    println!("Natural Language Query Interface");
    println!("=================================");
    println!();
    println!("Input: \"{}\"", query_str);
    println!();

    // Show the translated query
    println!("Translated Query:");
    println!("  NeedleQL: {}", build_result.needleql);
    println!(
        "  Quality Score: {:.1}%",
        build_result.quality_score * 100.0
    );
    println!();

    println!("Query Analysis:");
    println!("  Class: {:?}", build_result.analysis.class);
    println!("  Complexity: {:?}", build_result.analysis.complexity);
    println!(
        "  Confidence: {:.1}%",
        build_result.analysis.confidence * 100.0
    );
    println!();

    if analyze {
        // Show optimization hints
        if !build_result.optimization_hints.is_empty() {
            println!("Optimization Hints:");
            for hint in &build_result.optimization_hints {
                let severity = match hint.severity {
                    needle::query_builder::HintSeverity::Info => "INFO",
                    needle::query_builder::HintSeverity::Suggestion => "SUGG",
                    needle::query_builder::HintSeverity::Warning => "WARN",
                    needle::query_builder::HintSeverity::Critical => "CRIT",
                };
                println!("  [{}] {:?}: {}", severity, hint.category, hint.message);
                println!("       Suggestion: {}", hint.suggestion);
            }
            println!();
        }

        // Analyze the generated NeedleQL for additional insights
        let further_analysis = analyzer.analyze(&build_result.needleql);
        if !further_analysis.patterns.is_empty() {
            println!("Detected Patterns:");
            for pattern in &further_analysis.patterns {
                println!(
                    "  - {:?}: \"{}\"",
                    pattern.pattern_type, pattern.matched_text
                );
            }
            println!();
        }
    }

    if !build_result.suggestions.is_empty() {
        println!("Suggestions:");
        for suggestion in &build_result.suggestions {
            println!(
                "  - {:?}: {}",
                suggestion.suggestion_type, suggestion.message
            );
        }
        println!();
    }

    if !build_result.alternatives.is_empty() {
        println!("Alternative Queries:");
        for alt in &build_result.alternatives {
            println!("  - {}", alt.needleql);
            println!("    {}", alt.description);
        }
        println!();
    }

    if build_result.parsed.is_some() {
        println!("Query parsed successfully - ready for execution.");
    } else {
        println!("Note: Query could not be fully parsed. Review the NeedleQL syntax.");
    }

    Ok(())
}

/// Run diagnostics on a database, checking file integrity, collection health,
/// and metadata consistency.
pub(crate) fn diagnose_command(path: &str, extended: bool, json_output: bool) -> Result<()> {
    use std::time::Instant;

    let start = Instant::now();
    let mut issues: Vec<serde_json::Value> = Vec::new();
    let mut checks_passed = 0u32;
    let mut checks_failed = 0u32;

    // Check 1: Can we open the database?
    let db = match Database::open(path) {
        Ok(db) => {
            checks_passed += 1;
            if !json_output { println!("✓ Database file opens successfully"); }
            db
        }
        Err(e) => {
            checks_failed += 1;
            if json_output {
                issues.push(json!({"check": "database_open", "severity": "critical", "message": format!("{}", e)}));
                println!("{}", serde_json::to_string_pretty(&json!({
                    "database": path,
                    "status": "error",
                    "checks_passed": checks_passed,
                    "checks_failed": checks_failed,
                    "issues": issues,
                    "elapsed_ms": start.elapsed().as_millis(),
                })).unwrap_or_default());
            } else {
                println!("✗ Failed to open database: {}", e);
            }
            return Err(e);
        }
    };

    // Check 2: File metadata
    if let Ok(meta) = std::fs::metadata(path) {
        checks_passed += 1;
        let size_mb = meta.len() as f64 / 1_048_576.0;
        if !json_output { println!("✓ File size: {:.2} MB", size_mb); }
    }

    // Check 3: Collections integrity
    let collections = db.list_collections();
    if !json_output { println!("✓ Collections found: {}", collections.len()); }
    checks_passed += 1;

    for name in &collections {
        match db.collection(name) {
            Ok(coll) => {
                let active = coll.len();
                let deleted = coll.deleted_count();
                let dims = coll.dimensions().unwrap_or(0);
                checks_passed += 1;

                if !json_output {
                    println!("  ✓ '{}': {} active, {} deleted, {} dims", name, active, deleted, dims);
                }

                // Check deletion ratio
                let total = active + deleted;
                if total > 0 {
                    let delete_ratio = deleted as f64 / total as f64;
                    if delete_ratio > 0.5 {
                        let msg = format!("Collection '{}' has {:.0}% deleted vectors — compaction recommended", name, delete_ratio * 100.0);
                        if !json_output { println!("  ⚠ {}", msg); }
                        issues.push(json!({"check": "deletion_ratio", "severity": "warning", "collection": name, "message": msg}));
                    }
                }

                // Extended: verify a sample vector is retrievable
                if extended && active > 0 {
                    if let Ok(all_ids) = coll.ids() {
                        if let Some(id) = all_ids.first() {
                            match coll.get(id) {
                                Some((vec, _)) => {
                                    if vec.len() == dims {
                                        checks_passed += 1;
                                        if !json_output { println!("    ✓ Sample vector '{}' valid ({} dims)", id, vec.len()); }
                                    } else {
                                        checks_failed += 1;
                                        let msg = format!("Vector '{}' has {} dims, expected {}", id, vec.len(), dims);
                                        if !json_output { println!("    ✗ {}", msg); }
                                        issues.push(json!({"check": "vector_dimensions", "severity": "error", "collection": name, "message": msg}));
                                    }
                                }
                                None => {
                                    checks_failed += 1;
                                    let msg = format!("Could not retrieve vector '{}' despite being listed", id);
                                    if !json_output { println!("    ✗ {}", msg); }
                                    issues.push(json!({"check": "vector_retrieval", "severity": "error", "collection": name, "message": msg}));
                                }
                            }
                        }
                    }
                }
            }
            Err(e) => {
                checks_failed += 1;
                let msg = format!("Failed to access collection '{}': {}", name, e);
                if !json_output { println!("  ✗ {}", msg); }
                issues.push(json!({"check": "collection_access", "severity": "error", "collection": name, "message": msg}));
            }
        }
    }

    let elapsed = start.elapsed();
    let status = if checks_failed == 0 { "healthy" } else { "issues_found" };

    if json_output {
        println!("{}", serde_json::to_string_pretty(&json!({
            "database": path,
            "status": status,
            "checks_passed": checks_passed,
            "checks_failed": checks_failed,
            "issues": issues,
            "collections": collections.len(),
            "total_vectors": db.total_vectors(),
            "elapsed_ms": elapsed.as_millis(),
        })).unwrap_or_default());
    } else {
        println!();
        println!("Diagnosis complete in {:.0}ms", elapsed.as_millis());
        println!("  Checks passed: {}", checks_passed);
        println!("  Checks failed: {}", checks_failed);
        if checks_failed == 0 {
            println!("  Status: ✓ Healthy");
        } else {
            println!("  Status: ⚠ Issues found ({})", issues.len());
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_distance_all_variants() {
        assert!(matches!(parse_distance("cosine"), Some(DistanceFunction::Cosine)));
        assert!(matches!(parse_distance("COSINE"), Some(DistanceFunction::Cosine)));
        assert!(matches!(parse_distance("euclidean"), Some(DistanceFunction::Euclidean)));
        assert!(matches!(parse_distance("l2"), Some(DistanceFunction::Euclidean)));
        assert!(matches!(parse_distance("dot"), Some(DistanceFunction::DotProduct)));
        assert!(matches!(parse_distance("dotproduct"), Some(DistanceFunction::DotProduct)));
        assert!(matches!(parse_distance("manhattan"), Some(DistanceFunction::Manhattan)));
        assert!(matches!(parse_distance("l1"), Some(DistanceFunction::Manhattan)));
        assert!(parse_distance("unknown").is_none());
        assert!(parse_distance("").is_none());
    }

    #[test]
    fn test_parse_query_vector_valid() {
        assert_eq!(parse_query_vector("1.0,2.0,3.0").unwrap(), vec![1.0, 2.0, 3.0]);
        assert_eq!(parse_query_vector(" 1.0 , 2.0 ").unwrap(), vec![1.0, 2.0]);
        assert_eq!(parse_query_vector("42.5").unwrap(), vec![42.5]);
        assert_eq!(parse_query_vector("-1.0,0.0,1.0").unwrap(), vec![-1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_parse_query_vector_empty() {
        assert!(parse_query_vector("").is_err());
    }

    #[test]
    fn test_parse_query_vector_invalid() {
        assert!(parse_query_vector("1.0,abc,3.0").is_err());
    }

    #[test]
    fn test_parse_query_vector_trailing_comma() {
        assert_eq!(parse_query_vector("1.0,2.0,").unwrap(), vec![1.0, 2.0]);
    }

    #[test]
    fn test_create_collection_zero_dims() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.needle");
        let path_str = path.to_str().unwrap();
        let result = create_collection_command(path_str, "test", 0, "cosine");
        assert!(result.is_err());
    }

    #[test]
    fn test_create_collection_valid() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.needle");
        let path_str = path.to_str().unwrap();
        assert!(create_collection_command(path_str, "docs", 128, "cosine").is_ok());
    }

    #[test]
    fn test_info_command() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.needle");
        let path_str = path.to_str().unwrap();
        let mut db = Database::open(path_str).unwrap();
        db.create_collection("test", 64).unwrap();
        db.save().unwrap();
        assert!(info_command(path_str, false).is_ok());
    }

    #[test]
    fn test_count_and_get_commands() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.needle");
        let path_str = path.to_str().unwrap();
        let mut db = Database::open(path_str).unwrap();
        db.create_collection("test", 3).unwrap();
        let coll = db.collection("test").unwrap();
        coll.insert("v1", &[1.0, 0.0, 0.0], None).unwrap();
        db.save().unwrap();

        assert!(count_command(path_str, "test", false).is_ok());
        assert!(get_command(path_str, "test", "v1", false).is_ok());
        assert!(get_command(path_str, "test", "missing", false).is_ok());
    }

    #[test]
    fn test_delete_command() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.needle");
        let path_str = path.to_str().unwrap();
        let mut db = Database::open(path_str).unwrap();
        db.create_collection("test", 3).unwrap();
        let coll = db.collection("test").unwrap();
        coll.insert("v1", &[1.0, 0.0, 0.0], None).unwrap();
        db.save().unwrap();

        assert!(delete_command(path_str, "test", "v1").is_ok());
        assert!(delete_command(path_str, "test", "nonexistent").is_ok());
    }

    #[test]
    fn test_compact_command() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.needle");
        let path_str = path.to_str().unwrap();
        let mut db = Database::open(path_str).unwrap();
        db.create_collection("test", 3).unwrap();
        db.save().unwrap();
        assert!(compact_command(path_str).is_ok());
    }

    #[test]
    fn test_diagnose_command_healthy() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.needle");
        let path_str = path.to_str().unwrap();
        let mut db = Database::open(path_str).unwrap();
        db.create_collection("test", 3).unwrap();
        let coll = db.collection("test").unwrap();
        coll.insert("v1", &[1.0, 0.0, 0.0], None).unwrap();
        db.save().unwrap();
        assert!(diagnose_command(path_str, false, false).is_ok());
    }

    #[test]
    fn test_diagnose_command_json() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.needle");
        let path_str = path.to_str().unwrap();
        let mut db = Database::open(path_str).unwrap();
        db.create_collection("test", 3).unwrap();
        db.save().unwrap();
        assert!(diagnose_command(path_str, false, true).is_ok());
    }

    #[test]
    fn test_diagnose_command_extended() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.needle");
        let path_str = path.to_str().unwrap();
        let mut db = Database::open(path_str).unwrap();
        db.create_collection("test", 3).unwrap();
        let coll = db.collection("test").unwrap();
        coll.insert("v1", &[1.0, 0.0, 0.0], None).unwrap();
        db.save().unwrap();
        assert!(diagnose_command(path_str, true, false).is_ok());
    }
}
