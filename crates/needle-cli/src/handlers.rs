use needle::{CollectionConfig, Database, DistanceFunction, NeedleError, Result};
use serde_json::json;
use std::io::{self, BufRead};

#[cfg(feature = "server")]
use needle::server::{serve, ServerConfig};

use needle::query_builder::{QueryAnalyzer, VisualQueryBuilder};

pub(crate) fn info_command(path: &str) -> Result<()> {
    let db = Database::open(path)?;

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

pub(crate) fn collections_command(path: &str) -> Result<()> {
    let db = Database::open(path)?;

    let collections = db.list_collections();
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
    let mut db = Database::open(path)?;

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

pub(crate) fn stats_command(path: &str, collection_name: &str) -> Result<()> {
    let db = Database::open(path)?;
    let coll = db.collection(collection_name)?;

    let active_count = coll.len();
    let deleted_count = coll.deleted_count();
    let total_stored = active_count + deleted_count;

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
    let mut db = Database::open(path)?;
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
) -> Result<()> {
    let db = Database::open(path)?;
    let coll = db.collection(collection_name)?;

    let query = parse_query_vector(query_str)?;

    // Parse distance override if provided
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
        // Note: explain mode doesn't support distance override (uses HNSW stats)
        if distance_override.is_some() {
            eprintln!("Warning: --explain and --distance cannot be combined; ignoring --distance");
        }
        let (results, explain_data) = coll.search_explain(&query, k)?;

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

        // Print profiling information
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
    let mut db = Database::open(path)?;
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

pub(crate) fn get_command(path: &str, collection_name: &str, id: &str) -> Result<()> {
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
            println!("Vector '{}' not found", id);
        }
    }

    Ok(())
}

pub(crate) fn compact_command(path: &str) -> Result<()> {
    let mut db = Database::open(path)?;

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
    let mut db = Database::open(path)?;
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

pub(crate) fn count_command(path: &str, collection_name: &str) -> Result<()> {
    let db = Database::open(path)?;
    let coll = db.collection(collection_name)?;

    println!("{}", coll.len());

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

    let mut db = Database::open(path)?;
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
