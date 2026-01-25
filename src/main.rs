//! Needle CLI - Command line interface for the Needle vector database

use clap::{Parser, Subcommand};
use needle::{CollectionConfig, Database, DistanceFunction, Result};
use serde_json::json;
use std::io::{self, BufRead};

#[cfg(feature = "server")]
use needle::server::{ServerConfig, serve};

// Import new features for CLI
use needle::{
    BackupConfig, BackupManager, BackupType,
    DriftConfig, DriftDetector,
    QueryAnalyzer, VisualQueryBuilder,
};

#[derive(Parser)]
#[command(name = "needle")]
#[command(author, version, about = "Embedded Vector Database - SQLite for Vectors", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Show information about a database
    Info {
        /// Path to the database file
        database: String,
    },

    /// Create a new database
    Create {
        /// Path to the database file
        database: String,
    },

    /// List all collections in a database
    Collections {
        /// Path to the database file
        database: String,
    },

    /// Create a new collection
    CreateCollection {
        /// Path to the database file
        database: String,

        /// Collection name
        #[arg(short, long)]
        name: String,

        /// Vector dimensions
        #[arg(short, long)]
        dimensions: usize,

        /// Distance function (cosine, euclidean, dot, manhattan)
        #[arg(long, default_value = "cosine")]
        distance: String,
    },

    /// Show collection statistics
    Stats {
        /// Path to the database file
        database: String,

        /// Collection name
        #[arg(short, long)]
        collection: String,
    },

    /// Insert vectors from stdin (JSON format)
    Insert {
        /// Path to the database file
        database: String,

        /// Collection name
        #[arg(short, long)]
        collection: String,
    },

    /// Search for similar vectors
    Search {
        /// Path to the database file
        database: String,

        /// Collection name
        #[arg(short, long)]
        collection: String,

        /// Query vector (comma-separated floats)
        #[arg(short, long)]
        query: String,

        /// Number of results to return
        #[arg(short, long, default_value = "10")]
        k: usize,

        /// Show detailed query profiling information
        #[arg(short, long, default_value = "false")]
        explain: bool,

        /// Override distance function (cosine, euclidean, dot, manhattan)
        /// When different from the collection's index, uses brute-force search
        #[arg(long)]
        distance: Option<String>,
    },

    /// Delete a vector by ID
    Delete {
        /// Path to the database file
        database: String,

        /// Collection name
        #[arg(short, long)]
        collection: String,

        /// Vector ID to delete
        #[arg(short, long)]
        id: String,
    },

    /// Get a vector by ID
    Get {
        /// Path to the database file
        database: String,

        /// Collection name
        #[arg(short, long)]
        collection: String,

        /// Vector ID
        #[arg(short, long)]
        id: String,
    },

    /// Compact the database (remove deleted vectors)
    Compact {
        /// Path to the database file
        database: String,
    },

    /// Export collection to JSON
    Export {
        /// Path to the database file
        database: String,

        /// Collection name
        #[arg(short, long)]
        collection: String,
    },

    /// Import vectors from JSON file
    Import {
        /// Path to the database file
        database: String,

        /// Collection name
        #[arg(short, long)]
        collection: String,

        /// JSON file to import (use - for stdin)
        #[arg(short, long)]
        file: String,
    },

    /// Count vectors in a collection
    Count {
        /// Path to the database file
        database: String,

        /// Collection name
        #[arg(short, long)]
        collection: String,
    },

    /// Clear all vectors from a collection
    Clear {
        /// Path to the database file
        database: String,

        /// Collection name
        #[arg(short, long)]
        collection: String,

        /// Skip confirmation prompt
        #[arg(long)]
        force: bool,
    },

    /// Start HTTP API server (requires 'server' feature)
    #[cfg(feature = "server")]
    Serve {
        /// Address to bind to
        #[arg(short, long, default_value = "127.0.0.1:8080")]
        address: String,

        /// Database file path (omit for in-memory)
        #[arg(short, long)]
        database: Option<String>,
    },

    /// Auto-tune HNSW parameters for a workload
    Tune {
        /// Expected number of vectors
        #[arg(short, long)]
        vectors: usize,

        /// Vector dimensions
        #[arg(short, long)]
        dimensions: usize,

        /// Performance profile (low-latency, balanced, high-recall, low-memory)
        #[arg(short, long, default_value = "balanced")]
        profile: String,

        /// Memory budget in MB (optional)
        #[arg(short, long)]
        memory_mb: Option<usize>,
    },

    /// Natural language query interface
    Query {
        /// Path to the database file
        database: String,

        /// Collection name
        #[arg(short, long)]
        collection: String,

        /// Natural language query
        #[arg(short, long)]
        query: String,

        /// Number of results to return
        #[arg(short, long, default_value = "10")]
        k: usize,

        /// Show query analysis and optimization hints
        #[arg(long)]
        analyze: bool,
    },

    /// Backup management commands
    #[command(subcommand)]
    Backup(BackupCommands),

    /// Drift detection commands
    #[command(subcommand)]
    Drift(DriftCommands),

    /// Federated search commands
    #[command(subcommand)]
    Federate(FederateCommands),

    /// Collection alias management
    #[command(subcommand)]
    Alias(AliasCommands),

    /// TTL (time-to-live) management for vectors
    #[command(subcommand)]
    Ttl(TtlCommands),
}

/// Backup subcommands
#[derive(Subcommand)]
enum BackupCommands {
    /// Create a backup of the database
    Create {
        /// Path to the database file
        database: String,

        /// Backup destination path
        #[arg(short, long)]
        output: String,

        /// Backup type (full, incremental, differential)
        #[arg(short, long, default_value = "full")]
        backup_type: String,

        /// Enable compression
        #[arg(long)]
        compress: bool,
    },

    /// List available backups
    List {
        /// Backup directory path
        path: String,
    },

    /// Restore from a backup
    Restore {
        /// Backup file path
        backup: String,

        /// Database destination path
        #[arg(short, long)]
        output: String,

        /// Force overwrite if exists
        #[arg(long)]
        force: bool,
    },

    /// Verify backup integrity
    Verify {
        /// Backup file path
        backup: String,
    },

    /// Clean up old backups
    Cleanup {
        /// Backup directory path
        path: String,

        /// Keep last N backups
        #[arg(short, long, default_value = "5")]
        keep: usize,
    },
}

/// Drift detection subcommands
#[derive(Subcommand)]
enum DriftCommands {
    /// Create a baseline snapshot for drift detection
    Baseline {
        /// Path to the database file
        database: String,

        /// Collection name
        #[arg(short, long)]
        collection: String,

        /// Output file for baseline
        #[arg(short, long)]
        output: String,

        /// Sample size (0 for all vectors)
        #[arg(long, default_value = "1000")]
        sample_size: usize,
    },

    /// Detect drift from baseline
    Detect {
        /// Path to the database file
        database: String,

        /// Collection name
        #[arg(short, long)]
        collection: String,

        /// Baseline file path
        #[arg(short, long)]
        baseline: String,

        /// Drift threshold (0.0-1.0)
        #[arg(long, default_value = "0.1")]
        threshold: f64,
    },

    /// Generate drift report
    Report {
        /// Path to the database file
        database: String,

        /// Collection name
        #[arg(short, long)]
        collection: String,

        /// Baseline file path
        #[arg(short, long)]
        baseline: String,

        /// Output format (text, json)
        #[arg(long, default_value = "text")]
        format: String,
    },
}

/// Alias subcommands
#[derive(Subcommand)]
enum AliasCommands {
    /// Create a new alias for a collection
    Create {
        /// Path to the database file
        #[arg(short, long)]
        database: String,

        /// Alias name
        #[arg(short, long)]
        alias: String,

        /// Target collection name
        #[arg(short, long)]
        collection: String,
    },

    /// Delete an alias
    Delete {
        /// Path to the database file
        #[arg(short, long)]
        database: String,

        /// Alias name
        #[arg(short, long)]
        alias: String,
    },

    /// List all aliases
    List {
        /// Path to the database file
        #[arg(short, long)]
        database: String,
    },

    /// Resolve an alias to its target collection
    Resolve {
        /// Path to the database file
        #[arg(short, long)]
        database: String,

        /// Alias name
        #[arg(short, long)]
        alias: String,
    },

    /// Update an alias to point to a different collection
    Update {
        /// Path to the database file
        #[arg(short, long)]
        database: String,

        /// Alias name
        #[arg(short, long)]
        alias: String,

        /// New target collection name
        #[arg(short, long)]
        collection: String,
    },
}

/// TTL (time-to-live) subcommands
#[derive(Subcommand)]
enum TtlCommands {
    /// Sweep and delete all expired vectors in a collection
    Sweep {
        /// Path to the database file
        #[arg(short, long)]
        database: String,

        /// Collection name
        #[arg(short, long)]
        collection: String,
    },

    /// Show TTL statistics for a collection
    Stats {
        /// Path to the database file
        #[arg(short, long)]
        database: String,

        /// Collection name
        #[arg(short, long)]
        collection: String,
    },
}

/// Federated search subcommands
#[derive(Subcommand)]
enum FederateCommands {
    /// Search across multiple instances
    Search {
        /// Query vector (comma-separated floats)
        #[arg(short, long)]
        query: String,

        /// Collection name
        #[arg(short, long)]
        collection: String,

        /// Number of results
        #[arg(short, long, default_value = "10")]
        k: usize,

        /// Instance URLs (comma-separated)
        #[arg(short, long)]
        instances: String,

        /// Routing strategy (broadcast, latency-aware, round-robin)
        #[arg(long, default_value = "broadcast")]
        routing: String,

        /// Merge strategy (distance, rrf, consensus)
        #[arg(long, default_value = "distance")]
        merge: String,
    },

    /// Check health of federated instances
    Health {
        /// Instance URLs (comma-separated)
        #[arg(short, long)]
        instances: String,
    },

    /// Show federation statistics
    Stats {
        /// Instance URLs (comma-separated)
        #[arg(short, long)]
        instances: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Info { database } => info_command(&database),
        Commands::Create { database } => create_command(&database),
        Commands::Collections { database } => collections_command(&database),
        Commands::CreateCollection {
            database,
            name,
            dimensions,
            distance,
        } => create_collection_command(&database, &name, dimensions, &distance),
        Commands::Stats {
            database,
            collection,
        } => stats_command(&database, &collection),
        Commands::Insert {
            database,
            collection,
        } => insert_command(&database, &collection),
        Commands::Search {
            database,
            collection,
            query,
            k,
            explain,
            distance,
        } => search_command(&database, &collection, &query, k, explain, distance.as_deref()),
        Commands::Delete {
            database,
            collection,
            id,
        } => delete_command(&database, &collection, &id),
        Commands::Get {
            database,
            collection,
            id,
        } => get_command(&database, &collection, &id),
        Commands::Compact { database } => compact_command(&database),
        Commands::Export {
            database,
            collection,
        } => export_command(&database, &collection),
        Commands::Import {
            database,
            collection,
            file,
        } => import_command(&database, &collection, &file),
        Commands::Count {
            database,
            collection,
        } => count_command(&database, &collection),
        Commands::Clear {
            database,
            collection,
            force,
        } => clear_command(&database, &collection, force),
        #[cfg(feature = "server")]
        Commands::Serve { address, database } => serve_command(&address, database),
        Commands::Tune {
            vectors,
            dimensions,
            profile,
            memory_mb,
        } => tune_command(vectors, dimensions, &profile, memory_mb),
        Commands::Query {
            database,
            collection,
            query,
            k,
            analyze,
        } => query_command(&database, &collection, &query, k, analyze),
        Commands::Backup(cmd) => backup_command(cmd),
        Commands::Drift(cmd) => drift_command(cmd),
        Commands::Federate(cmd) => federate_command(cmd),
        Commands::Alias(cmd) => alias_command(cmd),
        Commands::Ttl(cmd) => ttl_command(cmd),
    }
}

fn info_command(path: &str) -> Result<()> {
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

fn create_command(path: &str) -> Result<()> {
    let _db = Database::open(path)?;
    println!("Created database: {}", path);
    Ok(())
}

fn collections_command(path: &str) -> Result<()> {
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

fn create_collection_command(
    path: &str,
    name: &str,
    dimensions: usize,
    distance: &str,
) -> Result<()> {
    let mut db = Database::open(path)?;

    let dist_fn = match distance.to_lowercase().as_str() {
        "cosine" => DistanceFunction::Cosine,
        "euclidean" | "l2" => DistanceFunction::Euclidean,
        "dot" | "dotproduct" => DistanceFunction::DotProduct,
        "manhattan" | "l1" => DistanceFunction::Manhattan,
        _ => {
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

fn stats_command(path: &str, collection_name: &str) -> Result<()> {
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

fn insert_command(path: &str, collection_name: &str) -> Result<()> {
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
            Some(arr) => arr.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect(),
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

fn search_command(
    path: &str,
    collection_name: &str,
    query_str: &str,
    k: usize,
    explain: bool,
    distance_override: Option<&str>,
) -> Result<()> {
    use needle::DistanceFunction;

    let db = Database::open(path)?;
    let coll = db.collection(collection_name)?;

    let query: Vec<f32> = query_str
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    if query.is_empty() {
        eprintln!("Invalid query vector. Use comma-separated floats.");
        return Ok(());
    }

    // Parse distance override if provided
    let distance_fn = distance_override.map(|d| match d.to_lowercase().as_str() {
        "cosine" => DistanceFunction::Cosine,
        "euclidean" => DistanceFunction::Euclidean,
        "dot" | "dotproduct" => DistanceFunction::DotProduct,
        "manhattan" => DistanceFunction::Manhattan,
        _ => {
            eprintln!("Warning: Unknown distance function '{}', using collection default", d);
            DistanceFunction::Cosine
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
                .map(|m| m.to_string())
                .unwrap_or_else(|| "null".to_string());
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
        println!("  Layers traversed: {}", explain_data.hnsw_stats.layers_traversed);
        println!("  Distance computations: {}", explain_data.hnsw_stats.distance_computations);
        println!("  Traversal time: {}μs", explain_data.hnsw_stats.traversal_time_us);
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
        if distance_fn.is_some() {
            println!("  (using distance override: {:?})", distance_fn.unwrap());
        }
        for result in results {
            let meta = result
                .metadata
                .as_ref()
                .map(|m| m.to_string())
                .unwrap_or_else(|| "null".to_string());
            println!(
                "  ID: {}, Distance: {:.6}, Metadata: {}",
                result.id, result.distance, meta
            );
        }
    }

    Ok(())
}

fn delete_command(path: &str, collection_name: &str, id: &str) -> Result<()> {
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

fn get_command(path: &str, collection_name: &str, id: &str) -> Result<()> {
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

fn compact_command(path: &str) -> Result<()> {
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
        println!("Compaction complete: removed {} total deleted vectors", total_deleted);
    } else {
        println!("No deleted vectors to compact");
    }

    Ok(())
}

fn export_command(path: &str, collection_name: &str) -> Result<()> {
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

fn import_command(path: &str, collection_name: &str, file_path: &str) -> Result<()> {
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

fn count_command(path: &str, collection_name: &str) -> Result<()> {
    let db = Database::open(path)?;
    let coll = db.collection(collection_name)?;

    println!("{}", coll.len());

    Ok(())
}

fn clear_command(path: &str, collection_name: &str, force: bool) -> Result<()> {
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
fn serve_command(address: &str, database: Option<String>) -> Result<()> {
    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| needle::NeedleError::InvalidConfig(e.to_string()))?;

    rt.block_on(async {
        let mut config = ServerConfig::new(address)
            .map_err(|e| needle::NeedleError::InvalidConfig(format!("Invalid address: {}", e)))?;
        if let Some(db_path) = database {
            config = config.with_db_path(db_path);
        }

        serve(config).await.map_err(|e| {
            needle::NeedleError::InvalidConfig(e.to_string())
        })
    })
}

fn tune_command(vectors: usize, dimensions: usize, profile: &str, memory_mb: Option<usize>) -> Result<()> {
    use needle::tuning::{auto_tune, TuningConstraints, PerformanceProfile};

    let perf_profile = match profile.to_lowercase().as_str() {
        "low-latency" | "lowlatency" | "fast" => PerformanceProfile::LowLatency,
        "high-recall" | "highrecall" | "accurate" => PerformanceProfile::HighRecall,
        "low-memory" | "lowmemory" | "compact" => PerformanceProfile::LowMemory,
        _ => PerformanceProfile::Balanced,
    };

    let mut constraints = TuningConstraints::new(vectors, dimensions)
        .with_profile(perf_profile);

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
    println!("  Memory per vector: {} bytes", result.estimated_memory_per_vector);
    println!("  Total memory: {:.1} MB", result.estimated_total_memory as f64 / 1024.0 / 1024.0);
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

fn query_command(path: &str, collection_name: &str, query_str: &str, _k: usize, analyze: bool) -> Result<()> {
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
    println!("  Quality Score: {:.1}%", build_result.quality_score * 100.0);
    println!();

    println!("Query Analysis:");
    println!("  Class: {:?}", build_result.analysis.class);
    println!("  Complexity: {:?}", build_result.analysis.complexity);
    println!("  Confidence: {:.1}%", build_result.analysis.confidence * 100.0);
    println!();

    if analyze {
        // Show optimization hints
        if !build_result.optimization_hints.is_empty() {
            println!("Optimization Hints:");
            for hint in &build_result.optimization_hints {
                let severity = match hint.severity {
                    needle::HintSeverity::Info => "INFO",
                    needle::HintSeverity::Suggestion => "SUGG",
                    needle::HintSeverity::Warning => "WARN",
                    needle::HintSeverity::Critical => "CRIT",
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
                println!("  - {:?}: \"{}\"", pattern.pattern_type, pattern.matched_text);
            }
            println!();
        }
    }

    if !build_result.suggestions.is_empty() {
        println!("Suggestions:");
        for suggestion in &build_result.suggestions {
            println!("  - {:?}: {}", suggestion.suggestion_type, suggestion.message);
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

// ============================================================================
// Backup Commands
// ============================================================================

fn backup_command(cmd: BackupCommands) -> Result<()> {
    match cmd {
        BackupCommands::Create {
            database,
            output,
            backup_type,
            compress,
        } => backup_create(&database, &output, &backup_type, compress),
        BackupCommands::List { path } => backup_list(&path),
        BackupCommands::Restore {
            backup,
            output,
            force,
        } => backup_restore(&backup, &output, force),
        BackupCommands::Verify { backup } => backup_verify(&backup),
        BackupCommands::Cleanup { path, keep } => backup_cleanup(&path, keep),
    }
}

fn backup_create(database: &str, output: &str, backup_type: &str, compress: bool) -> Result<()> {
    let db = Database::open(database)?;

    let _btype = match backup_type.to_lowercase().as_str() {
        "incremental" => BackupType::Incremental,
        "snapshot" => BackupType::Snapshot,
        _ => BackupType::Full,
    };

    let config = BackupConfig {
        compression: compress,
        verify: true,
        max_backups: Some(10),
        include_metadata: true,
    };

    let manager = BackupManager::new(output, config);
    let metadata = manager.create_backup(&db)?;

    println!("Backup created successfully!");
    println!();
    println!("Backup Details:");
    println!("  ID: {}", metadata.id);
    println!("  Type: {:?}", metadata.backup_type);
    println!("  Collections: {}", metadata.num_collections);
    println!("  Total vectors: {}", metadata.total_vectors);
    println!("  Size: {} bytes", metadata.size_bytes);
    println!("  Checksum: {}", metadata.checksum);

    Ok(())
}

fn backup_list(path: &str) -> Result<()> {
    let config = BackupConfig::default();
    let manager = BackupManager::new(path, config);
    let backups = manager.list_backups()?;

    if backups.is_empty() {
        println!("No backups found in: {}", path);
        return Ok(());
    }

    println!("Available Backups:");
    println!("{:-<80}", "");
    println!(
        "{:<36} {:<12} {:<10} {:<12}",
        "ID", "Type", "Vectors", "Size"
    );
    println!("{:-<80}", "");

    for backup in backups {
        let size_str = if backup.size_bytes > 1024 * 1024 {
            format!("{:.1} MB", backup.size_bytes as f64 / 1024.0 / 1024.0)
        } else if backup.size_bytes > 1024 {
            format!("{:.1} KB", backup.size_bytes as f64 / 1024.0)
        } else {
            format!("{} B", backup.size_bytes)
        };

        println!(
            "{:<36} {:<12} {:<10} {:<12}",
            backup.id,
            format!("{:?}", backup.backup_type),
            backup.total_vectors,
            size_str
        );
    }

    Ok(())
}

fn backup_restore(backup_path: &str, output: &str, force: bool) -> Result<()> {
    if std::path::Path::new(output).exists() && !force {
        eprintln!("Error: Destination '{}' already exists. Use --force to overwrite.", output);
        return Ok(());
    }

    // Get backup directory from backup_path
    let backup_dir = std::path::Path::new(backup_path)
        .parent()
        .unwrap_or(std::path::Path::new("."));

    let config = BackupConfig::default();
    let manager = BackupManager::new(backup_dir, config);
    let db = manager.restore_backup(backup_path)?;

    // The restored database is in-memory, we need to export and re-import
    // For now, just show a message about the restored data
    println!("Backup restored successfully!");
    println!("  Collections: {}", db.list_collections().len());
    println!("  Total vectors: {}", db.total_vectors());
    println!();
    println!("Note: To save to '{}', use the database normally and call save().", output);

    Ok(())
}

fn backup_verify(backup_path: &str) -> Result<()> {
    let config = BackupConfig::default();
    let manager = BackupManager::new(backup_path, config);
    let valid = manager.verify_backup(backup_path)?;

    if valid {
        println!("Backup verification: PASSED");
        println!("  Checksum: Valid");
        println!("  Structure: Valid");
    } else {
        println!("Backup verification: FAILED");
        println!("  The backup file may be corrupted.");
    }

    Ok(())
}

fn backup_cleanup(path: &str, keep: usize) -> Result<()> {
    let config = BackupConfig {
        max_backups: Some(keep),
        ..Default::default()
    };
    let manager = BackupManager::new(path, config);

    // List backups and manually clean up old ones
    let backups = manager.list_backups()?;

    if backups.len() <= keep {
        println!("No backups to clean up (have {}, keeping {}).", backups.len(), keep);
        return Ok(());
    }

    let to_remove = backups.len() - keep;
    println!("Would remove {} old backup(s), keeping last {}.", to_remove, keep);
    println!("Note: Manual cleanup - delete old backup files from: {}", path);

    Ok(())
}

// ============================================================================
// Drift Detection Commands
// ============================================================================

fn drift_command(cmd: DriftCommands) -> Result<()> {
    match cmd {
        DriftCommands::Baseline {
            database,
            collection,
            output,
            sample_size,
        } => drift_baseline(&database, &collection, &output, sample_size),
        DriftCommands::Detect {
            database,
            collection,
            baseline,
            threshold,
        } => drift_detect(&database, &collection, &baseline, threshold),
        DriftCommands::Report {
            database,
            collection,
            baseline,
            format,
        } => drift_report(&database, &collection, &baseline, &format),
    }
}

fn drift_baseline(database: &str, collection_name: &str, output: &str, sample_size: usize) -> Result<()> {
    let db = Database::open(database)?;
    let coll = db.collection(collection_name)?;

    // Get dimensions from collection
    let dimensions = coll.dimensions().unwrap_or(0);
    if dimensions == 0 {
        return Err(needle::NeedleError::InvalidInput(
            "Cannot determine vector dimensions".to_string(),
        ));
    }

    // Export vectors for baseline
    let vectors = coll.export_all()?;
    let sample: Vec<Vec<f32>> = if sample_size > 0 && sample_size < vectors.len() {
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        let mut sampled: Vec<_> = vectors.iter().map(|(_, v, _)| v.clone()).collect();
        sampled.shuffle(&mut rng);
        sampled.truncate(sample_size);
        sampled
    } else {
        vectors.iter().map(|(_, v, _)| v.clone()).collect()
    };

    // Create detector and add baseline
    let config = DriftConfig::default();
    let mut detector = DriftDetector::new(dimensions, config);
    detector.add_baseline(&sample)?;

    // Compute baseline statistics for saving
    let centroid: Vec<f32> = (0..dimensions)
        .map(|d| sample.iter().map(|v| v[d]).sum::<f32>() / sample.len() as f32)
        .collect();

    let variance: Vec<f32> = (0..dimensions)
        .map(|d| {
            let mean = centroid[d];
            sample.iter().map(|v| (v[d] - mean).powi(2)).sum::<f32>() / sample.len() as f32
        })
        .collect();

    // Save baseline to file
    let baseline_json = serde_json::to_string_pretty(&json!({
        "collection": collection_name,
        "sample_size": sample.len(),
        "dimensions": dimensions,
        "created_at": chrono::Utc::now().to_rfc3339(),
        "centroid": centroid,
        "variance": variance
    }))?;

    std::fs::write(output, baseline_json)?;

    println!("Baseline created successfully!");
    println!();
    println!("Details:");
    println!("  Collection: {}", collection_name);
    println!("  Sample size: {}", sample.len());
    println!("  Dimensions: {}", dimensions);
    println!("  Output: {}", output);

    Ok(())
}

fn drift_detect(database: &str, collection_name: &str, baseline_path: &str, threshold: f64) -> Result<()> {
    let db = Database::open(database)?;
    let coll = db.collection(collection_name)?;

    // Load baseline
    let baseline_content = std::fs::read_to_string(baseline_path)?;
    let baseline_data: serde_json::Value = serde_json::from_str(&baseline_content)?;

    let dimensions = baseline_data["dimensions"]
        .as_u64()
        .ok_or_else(|| needle::NeedleError::InvalidInput("Invalid baseline: missing dimensions".to_string()))?
        as usize;

    let baseline_centroid: Vec<f32> = baseline_data["centroid"]
        .as_array()
        .ok_or_else(|| needle::NeedleError::InvalidInput("Invalid baseline: missing centroid".to_string()))?
        .iter()
        .filter_map(|v| v.as_f64().map(|f| f as f32))
        .collect();

    let config = DriftConfig {
        centroid_threshold: threshold as f32,
        variance_threshold: threshold as f32,
        ..Default::default()
    };
    let mut detector = DriftDetector::new(dimensions, config);

    // Reconstruct baseline from saved stats
    // We create synthetic baseline vectors around the saved centroid
    let baseline_variance: Vec<f32> = baseline_data["variance"]
        .as_array()
        .map(|arr| arr.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect())
        .unwrap_or_else(|| vec![0.1; dimensions]);

    // Create baseline vectors around centroid
    let mut baseline_vectors = Vec::new();
    for _ in 0..100 {
        let vec: Vec<f32> = baseline_centroid
            .iter()
            .zip(baseline_variance.iter())
            .map(|(&c, &v)| c + (rand::random::<f32>() - 0.5) * v.sqrt() * 2.0)
            .collect();
        baseline_vectors.push(vec);
    }
    detector.add_baseline(&baseline_vectors)?;

    // Get current vectors and check for drift
    let vectors = coll.export_all()?;
    let mut drift_detected = false;
    let mut total_drift_score = 0.0f32;
    let mut samples_checked = 0;

    for (_, vec, _) in vectors.iter().take(1000) {
        let report = detector.check(vec)?;
        if report.is_drifting {
            drift_detected = true;
        }
        total_drift_score += report.drift_score;
        samples_checked += 1;
    }

    let avg_drift_score = if samples_checked > 0 {
        total_drift_score / samples_checked as f32
    } else {
        0.0
    };

    println!("Drift Detection Results");
    println!("=======================");
    println!();
    println!("Threshold: {:.2}", threshold);
    println!("Drift detected: {}", if drift_detected { "YES" } else { "NO" });
    println!();
    println!("Metrics:");
    println!("  Samples checked: {}", samples_checked);
    println!("  Average drift score: {:.4}", avg_drift_score);

    if drift_detected {
        println!();
        println!("Warning: Significant drift detected!");
        println!("  Consider retraining models or investigating data quality.");
    }

    Ok(())
}

fn drift_report(database: &str, collection_name: &str, baseline_path: &str, format: &str) -> Result<()> {
    let db = Database::open(database)?;
    let coll = db.collection(collection_name)?;

    // Load baseline
    let baseline_content = std::fs::read_to_string(baseline_path)?;
    let baseline_data: serde_json::Value = serde_json::from_str(&baseline_content)?;

    let dimensions = baseline_data["dimensions"]
        .as_u64()
        .ok_or_else(|| needle::NeedleError::InvalidInput("Invalid baseline: missing dimensions".to_string()))?
        as usize;

    let baseline_centroid: Vec<f32> = baseline_data["centroid"]
        .as_array()
        .ok_or_else(|| needle::NeedleError::InvalidInput("Invalid baseline: missing centroid".to_string()))?
        .iter()
        .filter_map(|v| v.as_f64().map(|f| f as f32))
        .collect();

    let baseline_variance: Vec<f32> = baseline_data["variance"]
        .as_array()
        .map(|arr| arr.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect())
        .unwrap_or_else(|| vec![0.1; dimensions]);

    // Get current vectors
    let vectors = coll.export_all()?;
    let current_vecs: Vec<Vec<f32>> = vectors.iter().map(|(_, v, _)| v.clone()).collect();

    // Compute current statistics
    let current_centroid: Vec<f32> = (0..dimensions)
        .map(|d| current_vecs.iter().map(|v| v[d]).sum::<f32>() / current_vecs.len().max(1) as f32)
        .collect();

    let current_variance: Vec<f32> = (0..dimensions)
        .map(|d| {
            let mean = current_centroid[d];
            current_vecs.iter().map(|v| (v[d] - mean).powi(2)).sum::<f32>() / current_vecs.len().max(1) as f32
        })
        .collect();

    // Compute drift metrics
    let centroid_shift: f32 = baseline_centroid
        .iter()
        .zip(current_centroid.iter())
        .map(|(b, c)| (b - c).powi(2))
        .sum::<f32>()
        .sqrt();

    let variance_change: f32 = baseline_variance
        .iter()
        .zip(current_variance.iter())
        .map(|(b, c)| ((c / b.max(0.0001)) - 1.0).abs())
        .sum::<f32>()
        / dimensions as f32;

    let drift_score = (centroid_shift * 0.6 + variance_change * 0.4).min(1.0);
    let has_drift = drift_score > 0.1;

    // Find top drifting dimensions
    let mut dimension_drifts: Vec<(usize, f32)> = (0..dimensions)
        .map(|d| {
            let shift = (baseline_centroid[d] - current_centroid[d]).abs();
            let var_change = ((current_variance[d] / baseline_variance[d].max(0.0001)) - 1.0).abs();
            (d, shift + var_change)
        })
        .collect();
    dimension_drifts.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let recommendations: Vec<String> = if has_drift {
        vec![
            "Review recent data ingestion for quality issues".to_string(),
            "Consider retraining embedding models".to_string(),
            "Investigate top drifting dimensions".to_string(),
        ]
    } else {
        vec!["Data distribution is stable".to_string()]
    };

    if format == "json" {
        let json_report = json!({
            "collection": collection_name,
            "baseline_file": baseline_path,
            "current_count": current_vecs.len(),
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "drift_detected": has_drift,
            "drift_score": drift_score,
            "metrics": {
                "centroid_shift": centroid_shift,
                "variance_change": variance_change
            },
            "dimension_drifts": dimension_drifts.iter().take(10).collect::<Vec<_>>(),
            "recommendations": recommendations
        });
        println!("{}", serde_json::to_string_pretty(&json_report)?);
    } else {
        println!("Drift Analysis Report");
        println!("=====================");
        println!();
        println!("Collection: {}", collection_name);
        println!("Baseline: {}", baseline_path);
        println!("Current vectors: {}", current_vecs.len());
        println!("Analysis time: {}", chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC"));
        println!();
        println!("Overall Assessment:");
        println!("  Drift detected: {}", if has_drift { "YES" } else { "NO" });
        println!("  Drift score: {:.4}", drift_score);
        println!();
        println!("Detailed Metrics:");
        println!("  Centroid shift: {:.4}", centroid_shift);
        println!("  Variance change: {:.4}", variance_change);

        if !dimension_drifts.is_empty() {
            println!();
            println!("Top Drifting Dimensions:");
            for (i, (dim, drift)) in dimension_drifts.iter().take(5).enumerate() {
                println!("  {}. Dimension {}: {:.4}", i + 1, dim, drift);
            }
        }

        println!();
        println!("Recommendations:");
        for rec in &recommendations {
            println!("  - {}", rec);
        }
    }

    Ok(())
}

// ============================================================================
// Federated Search Commands
// ============================================================================

fn federate_command(cmd: FederateCommands) -> Result<()> {
    match cmd {
        FederateCommands::Search {
            query,
            collection,
            k,
            instances,
            routing,
            merge,
        } => federate_search(&query, &collection, k, &instances, &routing, &merge),
        FederateCommands::Health { instances } => federate_health(&instances),
        FederateCommands::Stats { instances } => federate_stats(&instances),
    }
}

fn federate_search(
    query_str: &str,
    collection: &str,
    k: usize,
    instances_str: &str,
    routing: &str,
    merge: &str,
) -> Result<()> {
    use needle::{
        Federation, FederationConfig, InstanceConfig,
        RoutingStrategy, MergeStrategy,
    };

    // Parse query vector
    let query: Vec<f32> = query_str
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    if query.is_empty() {
        eprintln!("Invalid query vector. Use comma-separated floats.");
        return Ok(());
    }

    // Parse instances
    let instance_urls: Vec<&str> = instances_str.split(',').map(|s| s.trim()).collect();

    if instance_urls.is_empty() {
        eprintln!("No instances specified.");
        return Ok(());
    }

    // Parse routing strategy
    let routing_strategy = match routing.to_lowercase().as_str() {
        "latency-aware" | "latency" => RoutingStrategy::LatencyAware,
        "round-robin" | "roundrobin" => RoutingStrategy::RoundRobin,
        "geographic" | "geo" => RoutingStrategy::GeographicProximity,
        _ => RoutingStrategy::Broadcast,
    };

    // Parse merge strategy
    let merge_strategy = match merge.to_lowercase().as_str() {
        "rrf" | "reciprocal" => MergeStrategy::ReciprocalRankFusion,
        "consensus" => MergeStrategy::Consensus,
        "first" => MergeStrategy::FirstResponse,
        _ => MergeStrategy::DistanceBased,
    };

    // Create federation
    let config = FederationConfig::default()
        .with_routing(routing_strategy)
        .with_merge(merge_strategy);

    let federation = Federation::new(config);

    // Register instances
    for (i, url) in instance_urls.iter().enumerate() {
        let instance_config = InstanceConfig::new(format!("instance-{}", i), *url);
        federation.register_instance(instance_config);
    }

    println!("Federated Search");
    println!("================");
    println!();
    println!("Query: {} dimensions", query.len());
    println!("Collection: {}", collection);
    println!("K: {}", k);
    println!("Instances: {}", instance_urls.len());
    println!("Routing: {:?}", routing_strategy);
    println!("Merge: {:?}", merge_strategy);
    println!();

    // Note: Actual federated search requires async runtime and HTTP client
    // This demonstrates the CLI interface
    println!("Note: Federated search requires the 'server' feature and running instances.");
    println!("      Use 'needle serve' to start instances, then use this command to query them.");
    println!();
    println!("Configured instances:");
    for url in &instance_urls {
        println!("  - {}", url);
    }

    Ok(())
}

fn federate_health(instances_str: &str) -> Result<()> {
    use needle::{Federation, FederationConfig, InstanceConfig};

    let instance_urls: Vec<&str> = instances_str.split(',').map(|s| s.trim()).collect();

    let config = FederationConfig::default();
    let federation = Federation::new(config);

    for (i, url) in instance_urls.iter().enumerate() {
        let instance_config = InstanceConfig::new(format!("instance-{}", i), *url);
        federation.register_instance(instance_config);
    }

    let health = federation.health();

    println!("Federation Health Status");
    println!("========================");
    println!();
    println!("Overall: {:?}", health.status);
    println!("Healthy instances: {}/{}", health.healthy_instances, health.total_instances);
    println!("Degraded instances: {}", health.degraded_instances);
    println!("Unhealthy instances: {}", health.unhealthy_instances);
    println!("Average latency: {:.2} ms", health.avg_latency_ms);

    Ok(())
}

fn federate_stats(instances_str: &str) -> Result<()> {
    use needle::{Federation, FederationConfig, InstanceConfig};

    let instance_urls: Vec<&str> = instances_str.split(',').map(|s| s.trim()).collect();

    let config = FederationConfig::default();
    let federation = Federation::new(config);

    for (i, url) in instance_urls.iter().enumerate() {
        let instance_config = InstanceConfig::new(format!("instance-{}", i), *url);
        federation.register_instance(instance_config);
    }

    let stats = federation.stats();

    println!("Federation Statistics");
    println!("=====================");
    println!();
    println!("Total queries: {}", stats.total_queries);
    println!("Failed queries: {}", stats.failed_queries);
    println!("Partial results: {}", stats.partial_results);
    println!("Timeouts: {}", stats.timeouts);

    Ok(())
}

// ============================================================================
// Alias Commands
// ============================================================================

fn alias_command(cmd: AliasCommands) -> Result<()> {
    match cmd {
        AliasCommands::Create {
            database,
            alias,
            collection,
        } => alias_create(&database, &alias, &collection),
        AliasCommands::Delete { database, alias } => alias_delete(&database, &alias),
        AliasCommands::List { database } => alias_list(&database),
        AliasCommands::Resolve { database, alias } => alias_resolve(&database, &alias),
        AliasCommands::Update {
            database,
            alias,
            collection,
        } => alias_update(&database, &alias, &collection),
    }
}

fn alias_create(path: &str, alias: &str, collection: &str) -> Result<()> {
    let mut db = Database::open(path)?;
    db.create_alias(alias, collection)?;
    db.save()?;

    println!("Created alias '{}' -> '{}'", alias, collection);
    Ok(())
}

fn alias_delete(path: &str, alias: &str) -> Result<()> {
    let mut db = Database::open(path)?;
    let deleted = db.delete_alias(alias)?;
    db.save()?;

    if deleted {
        println!("Deleted alias '{}'", alias);
    } else {
        println!("Alias '{}' not found", alias);
    }
    Ok(())
}

fn alias_list(path: &str) -> Result<()> {
    let db = Database::open(path)?;
    let aliases = db.list_aliases();

    if aliases.is_empty() {
        println!("No aliases defined.");
    } else {
        println!("Aliases:");
        println!("{:-<50}", "");
        println!("{:<25} {:<25}", "Alias", "Collection");
        println!("{:-<50}", "");
        for (alias, collection) in aliases {
            println!("{:<25} {:<25}", alias, collection);
        }
    }

    Ok(())
}

fn alias_resolve(path: &str, alias: &str) -> Result<()> {
    let db = Database::open(path)?;

    match db.get_canonical_name(alias) {
        Some(collection) => {
            println!("{}", collection);
        }
        None => {
            println!("Alias '{}' not found", alias);
        }
    }

    Ok(())
}

fn alias_update(path: &str, alias: &str, collection: &str) -> Result<()> {
    let mut db = Database::open(path)?;
    db.update_alias(alias, collection)?;
    db.save()?;

    println!("Updated alias '{}' -> '{}'", alias, collection);
    Ok(())
}

// ============ TTL Commands ============

fn ttl_command(cmd: TtlCommands) -> Result<()> {
    match cmd {
        TtlCommands::Sweep {
            database,
            collection,
        } => ttl_sweep(&database, &collection),
        TtlCommands::Stats {
            database,
            collection,
        } => ttl_stats(&database, &collection),
    }
}

fn ttl_sweep(path: &str, collection_name: &str) -> Result<()> {
    let mut db = Database::open(path)?;
    let collection = db.collection(collection_name)?;
    let expired = collection.expire_vectors()?;
    db.save()?;

    if expired > 0 {
        println!("Expired {} vectors from '{}'", expired, collection_name);
    } else {
        println!("No expired vectors found in '{}'", collection_name);
    }
    Ok(())
}

fn ttl_stats(path: &str, collection_name: &str) -> Result<()> {
    let db = Database::open(path)?;
    let collection = db.collection(collection_name)?;
    let (total, expired, earliest, latest) = collection.ttl_stats();

    println!("TTL Statistics for '{}':", collection_name);
    println!("{:-<50}", "");
    println!("Vectors with TTL:        {}", total);
    println!("Currently expired:       {}", expired);

    if let Some(earliest_ts) = earliest {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        if earliest_ts > now {
            println!("Next expiration in:      {} seconds", earliest_ts - now);
        } else {
            println!("Oldest expired:          {} seconds ago", now - earliest_ts);
        }
    }

    if let Some(latest_ts) = latest {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        if latest_ts > now {
            println!("Latest expiration in:    {} seconds", latest_ts - now);
        }
    }

    if collection.needs_expiration_sweep(0.1) {
        println!("\nRecommendation: Run 'needle ttl sweep' to clean up expired vectors.");
    }

    Ok(())
}
