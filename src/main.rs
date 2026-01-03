//! Needle CLI - Command line interface for the Needle vector database

use clap::{Parser, Subcommand};
use needle::{CollectionConfig, Database, DistanceFunction, Result};
use serde_json::json;
use std::io::{self, BufRead};

#[cfg(feature = "server")]
use needle::server::{ServerConfig, serve};

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
        } => search_command(&database, &collection, &query, k),
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

fn search_command(path: &str, collection_name: &str, query_str: &str, k: usize) -> Result<()> {
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

    let results = coll.search(&query, k)?;

    println!("Search results (k={}):", k);
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
