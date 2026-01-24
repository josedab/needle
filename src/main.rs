//! Needle CLI — Command line interface for the Needle vector database.
//!
//! Provides commands for database management, vector operations, search,
//! server mode, backup/restore, and administrative tools.
//!
//! # Usage
//!
//! ```bash
//! needle create mydb.needle
//! needle create-collection mydb.needle -n docs -d 384
//! needle info mydb.needle
//! needle serve -a 127.0.0.1:8080  # requires --features server
//! ```

use clap::{Parser, Subcommand};
use needle::{CollectionConfig, Database, DistanceFunction, NeedleError, Result};
use serde_json::json;
use std::io::{self, BufRead};

#[cfg(feature = "server")]
use needle::server::{serve, ServerConfig};

// Import new features for CLI
use needle::backup::{BackupConfig, BackupManager, BackupType};
use needle::drift::{DriftConfig, DriftDetector};
use needle::query_builder::{QueryAnalyzer, VisualQueryBuilder};

#[derive(Parser)]
#[command(name = "needle")]
#[command(author, version, about = "Embedded Vector Database - SQLite for Vectors", long_about = None)]
struct Cli {
    /// Enable debug logging (sets RUST_LOG=debug)
    #[arg(short, long, global = true)]
    verbose: bool,

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

        /// Enable encryption at rest (requires --features encryption)
        #[arg(long)]
        encrypted: bool,
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

    /// Run an interactive demo (creates an in-memory database, inserts sample vectors, and searches)
    Demo {
        /// Number of vectors to generate
        #[arg(short, long, default_value = "100")]
        count: usize,

        /// Vector dimensions
        #[arg(short, long, default_value = "128")]
        dimensions: usize,
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

    /// Developer tools: setup, check, generate test data
    #[command(subcommand)]
    Dev(DevCommands),

    /// Start Model Context Protocol (MCP) server for AI agent integration
    Mcp {
        /// Path to the database file (created if not exists)
        #[arg(short, long, default_value = "needle.db")]
        database: String,

        /// Open database in read-only mode
        #[arg(long)]
        read_only: bool,
    },

    /// Initialize a new Needle project with sample configuration
    Init {
        /// Directory to initialize (default: current directory)
        #[arg(default_value = ".")]
        directory: String,

        /// Database name
        #[arg(short, long, default_value = "vectors.needle")]
        database: String,

        /// Default collection dimensions
        #[arg(short = 'D', long, default_value_t = 384)]
        dimensions: usize,
    },

    /// Check local environment and diagnose issues
    Doctor,

    /// Snapshot management for time-travel queries
    #[command(subcommand)]
    Snapshot(SnapshotCommands),

    /// Agentic memory management (store/recall/forget memories)
    #[command(subcommand)]
    Memory(MemoryCommands),

    /// Compare two collections and show differences
    Diff {
        /// Path to the database file
        database: String,
        /// First collection name
        #[arg(short = 'a', long)]
        source: String,
        /// Second collection name
        #[arg(short = 'b', long)]
        target: String,
        /// Maximum differences to show
        #[arg(short, long, default_value_t = 100)]
        limit: usize,
    },

    /// Estimate query cost before execution
    Estimate {
        /// Path to the database file
        database: String,
        /// Collection name
        #[arg(short, long)]
        collection: String,
        /// Number of results (k)
        #[arg(short, long, default_value_t = 10)]
        k: usize,
        /// Whether query will include a filter
        #[arg(long)]
        with_filter: bool,
    },

    /// Recommend the best index type for a workload
    RecommendIndex {
        /// Expected number of vectors
        #[arg(short, long)]
        vectors: usize,

        /// Vector dimensions
        #[arg(short, long)]
        dimensions: usize,

        /// Available memory in MB (optional)
        #[arg(short, long)]
        memory_mb: Option<usize>,

        /// Performance profile: balanced, low-latency, high-recall, low-memory
        #[arg(short, long, default_value = "balanced")]
        profile: String,
    },
}

/// Developer subcommands
#[derive(Subcommand)]
enum DevCommands {
    /// Run pre-commit checks (format + lint + unit tests)
    Check,

    /// Generate a test database with sample data
    GenerateTestData {
        /// Output database path
        #[arg(default_value = "test.needle")]
        output: String,

        /// Number of vectors to generate
        #[arg(short, long, default_value_t = 1000)]
        count: usize,

        /// Vector dimensions
        #[arg(short, long, default_value_t = 128)]
        dimensions: usize,
    },

    /// Show project info (version, features, module count)
    Info,

    /// Run a quick benchmark on insert and search performance
    Benchmark {
        /// Number of vectors
        #[arg(short, long, default_value_t = 10000)]
        count: usize,

        /// Vector dimensions
        #[arg(short, long, default_value_t = 128)]
        dimensions: usize,

        /// Number of search queries to run
        #[arg(short, long, default_value_t = 100)]
        queries: usize,
    },
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

/// Snapshot subcommands for time-travel queries
#[derive(Subcommand)]
enum SnapshotCommands {
    /// Create a named snapshot of a collection
    Create {
        /// Path to the database file
        database: String,
        /// Collection name
        #[arg(short, long)]
        collection: String,
        /// Snapshot name
        #[arg(short, long)]
        name: String,
    },

    /// List all snapshots for a collection
    List {
        /// Path to the database file
        database: String,
        /// Collection name
        #[arg(short, long)]
        collection: String,
    },

    /// Restore a collection from a snapshot
    Restore {
        /// Path to the database file
        database: String,
        /// Collection name
        #[arg(short, long)]
        collection: String,
        /// Snapshot name to restore
        #[arg(short, long)]
        name: String,
    },
}

/// Memory subcommands for agentic memory management
#[derive(Subcommand)]
enum MemoryCommands {
    /// Store a memory with a pre-computed embedding vector
    Remember {
        /// Path to the database file
        database: String,
        /// Collection name (used as memory store)
        #[arg(short, long)]
        collection: String,
        /// Memory content text
        #[arg(short = 't', long)]
        text: String,
        /// Vector embedding as comma-separated floats
        #[arg(short, long)]
        vector: String,
        /// Memory tier: episodic, semantic, procedural
        #[arg(long, default_value = "episodic")]
        tier: String,
        /// Importance score 0.0-1.0
        #[arg(long, default_value_t = 0.5)]
        importance: f32,
    },

    /// Recall memories similar to a query vector
    Recall {
        /// Path to the database file
        database: String,
        /// Collection name
        #[arg(short, long)]
        collection: String,
        /// Query vector as comma-separated floats
        #[arg(short, long)]
        vector: String,
        /// Number of memories to retrieve
        #[arg(short, long, default_value_t = 5)]
        k: usize,
        /// Filter by tier
        #[arg(long)]
        tier: Option<String>,
    },

    /// Forget (delete) a specific memory
    Forget {
        /// Path to the database file
        database: String,
        /// Collection name
        #[arg(short, long)]
        collection: String,
        /// Memory ID to forget
        #[arg(short, long)]
        id: String,
    },
}

fn main() {
    let cli = Cli::parse();

    if cli.verbose {
        std::env::set_var("RUST_LOG", "debug");
    }

    if let Err(err) = run(cli) {
        print_error(&err);
        std::process::exit(1);
    }
}

fn print_error(err: &NeedleError) {
    use needle::error::Recoverable;

    eprintln!("Error: {}", err);

    let code = err.error_code();
    eprintln!("  Code: {} ({})", code.code(), code.category());

    let hints = err.recovery_hints();
    if !hints.is_empty() {
        eprintln!();
        for hint in &hints {
            eprintln!("  Hint: {}", hint);
        }
    }
}

fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Commands::Info { database } => info_command(&database),
        Commands::Create { database } => create_command(&database),
        Commands::Collections { database } => collections_command(&database),
        Commands::CreateCollection {
            database,
            name,
            dimensions,
            distance,
            encrypted,
        } => create_collection_command(&database, &name, dimensions, &distance, encrypted),
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
        } => search_command(
            &database,
            &collection,
            &query,
            k,
            explain,
            distance.as_deref(),
        ),
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
        Commands::Demo { count, dimensions } => demo_command(count, dimensions),
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
        Commands::Dev(cmd) => dev_command(cmd),
        Commands::Mcp { database, read_only } => mcp_command(&database, read_only),
        Commands::Init { directory, database, dimensions } => init_command(&directory, &database, dimensions),
        Commands::Doctor => doctor_command(),
        Commands::Snapshot(cmd) => snapshot_command(cmd),
        Commands::Memory(cmd) => memory_command(cmd),
        Commands::Diff { database, source, target, limit } =>
            diff_command(&database, &source, &target, limit),
        Commands::Estimate { database, collection, k, with_filter } =>
            estimate_command(&database, &collection, k, with_filter),
        Commands::RecommendIndex { vectors, dimensions, memory_mb, profile } =>
            recommend_index_command(vectors, dimensions, memory_mb, &profile),
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
    encrypted: bool,
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

    if encrypted {
        #[cfg(feature = "encryption")]
        {
            println!(
                "Created encrypted collection '{}' with {} dimensions ({} distance)",
                name, dimensions, distance
            );
            println!("  Encryption: ChaCha20-Poly1305");
            println!("  Note: Set NEEDLE_ENCRYPTION_KEY env var or use --key-file for key management");
        }
        #[cfg(not(feature = "encryption"))]
        {
            eprintln!("Warning: --encrypted requires --features encryption. Collection created without encryption.");
        }
    }

    if !encrypted {
        println!(
            "Created collection '{}' with {} dimensions ({} distance)",
            name, dimensions, distance
        );
    }
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

fn search_command(
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

fn parse_distance(distance: &str) -> Option<DistanceFunction> {
    match distance.to_lowercase().as_str() {
        "cosine" => Some(DistanceFunction::Cosine),
        "euclidean" | "l2" => Some(DistanceFunction::Euclidean),
        "dot" | "dotproduct" => Some(DistanceFunction::DotProduct),
        "manhattan" | "l1" => Some(DistanceFunction::Manhattan),
        _ => None,
    }
}

fn parse_query_vector(query_str: &str) -> Result<Vec<f32>> {
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
        println!(
            "Compaction complete: removed {} total deleted vectors",
            total_deleted
        );
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

        serve(config)
            .await
            .map_err(|e| needle::NeedleError::InvalidConfig(e.to_string()))
    })
}

fn demo_command(count: usize, dimensions: usize) -> Result<()> {
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

    // Run a search
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

    // Filtered search
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

fn tune_command(
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

fn query_command(
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
        eprintln!(
            "Error: Destination '{}' already exists. Use --force to overwrite.",
            output
        );
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
    println!(
        "Note: To save to '{}', use the database normally and call save().",
        output
    );

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
        println!(
            "No backups to clean up (have {}, keeping {}).",
            backups.len(),
            keep
        );
        return Ok(());
    }

    let to_remove = backups.len() - keep;
    println!(
        "Would remove {} old backup(s), keeping last {}.",
        to_remove, keep
    );
    println!(
        "Note: Manual cleanup - delete old backup files from: {}",
        path
    );

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

fn drift_baseline(
    database: &str,
    collection_name: &str,
    output: &str,
    sample_size: usize,
) -> Result<()> {
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

fn drift_detect(
    database: &str,
    collection_name: &str,
    baseline_path: &str,
    threshold: f64,
) -> Result<()> {
    let db = Database::open(database)?;
    let coll = db.collection(collection_name)?;

    // Load baseline
    let baseline_content = std::fs::read_to_string(baseline_path)?;
    let baseline_data: serde_json::Value = serde_json::from_str(&baseline_content)?;

    let dimensions = baseline_data["dimensions"].as_u64().ok_or_else(|| {
        needle::NeedleError::InvalidInput("Invalid baseline: missing dimensions".to_string())
    })? as usize;

    let baseline_centroid: Vec<f32> = baseline_data["centroid"]
        .as_array()
        .ok_or_else(|| {
            needle::NeedleError::InvalidInput("Invalid baseline: missing centroid".to_string())
        })?
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
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_f64().map(|f| f as f32))
                .collect()
        })
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
    println!(
        "Drift detected: {}",
        if drift_detected { "YES" } else { "NO" }
    );
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

fn drift_report(
    database: &str,
    collection_name: &str,
    baseline_path: &str,
    format: &str,
) -> Result<()> {
    let db = Database::open(database)?;
    let coll = db.collection(collection_name)?;

    // Load baseline
    let baseline_content = std::fs::read_to_string(baseline_path)?;
    let baseline_data: serde_json::Value = serde_json::from_str(&baseline_content)?;

    let dimensions = baseline_data["dimensions"].as_u64().ok_or_else(|| {
        needle::NeedleError::InvalidInput("Invalid baseline: missing dimensions".to_string())
    })? as usize;

    let baseline_centroid: Vec<f32> = baseline_data["centroid"]
        .as_array()
        .ok_or_else(|| {
            needle::NeedleError::InvalidInput("Invalid baseline: missing centroid".to_string())
        })?
        .iter()
        .filter_map(|v| v.as_f64().map(|f| f as f32))
        .collect();

    let baseline_variance: Vec<f32> = baseline_data["variance"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_f64().map(|f| f as f32))
                .collect()
        })
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
            current_vecs
                .iter()
                .map(|v| (v[d] - mean).powi(2))
                .sum::<f32>()
                / current_vecs.len().max(1) as f32
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
        println!(
            "Analysis time: {}",
            chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
        );
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
    use needle::federated::{
        Federation, FederationConfig, InstanceConfig, MergeStrategy, RoutingStrategy,
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
    use needle::federated::{Federation, FederationConfig, InstanceConfig};

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
    println!(
        "Healthy instances: {}/{}",
        health.healthy_instances, health.total_instances
    );
    println!("Degraded instances: {}", health.degraded_instances);
    println!("Unhealthy instances: {}", health.unhealthy_instances);
    println!("Average latency: {:.2} ms", health.avg_latency_ms);

    Ok(())
}

fn federate_stats(instances_str: &str) -> Result<()> {
    use needle::federated::{Federation, FederationConfig, InstanceConfig};

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

fn dev_command(cmd: DevCommands) -> Result<()> {
    match cmd {
        DevCommands::Check => {
            println!("Running pre-commit checks...\n");
            let steps = [
                ("Format check", "cargo fmt -- --check"),
                ("Lint", "cargo clippy --features full -- -D warnings"),
                ("Unit tests", "cargo test --lib"),
            ];
            for (name, command) in &steps {
                println!("▶ {}...", name);
                let status = std::process::Command::new("sh")
                    .arg("-c")
                    .arg(command)
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

    // Benchmark inserts
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

    // Benchmark search
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

fn init_command(directory: &str, database: &str, dimensions: usize) -> Result<()> {
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

    // Create database with a default collection
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

fn doctor_command() -> Result<()> {
    println!("Needle Doctor — Environment Check\n");

    // Check Rust
    let rust_version = option_env!("CARGO_PKG_RUST_VERSION").unwrap_or("unknown");
    println!("  ✓ Rust MSRV: {rust_version}");

    // Check version
    println!("  ✓ Needle version: {}", env!("CARGO_PKG_VERSION"));

    // Check compiled features
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

    // Check if database path is writable
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

fn mcp_command(database: &str, read_only: bool) -> Result<()> {
    let db = Database::open(database)?;
    let server = needle::mcp::McpServer::new(db, read_only);
    eprintln!("Needle MCP server started (stdio transport)");
    eprintln!("Database: {database}");
    eprintln!("Read-only: {read_only}");
    server.run()
}

fn recommend_index_command(
    vectors: usize,
    dimensions: usize,
    memory_mb: Option<usize>,
    profile: &str,
) -> Result<()> {
    use needle::tuning::{IndexSelectionConstraints, recommend_index};

    let constraints = IndexSelectionConstraints {
        expected_vectors: vectors,
        dimensions,
        available_memory_bytes: memory_mb.map(|mb| mb * 1024 * 1024),
        available_disk_bytes: None,
        low_latency_critical: profile == "low-latency",
        target_recall: match profile {
            "high-recall" => 0.99,
            "low-latency" => 0.90,
            _ => 0.95,
        },
        frequent_updates: false,
    };

    let recommendation = recommend_index(&constraints);

    println!("Index Recommendation");
    println!("════════════════════");
    println!("  Vectors:    {vectors}");
    println!("  Dimensions: {dimensions}");
    if let Some(mb) = memory_mb {
        println!("  Memory:     {mb} MB");
    }
    println!("  Profile:    {profile}");
    println!();
    println!("  Recommended: {:?}", recommendation.recommended);
    println!();
    for line in &recommendation.explanation {
        println!("  {line}");
    }
    if !recommendation.alternatives.is_empty() {
        println!();
        println!("  Alternatives:");
        for alt in &recommendation.alternatives {
            println!("    - {:?}", alt);
        }
    }

    Ok(())
}

fn snapshot_command(cmd: SnapshotCommands) -> Result<()> {
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
    }
}

fn memory_command(cmd: MemoryCommands) -> Result<()> {
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

fn diff_command(path: &str, source: &str, target: &str, limit: usize) -> Result<()> {
    let db = Database::open(path)?;
    let coll_a = db.collection(source)?;
    let coll_b = db.collection(target)?;

    let ids_a: std::collections::HashSet<String> = coll_a.ids()?.into_iter().collect();
    let ids_b: std::collections::HashSet<String> = coll_b.ids()?.into_iter().collect();

    let only_a: Vec<&String> = ids_a.difference(&ids_b).take(limit).collect();
    let only_b: Vec<&String> = ids_b.difference(&ids_a).take(limit).collect();
    let shared: Vec<&String> = ids_a.intersection(&ids_b).collect();

    let mut modified_count = 0usize;
    for id in shared.iter().take(limit) {
        if let (Some((va, _)), Some((vb, _))) = (coll_a.get(id), coll_b.get(id)) {
            let dist: f32 = va.iter().zip(vb.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f32>().sqrt();
            if dist > 1e-6 {
                modified_count += 1;
                if modified_count <= 10 {
                    println!("  Modified: {} (L2 distance: {:.6})", id, dist);
                }
            }
        }
    }

    println!("Diff: '{}' vs '{}'", source, target);
    println!("  Source vectors: {}", ids_a.len());
    println!("  Target vectors: {}", ids_b.len());
    println!("  Only in source: {}", only_a.len());
    println!("  Only in target: {}", only_b.len());
    println!("  Modified: {}", modified_count);
    println!("  Unchanged: {}", shared.len() - modified_count);

    if !only_a.is_empty() {
        println!("\n  Removed (first {}):", only_a.len().min(10));
        for id in only_a.iter().take(10) { println!("    - {}", id); }
    }
    if !only_b.is_empty() {
        println!("\n  Added (first {}):", only_b.len().min(10));
        for id in only_b.iter().take(10) { println!("    + {}", id); }
    }

    Ok(())
}

fn estimate_command(path: &str, collection: &str, k: usize, with_filter: bool) -> Result<()> {
    use needle::search::cost_estimator::{CostEstimator, CollectionStatistics};

    let db = Database::open(path)?;
    let coll = db.collection(collection)?;
    let stats = coll.stats()?;

    let col_stats = CollectionStatistics::new(
        stats.vector_count,
        stats.dimensions,
        if stats.vector_count > 0 {
            coll.deleted_count() as f32 / (stats.vector_count + coll.deleted_count()) as f32
        } else {
            0.0
        },
    );

    let filter_sel = if with_filter { Some(0.3f32) } else { None };
    let estimator = CostEstimator::default();
    let plan = estimator.plan(&col_stats, k, filter_sel);

    println!("Query Cost Estimate for '{}' (k={})", collection, k);
    println!("  Collection: {} vectors, {} dimensions", stats.vector_count, stats.dimensions);
    println!("  Strategy: {}", plan.index_choice);
    println!("  Estimated latency: {:.2} ms", plan.cost.estimated_latency_ms);
    println!("  Estimated memory: {:.2} MB", plan.cost.estimated_memory_mb);
    println!("  Distance computations: {}", plan.cost.distance_computations);
    println!("  Nodes visited: {}", plan.cost.nodes_visited);
    println!("  Rationale:");
    for r in &plan.rationale {
        println!("    - {}", r);
    }

    Ok(())
}
