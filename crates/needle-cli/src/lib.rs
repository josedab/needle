//! Needle CLI - Command line interface for the Needle vector database

mod commands;
mod features;
mod handlers;

use clap::{Parser, Subcommand};
use commands::{AliasCommands, BackupCommands, DriftCommands, FederateCommands, TtlCommands};
use needle::Result;

#[derive(Parser)]
#[command(name = "needle")]
#[command(author, version, about = "Embedded Vector Database - SQLite for Vectors", long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Output format: "text" (default) or "json" for machine-readable output
    #[arg(long, global = true, default_value = "text")]
    output: OutputFormat,
}

/// Output format for CLI commands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum OutputFormat {
    /// Human-readable text output (default)
    Text,
    /// JSON output for scripting and automation
    Json,
}

#[derive(Subcommand)]
pub enum Commands {
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

    /// Run diagnostics on a database to check integrity and health
    Diagnose {
        /// Path to the database file
        database: String,

        /// Run extended checks (slower but more thorough)
        #[arg(long)]
        extended: bool,
    },
}

pub fn run() -> Result<()> {
    let cli = Cli::parse();
    let json_output = cli.output == OutputFormat::Json;

    match cli.command {
        Commands::Info { database } => handlers::info_command(&database, json_output),
        Commands::Create { database } => handlers::create_command(&database),
        Commands::Collections { database } => handlers::collections_command(&database, json_output),
        Commands::CreateCollection {
            database,
            name,
            dimensions,
            distance,
        } => handlers::create_collection_command(&database, &name, dimensions, &distance),
        Commands::Stats {
            database,
            collection,
        } => handlers::stats_command(&database, &collection, json_output),
        Commands::Insert {
            database,
            collection,
        } => handlers::insert_command(&database, &collection),
        Commands::Search {
            database,
            collection,
            query,
            k,
            explain,
            distance,
        } => handlers::search_command(
            &database,
            &collection,
            &query,
            k,
            explain,
            distance.as_deref(),
            json_output,
        ),
        Commands::Delete {
            database,
            collection,
            id,
        } => handlers::delete_command(&database, &collection, &id),
        Commands::Get {
            database,
            collection,
            id,
        } => handlers::get_command(&database, &collection, &id, json_output),
        Commands::Compact { database } => handlers::compact_command(&database),
        Commands::Export {
            database,
            collection,
        } => handlers::export_command(&database, &collection),
        Commands::Import {
            database,
            collection,
            file,
        } => handlers::import_command(&database, &collection, &file),
        Commands::Count {
            database,
            collection,
        } => handlers::count_command(&database, &collection, json_output),
        Commands::Clear {
            database,
            collection,
            force,
        } => handlers::clear_command(&database, &collection, force),
        #[cfg(feature = "server")]
        Commands::Serve { address, database } => handlers::serve_command(&address, database),
        Commands::Tune {
            vectors,
            dimensions,
            profile,
            memory_mb,
        } => handlers::tune_command(vectors, dimensions, &profile, memory_mb),
        Commands::Query {
            database,
            collection,
            query,
            k,
            analyze,
        } => handlers::query_command(&database, &collection, &query, k, analyze),
        Commands::Backup(cmd) => features::backup_command(cmd),
        #[cfg(feature = "observability")]
        Commands::Drift(cmd) => features::drift_command(cmd),
        #[cfg(not(feature = "observability"))]
        Commands::Drift(_) => {
            eprintln!("Drift detection requires the 'observability' feature. Rebuild with --features observability");
            std::process::exit(1);
        }
        Commands::Federate(cmd) => features::federate_command(cmd),
        Commands::Alias(cmd) => features::alias_command(cmd),
        Commands::Ttl(cmd) => features::ttl_command(cmd),
        Commands::Diagnose { database, extended } => handlers::diagnose_command(&database, extended, json_output),
    }
}
