use clap::Subcommand;

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

    /// Serverless function management (deploy/list/logs/remove)
    #[command(subcommand)]
    Function(FunctionCommands),

    /// Materialized view management (create/list/drop/refresh)
    #[command(subcommand)]
    Views(ViewsCommands),

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
        /// Similarity threshold for considering vectors "modified" (L2 distance)
        #[arg(long, default_value_t = 1e-6)]
        threshold: f32,
    },

    /// Merge vectors from source collection into target collection
    Merge {
        /// Path to the database file
        database: String,
        /// Source collection to merge from
        #[arg(short = 'a', long)]
        source: String,
        /// Target collection to merge into
        #[arg(short = 'b', long)]
        target: String,
        /// Base collection for 3-way merge (common ancestor)
        #[arg(long)]
        base: Option<String>,
        /// Conflict strategy: source-wins, target-wins, skip
        #[arg(long, default_value = "source-wins")]
        strategy: String,
        /// Dry run (show what would change without applying)
        #[arg(long)]
        dry_run: bool,
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

    /// Execute a SQL-compatible query against the database
    Sql {
        /// Path to the database file
        database: String,

        /// SQL query to execute (e.g., "SELECT * FROM docs WHERE vector SIMILAR TO $query LIMIT 10")
        #[arg(short, long)]
        query: String,

        /// Output format (json, table, csv)
        #[arg(long, default_value = "json")]
        format: String,

        /// Query vector as comma-separated floats (e.g., "0.1,0.2,0.3")
        #[arg(long)]
        vector: Option<String>,
    },

    /// Show provenance information for a vector
    Provenance {
        /// Path to the database file
        database: String,
        /// Collection name
        #[arg(short, long)]
        collection: String,
        /// Vector ID to get provenance for
        #[arg(short, long)]
        id: String,
    },

    /// Evaluate search quality against ground truth data
    Evaluate {
        /// Path to the database file
        database: String,

        /// Collection name
        #[arg(short, long)]
        collection: String,

        /// Path to ground truth JSON file
        #[arg(short, long)]
        ground_truth: String,

        /// Number of results to retrieve per query
        #[arg(short, long, default_value = "10")]
        k: usize,
    },

    /// Export a collection as a portable bundle
    ExportBundle {
        /// Path to the database file
        database: String,

        /// Collection name
        #[arg(short, long)]
        collection: String,

        /// Output bundle file path
        #[arg(short, long)]
        output: String,
    },

    /// Import a collection from a portable bundle
    ImportBundle {
        /// Path to the database file
        database: String,

        /// Bundle file path to import
        #[arg(short, long)]
        bundle: String,

        /// Override collection name (optional)
        #[arg(short, long)]
        name: Option<String>,
    },

    /// Analyze collection and recommend optimal compression strategy
    AdviseCompression {
        /// Path to the database file
        database: String,

        /// Collection name
        #[arg(short, long)]
        collection: String,

        /// Number of recall test queries
        #[arg(long, default_value = "100")]
        test_queries: usize,

        /// Recall k for evaluation
        #[arg(short, long, default_value = "10")]
        k: usize,

        /// Target recall levels (comma-separated, e.g. "0.99,0.95,0.90")
        #[arg(long, default_value = "0.99,0.95,0.90")]
        targets: String,

        /// Apply the recommended compression strategy immediately
        #[arg(long)]
        apply: bool,
    },

    /// Migrate vectors from external vector database
    Migrate {
        /// Path to the target database file
        database: String,

        /// Source system (qdrant, chromadb, milvus, pinecone)
        #[arg(short, long)]
        source: String,

        /// Source connection URL
        #[arg(long)]
        url: String,

        /// Target collection name
        #[arg(short, long)]
        collection: String,

        /// Dry run (validate without importing)
        #[arg(long)]
        dry_run: bool,

        /// Batch size for streaming transfer
        #[arg(long, default_value = "1000")]
        batch_size: usize,

        /// Resume from a previous migration checkpoint
        #[arg(long)]
        resume: Option<String>,

        /// Rollback a previously completed migration
        #[arg(long)]
        rollback: bool,
    },

    /// Visualize HNSW graph traversal for a search query (debug tool)
    ExplainSearch {
        /// Path to the database file
        database: String,

        /// Collection name
        #[arg(short, long)]
        collection: String,

        /// Query vector (comma-separated floats)
        #[arg(short, long)]
        query: String,

        /// Number of results
        #[arg(short, long, default_value = "10")]
        k: usize,

        /// Output format (text, json)
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Recommend optimal index type with what-if analysis
    Advise {
        /// Path to the database file
        database: String,

        /// Collection name
        #[arg(short, long)]
        collection: String,

        /// Number of sample queries for what-if analysis
        #[arg(long, default_value = "50")]
        sample_queries: usize,
    },

    /// Watch collection for real-time change events (CDC)
    Watch {
        /// Path to the database file
        database: String,

        /// Collection name
        #[arg(short, long)]
        collection: String,

        /// Resume from a specific sequence number (0 = from beginning)
        #[arg(long, default_value = "0")]
        from_sequence: u64,

        /// Maximum events per poll batch
        #[arg(long, default_value = "100")]
        batch_size: usize,

        /// Consumer ID for offset tracking
        #[arg(long, default_value = "cli-watcher")]
        consumer_id: String,
    },

    /// Show incremental sync status and manage replication
    Sync {
        /// Path to the database file
        database: String,

        /// Replica ID for this node
        #[arg(long, default_value = "replica-0")]
        replica_id: String,

        /// Show sync status (latest LSN, buffer size, replicas)
        #[arg(long)]
        status: bool,
    },

    /// Deduplicate near-duplicate vectors in a collection
    Dedup {
        /// Path to the database file
        database: String,

        /// Collection name
        #[arg(short, long)]
        collection: String,

        /// Similarity threshold (0.0-1.0, default: 0.95)
        #[arg(short, long, default_value = "0.95")]
        threshold: f32,

        /// Merge strategy: keep-first, keep-latest, merge-metadata
        #[arg(long, default_value = "keep-first")]
        strategy: String,

        /// Dry run (show duplicates without removing)
        #[arg(long)]
        dry_run: bool,
    },

    /// Show collection health score and anomaly detection
    Health {
        /// Path to the database file
        database: String,

        /// Collection name
        #[arg(short, long)]
        collection: String,

        /// Output format (text, json)
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Interactive REPL/playground for vector exploration
    Playground {
        /// Path to the database file (optional, starts in-memory)
        #[arg(short, long)]
        database: Option<String>,
    },

    /// Run standardized ANN benchmarks with recall and QPS metrics
    Bench {
        /// Number of vectors (default: 10000)
        #[arg(short, long, default_value = "10000")]
        vectors: usize,

        /// Vector dimensions (default: 128)
        #[arg(short, long, default_value = "128")]
        dimensions: usize,

        /// Number of queries to run (default: 100)
        #[arg(short, long, default_value = "100")]
        queries: usize,

        /// K values for recall measurement (comma-separated)
        #[arg(long, default_value = "1,10,100")]
        k_values: String,

        /// Output format (text, json, html)
        #[arg(long, default_value = "text")]
        format: String,

        /// Output file path for report
        #[arg(short, long)]
        output: Option<String>,

        /// Compare with a previous benchmark report (JSON file)
        #[arg(long)]
        compare: Option<String>,
    },

    /// Streaming ingestion management
    #[command(subcommand)]
    Ingestion(IngestionCommands),

    /// Semantic cache management
    #[command(subcommand)]
    Cache(CacheCommands),

    /// Analyze collection for auto-partitioning recommendations
    Partition {
        /// Path to the database file
        database: String,

        /// Collection name
        #[arg(short, long)]
        collection: String,

        /// Analyze only (don't apply)
        #[arg(long)]
        analyze: bool,

        /// Target partition size (number of vectors per partition)
        #[arg(long, default_value = "100000")]
        target_size: usize,
    },
}

/// Cache management subcommands
#[derive(Subcommand)]
pub enum CacheCommands {
    /// Show cache statistics
    Stats {
        /// Path to the database file
        database: String,
    },
    /// Clear all cache entries
    Clear {
        /// Path to the database file
        database: String,
        /// Skip confirmation prompt
        #[arg(long)]
        force: bool,
    },
}

/// Ingestion subcommands
#[derive(Subcommand)]
pub enum IngestionCommands {
    /// Show ingestion pipeline status
    Status {
        /// Path to the database file
        database: String,
    },
}

/// Developer subcommands
#[derive(Subcommand)]
pub enum DevCommands {
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
pub enum BackupCommands {
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
pub enum DriftCommands {
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
pub enum AliasCommands {
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
pub enum TtlCommands {
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
pub enum FederateCommands {
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
pub enum SnapshotCommands {
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

    /// Prune old snapshots beyond retention window
    Prune {
        /// Path to the database file
        database: String,
        /// Collection name
        #[arg(short, long)]
        collection: String,
        /// Retention window in seconds (snapshots older than this are pruned)
        #[arg(short, long)]
        retention_secs: u64,
        /// Dry run (show what would be pruned without actually pruning)
        #[arg(long)]
        dry_run: bool,
    },
}

/// Memory subcommands for agentic memory management
#[derive(Subcommand)]
pub enum MemoryCommands {
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

/// Serverless function management subcommands
#[derive(Subcommand)]
pub enum FunctionCommands {
    /// Deploy a new serverless function
    Deploy {
        /// Path to the database file
        database: String,
        /// Function name
        #[arg(short, long)]
        name: String,
        /// Event filters (comma-separated, e.g. "vector.inserted,vector.deleted")
        #[arg(short, long, default_value = "*")]
        events: String,
        /// Collection filter (only trigger for this collection)
        #[arg(short, long)]
        collection: Option<String>,
    },

    /// List deployed functions
    List {
        /// Path to the database file
        database: String,
    },

    /// Show function invocation logs
    Logs {
        /// Path to the database file
        database: String,
        /// Function name (optional, shows all if omitted)
        #[arg(short, long)]
        name: Option<String>,
        /// Maximum number of log entries
        #[arg(short, long, default_value_t = 50)]
        limit: usize,
    },

    /// Remove a deployed function
    Remove {
        /// Path to the database file
        database: String,
        /// Function name
        #[arg(short, long)]
        name: String,
    },
}

/// Materialized view management subcommands
#[derive(Subcommand)]
pub enum ViewsCommands {
    /// Create a materialized view from a NeedleQL query
    Create {
        /// Path to the database file
        database: String,
        /// View definition (NeedleQL CREATE VIEW statement)
        #[arg(short, long)]
        query: String,
    },

    /// List all materialized views
    List {
        /// Path to the database file
        database: String,
    },

    /// Drop a materialized view
    Drop {
        /// Path to the database file
        database: String,
        /// View name
        #[arg(short, long)]
        name: String,
    },

    /// Refresh a materialized view
    Refresh {
        /// Path to the database file
        database: String,
        /// View name (or "all" to refresh all stale views)
        #[arg(short, long)]
        name: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
}
