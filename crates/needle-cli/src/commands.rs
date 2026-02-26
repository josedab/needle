use clap::Subcommand;

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
