//! Error Types and Handling
//!
//! Comprehensive error types for the Needle vector database with structured
//! error codes for programmatic handling and detailed error messages for debugging.
//!
//! # Error Categories
//!
//! Errors are organized into categories with numeric codes:
//!
//! | Range | Category | Examples |
//! |-------|----------|----------|
//! | 1xxx | I/O errors | Read, Write, Permission, DiskFull |
//! | 2xxx | Serialization | Serialize, Deserialize, InvalidFormat |
//! | 3xxx | Collection | NotFound, AlreadyExists, Corrupted |
//! | 4xxx | Vector | NotFound, DimensionMismatch, Invalid |
//! | 5xxx | Database | Invalid, Corrupted, Locked |
//! | 6xxx | Index | Error, Corrupted, BuildFailed |
//! | 7xxx | Configuration | Invalid, Missing |
//! | 8xxx | Resource | Capacity, Quota, Memory |
//! | 9xxx | Operational | Timeout, Lock, Conflict |
//! | 10xxx | Security | Encryption, Decryption, Auth |
//! | 11xxx | Distributed | Consensus, Replication, Network |
//! | 12xxx | Backup | Failed, Restore, Corrupted |
//! | 13xxx | State | InvalidOperation, InvalidState |
//!
//! # Example
//!
//! ```rust
//! use needle::error::{NeedleError, Result, ErrorCode, Recoverable};
//!
//! fn example_operation() -> Result<()> {
//!     // Use Result<T> which is an alias for std::result::Result<T, NeedleError>
//!     Err(NeedleError::CollectionNotFound("my_collection".to_string()))
//! }
//!
//! fn handle_error(err: NeedleError) {
//!     // Get the error code for programmatic handling
//!     let code = err.error_code();
//!     println!("Error code: {:?} ({})", code, code.code());
//!
//!     // Match on specific error variants
//!     match err {
//!         NeedleError::CollectionNotFound(name) => {
//!             println!("Collection '{}' not found", name);
//!         }
//!         NeedleError::DimensionMismatch { expected, got } => {
//!             println!("Expected {} dimensions, got {}", expected, got);
//!         }
//!         _ => println!("Other error: {}", err),
//!     }
//! }
//! ```
//!
//! # Error Propagation
//!
//! Use the `?` operator to propagate errors:
//!
//! ```rust,ignore
//! use needle::{Database, Result};
//!
//! fn load_and_search(path: &str, query: &[f32]) -> Result<Vec<String>> {
//!     let db = Database::open(path)?;  // Propagates IoError, etc.
//!     let coll = db.collection("docs")?;  // Propagates CollectionNotFound
//!     let results = coll.search(query, 10)?;  // Propagates DimensionMismatch, etc.
//!     Ok(results.iter().map(|r| r.id.clone()).collect())
//! }
//! ```

use thiserror::Error;

/// Error code categories for programmatic error handling.
///
/// Each error code belongs to a category indicated by its numeric range.
/// Use [`ErrorCode::category()`] to get the human-readable category name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorCode {
    /// Failed to read from disk or network
    IoRead = 1001,
    /// Failed to write to disk or network
    IoWrite = 1002,
    /// Insufficient file system permissions
    IoPermission = 1003,
    /// Disk is full or quota exceeded
    IoDiskFull = 1004,

    /// Failed to serialize data (e.g., JSON encoding)
    SerializationFailed = 2001,
    /// Failed to deserialize data (e.g., corrupt JSON)
    DeserializationFailed = 2002,
    /// Data format is invalid or unsupported
    InvalidFormat = 2003,

    /// Referenced collection does not exist
    CollectionNotFound = 3001,
    /// A collection with this name already exists
    CollectionAlreadyExists = 3002,
    /// Collection data is corrupted
    CollectionCorrupted = 3003,
    /// Referenced alias does not exist
    AliasNotFound = 3004,
    /// An alias with this name already exists
    AliasAlreadyExists = 3005,
    /// Cannot drop collection because aliases reference it
    AliasTargetHasAliases = 3006,

    /// Referenced vector ID does not exist
    VectorNotFound = 4001,
    /// A vector with this ID already exists
    VectorAlreadyExists = 4002,
    /// Vector dimensions do not match collection
    DimensionMismatch = 4003,
    /// Vector contains invalid values (NaN, Infinity)
    InvalidVector = 4004,

    /// Database file is invalid or unrecognized
    InvalidDatabase = 5001,
    /// Database file is corrupted
    DatabaseCorrupted = 5002,
    /// Database is locked by another process
    DatabaseLocked = 5003,

    /// General index operation failure
    IndexError = 6001,
    /// Index data is corrupted
    IndexCorrupted = 6002,
    /// Index construction failed
    IndexBuildFailed = 6003,

    /// Configuration value is invalid
    InvalidConfig = 7001,
    /// Required configuration is missing
    MissingConfig = 7002,

    /// Collection or database capacity limit reached
    CapacityExceeded = 8001,
    /// Usage quota exceeded
    QuotaExceeded = 8002,
    /// System memory exhausted
    MemoryExhausted = 8003,

    /// Operation timed out
    Timeout = 9001,
    /// Lock acquisition timed out
    LockTimeout = 9002,
    /// Conflicting concurrent operation
    Conflict = 9003,
    /// Generic resource not found
    NotFound = 9004,

    /// Encryption operation failed
    EncryptionError = 10001,
    /// Decryption operation failed
    DecryptionError = 10002,
    /// Authentication check failed
    AuthenticationFailed = 10003,

    /// Raft consensus failure
    ConsensusError = 11001,
    /// Data replication failure
    ReplicationError = 11002,
    /// Network communication failure
    NetworkError = 11003,

    /// Backup creation failed
    BackupFailed = 12001,
    /// Backup restoration failed
    RestoreFailed = 12002,
    /// Backup file is corrupted
    BackupCorrupted = 12003,

    /// Operation is not valid in current context
    InvalidOperation = 13001,
    /// System is in an invalid state
    InvalidState = 13002,
    /// Caller lacks required authorization
    Unauthorized = 13003,
}

impl ErrorCode {
    /// Get the numeric error code
    pub fn code(&self) -> u32 {
        *self as u32
    }

    /// Get a brief description of the error category
    pub fn category(&self) -> &'static str {
        match self {
            ErrorCode::IoRead
            | ErrorCode::IoWrite
            | ErrorCode::IoPermission
            | ErrorCode::IoDiskFull => "I/O",
            ErrorCode::SerializationFailed
            | ErrorCode::DeserializationFailed
            | ErrorCode::InvalidFormat => "Serialization",
            ErrorCode::CollectionNotFound
            | ErrorCode::CollectionAlreadyExists
            | ErrorCode::CollectionCorrupted
            | ErrorCode::AliasNotFound
            | ErrorCode::AliasAlreadyExists
            | ErrorCode::AliasTargetHasAliases => "Collection",
            ErrorCode::VectorNotFound
            | ErrorCode::VectorAlreadyExists
            | ErrorCode::DimensionMismatch
            | ErrorCode::InvalidVector => "Vector",
            ErrorCode::InvalidDatabase
            | ErrorCode::DatabaseCorrupted
            | ErrorCode::DatabaseLocked => "Database",
            ErrorCode::IndexError | ErrorCode::IndexCorrupted | ErrorCode::IndexBuildFailed => {
                "Index"
            }
            ErrorCode::InvalidConfig | ErrorCode::MissingConfig => "Configuration",
            ErrorCode::CapacityExceeded | ErrorCode::QuotaExceeded | ErrorCode::MemoryExhausted => {
                "Resource"
            }
            ErrorCode::Timeout
            | ErrorCode::LockTimeout
            | ErrorCode::Conflict
            | ErrorCode::NotFound => "Operational",
            ErrorCode::EncryptionError
            | ErrorCode::DecryptionError
            | ErrorCode::AuthenticationFailed => "Security",
            ErrorCode::ConsensusError | ErrorCode::ReplicationError | ErrorCode::NetworkError => {
                "Distributed"
            }
            ErrorCode::BackupFailed | ErrorCode::RestoreFailed | ErrorCode::BackupCorrupted => {
                "Backup"
            }
            ErrorCode::InvalidOperation | ErrorCode::InvalidState | ErrorCode::Unauthorized => {
                "State"
            }
        }
    }
}

/// A recovery hint providing actionable guidance for resolving errors
#[derive(Debug, Clone)]
pub struct RecoveryHint {
    /// Short summary of the recovery action
    pub summary: String,
    /// Detailed steps or explanation
    pub details: Option<String>,
    /// Related documentation or reference
    pub doc_ref: Option<String>,
}

impl RecoveryHint {
    /// Create a new recovery hint with just a summary
    pub fn new(summary: impl Into<String>) -> Self {
        Self {
            summary: summary.into(),
            details: None,
            doc_ref: None,
        }
    }

    /// Add detailed recovery steps
    pub fn with_details(mut self, details: impl Into<String>) -> Self {
        self.details = Some(details.into());
        self
    }

    /// Add documentation reference
    pub fn with_doc(mut self, doc_ref: impl Into<String>) -> Self {
        self.doc_ref = Some(doc_ref.into());
        self
    }
}

impl std::fmt::Display for RecoveryHint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.summary)?;
        if let Some(details) = &self.details {
            write!(f, "\n  Details: {}", details)?;
        }
        if let Some(doc) = &self.doc_ref {
            write!(f, "\n  See: {}", doc)?;
        }
        Ok(())
    }
}

/// Trait for errors that can provide recovery hints
pub trait Recoverable {
    /// Get the error code for this error
    fn error_code(&self) -> ErrorCode;

    /// Get recovery hints for this error
    fn recovery_hints(&self) -> Vec<RecoveryHint>;

    /// Check if the error is retryable
    fn is_retryable(&self) -> bool;

    /// Get suggested retry delay in milliseconds
    fn suggested_retry_delay_ms(&self) -> Option<u64>;
}

/// Error types for Needle database operations.
///
/// Each variant maps to an [`ErrorCode`] for programmatic handling. Use
/// [`Recoverable::error_code()`] to get the code and
/// [`Recoverable::recovery_hints()`] for actionable remediation steps.
///
/// # Matching
///
/// ```rust,ignore
/// match err {
///     NeedleError::CollectionNotFound(name) => { /* create collection or correct name */ }
///     NeedleError::DimensionMismatch { expected, got } => { /* fix vector size */ }
///     other => eprintln!("error {}: {}", other.error_code().code(), other),
/// }
/// ```
#[must_use]
#[derive(Error, Debug)]
pub enum NeedleError {
    /// An I/O error occurred during file or network operations.
    ///
    /// Returned when Needle cannot read from or write to the database file,
    /// typically caused by file system issues, permission errors, or disk full
    /// conditions.  Inspect the inner [`std::io::Error`] for the root cause.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Failed to serialize or deserialize data (JSON).
    ///
    /// Usually indicates corrupted data, an incompatible schema version, or
    /// a malformed JSON payload sent to the REST API.  Try re-creating the
    /// database from a backup if the on-disk file is corrupt.
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// The vector dimension does not match the collection's configured dimension.
    ///
    /// Returned by `insert`, `update`, `search`, and their batch variants when
    /// the supplied vector length differs from the collection's `dimensions`.
    /// Verify the embedding model output size matches the collection config.
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    /// The requested collection does not exist in the database.
    ///
    /// Returned by `Database::collection()` and any method that references a
    /// collection by name (including aliases).  Call
    /// `Database::list_collections()` to see available collections, or create
    /// the collection first with `Database::create_collection()`.
    #[error("Collection '{0}' not found")]
    CollectionNotFound(String),

    /// A collection with the given name already exists.
    ///
    /// Returned by `Database::create_collection()`.  Use a different name,
    /// delete the existing collection with `Database::drop_collection()`, or
    /// use `Database::collection()` to access the existing one.
    #[error("Collection '{0}' already exists")]
    CollectionAlreadyExists(String),

    /// The requested alias does not exist.
    ///
    /// Returned by `Database::update_alias()` or `Database::delete_alias()`
    /// when the alias name has not been registered.  Create the alias first
    /// with `Database::create_alias()`.
    #[error("Alias '{0}' not found")]
    AliasNotFound(String),

    /// An alias with the given name already exists.
    ///
    /// Returned by `Database::create_alias()`.  Use `update_alias()` to
    /// re-point an existing alias to a different collection.
    #[error("Alias '{0}' already exists")]
    AliasAlreadyExists(String),

    /// Cannot drop a collection because one or more aliases still reference it.
    ///
    /// Remove or re-point all aliases that target this collection before
    /// calling `Database::drop_collection()`.
    #[error("Cannot drop collection '{0}': aliases still reference it")]
    CollectionHasAliases(String),

    /// The requested vector ID was not found in the collection.
    ///
    /// Returned by `get`, `update`, `delete`, `set_ttl`, and similar
    /// single-vector operations.  The string payload is the missing ID.
    #[error("Vector '{0}' not found")]
    VectorNotFound(String),

    /// A vector with the given ID already exists in the collection.
    ///
    /// Returned by `insert` and `insert_batch`.  Use `upsert` if you want
    /// insert-or-update semantics, or `delete` the existing vector first.
    #[error("Vector '{0}' already exists")]
    VectorAlreadyExists(String),

    /// A duplicate ID was encountered during a batch operation.
    ///
    /// Returned by `insert_batch` when the input contains the same ID more
    /// than once.  De-duplicate IDs before calling batch methods.
    #[error("Duplicate ID: '{0}'")]
    DuplicateId(String),

    /// Another operation is currently in progress and prevents this one.
    ///
    /// Typically transient — retry after the ongoing operation completes.
    #[error("Operation in progress: {0}")]
    OperationInProgress(String),

    /// The database file is not a valid Needle database or has an unsupported format version.
    ///
    /// Returned by `Database::open()`.  The file may belong to a different
    /// application, or the Needle version may be too old to read this format.
    #[error("Invalid database file: {0}")]
    InvalidDatabase(String),

    /// Data corruption was detected (e.g., checksum mismatch).
    ///
    /// The database may need to be restored from a backup.  If the corruption
    /// is limited to a single collection, try restoring from a snapshot.
    #[error("Database corruption detected: {0}")]
    Corruption(String),

    /// An error occurred within an index operation (HNSW, IVF, etc.).
    ///
    /// This is a catch-all for internal index failures.  Check the message
    /// string for details and consider rebuilding the index if it persists.
    #[error("Index error: {0}")]
    Index(String),

    /// The provided configuration is invalid.
    ///
    /// Returned when `CollectionConfig`, `HnswConfig`, or other config structs
    /// contain out-of-range values.  Check the field values against the
    /// documented constraints (e.g., dimensions > 0, M ≥ 4).
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// A capacity limit has been exceeded (e.g., max vectors per collection).
    ///
    /// Consider splitting data across multiple collections, enabling
    /// quantization to reduce memory, or increasing the configured limit.
    #[error("Capacity exceeded: {0}")]
    CapacityExceeded(String),

    /// The provided vector data is invalid (e.g., contains NaN or Inf values).
    ///
    /// Returned during insert or search when vector validation fails.  Ensure
    /// your embedding pipeline produces finite floating-point values.
    #[error("Invalid vector: {0}")]
    InvalidVector(String),

    /// The input to an operation is invalid.
    ///
    /// A general validation error for non-vector inputs such as malformed
    /// filter expressions or snapshot names.
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// A resource quota has been exceeded (e.g., storage or request limits).
    ///
    /// Reduce usage or increase the quota in your configuration.
    #[error("Quota exceeded: {0}")]
    QuotaExceeded(String),

    /// An error occurred during a backup or restore operation.
    ///
    /// Check that the backup path is accessible and the backup file is not
    /// corrupt.  See [`BackupManager`](crate::BackupManager) for retry options.
    #[error("Backup error: {0}")]
    BackupError(String),

    /// A generic not-found error for resources other than collections or vectors.
    ///
    /// Used for snapshots, namespaces, and other named resources.
    #[error("Not found: {0}")]
    NotFound(String),

    /// A conflicting operation was detected (e.g., concurrent modification).
    ///
    /// Retry the operation.  If the conflict persists, check for concurrent
    /// writers that may need coordination via the `CollectionRef` API.
    #[error("Conflict: {0}")]
    Conflict(String),

    /// An error occurred during encryption or decryption.
    ///
    /// Verify that the correct encryption key is being used and that the
    /// `encryption` feature is enabled.  Returned by enterprise encryption APIs.
    #[error("Encryption error: {0}")]
    EncryptionError(String),

    /// An error occurred in the Raft consensus protocol (enterprise distributed mode).
    ///
    /// Check network connectivity between cluster nodes and inspect Raft logs
    /// for leader election or log replication failures.
    #[error("Consensus error: {0}")]
    ConsensusError(String),

    /// Failed to acquire an internal lock.
    ///
    /// This is typically transient and caused by high contention.  Retry the
    /// operation.  If it persists, check for deadlocks or long-running
    /// operations holding the lock.
    #[error("Lock error: failed to acquire lock")]
    LockError,

    /// The operation exceeded the configured timeout duration.
    ///
    /// The inner [`Duration`](std::time::Duration) indicates how long the
    /// operation waited.  Increase the timeout or optimize the operation
    /// (e.g., reduce `ef_search` or batch size).
    #[error("Operation timed out after {0:?}")]
    Timeout(std::time::Duration),

    /// Timed out while waiting to acquire a lock.
    ///
    /// Indicates heavy contention on a collection.  Consider increasing the
    /// lock timeout, reducing write concurrency, or sharding data across
    /// multiple collections.
    #[error("Lock acquisition timed out after {0:?}")]
    LockTimeout(std::time::Duration),

    /// The requested operation is not valid in the current context.
    ///
    /// For example, calling `compact()` while a compaction is already running,
    /// or attempting a write on a read-only snapshot.
    #[error("Invalid operation: {0}")]
    InvalidOperation(String),

    /// The system is in an invalid state for the requested operation.
    ///
    /// This usually indicates a programming error or an unexpected internal
    /// condition.  If reproducible, please file a bug report.
    #[error("Invalid state: {0}")]
    InvalidState(String),

    /// The request lacks valid authentication or authorization credentials.
    ///
    /// Returned by the REST API when an API key or RBAC token is missing or
    /// invalid.  Check the `Authorization` header or RBAC configuration.
    #[error("Unauthorized: {0}")]
    Unauthorized(String),

    /// An argument passed to a function or API is invalid.
    ///
    /// Similar to [`InvalidInput`](Self::InvalidInput) but specific to
    /// function parameters (e.g., negative `k` value, empty ID string).
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),
}

impl Recoverable for NeedleError {
    fn error_code(&self) -> ErrorCode {
        match self {
            NeedleError::Io(source) => match source.kind() {
                std::io::ErrorKind::NotFound => ErrorCode::IoRead,
                std::io::ErrorKind::PermissionDenied => ErrorCode::IoPermission,
                std::io::ErrorKind::WriteZero => ErrorCode::IoDiskFull,
                _ => ErrorCode::IoWrite,
            },
            NeedleError::Serialization(_) => ErrorCode::SerializationFailed,
            NeedleError::DimensionMismatch { .. } => ErrorCode::DimensionMismatch,
            NeedleError::CollectionNotFound(_) => ErrorCode::CollectionNotFound,
            NeedleError::CollectionAlreadyExists(_) => ErrorCode::CollectionAlreadyExists,
            NeedleError::AliasNotFound(_) => ErrorCode::AliasNotFound,
            NeedleError::AliasAlreadyExists(_) => ErrorCode::AliasAlreadyExists,
            NeedleError::CollectionHasAliases(_) => ErrorCode::AliasTargetHasAliases,
            NeedleError::VectorNotFound(_) => ErrorCode::VectorNotFound,
            NeedleError::VectorAlreadyExists(_) => ErrorCode::VectorAlreadyExists,
            NeedleError::DuplicateId(_) => ErrorCode::VectorAlreadyExists,
            NeedleError::OperationInProgress(_) => ErrorCode::Conflict,
            NeedleError::InvalidDatabase(_) => ErrorCode::InvalidDatabase,
            NeedleError::Corruption(_) => ErrorCode::DatabaseCorrupted,
            NeedleError::Index(_) => ErrorCode::IndexError,
            NeedleError::InvalidConfig(_) => ErrorCode::InvalidConfig,
            NeedleError::CapacityExceeded(_) => ErrorCode::CapacityExceeded,
            NeedleError::InvalidVector(_) => ErrorCode::InvalidVector,
            NeedleError::InvalidInput(_) => ErrorCode::InvalidConfig,
            NeedleError::QuotaExceeded(_) => ErrorCode::QuotaExceeded,
            NeedleError::BackupError(_) => ErrorCode::BackupFailed,
            NeedleError::NotFound(_) => ErrorCode::NotFound,
            NeedleError::Conflict(_) => ErrorCode::Conflict,
            NeedleError::EncryptionError(_) => ErrorCode::EncryptionError,
            NeedleError::ConsensusError(_) => ErrorCode::ConsensusError,
            NeedleError::LockError => ErrorCode::DatabaseLocked,
            NeedleError::Timeout(_) => ErrorCode::Timeout,
            NeedleError::LockTimeout(_) => ErrorCode::LockTimeout,
            NeedleError::InvalidOperation(_) => ErrorCode::InvalidOperation,
            NeedleError::InvalidState(_) => ErrorCode::InvalidState,
            NeedleError::Unauthorized(_) => ErrorCode::Unauthorized,
            NeedleError::InvalidArgument(_) => ErrorCode::InvalidConfig,
        }
    }

    #[allow(clippy::too_many_lines)]
    fn recovery_hints(&self) -> Vec<RecoveryHint> {
        match self {
            NeedleError::Io(source) => {
                match source.kind() {
                    std::io::ErrorKind::NotFound => vec![
                        RecoveryHint::new("Verify the file or directory exists")
                            .with_details("Check file path spelling and ensure parent directories exist"),
                        RecoveryHint::new("Create the database file using Database::open() or Database::in_memory()"),
                    ],
                    std::io::ErrorKind::PermissionDenied => vec![
                        RecoveryHint::new("Check file permissions")
                            .with_details("Ensure the process has read/write access to the file and directory"),
                        RecoveryHint::new("Try running with elevated permissions or change file ownership"),
                    ],
                    std::io::ErrorKind::WriteZero | std::io::ErrorKind::StorageFull => vec![
                        RecoveryHint::new("Free up disk space")
                            .with_details("The disk is full; delete unnecessary files or expand storage"),
                        RecoveryHint::new("Run database compact() to reclaim space from deleted vectors"),
                    ],
                    _ => vec![
                        RecoveryHint::new("Check disk health and file system integrity"),
                        RecoveryHint::new("Ensure no other process has exclusive access to the file"),
                    ],
                }
            }

            NeedleError::Serialization(_) => vec![
                RecoveryHint::new("Check data format compatibility")
                    .with_details("Ensure data matches expected JSON schema"),
                RecoveryHint::new("Verify metadata values are valid JSON types"),
                RecoveryHint::new("If upgrading, check for breaking changes in data format"),
            ],

            NeedleError::DimensionMismatch { expected, got } => vec![
                RecoveryHint::new(format!("Resize vector to {} dimensions (currently {})", expected, got))
                    .with_details("All vectors in a collection must have the same dimensionality"),
                RecoveryHint::new("Verify your embedding model produces the expected dimensions"),
                RecoveryHint::new("If using a different model, create a new collection with the correct dimensions"),
            ],

            NeedleError::CollectionNotFound(name) => vec![
                RecoveryHint::new(format!("Create collection '{}' first using db.create_collection()", name)),
                RecoveryHint::new("Check collection name spelling (case-sensitive)"),
                RecoveryHint::new("Use db.list_collections() to see available collections"),
            ],

            NeedleError::CollectionAlreadyExists(name) => vec![
                RecoveryHint::new(format!("Use db.collection(\"{}\") to access the existing collection", name)),
                RecoveryHint::new("Delete the existing collection first if you need to recreate it"),
                RecoveryHint::new("Choose a different collection name"),
            ],

            NeedleError::AliasNotFound(name) => vec![
                RecoveryHint::new(format!("Create alias '{}' first using db.create_alias()", name)),
                RecoveryHint::new("Check alias name spelling (case-sensitive)"),
                RecoveryHint::new("Use db.list_aliases() to see available aliases"),
            ],

            NeedleError::AliasAlreadyExists(name) => vec![
                RecoveryHint::new(format!("Alias '{}' already exists", name)),
                RecoveryHint::new("Delete the existing alias first if you need to recreate it"),
                RecoveryHint::new("Choose a different alias name"),
            ],

            NeedleError::CollectionHasAliases(name) => vec![
                RecoveryHint::new(format!("Collection '{}' has aliases pointing to it", name)),
                RecoveryHint::new("Delete all aliases first using db.delete_alias()"),
                RecoveryHint::new("Use db.aliases_for_collection() to see which aliases reference this collection"),
            ],

            NeedleError::VectorNotFound(id) => vec![
                RecoveryHint::new(format!("Insert vector '{}' before accessing it", id)),
                RecoveryHint::new("Verify the vector ID is correct"),
                RecoveryHint::new("Check if the vector was previously deleted"),
            ],

            NeedleError::VectorAlreadyExists(id) => vec![
                RecoveryHint::new(format!("Use a unique ID instead of '{}'", id)),
                RecoveryHint::new(format!("Delete the existing vector first: collection.delete(\"{}\")", id)),
                RecoveryHint::new("Use upsert semantics if you want to replace existing vectors"),
            ],

            NeedleError::InvalidDatabase(reason) => vec![
                RecoveryHint::new("The database file appears corrupted or incompatible")
                    .with_details(reason.clone()),
                RecoveryHint::new("Restore from a recent backup if available"),
                RecoveryHint::new("Create a new database if no backup exists"),
                RecoveryHint::new("Check if the file was created by a different version of Needle"),
            ],

            NeedleError::Corruption(reason) => vec![
                RecoveryHint::new("Restore from the most recent backup")
                    .with_details(format!("Corruption detected: {}", reason)),
                RecoveryHint::new("Run database repair utility if available"),
                RecoveryHint::new("Consider enabling write-ahead logging (WAL) for crash recovery"),
            ],

            NeedleError::Index(reason) => vec![
                RecoveryHint::new("Rebuild the index")
                    .with_details(reason.clone()),
                RecoveryHint::new("Check HNSW parameters (M, ef_construction) are within valid ranges"),
                RecoveryHint::new("Ensure sufficient memory for index construction"),
            ],

            NeedleError::InvalidConfig(reason) => vec![
                RecoveryHint::new(format!("Fix configuration: {}", reason)),
                RecoveryHint::new("Use default configuration as a starting point")
                    .with_doc("See CLAUDE.md for configuration guidelines"),
            ],

            NeedleError::CapacityExceeded(reason) => vec![
                RecoveryHint::new(reason.clone()),
                RecoveryHint::new("Delete unused vectors to free capacity"),
                RecoveryHint::new("Use quantization to reduce memory per vector"),
                RecoveryHint::new("Consider sharding data across multiple collections"),
            ],

            NeedleError::InvalidVector(reason) => vec![
                RecoveryHint::new(format!("Fix vector data: {}", reason)),
                RecoveryHint::new("Ensure vector contains no NaN or Infinity values"),
                RecoveryHint::new("Verify vector dimensions match collection configuration"),
                RecoveryHint::new("Normalize vectors if using cosine distance"),
            ],

            NeedleError::InvalidInput(reason) => vec![
                RecoveryHint::new(format!("Check input value: {}", reason)),
                RecoveryHint::new("Verify input types match expected signatures"),
            ],

            NeedleError::QuotaExceeded(reason) => vec![
                RecoveryHint::new(reason.clone()),
                RecoveryHint::new("Request a quota increase if needed"),
                RecoveryHint::new("Delete unused resources to stay within limits"),
            ],

            NeedleError::BackupError(reason) => vec![
                RecoveryHint::new(format!("Backup operation failed: {}", reason)),
                RecoveryHint::new("Verify backup directory has sufficient space and permissions"),
                RecoveryHint::new("Check backup integrity with verify_backup()"),
                RecoveryHint::new("Retry the backup operation"),
            ],

            NeedleError::NotFound(resource) => vec![
                RecoveryHint::new(format!("Resource '{}' not found", resource)),
                RecoveryHint::new("Verify the resource identifier is correct"),
                RecoveryHint::new("Create the resource before accessing it"),
            ],

            NeedleError::Conflict(reason) => vec![
                RecoveryHint::new(format!("Resolve conflict: {}", reason)),
                RecoveryHint::new("Retry the operation after a short delay"),
                RecoveryHint::new("Use optimistic locking for concurrent operations"),
            ],

            NeedleError::EncryptionError(reason) => vec![
                RecoveryHint::new(format!("Encryption operation failed: {}", reason)),
                RecoveryHint::new("Verify the encryption key is correct"),
                RecoveryHint::new("Ensure the key was generated with KeyManager::new()"),
                RecoveryHint::new("Check that encrypted data hasn't been corrupted"),
            ],

            NeedleError::ConsensusError(reason) => vec![
                RecoveryHint::new(format!("Consensus failed: {}", reason)),
                RecoveryHint::new("Check network connectivity between cluster nodes"),
                RecoveryHint::new("Verify quorum requirements are met"),
                RecoveryHint::new("Wait for cluster to stabilize and retry"),
            ],

            NeedleError::LockError => vec![
                RecoveryHint::new("Failed to acquire lock"),
                RecoveryHint::new("Another operation may be holding the lock"),
                RecoveryHint::new("Wait for the other operation to complete and retry"),
                RecoveryHint::new("Check for deadlock conditions in concurrent code"),
            ],

            NeedleError::Timeout(duration) => vec![
                RecoveryHint::new(format!("Operation timed out after {:?}", duration)),
                RecoveryHint::new("Increase the timeout value for long-running operations"),
                RecoveryHint::new("Check for performance bottlenecks or resource contention"),
                RecoveryHint::new("Consider breaking large operations into smaller batches"),
            ],

            NeedleError::LockTimeout(duration) => vec![
                RecoveryHint::new(format!("Lock acquisition timed out after {:?}", duration)),
                RecoveryHint::new("Another process may be holding the lock"),
                RecoveryHint::new("Retry after a short delay"),
                RecoveryHint::new("Consider using shorter lock hold times"),
            ],

            NeedleError::InvalidOperation(msg) => vec![
                RecoveryHint::new(format!("Invalid operation: {}", msg)),
                RecoveryHint::new("Check the operation preconditions"),
                RecoveryHint::new("Ensure the current state allows this operation"),
            ],

            NeedleError::InvalidState(msg) => vec![
                RecoveryHint::new(format!("Invalid state: {}", msg)),
                RecoveryHint::new("The system is in an unexpected state"),
                RecoveryHint::new("Try reinitializing the component"),
            ],

            NeedleError::Unauthorized(msg) => vec![
                RecoveryHint::new(format!("Unauthorized: {}", msg)),
                RecoveryHint::new("Check your credentials and permissions"),
                RecoveryHint::new("Ensure you have access to the requested resource"),
            ],

            NeedleError::InvalidArgument(msg) => vec![
                RecoveryHint::new(format!("Invalid argument: {}", msg)),
                RecoveryHint::new("Check the argument values and try again"),
            ],

            NeedleError::DuplicateId(id) => vec![
                RecoveryHint::new(format!("ID '{}' already exists", id)),
                RecoveryHint::new("Use a different ID or update the existing vector"),
            ],

            NeedleError::OperationInProgress(msg) => vec![
                RecoveryHint::new(format!("Operation in progress: {}", msg)),
                RecoveryHint::new("Wait for the current operation to complete"),
                RecoveryHint::new("Consider using async APIs for long-running operations"),
            ],
        }
    }

    fn is_retryable(&self) -> bool {
        matches!(
            self,
            NeedleError::Timeout(_)
                | NeedleError::LockTimeout(_)
                | NeedleError::LockError
                | NeedleError::Conflict(_)
                | NeedleError::ConsensusError(_)
        )
    }

    fn suggested_retry_delay_ms(&self) -> Option<u64> {
        match self {
            NeedleError::Timeout(_) => Some(1000),
            NeedleError::LockTimeout(_) => Some(100),
            NeedleError::LockError => Some(50),
            NeedleError::Conflict(_) => Some(100),
            NeedleError::ConsensusError(_) => Some(500),
            _ => None,
        }
    }
}

impl NeedleError {
    /// Get a formatted error message with recovery hints
    pub fn format_with_hints(&self) -> String {
        let hints = self.recovery_hints();
        let mut output = format!("Error [{}]: {}", self.error_code().code(), self);

        if !hints.is_empty() {
            output.push_str("\n\nRecovery suggestions:");
            for (i, hint) in hints.iter().enumerate() {
                output.push_str(&format!("\n  {}. {}", i + 1, hint));
            }
        }

        if self.is_retryable() {
            if let Some(delay) = self.suggested_retry_delay_ms() {
                output.push_str(&format!(
                    "\n\nThis error is retryable. Suggested delay: {}ms",
                    delay
                ));
            } else {
                output.push_str("\n\nThis error is retryable.");
            }
        }

        output
    }

    /// Returns a concise, actionable help string for the most common errors.
    ///
    /// Designed for display in CLI output and HTTP error responses. Returns
    /// the single most useful suggestion for fixing the error.
    pub fn help(&self) -> String {
        match self {
            NeedleError::DimensionMismatch { expected, got } => format!(
                "The collection expects {}-dimensional vectors, but got {}. \
                 Ensure your embedding model output matches the collection dimensions. \
                 Check with: collection.dimensions()",
                expected, got
            ),
            NeedleError::CollectionNotFound(name) => format!(
                "Collection '{}' does not exist. Create it with \
                 db.create_collection(\"{}\", dims) or POST /collections. \
                 Use db.list_collections() or GET /collections to see available collections.",
                name, name
            ),
            NeedleError::CollectionAlreadyExists(name) => format!(
                "Collection '{}' already exists. Use db.collection(\"{}\") to access it, \
                 or delete it first with db.delete_collection(\"{}\").",
                name, name, name
            ),
            NeedleError::VectorNotFound(id) => format!(
                "Vector '{}' does not exist. It may have been deleted or never inserted. \
                 Use collection.contains(\"{}\") to check before accessing.",
                id, id
            ),
            NeedleError::InvalidVector(reason) => format!(
                "Vector data is invalid: {}. \
                 Ensure the vector contains only finite f32 values (no NaN or Infinity).",
                reason
            ),
            NeedleError::InvalidInput(reason) => format!(
                "Invalid input: {}. Check the JSON format and field types match the API spec.",
                reason
            ),
            NeedleError::Serialization(_) => String::from(
                "Failed to parse JSON input. Verify the request body is valid JSON \
                 and matches the expected schema.",
            ),
            _ => {
                let hints = self.recovery_hints();
                hints.first().map(|h| h.to_string()).unwrap_or_default()
            }
        }
    }
}

/// Result type alias for Needle operations
pub type Result<T> = std::result::Result<T, NeedleError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_codes() {
        let error = NeedleError::CollectionNotFound("test".to_string());
        assert_eq!(error.error_code(), ErrorCode::CollectionNotFound);
        assert_eq!(error.error_code().code(), 3001);
        assert_eq!(error.error_code().category(), "Collection");
    }

    #[test]
    fn test_recovery_hints() {
        let error = NeedleError::CollectionNotFound("missing".to_string());
        let hints = error.recovery_hints();
        assert!(!hints.is_empty());
        assert!(hints.iter().any(|h| h.summary.contains("missing")));
    }

    #[test]
    fn test_retryable_errors() {
        let timeout = NeedleError::Timeout(std::time::Duration::from_secs(5));
        assert!(timeout.is_retryable());
        assert!(timeout.suggested_retry_delay_ms().is_some());

        let not_found = NeedleError::CollectionNotFound("test".to_string());
        assert!(!not_found.is_retryable());
        assert!(not_found.suggested_retry_delay_ms().is_none());
    }

    #[test]
    fn test_dimension_mismatch_hints() {
        let error = NeedleError::DimensionMismatch {
            expected: 128,
            got: 256,
        };
        let hints = error.recovery_hints();
        assert!(hints
            .iter()
            .any(|h| h.summary.contains("128") && h.summary.contains("256")));
    }

    #[test]
    fn test_format_with_hints() {
        let error = NeedleError::CollectionNotFound("test_collection".to_string());
        let formatted = error.format_with_hints();
        assert!(formatted.contains("Error [3001]"));
        assert!(formatted.contains("test_collection"));
        assert!(formatted.contains("Recovery suggestions"));
    }

    #[test]
    fn test_io_error_hints_by_kind() {
        // Test NotFound
        let error = NeedleError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "file not found",
        ));
        let hints = error.recovery_hints();
        assert!(hints.iter().any(|h| h.summary.contains("exist")));
        assert_eq!(error.error_code(), ErrorCode::IoRead);

        // Test PermissionDenied
        let error = NeedleError::Io(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "access denied",
        ));
        let hints = error.recovery_hints();
        assert!(hints.iter().any(|h| h.summary.contains("permission")));
        assert_eq!(error.error_code(), ErrorCode::IoPermission);
    }

    #[test]
    fn test_help_dimension_mismatch() {
        let error = NeedleError::DimensionMismatch {
            expected: 384,
            got: 128,
        };
        let help = error.help();
        assert!(help.contains("384"));
        assert!(help.contains("128"));
        assert!(help.contains("embedding model"));
    }

    #[test]
    fn test_help_collection_not_found() {
        let error = NeedleError::CollectionNotFound("docs".to_string());
        let help = error.help();
        assert!(help.contains("docs"));
        assert!(help.contains("create_collection"));
        assert!(help.contains("list_collections"));
    }

    #[test]
    fn test_help_collection_already_exists() {
        let error = NeedleError::CollectionAlreadyExists("mydata".to_string());
        let help = error.help();
        assert!(help.contains("mydata"));
        assert!(help.contains("already exists"));
    }

    #[test]
    fn test_help_vector_not_found() {
        let error = NeedleError::VectorNotFound("doc42".to_string());
        let help = error.help();
        assert!(help.contains("doc42"));
    }

    #[test]
    fn test_help_fallback_to_recovery_hints() {
        let error = NeedleError::LockError;
        let help = error.help();
        assert!(!help.is_empty());
    }

    #[test]
    fn test_all_error_variants_have_error_codes() {
        let variants: Vec<NeedleError> = vec![
            NeedleError::Io(std::io::Error::new(std::io::ErrorKind::Other, "test")),
            NeedleError::Serialization(
                serde_json::from_str::<serde_json::Value>("invalid").unwrap_err(),
            ),
            NeedleError::DimensionMismatch {
                expected: 128,
                got: 256,
            },
            NeedleError::CollectionNotFound("c".into()),
            NeedleError::CollectionAlreadyExists("c".into()),
            NeedleError::AliasNotFound("a".into()),
            NeedleError::AliasAlreadyExists("a".into()),
            NeedleError::CollectionHasAliases("c".into()),
            NeedleError::VectorNotFound("v".into()),
            NeedleError::VectorAlreadyExists("v".into()),
            NeedleError::DuplicateId("d".into()),
            NeedleError::OperationInProgress("op".into()),
            NeedleError::InvalidDatabase("bad".into()),
            NeedleError::Corruption("corrupt".into()),
            NeedleError::Index("idx".into()),
            NeedleError::InvalidConfig("cfg".into()),
            NeedleError::CapacityExceeded("cap".into()),
            NeedleError::InvalidVector("vec".into()),
            NeedleError::InvalidInput("inp".into()),
            NeedleError::QuotaExceeded("quota".into()),
            NeedleError::BackupError("bak".into()),
            NeedleError::NotFound("nf".into()),
            NeedleError::Conflict("conflict".into()),
            NeedleError::EncryptionError("enc".into()),
            NeedleError::ConsensusError("raft".into()),
            NeedleError::LockError,
            NeedleError::Timeout(std::time::Duration::from_secs(5)),
            NeedleError::LockTimeout(std::time::Duration::from_millis(100)),
            NeedleError::InvalidOperation("op".into()),
            NeedleError::InvalidState("state".into()),
            NeedleError::Unauthorized("unauth".into()),
            NeedleError::InvalidArgument("arg".into()),
        ];

        for error in &variants {
            let code = error.error_code();
            assert!(code.code() > 0, "Error {:?} has zero error code", error);
            assert!(
                !code.category().is_empty(),
                "Error {:?} has empty category",
                error
            );
        }
    }

    #[test]
    fn test_all_error_variants_have_recovery_hints() {
        let variants: Vec<NeedleError> = vec![
            NeedleError::Serialization(
                serde_json::from_str::<serde_json::Value>("bad").unwrap_err(),
            ),
            NeedleError::AliasNotFound("a".into()),
            NeedleError::AliasAlreadyExists("a".into()),
            NeedleError::CollectionHasAliases("c".into()),
            NeedleError::VectorAlreadyExists("v".into()),
            NeedleError::DuplicateId("d".into()),
            NeedleError::OperationInProgress("op".into()),
            NeedleError::InvalidDatabase("bad".into()),
            NeedleError::Corruption("corrupt".into()),
            NeedleError::Index("idx".into()),
            NeedleError::InvalidConfig("cfg".into()),
            NeedleError::CapacityExceeded("cap".into()),
            NeedleError::InvalidVector("vec".into()),
            NeedleError::InvalidInput("inp".into()),
            NeedleError::QuotaExceeded("quota".into()),
            NeedleError::BackupError("bak".into()),
            NeedleError::NotFound("nf".into()),
            NeedleError::Conflict("conflict".into()),
            NeedleError::EncryptionError("enc".into()),
            NeedleError::ConsensusError("raft".into()),
            NeedleError::LockTimeout(std::time::Duration::from_millis(100)),
            NeedleError::InvalidOperation("op".into()),
            NeedleError::InvalidState("state".into()),
            NeedleError::Unauthorized("unauth".into()),
            NeedleError::InvalidArgument("arg".into()),
        ];

        for error in &variants {
            let hints = error.recovery_hints();
            assert!(
                !hints.is_empty(),
                "Error {:?} has no recovery hints",
                error
            );
        }
    }

    #[test]
    fn test_all_error_variants_display() {
        assert_eq!(
            NeedleError::AliasNotFound("test".into()).to_string(),
            "Alias 'test' not found"
        );
        assert_eq!(
            NeedleError::AliasAlreadyExists("test".into()).to_string(),
            "Alias 'test' already exists"
        );
        assert_eq!(
            NeedleError::CollectionHasAliases("col".into()).to_string(),
            "Cannot drop collection 'col': aliases still reference it"
        );
        assert_eq!(
            NeedleError::VectorAlreadyExists("v1".into()).to_string(),
            "Vector 'v1' already exists"
        );
        assert_eq!(
            NeedleError::DuplicateId("dup".into()).to_string(),
            "Duplicate ID: 'dup'"
        );
        assert_eq!(
            NeedleError::InvalidDatabase("bad".into()).to_string(),
            "Invalid database file: bad"
        );
        assert_eq!(
            NeedleError::Corruption("crc".into()).to_string(),
            "Database corruption detected: crc"
        );
        assert_eq!(
            NeedleError::InvalidConfig("m=0".into()).to_string(),
            "Invalid configuration: m=0"
        );
        assert_eq!(
            NeedleError::CapacityExceeded("max".into()).to_string(),
            "Capacity exceeded: max"
        );
        assert_eq!(
            NeedleError::QuotaExceeded("limit".into()).to_string(),
            "Quota exceeded: limit"
        );
        assert_eq!(
            NeedleError::BackupError("fail".into()).to_string(),
            "Backup error: fail"
        );
        assert_eq!(
            NeedleError::EncryptionError("key".into()).to_string(),
            "Encryption error: key"
        );
        assert_eq!(
            NeedleError::ConsensusError("raft".into()).to_string(),
            "Consensus error: raft"
        );
        assert_eq!(
            NeedleError::Unauthorized("no token".into()).to_string(),
            "Unauthorized: no token"
        );
        assert_eq!(
            NeedleError::InvalidArgument("k<0".into()).to_string(),
            "Invalid argument: k<0"
        );
    }

    #[test]
    fn test_error_code_mapping_specific() {
        assert_eq!(
            NeedleError::AliasNotFound("a".into()).error_code(),
            ErrorCode::AliasNotFound
        );
        assert_eq!(
            NeedleError::AliasAlreadyExists("a".into()).error_code(),
            ErrorCode::AliasAlreadyExists
        );
        assert_eq!(
            NeedleError::CollectionHasAliases("c".into()).error_code(),
            ErrorCode::AliasTargetHasAliases
        );
        assert_eq!(
            NeedleError::VectorAlreadyExists("v".into()).error_code(),
            ErrorCode::VectorAlreadyExists
        );
        assert_eq!(
            NeedleError::DuplicateId("d".into()).error_code(),
            ErrorCode::VectorAlreadyExists
        );
        assert_eq!(
            NeedleError::OperationInProgress("op".into()).error_code(),
            ErrorCode::Conflict
        );
        assert_eq!(
            NeedleError::InvalidDatabase("bad".into()).error_code(),
            ErrorCode::InvalidDatabase
        );
        assert_eq!(
            NeedleError::Corruption("c".into()).error_code(),
            ErrorCode::DatabaseCorrupted
        );
        assert_eq!(
            NeedleError::Index("i".into()).error_code(),
            ErrorCode::IndexError
        );
        assert_eq!(
            NeedleError::InvalidConfig("c".into()).error_code(),
            ErrorCode::InvalidConfig
        );
        assert_eq!(
            NeedleError::CapacityExceeded("c".into()).error_code(),
            ErrorCode::CapacityExceeded
        );
        assert_eq!(
            NeedleError::InvalidVector("v".into()).error_code(),
            ErrorCode::InvalidVector
        );
        assert_eq!(
            NeedleError::QuotaExceeded("q".into()).error_code(),
            ErrorCode::QuotaExceeded
        );
        assert_eq!(
            NeedleError::BackupError("b".into()).error_code(),
            ErrorCode::BackupFailed
        );
        assert_eq!(
            NeedleError::NotFound("n".into()).error_code(),
            ErrorCode::NotFound
        );
        assert_eq!(
            NeedleError::Conflict("c".into()).error_code(),
            ErrorCode::Conflict
        );
        assert_eq!(
            NeedleError::EncryptionError("e".into()).error_code(),
            ErrorCode::EncryptionError
        );
        assert_eq!(
            NeedleError::ConsensusError("r".into()).error_code(),
            ErrorCode::ConsensusError
        );
        assert_eq!(
            NeedleError::LockError.error_code(),
            ErrorCode::DatabaseLocked
        );
        assert_eq!(
            NeedleError::LockTimeout(std::time::Duration::from_millis(50)).error_code(),
            ErrorCode::LockTimeout
        );
        assert_eq!(
            NeedleError::InvalidOperation("o".into()).error_code(),
            ErrorCode::InvalidOperation
        );
        assert_eq!(
            NeedleError::InvalidState("s".into()).error_code(),
            ErrorCode::InvalidState
        );
        assert_eq!(
            NeedleError::Unauthorized("u".into()).error_code(),
            ErrorCode::Unauthorized
        );
        assert_eq!(
            NeedleError::InvalidArgument("a".into()).error_code(),
            ErrorCode::InvalidConfig
        );
    }

    #[test]
    fn test_retryable_variants_comprehensive() {
        // All retryable errors
        assert!(NeedleError::Timeout(std::time::Duration::from_secs(1)).is_retryable());
        assert!(NeedleError::LockTimeout(std::time::Duration::from_millis(50)).is_retryable());
        assert!(NeedleError::LockError.is_retryable());
        assert!(NeedleError::Conflict("c".into()).is_retryable());
        assert!(NeedleError::ConsensusError("r".into()).is_retryable());

        // Non-retryable errors
        assert!(!NeedleError::InvalidConfig("c".into()).is_retryable());
        assert!(!NeedleError::Corruption("c".into()).is_retryable());
        assert!(!NeedleError::Unauthorized("u".into()).is_retryable());
        assert!(!NeedleError::InvalidArgument("a".into()).is_retryable());
    }

    #[test]
    fn test_suggested_retry_delays() {
        assert_eq!(
            NeedleError::Timeout(std::time::Duration::from_secs(1)).suggested_retry_delay_ms(),
            Some(1000)
        );
        assert_eq!(
            NeedleError::LockTimeout(std::time::Duration::from_millis(50))
                .suggested_retry_delay_ms(),
            Some(100)
        );
        assert_eq!(
            NeedleError::LockError.suggested_retry_delay_ms(),
            Some(50)
        );
        assert_eq!(
            NeedleError::Conflict("c".into()).suggested_retry_delay_ms(),
            Some(100)
        );
        assert_eq!(
            NeedleError::ConsensusError("r".into()).suggested_retry_delay_ms(),
            Some(500)
        );
        assert_eq!(
            NeedleError::CollectionNotFound("c".into()).suggested_retry_delay_ms(),
            None
        );
    }

    #[test]
    fn test_from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
        let needle_err: NeedleError = io_err.into();
        assert!(matches!(needle_err, NeedleError::Io(_)));
        assert_eq!(needle_err.error_code(), ErrorCode::IoRead);
    }

    #[test]
    fn test_from_serde_error() {
        let json_err = serde_json::from_str::<serde_json::Value>("not json").unwrap_err();
        let needle_err: NeedleError = json_err.into();
        assert!(matches!(needle_err, NeedleError::Serialization(_)));
        assert_eq!(needle_err.error_code(), ErrorCode::SerializationFailed);
    }

    #[test]
    fn test_error_code_categories() {
        assert_eq!(ErrorCode::IoRead.category(), "I/O");
        assert_eq!(ErrorCode::IoPermission.category(), "I/O");
        assert_eq!(ErrorCode::SerializationFailed.category(), "Serialization");
        assert_eq!(ErrorCode::CollectionNotFound.category(), "Collection");
        assert_eq!(ErrorCode::AliasNotFound.category(), "Collection");
        assert_eq!(ErrorCode::VectorNotFound.category(), "Vector");
        assert_eq!(ErrorCode::DimensionMismatch.category(), "Vector");
        assert_eq!(ErrorCode::InvalidDatabase.category(), "Database");
        assert_eq!(ErrorCode::IndexError.category(), "Index");
        assert_eq!(ErrorCode::InvalidConfig.category(), "Configuration");
        assert_eq!(ErrorCode::CapacityExceeded.category(), "Resource");
        assert_eq!(ErrorCode::Timeout.category(), "Operational");
        assert_eq!(ErrorCode::EncryptionError.category(), "Security");
        assert_eq!(ErrorCode::ConsensusError.category(), "Distributed");
        assert_eq!(ErrorCode::BackupFailed.category(), "Backup");
        assert_eq!(ErrorCode::InvalidOperation.category(), "State");
        assert_eq!(ErrorCode::Unauthorized.category(), "State");
    }

    #[test]
    fn test_help_for_untested_variants() {
        // Alias errors
        let err = NeedleError::AliasNotFound("my_alias".into());
        let help = err.help();
        assert!(!help.is_empty());

        // Invalid input
        let err = NeedleError::InvalidInput("bad filter".into());
        let help = err.help();
        assert!(help.contains("bad filter"));

        // Serialization
        let err = NeedleError::Serialization(
            serde_json::from_str::<serde_json::Value>("x").unwrap_err(),
        );
        let help = err.help();
        assert!(help.contains("JSON"));
    }

    // ── Additional comprehensive tests ──────────────────────────────────

    #[test]
    fn test_display_remaining_variants() {
        assert_eq!(
            NeedleError::LockError.to_string(),
            "Lock error: failed to acquire lock"
        );
        let timeout = NeedleError::Timeout(std::time::Duration::from_secs(5));
        assert!(timeout.to_string().contains("5s"));

        let lock_timeout = NeedleError::LockTimeout(std::time::Duration::from_millis(200));
        assert!(lock_timeout.to_string().contains("200ms"));

        assert_eq!(
            NeedleError::OperationInProgress("compaction".into()).to_string(),
            "Operation in progress: compaction"
        );
        assert_eq!(
            NeedleError::NotFound("snapshot_1".into()).to_string(),
            "Not found: snapshot_1"
        );
        assert_eq!(
            NeedleError::Conflict("write conflict".into()).to_string(),
            "Conflict: write conflict"
        );
        assert_eq!(
            NeedleError::Index("rebuild needed".into()).to_string(),
            "Index error: rebuild needed"
        );
        assert_eq!(
            NeedleError::InvalidVector("NaN detected".into()).to_string(),
            "Invalid vector: NaN detected"
        );
        assert_eq!(
            NeedleError::InvalidInput("bad filter".into()).to_string(),
            "Invalid input: bad filter"
        );
        assert_eq!(
            NeedleError::InvalidOperation("read-only".into()).to_string(),
            "Invalid operation: read-only"
        );
        assert_eq!(
            NeedleError::InvalidState("uninitialized".into()).to_string(),
            "Invalid state: uninitialized"
        );
    }

    #[test]
    fn test_error_code_numeric_values() {
        assert_eq!(ErrorCode::IoRead.code(), 1001);
        assert_eq!(ErrorCode::IoWrite.code(), 1002);
        assert_eq!(ErrorCode::IoPermission.code(), 1003);
        assert_eq!(ErrorCode::IoDiskFull.code(), 1004);
        assert_eq!(ErrorCode::SerializationFailed.code(), 2001);
        assert_eq!(ErrorCode::DeserializationFailed.code(), 2002);
        assert_eq!(ErrorCode::InvalidFormat.code(), 2003);
        assert_eq!(ErrorCode::CollectionNotFound.code(), 3001);
        assert_eq!(ErrorCode::CollectionAlreadyExists.code(), 3002);
        assert_eq!(ErrorCode::CollectionCorrupted.code(), 3003);
        assert_eq!(ErrorCode::AliasNotFound.code(), 3004);
        assert_eq!(ErrorCode::AliasAlreadyExists.code(), 3005);
        assert_eq!(ErrorCode::AliasTargetHasAliases.code(), 3006);
        assert_eq!(ErrorCode::VectorNotFound.code(), 4001);
        assert_eq!(ErrorCode::VectorAlreadyExists.code(), 4002);
        assert_eq!(ErrorCode::DimensionMismatch.code(), 4003);
        assert_eq!(ErrorCode::InvalidVector.code(), 4004);
        assert_eq!(ErrorCode::InvalidDatabase.code(), 5001);
        assert_eq!(ErrorCode::DatabaseCorrupted.code(), 5002);
        assert_eq!(ErrorCode::DatabaseLocked.code(), 5003);
        assert_eq!(ErrorCode::IndexError.code(), 6001);
        assert_eq!(ErrorCode::IndexCorrupted.code(), 6002);
        assert_eq!(ErrorCode::IndexBuildFailed.code(), 6003);
        assert_eq!(ErrorCode::InvalidConfig.code(), 7001);
        assert_eq!(ErrorCode::MissingConfig.code(), 7002);
        assert_eq!(ErrorCode::CapacityExceeded.code(), 8001);
        assert_eq!(ErrorCode::QuotaExceeded.code(), 8002);
        assert_eq!(ErrorCode::MemoryExhausted.code(), 8003);
        assert_eq!(ErrorCode::Timeout.code(), 9001);
        assert_eq!(ErrorCode::LockTimeout.code(), 9002);
        assert_eq!(ErrorCode::Conflict.code(), 9003);
        assert_eq!(ErrorCode::NotFound.code(), 9004);
        assert_eq!(ErrorCode::EncryptionError.code(), 10001);
        assert_eq!(ErrorCode::DecryptionError.code(), 10002);
        assert_eq!(ErrorCode::AuthenticationFailed.code(), 10003);
        assert_eq!(ErrorCode::ConsensusError.code(), 11001);
        assert_eq!(ErrorCode::ReplicationError.code(), 11002);
        assert_eq!(ErrorCode::NetworkError.code(), 11003);
        assert_eq!(ErrorCode::BackupFailed.code(), 12001);
        assert_eq!(ErrorCode::RestoreFailed.code(), 12002);
        assert_eq!(ErrorCode::BackupCorrupted.code(), 12003);
        assert_eq!(ErrorCode::InvalidOperation.code(), 13001);
        assert_eq!(ErrorCode::InvalidState.code(), 13002);
        assert_eq!(ErrorCode::Unauthorized.code(), 13003);
    }

    #[test]
    fn test_recovery_hint_builder_and_display() {
        let hint = RecoveryHint::new("Check permissions")
            .with_details("Run chmod 644 on the file")
            .with_doc("https://docs.needle.dev/troubleshooting");

        assert_eq!(hint.summary, "Check permissions");
        assert_eq!(hint.details.as_deref(), Some("Run chmod 644 on the file"));
        assert_eq!(
            hint.doc_ref.as_deref(),
            Some("https://docs.needle.dev/troubleshooting")
        );

        let display = format!("{}", hint);
        assert!(display.contains("Check permissions"));
        assert!(display.contains("Details:"));
        assert!(display.contains("See:"));

        // Hint without details or doc
        let simple = RecoveryHint::new("Retry the operation");
        let simple_display = format!("{}", simple);
        assert_eq!(simple_display, "Retry the operation");
        assert!(!simple_display.contains("Details:"));
        assert!(!simple_display.contains("See:"));
    }

    #[test]
    fn test_io_error_kind_disk_full_mapping() {
        let err = NeedleError::Io(std::io::Error::new(
            std::io::ErrorKind::WriteZero,
            "disk full",
        ));
        assert_eq!(err.error_code(), ErrorCode::IoDiskFull);
        let hints = err.recovery_hints();
        assert!(hints.iter().any(|h| h.summary.contains("disk") || h.summary.contains("space")));

        // Generic IO error maps to IoWrite
        let err = NeedleError::Io(std::io::Error::new(
            std::io::ErrorKind::BrokenPipe,
            "pipe broken",
        ));
        assert_eq!(err.error_code(), ErrorCode::IoWrite);
    }

    #[test]
    fn test_format_with_hints_various_errors() {
        let err = NeedleError::DimensionMismatch {
            expected: 384,
            got: 768,
        };
        let formatted = err.format_with_hints();
        assert!(formatted.contains("Error [4003]"));
        assert!(formatted.contains("384"));
        assert!(formatted.contains("768"));

        let err = NeedleError::Unauthorized("missing token".into());
        let formatted = err.format_with_hints();
        assert!(formatted.contains("Error [13003]"));
        assert!(formatted.contains("missing token"));

        // LockError has no inner message but should still format
        let err = NeedleError::LockError;
        let formatted = err.format_with_hints();
        assert!(formatted.contains("Error [5003]"));
    }

    #[test]
    fn test_error_source_from_io() {
        use std::error::Error;
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
        let needle_err: NeedleError = io_err.into();
        // std::error::Error::source() should return the inner io::Error
        assert!(needle_err.source().is_some());
    }

    #[test]
    fn test_error_source_from_serde() {
        use std::error::Error;
        let json_err = serde_json::from_str::<serde_json::Value>("{bad}").unwrap_err();
        let needle_err: NeedleError = json_err.into();
        assert!(needle_err.source().is_some());

        // Non-From variants should have no source
        let err = NeedleError::CollectionNotFound("test".into());
        assert!(err.source().is_none());
    }

    #[test]
    fn test_all_error_code_categories_exhaustive() {
        // Verify all categories map correctly
        assert_eq!(ErrorCode::IoWrite.category(), "I/O");
        assert_eq!(ErrorCode::IoDiskFull.category(), "I/O");
        assert_eq!(ErrorCode::DeserializationFailed.category(), "Serialization");
        assert_eq!(ErrorCode::InvalidFormat.category(), "Serialization");
        assert_eq!(ErrorCode::CollectionAlreadyExists.category(), "Collection");
        assert_eq!(ErrorCode::CollectionCorrupted.category(), "Collection");
        assert_eq!(ErrorCode::AliasAlreadyExists.category(), "Collection");
        assert_eq!(ErrorCode::AliasTargetHasAliases.category(), "Collection");
        assert_eq!(ErrorCode::VectorAlreadyExists.category(), "Vector");
        assert_eq!(ErrorCode::InvalidVector.category(), "Vector");
        assert_eq!(ErrorCode::DatabaseCorrupted.category(), "Database");
        assert_eq!(ErrorCode::DatabaseLocked.category(), "Database");
        assert_eq!(ErrorCode::IndexCorrupted.category(), "Index");
        assert_eq!(ErrorCode::IndexBuildFailed.category(), "Index");
        assert_eq!(ErrorCode::MissingConfig.category(), "Configuration");
        assert_eq!(ErrorCode::QuotaExceeded.category(), "Resource");
        assert_eq!(ErrorCode::MemoryExhausted.category(), "Resource");
        assert_eq!(ErrorCode::LockTimeout.category(), "Operational");
        assert_eq!(ErrorCode::NotFound.category(), "Operational");
        assert_eq!(ErrorCode::DecryptionError.category(), "Security");
        assert_eq!(ErrorCode::AuthenticationFailed.category(), "Security");
        assert_eq!(ErrorCode::ReplicationError.category(), "Distributed");
        assert_eq!(ErrorCode::NetworkError.category(), "Distributed");
        assert_eq!(ErrorCode::RestoreFailed.category(), "Backup");
        assert_eq!(ErrorCode::BackupCorrupted.category(), "Backup");
        assert_eq!(ErrorCode::InvalidState.category(), "State");
    }

    #[test]
    fn test_non_retryable_errors_comprehensive() {
        let non_retryable = vec![
            NeedleError::DimensionMismatch { expected: 128, got: 256 },
            NeedleError::CollectionNotFound("c".into()),
            NeedleError::CollectionAlreadyExists("c".into()),
            NeedleError::AliasNotFound("a".into()),
            NeedleError::AliasAlreadyExists("a".into()),
            NeedleError::CollectionHasAliases("c".into()),
            NeedleError::VectorNotFound("v".into()),
            NeedleError::VectorAlreadyExists("v".into()),
            NeedleError::DuplicateId("d".into()),
            NeedleError::InvalidDatabase("bad".into()),
            NeedleError::Corruption("corrupt".into()),
            NeedleError::Index("idx".into()),
            NeedleError::InvalidConfig("cfg".into()),
            NeedleError::CapacityExceeded("cap".into()),
            NeedleError::InvalidVector("vec".into()),
            NeedleError::InvalidInput("inp".into()),
            NeedleError::QuotaExceeded("quota".into()),
            NeedleError::BackupError("bak".into()),
            NeedleError::NotFound("nf".into()),
            NeedleError::EncryptionError("enc".into()),
            NeedleError::OperationInProgress("op".into()),
            NeedleError::InvalidOperation("op".into()),
            NeedleError::InvalidState("state".into()),
            NeedleError::Unauthorized("unauth".into()),
            NeedleError::InvalidArgument("arg".into()),
        ];

        for error in &non_retryable {
            assert!(
                !error.is_retryable(),
                "Error {:?} should NOT be retryable",
                error
            );
            assert!(
                error.suggested_retry_delay_ms().is_none(),
                "Non-retryable error {:?} should have no retry delay",
                error
            );
        }
    }

    #[test]
    fn test_recovery_hints_content_for_key_errors() {
        // Collection not found hints should mention create_collection
        let err = NeedleError::CollectionNotFound("embeddings".into());
        let hints = err.recovery_hints();
        assert!(hints.iter().any(|h| h.summary.contains("create_collection")));
        assert!(hints.iter().any(|h| h.summary.contains("list_collections")));

        // Corruption hints should mention backup
        let err = NeedleError::Corruption("checksum mismatch".into());
        let hints = err.recovery_hints();
        assert!(hints.iter().any(|h| h.summary.contains("backup")));

        // Encryption error hints should mention key
        let err = NeedleError::EncryptionError("invalid key".into());
        let hints = err.recovery_hints();
        assert!(!hints.is_empty());

        // Timeout hints should mention retrying
        let err = NeedleError::Timeout(std::time::Duration::from_secs(30));
        let hints = err.recovery_hints();
        assert!(hints.iter().any(|h| {
            let lower = h.summary.to_lowercase();
            lower.contains("timeout") || lower.contains("retry") || lower.contains("increase")
        }));
    }
}
