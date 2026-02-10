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
            ErrorCode::IoRead | ErrorCode::IoWrite | ErrorCode::IoPermission | ErrorCode::IoDiskFull => "I/O",
            ErrorCode::SerializationFailed | ErrorCode::DeserializationFailed | ErrorCode::InvalidFormat => "Serialization",
            ErrorCode::CollectionNotFound | ErrorCode::CollectionAlreadyExists | ErrorCode::CollectionCorrupted | ErrorCode::AliasNotFound | ErrorCode::AliasAlreadyExists | ErrorCode::AliasTargetHasAliases => "Collection",
            ErrorCode::VectorNotFound | ErrorCode::VectorAlreadyExists | ErrorCode::DimensionMismatch | ErrorCode::InvalidVector => "Vector",
            ErrorCode::InvalidDatabase | ErrorCode::DatabaseCorrupted | ErrorCode::DatabaseLocked => "Database",
            ErrorCode::IndexError | ErrorCode::IndexCorrupted | ErrorCode::IndexBuildFailed => "Index",
            ErrorCode::InvalidConfig | ErrorCode::MissingConfig => "Configuration",
            ErrorCode::CapacityExceeded | ErrorCode::QuotaExceeded | ErrorCode::MemoryExhausted => "Resource",
            ErrorCode::Timeout | ErrorCode::LockTimeout | ErrorCode::Conflict | ErrorCode::NotFound => "Operational",
            ErrorCode::EncryptionError | ErrorCode::DecryptionError | ErrorCode::AuthenticationFailed => "Security",
            ErrorCode::ConsensusError | ErrorCode::ReplicationError | ErrorCode::NetworkError => "Distributed",
            ErrorCode::BackupFailed | ErrorCode::RestoreFailed | ErrorCode::BackupCorrupted => "Backup",
            ErrorCode::InvalidOperation | ErrorCode::InvalidState | ErrorCode::Unauthorized => "State",
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

/// Error types for Needle database operations
#[must_use]
#[derive(Error, Debug)]
pub enum NeedleError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("Collection '{0}' not found")]
    CollectionNotFound(String),

    #[error("Collection '{0}' already exists")]
    CollectionAlreadyExists(String),

    #[error("Alias '{0}' not found")]
    AliasNotFound(String),

    #[error("Alias '{0}' already exists")]
    AliasAlreadyExists(String),

    #[error("Cannot drop collection '{0}': aliases still reference it")]
    CollectionHasAliases(String),

    #[error("Vector '{0}' not found")]
    VectorNotFound(String),

    #[error("Vector '{0}' already exists")]
    VectorAlreadyExists(String),

    #[error("Duplicate ID: '{0}'")]
    DuplicateId(String),

    #[error("Operation in progress: {0}")]
    OperationInProgress(String),

    #[error("Invalid database file: {0}")]
    InvalidDatabase(String),

    #[error("Database corruption detected: {0}")]
    Corruption(String),

    #[error("Index error: {0}")]
    Index(String),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Capacity exceeded: {0}")]
    CapacityExceeded(String),

    #[error("Invalid vector: {0}")]
    InvalidVector(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Quota exceeded: {0}")]
    QuotaExceeded(String),

    #[error("Backup error: {0}")]
    BackupError(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Conflict: {0}")]
    Conflict(String),

    #[error("Encryption error: {0}")]
    EncryptionError(String),

    #[error("Consensus error: {0}")]
    ConsensusError(String),

    #[error("Lock error: failed to acquire lock")]
    LockError,

    #[error("Operation timed out after {0:?}")]
    Timeout(std::time::Duration),

    #[error("Lock acquisition timed out after {0:?}")]
    LockTimeout(std::time::Duration),

    #[error("Invalid operation: {0}")]
    InvalidOperation(String),

    #[error("Invalid state: {0}")]
    InvalidState(String),

    #[error("Unauthorized: {0}")]
    Unauthorized(String),

    #[error("Invalid argument: {0}")]
    InvalidArgument(String),
}

impl Recoverable for NeedleError {
    fn error_code(&self) -> ErrorCode {
        match self {
            NeedleError::Io(source) => {
                match source.kind() {
                    std::io::ErrorKind::NotFound => ErrorCode::IoRead,
                    std::io::ErrorKind::PermissionDenied => ErrorCode::IoPermission,
                    std::io::ErrorKind::WriteZero => ErrorCode::IoDiskFull,
                    _ => ErrorCode::IoWrite,
                }
            }
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
                output.push_str(&format!("\n\nThis error is retryable. Suggested delay: {}ms", delay));
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
                 and matches the expected schema."
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
        assert!(hints.iter().any(|h| h.summary.contains("128") && h.summary.contains("256")));
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
        let error = NeedleError::Io(std::io::Error::new(std::io::ErrorKind::NotFound, "file not found"));
        let hints = error.recovery_hints();
        assert!(hints.iter().any(|h| h.summary.contains("exist")));
        assert_eq!(error.error_code(), ErrorCode::IoRead);

        // Test PermissionDenied
        let error = NeedleError::Io(std::io::Error::new(std::io::ErrorKind::PermissionDenied, "access denied"));
        let hints = error.recovery_hints();
        assert!(hints.iter().any(|h| h.summary.contains("permission")));
        assert_eq!(error.error_code(), ErrorCode::IoPermission);
    }

    #[test]
    fn test_help_dimension_mismatch() {
        let error = NeedleError::DimensionMismatch { expected: 384, got: 128 };
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
}
