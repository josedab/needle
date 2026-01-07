use thiserror::Error;

/// Error types for Needle database operations
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

    #[error("Vector '{0}' not found")]
    VectorNotFound(String),

    #[error("Vector '{0}' already exists")]
    VectorAlreadyExists(String),

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
}

/// Result type alias for Needle operations
pub type Result<T> = std::result::Result<T, NeedleError>;
