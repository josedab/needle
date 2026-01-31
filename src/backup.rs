//! Backup and Restore
//!
//! Provides backup and restore functionality for Needle databases:
//! - Full database backups
//! - Incremental backups
//! - Point-in-time recovery
//! - Backup verification
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::backup::{BackupManager, BackupConfig};
//!
//! let manager = BackupManager::new("/path/to/backups", BackupConfig::default());
//!
//! // Create a backup
//! let backup = manager.create_backup(&database)?;
//!
//! // Restore from backup
//! let restored_db = manager.restore_backup(&backup.id)?;
//! ```

use crate::database::Database;
use crate::error::{NeedleError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

/// Validate a backup ID or snapshot name to prevent path traversal attacks.
/// Only allows alphanumeric characters, underscores, and hyphens.
fn validate_backup_id(id: &str) -> Result<()> {
    if id.is_empty() {
        return Err(NeedleError::InvalidInput(
            "Backup ID cannot be empty".to_string(),
        ));
    }

    // Maximum reasonable length for an ID
    if id.len() > 256 {
        return Err(NeedleError::InvalidInput(
            "Backup ID too long (max 256 characters)".to_string(),
        ));
    }

    // Only allow safe characters: alphanumeric, underscore, hyphen
    for c in id.chars() {
        if !c.is_ascii_alphanumeric() && c != '_' && c != '-' {
            return Err(NeedleError::InvalidInput(format!(
                "Invalid character '{}' in backup ID. Only alphanumeric, underscore, and hyphen allowed.",
                c
            )));
        }
    }

    // Explicitly reject common path traversal patterns
    if id.contains("..") || id.starts_with('.') || id.contains('/') || id.contains('\\') {
        return Err(NeedleError::InvalidInput(
            "Invalid backup ID: path traversal patterns detected".to_string(),
        ));
    }

    Ok(())
}

/// Verify that a path is safely contained within a base directory.
/// Returns the canonicalized path if valid, or an error if the path escapes.
fn ensure_path_contained(base: &Path, path: &Path) -> Result<PathBuf> {
    // Canonicalize both paths for comparison
    let canonical_base = base.canonicalize().unwrap_or_else(|_| base.to_path_buf());

    // For paths that don't exist yet, we check the parent
    let canonical_path = if path.exists() {
        path.canonicalize()?
    } else {
        // Get the parent directory and ensure it exists and is safe
        let parent = path.parent().ok_or_else(|| {
            NeedleError::InvalidInput("Invalid path: no parent directory".to_string())
        })?;
        let canonical_parent = parent.canonicalize().unwrap_or_else(|_| parent.to_path_buf());

        // Construct the full path with the filename
        let filename = path.file_name().ok_or_else(|| {
            NeedleError::InvalidInput("Invalid path: no filename".to_string())
        })?;
        canonical_parent.join(filename)
    };

    // Verify the path is within the base directory
    if !canonical_path.starts_with(&canonical_base) {
        return Err(NeedleError::InvalidInput("Path traversal detected: path escapes backup directory".to_string()));
    }

    Ok(canonical_path)
}

/// Backup configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupConfig {
    /// Enable compression
    pub compression: bool,
    /// Verify backup after creation
    pub verify: bool,
    /// Maximum number of backups to retain
    pub max_backups: Option<usize>,
    /// Include metadata in backup
    pub include_metadata: bool,
}

impl Default for BackupConfig {
    fn default() -> Self {
        Self {
            compression: true,
            verify: true,
            max_backups: Some(10),
            include_metadata: true,
        }
    }
}

/// Backup metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupMetadata {
    /// Unique backup ID
    pub id: String,
    /// Creation timestamp
    pub created_at: u64,
    /// Source database path (if any)
    pub source_path: Option<String>,
    /// Number of collections
    pub num_collections: usize,
    /// Total number of vectors
    pub total_vectors: usize,
    /// Backup file size in bytes
    pub size_bytes: u64,
    /// Checksum for verification
    pub checksum: String,
    /// Needle version
    pub version: String,
    /// Backup type
    pub backup_type: BackupType,
    /// Parent backup ID (for incremental)
    pub parent_id: Option<String>,
}

/// Type of backup
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum BackupType {
    /// Full database backup
    Full,
    /// Incremental backup (only changes since last backup)
    Incremental,
    /// Snapshot (point-in-time)
    Snapshot,
}

/// Backup manager
pub struct BackupManager {
    /// Backup directory
    backup_dir: PathBuf,
    /// Configuration
    config: BackupConfig,
}

impl BackupManager {
    /// Create a new backup manager
    pub fn new(backup_dir: impl AsRef<Path>, config: BackupConfig) -> Self {
        Self {
            backup_dir: backup_dir.as_ref().to_path_buf(),
            config,
        }
    }

    /// Initialize backup directory
    pub fn init(&self) -> Result<()> {
        fs::create_dir_all(&self.backup_dir)?;
        fs::create_dir_all(self.backup_dir.join("full"))?;
        fs::create_dir_all(self.backup_dir.join("incremental"))?;
        fs::create_dir_all(self.backup_dir.join("snapshots"))?;
        Ok(())
    }

    /// Create a full backup
    pub fn create_backup(&self, db: &Database) -> Result<BackupMetadata> {
        self.init()?;

        let backup_id = generate_backup_id();
        let timestamp = current_timestamp();

        // Export database to JSON
        let collections: Vec<_> = db.list_collections();
        let mut total_vectors = 0;

        let mut backup_data = BackupData {
            metadata: BackupMetadata {
                id: backup_id.clone(),
                created_at: timestamp,
                source_path: None,
                num_collections: collections.len(),
                total_vectors: 0,
                size_bytes: 0,
                checksum: String::new(),
                version: env!("CARGO_PKG_VERSION").to_string(),
                backup_type: BackupType::Full,
                parent_id: None,
            },
            collections: Vec::new(),
        };

        for name in &collections {
            if let Ok(coll_ref) = db.collection(name) {
                let entries = coll_ref.export_all().unwrap_or_default();
                total_vectors += entries.len();

                backup_data.collections.push(CollectionBackup {
                    name: name.clone(),
                    dimensions: coll_ref.dimensions().unwrap_or(0),
                    entries,
                });
            }
        }

        backup_data.metadata.total_vectors = total_vectors;

        // Write backup file
        let backup_path = self
            .backup_dir
            .join("full")
            .join(format!("{}.json", backup_id));

        let file = File::create(&backup_path)?;
        let writer = BufWriter::new(file);

        if self.config.compression {
            // Simple compression by not pretty-printing
            serde_json::to_writer(writer, &backup_data)?;
        } else {
            serde_json::to_writer_pretty(writer, &backup_data)?;
        }

        // Calculate file size and checksum
        let file_size = fs::metadata(&backup_path)?.len();
        backup_data.metadata.size_bytes = file_size;
        backup_data.metadata.checksum = calculate_file_checksum(&backup_path)?;

        // Update metadata file
        self.save_metadata(&backup_data.metadata)?;

        // Verify if configured
        if self.config.verify {
            self.verify_backup(&backup_id)?;
        }

        // Cleanup old backups
        if let Some(max) = self.config.max_backups {
            self.cleanup_old_backups(max)?;
        }

        Ok(backup_data.metadata)
    }

    /// Create an incremental backup
    pub fn create_incremental(&self, db: &Database, parent_id: &str) -> Result<BackupMetadata> {
        // Validate parent ID to prevent path traversal
        validate_backup_id(parent_id)?;

        // For simplicity, incremental backups are implemented as full backups
        // with parent reference. A production system would track changes.
        let mut metadata = self.create_backup(db)?;
        metadata.backup_type = BackupType::Incremental;
        metadata.parent_id = Some(parent_id.to_string());
        self.save_metadata(&metadata)?;
        Ok(metadata)
    }

    /// Create a snapshot
    pub fn create_snapshot(&self, db: &Database, name: &str) -> Result<BackupMetadata> {
        // Validate snapshot name to prevent path traversal
        validate_backup_id(name)?;

        self.init()?;

        let backup_id = format!("snapshot_{}", name);
        let timestamp = current_timestamp();

        let collections: Vec<_> = db.list_collections();
        let mut total_vectors = 0;

        let mut backup_data = BackupData {
            metadata: BackupMetadata {
                id: backup_id.clone(),
                created_at: timestamp,
                source_path: None,
                num_collections: collections.len(),
                total_vectors: 0,
                size_bytes: 0,
                checksum: String::new(),
                version: env!("CARGO_PKG_VERSION").to_string(),
                backup_type: BackupType::Snapshot,
                parent_id: None,
            },
            collections: Vec::new(),
        };

        for coll_name in &collections {
            if let Ok(coll_ref) = db.collection(coll_name) {
                let entries = coll_ref.export_all().unwrap_or_default();
                total_vectors += entries.len();

                backup_data.collections.push(CollectionBackup {
                    name: coll_name.to_string(),
                    dimensions: coll_ref.dimensions().unwrap_or(0),
                    entries,
                });
            }
        }

        backup_data.metadata.total_vectors = total_vectors;

        let backup_path = self
            .backup_dir
            .join("snapshots")
            .join(format!("{}.json", backup_id));

        let file = File::create(&backup_path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer(writer, &backup_data)?;

        let file_size = fs::metadata(&backup_path)?.len();
        backup_data.metadata.size_bytes = file_size;
        backup_data.metadata.checksum = calculate_file_checksum(&backup_path)?;

        self.save_metadata(&backup_data.metadata)?;

        Ok(backup_data.metadata)
    }

    /// Restore from backup
    pub fn restore_backup(&self, backup_id: &str) -> Result<Database> {
        let backup_path = self.find_backup_path(backup_id)?;

        let file = File::open(&backup_path)?;
        let reader = BufReader::new(file);

        let backup_data: BackupData = serde_json::from_reader(reader)
            ?;

        let db = Database::in_memory();

        for coll_backup in backup_data.collections {
            db.create_collection(&coll_backup.name, coll_backup.dimensions)?;
            let coll_ref = db.collection(&coll_backup.name)?;

            for (id, vector, metadata) in coll_backup.entries {
                coll_ref.insert(&id, &vector, metadata)?;
            }
        }

        Ok(db)
    }

    /// Verify backup integrity
    pub fn verify_backup(&self, backup_id: &str) -> Result<bool> {
        let backup_path = self.find_backup_path(backup_id)?;

        // Verify checksum
        let current_checksum = calculate_file_checksum(&backup_path)?;

        // Load and verify metadata
        if let Some(metadata) = self.get_metadata(backup_id)? {
            if metadata.checksum != current_checksum && !metadata.checksum.is_empty() {
                return Ok(false);
            }
        }

        // Try to parse the backup
        let file = File::open(&backup_path)?;
        let reader = BufReader::new(file);

        let _: BackupData = serde_json::from_reader(reader)
            ?;

        Ok(true)
    }

    /// List all backups
    pub fn list_backups(&self) -> Result<Vec<BackupMetadata>> {
        let metadata_path = self.backup_dir.join("metadata.json");

        if !metadata_path.exists() {
            return Ok(Vec::new());
        }

        let file = File::open(&metadata_path)?;
        let reader = BufReader::new(file);

        let all_metadata: Vec<BackupMetadata> = serde_json::from_reader(reader)
            ?;

        Ok(all_metadata)
    }

    /// Get metadata for a specific backup
    pub fn get_metadata(&self, backup_id: &str) -> Result<Option<BackupMetadata>> {
        let backups = self.list_backups()?;
        Ok(backups.into_iter().find(|b| b.id == backup_id))
    }

    /// Delete a backup
    pub fn delete_backup(&self, backup_id: &str) -> Result<bool> {
        // Validate backup ID first - propagate validation errors
        validate_backup_id(backup_id)?;

        let backup_path = match self.find_backup_path(backup_id) {
            Ok(p) => p,
            Err(NeedleError::BackupError(_)) => return Ok(false), // Not found is OK
            Err(e) => return Err(e), // Propagate other errors (validation, etc.)
        };

        fs::remove_file(&backup_path)?;

        // Update metadata
        let mut backups = self.list_backups()?;
        backups.retain(|b| b.id != backup_id);
        self.save_all_metadata(&backups)?;

        Ok(true)
    }

    /// Find the path of a backup
    fn find_backup_path(&self, backup_id: &str) -> Result<PathBuf> {
        // Validate backup ID to prevent path traversal
        validate_backup_id(backup_id)?;

        let paths = [
            self.backup_dir
                .join("full")
                .join(format!("{}.json", backup_id)),
            self.backup_dir
                .join("incremental")
                .join(format!("{}.json", backup_id)),
            self.backup_dir
                .join("snapshots")
                .join(format!("{}.json", backup_id)),
        ];

        for path in &paths {
            if path.exists() {
                // Double-check: ensure path is contained within backup directory
                let safe_path = ensure_path_contained(&self.backup_dir, path)?;
                return Ok(safe_path);
            }
        }

        Err(NeedleError::BackupError(format!(
            "Backup not found: {}",
            backup_id
        )))
    }

    /// Save metadata for a backup
    fn save_metadata(&self, metadata: &BackupMetadata) -> Result<()> {
        let mut all_metadata = self.list_backups().unwrap_or_default();

        // Update or add
        if let Some(existing) = all_metadata.iter_mut().find(|m| m.id == metadata.id) {
            *existing = metadata.clone();
        } else {
            all_metadata.push(metadata.clone());
        }

        self.save_all_metadata(&all_metadata)
    }

    /// Save all metadata
    fn save_all_metadata(&self, metadata: &[BackupMetadata]) -> Result<()> {
        let metadata_path = self.backup_dir.join("metadata.json");
        let file = File::create(&metadata_path)?;
        let writer = BufWriter::new(file);

        serde_json::to_writer_pretty(writer, metadata)
            ?;

        Ok(())
    }

    /// Cleanup old backups
    fn cleanup_old_backups(&self, max_backups: usize) -> Result<()> {
        let mut backups = self.list_backups()?;

        // Sort by creation time (newest first)
        backups.sort_by(|a, b| b.created_at.cmp(&a.created_at));

        // Keep only full backups in cleanup consideration
        let full_backups: Vec<_> = backups
            .iter()
            .filter(|b| b.backup_type == BackupType::Full)
            .collect();

        if full_backups.len() > max_backups {
            for backup in full_backups.iter().skip(max_backups) {
                self.delete_backup(&backup.id)?;
            }
        }

        Ok(())
    }
}

/// Internal backup data structure
#[derive(Debug, Serialize, Deserialize)]
struct BackupData {
    metadata: BackupMetadata,
    collections: Vec<CollectionBackup>,
}

/// Collection backup data
#[derive(Debug, Serialize, Deserialize)]
struct CollectionBackup {
    name: String,
    dimensions: usize,
    entries: Vec<crate::database::ExportEntry>,
}

/// Generate a unique backup ID
fn generate_backup_id() -> String {
    let timestamp = current_timestamp();
    let random: u32 = rand::random();
    format!("backup_{}_{:08x}", timestamp, random)
}

/// Get current timestamp
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Calculate file checksum (simple hash)
fn calculate_file_checksum(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut contents = Vec::new();
    file.read_to_end(&mut contents)?;

    // Simple checksum using FNV-1a
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in &contents {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }

    Ok(format!("{:016x}", hash))
}

/// Backup schedule configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupSchedule {
    /// Interval between backups in seconds
    pub interval_secs: u64,
    /// Time of day for daily backups (hour, minute)
    pub daily_time: Option<(u8, u8)>,
    /// Days of week for weekly backups (0 = Sunday)
    pub weekly_days: Vec<u8>,
    /// Day of month for monthly backups
    pub monthly_day: Option<u8>,
}

impl Default for BackupSchedule {
    fn default() -> Self {
        Self {
            interval_secs: 86400, // Daily
            daily_time: Some((3, 0)), // 3 AM
            weekly_days: vec![0],  // Sunday
            monthly_day: Some(1),  // 1st of month
        }
    }
}

// =============================================================================
// Advanced Incremental Backup & Sync (Next-Gen)
// =============================================================================

/// Incremental backup state tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncrementalState {
    /// Last full backup ID
    pub last_full_backup: Option<String>,
    /// Last incremental backup ID
    pub last_incremental: Option<String>,
    /// Last backed up LSN (log sequence number)
    pub last_lsn: u64,
    /// Changed collections since last backup
    pub changed_collections: Vec<String>,
    /// Changed vector IDs by collection
    pub changed_vectors: std::collections::HashMap<String, Vec<String>>,
    /// Timestamp of last backup
    pub last_backup_timestamp: u64,
}

impl Default for IncrementalState {
    fn default() -> Self {
        Self {
            last_full_backup: None,
            last_incremental: None,
            last_lsn: 0,
            changed_collections: Vec::new(),
            changed_vectors: std::collections::HashMap::new(),
            last_backup_timestamp: 0,
        }
    }
}

/// Cloud storage provider for remote backup sync
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CloudProvider {
    /// Amazon S3
    S3,
    /// Google Cloud Storage
    GCS,
    /// Azure Blob Storage
    Azure,
    /// Custom HTTP endpoint
    Custom,
    /// Local filesystem (for testing)
    Local,
}

/// Cloud sync configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudSyncConfig {
    /// Cloud provider
    pub provider: CloudProvider,
    /// Bucket/container name
    pub bucket: String,
    /// Prefix path for backups
    pub prefix: String,
    /// Region (for S3/GCS)
    pub region: Option<String>,
    /// Endpoint URL (for custom/minio)
    pub endpoint: Option<String>,
    /// Access key/credentials
    pub access_key: Option<String>,
    /// Secret key/credentials (should be handled securely)
    pub secret_key_env: Option<String>,
    /// Enable encryption at rest
    pub encrypt: bool,
    /// Compression level (0-9)
    pub compression_level: u32,
    /// Multipart upload threshold (bytes)
    pub multipart_threshold: u64,
}

impl Default for CloudSyncConfig {
    fn default() -> Self {
        Self {
            provider: CloudProvider::Local,
            bucket: "needle-backups".to_string(),
            prefix: "".to_string(),
            region: None,
            endpoint: None,
            access_key: None,
            secret_key_env: Some("NEEDLE_BACKUP_SECRET".to_string()),
            encrypt: true,
            compression_level: 6,
            multipart_threshold: 100 * 1024 * 1024, // 100MB
        }
    }
}

/// Point-in-time recovery configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PitrConfig {
    /// Enable point-in-time recovery
    pub enabled: bool,
    /// WAL retention period in seconds
    pub wal_retention_secs: u64,
    /// Maximum WAL size before rotation
    pub max_wal_size: u64,
    /// Checkpoint interval in seconds
    pub checkpoint_interval: u64,
}

impl Default for PitrConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            wal_retention_secs: 7 * 24 * 3600, // 7 days
            max_wal_size: 1024 * 1024 * 1024,  // 1GB
            checkpoint_interval: 300,           // 5 minutes
        }
    }
}

/// WAL entry for point-in-time recovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalEntry {
    /// Log sequence number
    pub lsn: u64,
    /// Operation type
    pub operation: WalOperation,
    /// Collection name
    pub collection: String,
    /// Vector ID (if applicable)
    pub vector_id: Option<String>,
    /// Timestamp
    pub timestamp: u64,
    /// Data payload (serialized)
    pub data: Option<Vec<u8>>,
}

/// WAL operation types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WalOperation {
    /// Insert vector
    Insert,
    /// Update vector
    Update,
    /// Delete vector
    Delete,
    /// Create collection
    CreateCollection,
    /// Delete collection
    DeleteCollection,
    /// Compact collection
    Compact,
    /// Checkpoint marker
    Checkpoint,
}

/// Incremental backup manager
pub struct IncrementalBackupManager {
    backup_dir: PathBuf,
    state: parking_lot::RwLock<IncrementalState>,
    cloud_config: Option<CloudSyncConfig>,
    pitr_config: PitrConfig,
    wal_entries: parking_lot::RwLock<Vec<WalEntry>>,
    current_lsn: std::sync::atomic::AtomicU64,
    /// Optional replication leader; receives segments when WAL entries accumulate.
    replication_leader: Option<Arc<ReplicationLeader>>,
    /// Tracks the LSN at which the last replication segment was produced.
    last_segment_lsn: std::sync::atomic::AtomicU64,
}

impl IncrementalBackupManager {
    /// Create a new incremental backup manager
    pub fn new(backup_dir: impl AsRef<Path>) -> Self {
        Self {
            backup_dir: backup_dir.as_ref().to_path_buf(),
            state: parking_lot::RwLock::new(IncrementalState::default()),
            cloud_config: None,
            pitr_config: PitrConfig::default(),
            wal_entries: parking_lot::RwLock::new(Vec::new()),
            current_lsn: std::sync::atomic::AtomicU64::new(1),
            replication_leader: None,
            last_segment_lsn: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Enable cloud sync
    pub fn with_cloud_sync(mut self, config: CloudSyncConfig) -> Self {
        self.cloud_config = Some(config);
        self
    }

    /// Configure PITR
    pub fn with_pitr(mut self, config: PitrConfig) -> Self {
        self.pitr_config = config;
        self
    }

    /// Attach a replication leader. WAL entries will automatically produce
    /// replication segments on the leader when the segment threshold is reached.
    pub fn with_replication_leader(mut self, leader: Arc<ReplicationLeader>) -> Self {
        self.replication_leader = Some(leader);
        self
    }

    /// Record a WAL entry
    pub fn record_wal(&self, operation: WalOperation, collection: &str, vector_id: Option<&str>) {
        if !self.pitr_config.enabled {
            return;
        }

        let lsn = self
            .current_lsn
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        let entry = WalEntry {
            lsn,
            operation,
            collection: collection.to_string(),
            vector_id: vector_id.map(String::from),
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            data: None,
        };

        self.wal_entries.write().push(entry);

        // Track changed collections
        let mut state = self.state.write();
        if !state.changed_collections.contains(&collection.to_string()) {
            state.changed_collections.push(collection.to_string());
        }
        if let Some(vid) = vector_id {
            state
                .changed_vectors
                .entry(collection.to_string())
                .or_default()
                .push(vid.to_string());
        }
        drop(state);

        // Produce a replication segment when enough WAL entries accumulate
        if let Some(leader) = &self.replication_leader {
            let last_seg = self.last_segment_lsn.load(std::sync::atomic::Ordering::Acquire);
            let entries = self.wal_entries.read();
            let pending: Vec<_> = entries.iter().filter(|e| e.lsn > last_seg).collect();
            let pending_bytes: usize = pending.iter().map(|e| {
                e.collection.len() + e.vector_id.as_ref().map_or(0, |v| v.len()) + 64
            }).sum();

            if pending.len() >= 32 || pending_bytes >= leader.segment_max_bytes() {
                let collections: Vec<String> = pending
                    .iter()
                    .map(|e| e.collection.clone())
                    .collect::<std::collections::HashSet<_>>()
                    .into_iter()
                    .collect();
                let lsn_start = pending.first().map(|e| e.lsn).unwrap_or(0);
                let lsn_end = pending.last().map(|e| e.lsn).unwrap_or(0);
                drop(entries);

                leader.produce_segment(lsn_start, lsn_end, collections, pending_bytes);
                self.last_segment_lsn.store(lsn_end, std::sync::atomic::Ordering::Release);
            }
        }
    }

    /// Create an incremental backup
    pub fn create_incremental(&self, db: &Database) -> Result<IncrementalBackupInfo> {
        let state = self.state.read();

        // Determine what changed
        let changed_collections = state.changed_collections.clone();
        let changed_vectors = state.changed_vectors.clone();
        let base_backup = state.last_full_backup.clone();

        drop(state);

        let backup_id = format!(
            "inc-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis()
        );

        let backup_dir = self.backup_dir.join(&backup_id);
        fs::create_dir_all(&backup_dir)?;

        // Export only changed vectors
        let mut exported_count = 0;
        for collection_name in &changed_collections {
            if let Ok(collection) = db.collection(collection_name) {
                let changes_file = backup_dir.join(format!("{}.json", collection_name));
                let mut changes = Vec::new();

                if let Some(vector_ids) = changed_vectors.get(collection_name) {
                    for id in vector_ids {
                        if let Some((vector, metadata)) = collection.get(id) {
                            changes.push(serde_json::json!({
                                "id": id,
                                "vector": vector,
                                "metadata": metadata
                            }));
                            exported_count += 1;
                        }
                    }
                }

                let file = File::create(&changes_file)?;
                serde_json::to_writer(BufWriter::new(file), &changes)?;
            }
        }

        // Save WAL entries
        let wal_file = backup_dir.join("wal.json");
        let wal_entries: Vec<_> = self.wal_entries.read().clone();
        let file = File::create(&wal_file)?;
        serde_json::to_writer(BufWriter::new(file), &wal_entries)?;

        // Update state
        let mut state = self.state.write();
        state.last_incremental = Some(backup_id.clone());
        state.last_lsn = self.current_lsn.load(std::sync::atomic::Ordering::Relaxed);
        state.changed_collections.clear();
        state.changed_vectors.clear();
        state.last_backup_timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Ok(IncrementalBackupInfo {
            backup_id,
            base_backup,
            collections_affected: changed_collections,
            vectors_exported: exported_count,
            wal_entries_count: wal_entries.len(),
            timestamp: state.last_backup_timestamp,
        })
    }

    /// Restore to a specific point in time
    pub fn restore_to_point_in_time(&self, target_timestamp: u64) -> Result<Vec<WalEntry>> {
        let entries = self.wal_entries.read();

        // Find entries up to target timestamp
        let applicable: Vec<_> = entries
            .iter()
            .filter(|e| e.timestamp <= target_timestamp)
            .cloned()
            .collect();

        Ok(applicable)
    }

    /// Get current WAL size
    pub fn wal_size(&self) -> usize {
        self.wal_entries.read().len()
    }

    /// Create checkpoint (flush WAL to backup)
    pub fn checkpoint(&self) -> Result<u64> {
        let lsn = self.current_lsn.load(std::sync::atomic::Ordering::Relaxed);

        self.record_wal(WalOperation::Checkpoint, "_system", None);

        // In a full implementation, this would:
        // 1. Flush WAL to disk
        // 2. Update checkpoint file
        // 3. Optionally sync to cloud

        Ok(lsn)
    }

    /// Prune old WAL entries
    pub fn prune_wal(&self, keep_after_lsn: u64) {
        self.wal_entries.write().retain(|e| e.lsn >= keep_after_lsn);
    }

    /// Sync backup to cloud
    pub fn sync_to_cloud(&self, backup_id: &str) -> Result<CloudSyncResult> {
        let cloud_config = self
            .cloud_config
            .as_ref()
            .ok_or_else(|| NeedleError::InvalidInput("Cloud sync not configured".to_string()))?;

        let backup_path = self.backup_dir.join(backup_id);
        if !backup_path.exists() {
            return Err(NeedleError::NotFound(format!(
                "Backup not found: {}",
                backup_id
            )));
        }

        // Count files and sizes
        let mut files_uploaded = 0;
        let mut bytes_uploaded = 0u64;

        for entry in fs::read_dir(&backup_path)? {
            let entry = entry?;
            if entry.file_type()?.is_file() {
                let size = entry.metadata()?.len();
                files_uploaded += 1;
                bytes_uploaded += size;

                // In a full implementation, this would:
                // - Read the file
                // - Optionally compress and encrypt
                // - Upload to cloud provider
            }
        }

        Ok(CloudSyncResult {
            backup_id: backup_id.to_string(),
            provider: cloud_config.provider.clone(),
            destination: format!(
                "{}/{}/{}",
                cloud_config.bucket, cloud_config.prefix, backup_id
            ),
            files_uploaded,
            bytes_uploaded,
            compressed: cloud_config.compression_level > 0,
            encrypted: cloud_config.encrypt,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        })
    }

    /// List available restore points
    pub fn list_restore_points(&self) -> Vec<RestorePoint> {
        let state = self.state.read();
        let entries = self.wal_entries.read();

        let mut points = Vec::new();

        // Add checkpoint restore points
        for entry in entries.iter() {
            if entry.operation == WalOperation::Checkpoint {
                points.push(RestorePoint {
                    lsn: entry.lsn,
                    timestamp: entry.timestamp,
                    point_type: RestorePointType::Checkpoint,
                    description: format!("Checkpoint at LSN {}", entry.lsn),
                });
            }
        }

        // Add backup restore points
        if let Some(ref full_backup) = state.last_full_backup {
            points.push(RestorePoint {
                lsn: 0,
                timestamp: state.last_backup_timestamp,
                point_type: RestorePointType::FullBackup,
                description: format!("Full backup: {}", full_backup),
            });
        }

        points.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        points
    }
}

/// Incremental backup info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncrementalBackupInfo {
    /// Backup ID
    pub backup_id: String,
    /// Base full backup ID
    pub base_backup: Option<String>,
    /// Collections affected
    pub collections_affected: Vec<String>,
    /// Number of vectors exported
    pub vectors_exported: usize,
    /// Number of WAL entries included
    pub wal_entries_count: usize,
    /// Timestamp
    pub timestamp: u64,
}

/// Cloud sync result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudSyncResult {
    /// Backup ID
    pub backup_id: String,
    /// Cloud provider
    pub provider: CloudProvider,
    /// Destination path
    pub destination: String,
    /// Number of files uploaded
    pub files_uploaded: usize,
    /// Bytes uploaded
    pub bytes_uploaded: u64,
    /// Whether data was compressed
    pub compressed: bool,
    /// Whether data was encrypted
    pub encrypted: bool,
    /// Timestamp
    pub timestamp: u64,
}

/// Restore point information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RestorePoint {
    /// LSN of the restore point
    pub lsn: u64,
    /// Timestamp
    pub timestamp: u64,
    /// Type of restore point
    pub point_type: RestorePointType,
    /// Description
    pub description: String,
}

/// Type of restore point
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RestorePointType {
    /// Full backup
    FullBackup,
    /// Incremental backup
    IncrementalBackup,
    /// WAL checkpoint
    Checkpoint,
    /// Named snapshot
    Snapshot,
}

// ---------------------------------------------------------------------------
// Snapshot Replication Protocol
// ---------------------------------------------------------------------------

/// Consistency level for replicated reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsistencyLevel {
    /// Read from any replica (fastest, may return stale data)
    Eventual,
    /// Read must be within `max_staleness_seconds` of leader
    BoundedStaleness { max_staleness_seconds: u64 },
    /// Read must reflect all writes acknowledged before the read began
    Strong,
}

impl Default for ConsistencyLevel {
    fn default() -> Self {
        ConsistencyLevel::Eventual
    }
}

/// A replication snapshot segment (binary diff).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotSegment {
    pub segment_id: u64,
    pub lsn_start: u64,
    pub lsn_end: u64,
    pub collections_affected: Vec<String>,
    pub data_bytes: usize,
    pub checksum: u64,
    pub created_at: u64,
}

/// Status of a replication follower.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FollowerStatus {
    Syncing,
    CaughtUp,
    Lagging,
    Disconnected,
}

/// Tracks the state of a single replication follower.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FollowerState {
    pub follower_id: String,
    pub last_applied_lsn: u64,
    pub status: FollowerStatus,
    pub last_heartbeat: u64,
    pub lag_bytes: u64,
    pub lag_seconds: u64,
}

/// Configuration for the replication leader.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicationConfig {
    pub segment_max_bytes: usize,
    pub heartbeat_interval_seconds: u64,
    pub follower_timeout_seconds: u64,
    pub max_lag_before_alert: u64,
}

impl Default for ReplicationConfig {
    fn default() -> Self {
        Self {
            segment_max_bytes: 16 * 1024 * 1024, // 16 MB
            heartbeat_interval_seconds: 5,
            follower_timeout_seconds: 30,
            max_lag_before_alert: 10,
        }
    }
}

/// Leader-side replication manager. Produces snapshot segments from WAL entries
/// and tracks follower progress.
pub struct ReplicationLeader {
    config: ReplicationConfig,
    segments: parking_lot::RwLock<Vec<SnapshotSegment>>,
    followers: parking_lot::RwLock<HashMap<String, FollowerState>>,
    current_lsn: std::sync::atomic::AtomicU64,
    segment_counter: std::sync::atomic::AtomicU64,
}

impl ReplicationLeader {
    pub fn new(config: ReplicationConfig) -> Self {
        Self {
            config,
            segments: parking_lot::RwLock::new(Vec::new()),
            followers: parking_lot::RwLock::new(HashMap::new()),
            current_lsn: std::sync::atomic::AtomicU64::new(0),
            segment_counter: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Maximum segment size in bytes (from config).
    pub fn segment_max_bytes(&self) -> usize {
        self.config.segment_max_bytes
    }

    /// Produce a new replication segment from a batch of WAL entries.
    pub fn produce_segment(
        &self,
        lsn_start: u64,
        lsn_end: u64,
        collections: Vec<String>,
        data_bytes: usize,
    ) -> SnapshotSegment {
        let seq = self.segment_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let segment = SnapshotSegment {
            segment_id: seq,
            lsn_start,
            lsn_end,
            collections_affected: collections,
            data_bytes,
            checksum: lsn_start ^ lsn_end ^ data_bytes as u64,
            created_at: now,
        };

        self.current_lsn.store(lsn_end, std::sync::atomic::Ordering::Release);
        self.segments.write().push(segment.clone());
        segment
    }

    /// Register a new follower.
    pub fn register_follower(&self, follower_id: &str) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        self.followers.write().insert(
            follower_id.to_string(),
            FollowerState {
                follower_id: follower_id.to_string(),
                last_applied_lsn: 0,
                status: FollowerStatus::Syncing,
                last_heartbeat: now,
                lag_bytes: 0,
                lag_seconds: 0,
            },
        );
    }

    /// Acknowledge that a follower has applied up to `lsn`.
    pub fn ack(&self, follower_id: &str, applied_lsn: u64) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let leader_lsn = self.current_lsn.load(std::sync::atomic::Ordering::Acquire);

        if let Some(f) = self.followers.write().get_mut(follower_id) {
            f.last_applied_lsn = applied_lsn;
            f.last_heartbeat = now;
            f.status = if applied_lsn >= leader_lsn {
                FollowerStatus::CaughtUp
            } else {
                FollowerStatus::Lagging
            };
        }
    }

    /// Get segments that a follower needs to catch up.
    pub fn segments_since(&self, from_lsn: u64) -> Vec<SnapshotSegment> {
        self.segments
            .read()
            .iter()
            .filter(|s| s.lsn_start >= from_lsn)
            .cloned()
            .collect()
    }

    /// Check if a read at `consistency` level is satisfied.
    pub fn can_read(&self, follower_id: &str, consistency: &ConsistencyLevel) -> bool {
        let followers = self.followers.read();
        let follower = match followers.get(follower_id) {
            Some(f) => f,
            None => return false,
        };
        let leader_lsn = self.current_lsn.load(std::sync::atomic::Ordering::Acquire);

        match consistency {
            ConsistencyLevel::Eventual => true,
            ConsistencyLevel::BoundedStaleness { max_staleness_seconds } => {
                follower.lag_seconds <= *max_staleness_seconds
            }
            ConsistencyLevel::Strong => follower.last_applied_lsn >= leader_lsn,
        }
    }

    /// Detect disconnected followers.
    pub fn detect_disconnected(&self) -> Vec<String> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let timeout = self.config.follower_timeout_seconds;
        let mut disconnected = Vec::new();

        let mut followers = self.followers.write();
        for f in followers.values_mut() {
            if now.saturating_sub(f.last_heartbeat) > timeout
                && f.status != FollowerStatus::Disconnected
            {
                f.status = FollowerStatus::Disconnected;
                disconnected.push(f.follower_id.clone());
            }
        }
        disconnected
    }

    /// List all followers with their status.
    pub fn list_followers(&self) -> Vec<FollowerState> {
        self.followers.read().values().cloned().collect()
    }

    /// Current leader LSN.
    pub fn current_lsn(&self) -> u64 {
        self.current_lsn.load(std::sync::atomic::Ordering::Acquire)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_db() -> Database {
        let db = Database::in_memory();
        db.create_collection("test", 8).unwrap();

        let coll = db.collection("test").unwrap();
        for i in 0..10 {
            coll.insert(
                format!("doc{}", i),
                &[i as f32; 8],
                Some(serde_json::json!({"index": i})),
            )
            .unwrap();
        }

        db
    }

    #[test]
    fn test_create_backup() {
        let temp_dir = TempDir::new().unwrap();
        let manager = BackupManager::new(temp_dir.path(), BackupConfig::default());

        let db = create_test_db();
        let metadata = manager.create_backup(&db).unwrap();

        assert!(!metadata.id.is_empty());
        assert_eq!(metadata.num_collections, 1);
        assert_eq!(metadata.total_vectors, 10);
        assert_eq!(metadata.backup_type, BackupType::Full);
    }

    #[test]
    fn test_restore_backup() {
        let temp_dir = TempDir::new().unwrap();
        let manager = BackupManager::new(temp_dir.path(), BackupConfig::default());

        let db = create_test_db();
        let metadata = manager.create_backup(&db).unwrap();

        let restored = manager.restore_backup(&metadata.id).unwrap();

        // Verify collections
        let collections = restored.list_collections();
        assert_eq!(collections.len(), 1);
        assert!(collections.contains(&"test".to_string()));

        // Verify vectors
        let coll = restored.collection("test").unwrap();
        assert_eq!(coll.len(), 10);
    }

    #[test]
    fn test_verify_backup() {
        let temp_dir = TempDir::new().unwrap();
        let config = BackupConfig {
            verify: false,
            ..Default::default()
        };
        let manager = BackupManager::new(temp_dir.path(), config);

        let db = create_test_db();
        let metadata = manager.create_backup(&db).unwrap();

        let is_valid = manager.verify_backup(&metadata.id).unwrap();
        assert!(is_valid);
    }

    #[test]
    fn test_list_backups() {
        let temp_dir = TempDir::new().unwrap();
        let manager = BackupManager::new(temp_dir.path(), BackupConfig::default());

        let db = create_test_db();

        // Create multiple backups
        manager.create_backup(&db).unwrap();
        manager.create_backup(&db).unwrap();
        manager.create_backup(&db).unwrap();

        let backups = manager.list_backups().unwrap();
        assert_eq!(backups.len(), 3);
    }

    #[test]
    fn test_delete_backup() {
        let temp_dir = TempDir::new().unwrap();
        let manager = BackupManager::new(temp_dir.path(), BackupConfig::default());

        let db = create_test_db();
        let metadata = manager.create_backup(&db).unwrap();

        assert!(manager.delete_backup(&metadata.id).unwrap());
        assert!(!manager.delete_backup(&metadata.id).unwrap());
    }

    #[test]
    fn test_create_snapshot() {
        let temp_dir = TempDir::new().unwrap();
        let manager = BackupManager::new(temp_dir.path(), BackupConfig::default());

        let db = create_test_db();
        let metadata = manager.create_snapshot(&db, "test_snapshot").unwrap();

        assert_eq!(metadata.backup_type, BackupType::Snapshot);
        assert!(metadata.id.contains("snapshot_test_snapshot"));
    }

    #[test]
    fn test_backup_cleanup() {
        let temp_dir = TempDir::new().unwrap();
        let config = BackupConfig {
            max_backups: Some(2),
            verify: false,
            ..Default::default()
        };
        let manager = BackupManager::new(temp_dir.path(), config);

        let db = create_test_db();

        // Create more backups than max
        manager.create_backup(&db).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
        manager.create_backup(&db).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
        manager.create_backup(&db).unwrap();

        let backups = manager.list_backups().unwrap();
        let full_backups: Vec<_> = backups
            .iter()
            .filter(|b| b.backup_type == BackupType::Full)
            .collect();

        assert!(full_backups.len() <= 2);
    }

    #[test]
    fn test_checksum() {
        let temp_dir = TempDir::new().unwrap();
        let test_file = temp_dir.path().join("test.txt");
        fs::write(&test_file, b"Hello, World!").unwrap();

        let checksum = calculate_file_checksum(&test_file).unwrap();
        assert!(!checksum.is_empty());

        // Same content should produce same checksum
        let checksum2 = calculate_file_checksum(&test_file).unwrap();
        assert_eq!(checksum, checksum2);
    }

    #[test]
    fn test_validate_backup_id_valid() {
        assert!(validate_backup_id("backup_123").is_ok());
        assert!(validate_backup_id("my-backup").is_ok());
        assert!(validate_backup_id("Test_Backup_2024").is_ok());
        assert!(validate_backup_id("a").is_ok());
    }

    #[test]
    fn test_validate_backup_id_path_traversal() {
        // Path traversal attempts should be rejected
        assert!(validate_backup_id("../../../etc/passwd").is_err());
        assert!(validate_backup_id("..").is_err());
        assert!(validate_backup_id("foo/../bar").is_err());
        assert!(validate_backup_id("foo/bar").is_err());
        assert!(validate_backup_id("foo\\bar").is_err());
        assert!(validate_backup_id(".hidden").is_err());
    }

    #[test]
    fn test_validate_backup_id_invalid_chars() {
        // Invalid characters should be rejected
        assert!(validate_backup_id("backup with space").is_err());
        assert!(validate_backup_id("backup@123").is_err());
        assert!(validate_backup_id("backup$test").is_err());
        assert!(validate_backup_id("backup;drop").is_err());
        assert!(validate_backup_id("").is_err());
    }

    #[test]
    fn test_validate_backup_id_too_long() {
        // Too long IDs should be rejected
        let long_id = "a".repeat(257);
        assert!(validate_backup_id(&long_id).is_err());
    }

    #[test]
    fn test_restore_path_traversal_rejected() {
        let temp_dir = TempDir::new().unwrap();
        let manager = BackupManager::new(temp_dir.path(), BackupConfig::default());

        // Attempt to restore with path traversal should fail
        let result = manager.restore_backup("../../../etc/passwd");
        assert!(result.is_err());
    }

    #[test]
    fn test_delete_path_traversal_rejected() {
        let temp_dir = TempDir::new().unwrap();
        let manager = BackupManager::new(temp_dir.path(), BackupConfig::default());

        // Attempt to delete with path traversal should fail
        let result = manager.delete_backup("../../../etc/passwd");
        assert!(result.is_err());
    }

    #[test]
    fn test_snapshot_path_traversal_rejected() {
        let temp_dir = TempDir::new().unwrap();
        let manager = BackupManager::new(temp_dir.path(), BackupConfig::default());
        let db = create_test_db();

        // Attempt to create snapshot with path traversal should fail
        let result = manager.create_snapshot(&db, "../../../etc/passwd");
        assert!(result.is_err());
    }

    #[test]
    fn test_replication_leader_produce_and_ack() {
        let leader = ReplicationLeader::new(ReplicationConfig::default());
        leader.register_follower("follower_1");

        let seg = leader.produce_segment(1, 100, vec!["docs".into()], 4096);
        assert_eq!(seg.lsn_start, 1);
        assert_eq!(seg.lsn_end, 100);
        assert_eq!(leader.current_lsn(), 100);

        // Follower acks up to 100
        leader.ack("follower_1", 100);
        let followers = leader.list_followers();
        assert_eq!(followers[0].status, FollowerStatus::CaughtUp);
    }

    #[test]
    fn test_replication_segments_since() {
        let leader = ReplicationLeader::new(ReplicationConfig::default());
        leader.produce_segment(1, 50, vec!["a".into()], 1024);
        leader.produce_segment(50, 100, vec!["b".into()], 2048);
        leader.produce_segment(100, 150, vec!["c".into()], 512);

        let catchup = leader.segments_since(50);
        assert_eq!(catchup.len(), 2); // segments starting at 50 and 100
    }

    #[test]
    fn test_consistency_levels() {
        let leader = ReplicationLeader::new(ReplicationConfig::default());
        leader.register_follower("f1");
        leader.produce_segment(1, 100, vec![], 0);

        // Eventual always readable
        assert!(leader.can_read("f1", &ConsistencyLevel::Eventual));

        // Strong requires caught-up
        assert!(!leader.can_read("f1", &ConsistencyLevel::Strong));

        leader.ack("f1", 100);
        assert!(leader.can_read("f1", &ConsistencyLevel::Strong));
    }
}
