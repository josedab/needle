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
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read};
use std::path::{Path, PathBuf};
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
}
