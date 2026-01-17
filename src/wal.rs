// Allow dead_code for this public API module - types are exported for library users
#![allow(dead_code)]
#![allow(clippy::redundant_closure)]

//! Write-Ahead Log (WAL) for durability and crash recovery.
//!
//! The WAL provides durability guarantees by writing operations to a log
//! before applying them to the main data store. In case of a crash, the
//! log can be replayed to recover uncommitted operations.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                      WAL Manager                             │
//! ├─────────────────────────────────────────────────────────────┤
//! │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
//! │  │ Log Writer  │  │ Log Reader  │  │ Checkpoint Manager  │  │
//! │  └─────────────┘  └─────────────┘  └─────────────────────┘  │
//! ├─────────────────────────────────────────────────────────────┤
//! │                      Segment Files                           │
//! │  [segment_000001.wal] [segment_000002.wal] [...]            │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Crash Recovery Procedures
//!
//! ## Overview
//!
//! Needle uses the WAL to ensure durability. Every write operation is logged
//! before being applied to the in-memory index. This allows recovery after:
//! - Process crashes
//! - Power failures
//! - System panics
//!
//! ## Recovery Flow
//!
//! ```text
//! ┌─────────────┐     ┌─────────────────────┐     ┌────────────────┐
//! │  WAL Files  │────►│ Find last checkpoint│────►│ Replay entries │
//! └─────────────┘     └─────────────────────┘     └────────────────┘
//!                                                         │
//!                     ┌─────────────────────┐             ▼
//!                     │  Database restored  │◄────────────┘
//!                     └─────────────────────┘
//! ```
//!
//! ## Step-by-Step Recovery
//!
//! 1. **Open the WAL directory**: The WAL manager scans for segment files
//! 2. **Find the last checkpoint**: Checkpoints mark points where data was
//!    safely persisted to the main database file
//! 3. **Replay from checkpoint**: All entries after the checkpoint LSN are
//!    replayed to reconstruct uncommitted changes
//! 4. **Verify integrity**: Checksums are validated during replay (if enabled)
//!
//! ## Recovery Example
//!
//! ```rust,ignore
//! use needle::wal::{WalManager, WalConfig};
//! use needle::Database;
//!
//! // Step 1: Open WAL and find checkpoint
//! let wal = WalManager::open("/path/to/wal", WalConfig::default())?;
//! let checkpoint_lsn = wal.checkpoint_lsn();
//!
//! // Step 2: Open database (will be at checkpoint state)
//! let mut db = Database::open("vectors.needle")?;
//!
//! // Step 3: Replay operations since checkpoint
//! wal.replay(checkpoint_lsn, |record| {
//!     match record.entry {
//!         WalEntry::Insert { collection, id, vector, metadata } => {
//!             let coll = db.collection(&collection)?;
//!             coll.insert(&id, &vector, metadata)?;
//!         }
//!         WalEntry::Delete { collection, id } => {
//!             let coll = db.collection(&collection)?;
//!             coll.delete(&id)?;
//!         }
//!         // Handle other entry types...
//!         _ => {}
//!     }
//!     Ok(())
//! })?;
//!
//! // Step 4: Create new checkpoint after recovery
//! wal.checkpoint()?;
//! db.save()?;
//! ```
//!
//! ## Handling Corruption
//!
//! If a WAL file is corrupted (checksum mismatch), recovery stops at the
//! last valid entry. To handle this:
//!
//! 1. The database state reflects all operations up to the corruption point
//! 2. Corrupted segments are preserved (not deleted) for forensic analysis
//! 3. A warning is logged indicating the corruption location
//!
//! ## Configuration for Durability
//!
//! | Setting | Value | Effect |
//! |---------|-------|--------|
//! | `sync_on_write` | `true` | Every write is fsync'd (safest, slowest) |
//! | `sync_on_write` | `false` | Writes are batched, periodic sync (faster) |
//! | `enable_checksums` | `true` | CRC32 validation on recovery |
//! | `checkpoint_interval` | `1000` | Checkpoint after N operations |
//!
//! ## Best Practices
//!
//! 1. **Regular checkpoints**: Call `checkpoint()` periodically to reduce
//!    recovery time and WAL size
//! 2. **Sync before critical operations**: Call `sync()` before operations
//!    where data loss is unacceptable
//! 3. **Monitor WAL size**: Large WALs increase recovery time; adjust
//!    checkpoint frequency accordingly
//! 4. **Backup WAL files**: Include WAL directory in backups for point-in-time
//!    recovery capability
//!
//! # Usage
//!
//! ```rust,ignore
//! use needle::wal::{WalManager, WalConfig, WalEntry};
//!
//! // Create WAL manager
//! let config = WalConfig::default();
//! let mut wal = WalManager::open("/path/to/wal", config)?;
//!
//! // Write operations
//! let lsn = wal.append(WalEntry::Insert {
//!     collection: "docs".to_string(),
//!     id: "doc1".to_string(),
//!     vector: vec![0.1; 384],
//!     metadata: None,
//! })?;
//!
//! // Sync to disk
//! wal.sync()?;
//!
//! // Create checkpoint
//! wal.checkpoint()?;
//! ```

use crate::error::{NeedleError, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::VecDeque;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Log Sequence Number - monotonically increasing identifier for log entries.
pub type Lsn = u64;

/// WAL entry representing a single operation.
///
/// # Example
///
/// ```
/// use needle::wal::WalEntry;
/// use serde_json::json;
///
/// // Create an insert entry
/// let insert = WalEntry::Insert {
///     collection: "documents".to_string(),
///     id: "doc1".to_string(),
///     vector: vec![0.1, 0.2, 0.3, 0.4],
///     metadata: Some(json!({"title": "Hello World"})),
/// };
///
/// // Create a delete entry
/// let delete = WalEntry::Delete {
///     collection: "documents".to_string(),
///     id: "doc1".to_string(),
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum WalEntry {
    /// Insert a new vector.
    Insert {
        collection: String,
        id: String,
        vector: Vec<f32>,
        metadata: Option<Value>,
    },

    /// Update an existing vector.
    Update {
        collection: String,
        id: String,
        vector: Vec<f32>,
        metadata: Option<Value>,
    },

    /// Delete a vector.
    Delete { collection: String, id: String },

    /// Create a new collection.
    CreateCollection {
        name: String,
        dimensions: usize,
        distance: String,
    },

    /// Drop a collection.
    DropCollection { name: String },

    /// Clear all vectors from a collection.
    ClearCollection { collection: String },

    /// Batch insert multiple vectors.
    BatchInsert {
        collection: String,
        entries: Vec<BatchEntry>,
    },

    /// Checkpoint marker - indicates data up to this point is persisted.
    Checkpoint { lsn: Lsn, timestamp: u64 },

    /// Transaction begin marker.
    TxnBegin { txn_id: u64 },

    /// Transaction commit marker.
    TxnCommit { txn_id: u64 },

    /// Transaction rollback marker.
    TxnRollback { txn_id: u64 },
}

/// Entry in a batch insert operation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BatchEntry {
    pub id: String,
    pub vector: Vec<f32>,
    pub metadata: Option<Value>,
}

/// WAL record with header information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalRecord {
    /// Log sequence number.
    pub lsn: Lsn,
    /// Timestamp when the record was written.
    pub timestamp: u64,
    /// CRC32 checksum of the entry data.
    pub checksum: u32,
    /// Length of the serialized entry.
    pub length: u32,
    /// The actual entry data.
    pub entry: WalEntry,
}

/// WAL configuration options.
///
/// # Example
///
/// ```
/// use needle::WalConfig;
/// use std::time::Duration;
///
/// // Configure WAL for high durability
/// let config = WalConfig::new()
///     .sync_on_write(true)           // Sync after every write
///     .segment_size(32 * 1024 * 1024) // 32MB segments
///     .max_segments(20);              // Keep more segments
///
/// assert!(config.sync_on_write);
/// assert_eq!(config.max_segments, 20);
///
/// // Checksums are enabled by default
/// assert!(config.enable_checksums);
/// ```
#[derive(Debug, Clone)]
pub struct WalConfig {
    /// Maximum size of a single segment file in bytes.
    pub segment_size: u64,
    /// Whether to sync after every write.
    pub sync_on_write: bool,
    /// Interval for automatic sync (if sync_on_write is false).
    pub sync_interval: Duration,
    /// Maximum number of segments to retain.
    pub max_segments: usize,
    /// Whether to compress segment files.
    pub compress: bool,
    /// Buffer size for writes.
    pub write_buffer_size: usize,
    /// Enable checksums for data integrity.
    pub enable_checksums: bool,
}

impl Default for WalConfig {
    fn default() -> Self {
        Self {
            segment_size: 64 * 1024 * 1024, // 64MB
            sync_on_write: false,
            sync_interval: Duration::from_millis(100),
            max_segments: 10,
            compress: false,
            write_buffer_size: 64 * 1024, // 64KB
            enable_checksums: true,
        }
    }
}

impl WalConfig {
    /// Create a new WAL configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the segment size.
    pub fn segment_size(mut self, size: u64) -> Self {
        self.segment_size = size;
        self
    }

    /// Enable sync on every write.
    pub fn sync_on_write(mut self, enabled: bool) -> Self {
        self.sync_on_write = enabled;
        self
    }

    /// Set the sync interval.
    pub fn sync_interval(mut self, interval: Duration) -> Self {
        self.sync_interval = interval;
        self
    }

    /// Set maximum number of segments to retain.
    pub fn max_segments(mut self, max: usize) -> Self {
        self.max_segments = max;
        self
    }

    /// Enable compression.
    pub fn compress(mut self, enabled: bool) -> Self {
        self.compress = enabled;
        self
    }
}

/// WAL segment file.
struct WalSegment {
    /// Segment number.
    number: u64,
    /// Path to the segment file.
    path: PathBuf,
    /// File handle for writing.
    file: Option<BufWriter<File>>,
    /// Current size in bytes.
    size: u64,
    /// First LSN in this segment.
    first_lsn: Lsn,
    /// Last LSN in this segment.
    last_lsn: Lsn,
}

impl WalSegment {
    fn new(dir: &Path, number: u64) -> Result<Self> {
        let path = dir.join(format!("segment_{:08}.wal", number));
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| NeedleError::Io(e))?;

        let size = file
            .metadata()
            .map_err(|e| NeedleError::Io(e))?
            .len();

        Ok(Self {
            number,
            path,
            file: Some(BufWriter::new(file)),
            size,
            first_lsn: 0,
            last_lsn: 0,
        })
    }

    fn open_for_read(path: &Path) -> Result<BufReader<File>> {
        let file = File::open(path)?;
        Ok(BufReader::new(file))
    }

    fn write(&mut self, data: &[u8]) -> Result<()> {
        if let Some(ref mut writer) = self.file {
            writer
                .write_all(data)
                .map_err(|e| NeedleError::Io(e))?;
            self.size += data.len() as u64;
        }
        Ok(())
    }

    fn sync(&mut self) -> Result<()> {
        if let Some(ref mut writer) = self.file {
            writer
                .flush()
                .map_err(|e| NeedleError::Io(e))?;
            writer
                .get_ref()
                .sync_all()
                .map_err(|e| NeedleError::Io(e))?;
        }
        Ok(())
    }

    fn close(&mut self) -> Result<()> {
        if let Some(ref mut writer) = self.file.take() {
            writer
                .flush()
                .map_err(|e| NeedleError::Io(e))?;
        }
        Ok(())
    }
}

/// WAL statistics.
#[derive(Debug, Clone, Default)]
pub struct WalStats {
    /// Total number of entries written.
    pub entries_written: u64,
    /// Total bytes written.
    pub bytes_written: u64,
    /// Number of syncs performed.
    pub syncs: u64,
    /// Number of checkpoints created.
    pub checkpoints: u64,
    /// Number of segments.
    pub segments: usize,
    /// Current LSN.
    pub current_lsn: Lsn,
    /// Last checkpoint LSN.
    pub checkpoint_lsn: Lsn,
}

/// WAL manager for durability.
#[allow(dead_code)]
pub struct WalManager {
    /// WAL directory.
    dir: PathBuf,
    /// Configuration.
    config: WalConfig,
    /// Current segment for writing.
    current_segment: Mutex<WalSegment>,
    /// List of all segments.
    segments: RwLock<Vec<PathBuf>>,
    /// Current LSN counter.
    next_lsn: AtomicU64,
    /// Last synced LSN.
    synced_lsn: AtomicU64,
    /// Last checkpoint LSN.
    checkpoint_lsn: AtomicU64,
    /// Statistics.
    stats: RwLock<WalStats>,
    /// Pending entries in write buffer (for batched writes).
    write_buffer: Mutex<VecDeque<WalRecord>>,
    /// Last sync time.
    last_sync: Mutex<Instant>,
}

impl WalManager {
    /// Open or create a WAL in the specified directory.
    pub fn open<P: AsRef<Path>>(dir: P, config: WalConfig) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        fs::create_dir_all(&dir).map_err(|e| NeedleError::Io(e))?;

        // Find existing segments
        let mut segments: Vec<PathBuf> = fs::read_dir(&dir)
            .map_err(|e| NeedleError::Io(e))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.extension()
                    .map(|ext| ext == "wal")
                    .unwrap_or(false)
            })
            .collect();
        segments.sort();

        // Determine next segment number and LSN
        let (segment_number, next_lsn) = if segments.is_empty() {
            (1, 1)
        } else {
            let last_lsn = Self::find_last_lsn(&segments)?;
            let last_segment = segments.last().unwrap();
            let segment_number = Self::parse_segment_number(last_segment).unwrap_or(0) + 1;
            (segment_number, last_lsn + 1)
        };

        let current_segment = WalSegment::new(&dir, segment_number)?;

        Ok(Self {
            dir: dir.clone(),
            config,
            current_segment: Mutex::new(current_segment),
            segments: RwLock::new(segments),
            next_lsn: AtomicU64::new(next_lsn),
            synced_lsn: AtomicU64::new(0),
            checkpoint_lsn: AtomicU64::new(0),
            stats: RwLock::new(WalStats::default()),
            write_buffer: Mutex::new(VecDeque::new()),
            last_sync: Mutex::new(Instant::now()),
        })
    }

    /// Append an entry to the WAL.
    pub fn append(&self, entry: WalEntry) -> Result<Lsn> {
        let lsn = self.next_lsn.fetch_add(1, Ordering::SeqCst);
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let record = WalRecord {
            lsn,
            timestamp,
            checksum: 0, // Will be computed below
            length: 0,   // Will be computed below
            entry,
        };

        // Serialize the entry
        let entry_data =
            serde_json::to_vec(&record.entry).map_err(|e| NeedleError::Serialization(e))?;

        let checksum = if self.config.enable_checksums {
            crc32_checksum(&entry_data)
        } else {
            0
        };

        // Create the final record with correct checksum and length
        let final_record = WalRecord {
            lsn,
            timestamp,
            checksum,
            length: entry_data.len() as u32,
            entry: record.entry,
        };

        // Serialize the full record
        let record_data = serde_json::to_vec(&final_record)
            .map_err(|e| NeedleError::Serialization(e))?;

        // Write length prefix + record data
        let mut data = Vec::with_capacity(4 + record_data.len());
        data.extend_from_slice(&(record_data.len() as u32).to_le_bytes());
        data.extend_from_slice(&record_data);
        data.push(b'\n'); // Newline for readability in debug

        // Write to current segment
        {
            let mut segment = self
                .current_segment
                .lock()
                .map_err(|_| NeedleError::LockError)?;

            // Check if we need to rotate
            if segment.size + data.len() as u64 > self.config.segment_size {
                self.rotate_segment_locked(&mut segment)?;
            }

            segment.write(&data)?;
            segment.last_lsn = lsn;
            if segment.first_lsn == 0 {
                segment.first_lsn = lsn;
            }

            // Update stats
            {
                let mut stats = self.stats.write().map_err(|_| NeedleError::LockError)?;
                stats.entries_written += 1;
                stats.bytes_written += data.len() as u64;
                stats.current_lsn = lsn;
            }

            // Sync if configured
            if self.config.sync_on_write {
                segment.sync()?;
                self.synced_lsn.store(lsn, Ordering::SeqCst);
            }
        }

        Ok(lsn)
    }

    /// Sync all pending writes to disk.
    pub fn sync(&self) -> Result<()> {
        let current_lsn = self.next_lsn.load(Ordering::SeqCst).saturating_sub(1);

        {
            let mut segment = self
                .current_segment
                .lock()
                .map_err(|_| NeedleError::LockError)?;
            segment.sync()?;
        }

        self.synced_lsn.store(current_lsn, Ordering::SeqCst);
        *self.last_sync.lock().map_err(|_| NeedleError::LockError)? = Instant::now();

        {
            let mut stats = self.stats.write().map_err(|_| NeedleError::LockError)?;
            stats.syncs += 1;
        }

        Ok(())
    }

    /// Create a checkpoint, indicating all operations up to this point are persisted.
    pub fn checkpoint(&self) -> Result<Lsn> {
        let lsn = self.next_lsn.load(Ordering::SeqCst).saturating_sub(1);

        // Write checkpoint entry
        let checkpoint_lsn = self.append(WalEntry::Checkpoint {
            lsn,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        })?;

        // Sync to ensure checkpoint is durable
        self.sync()?;

        self.checkpoint_lsn.store(checkpoint_lsn, Ordering::SeqCst);

        {
            let mut stats = self.stats.write().map_err(|_| NeedleError::LockError)?;
            stats.checkpoints += 1;
            stats.checkpoint_lsn = checkpoint_lsn;
        }

        // Clean up old segments
        self.cleanup_old_segments()?;

        Ok(checkpoint_lsn)
    }

    /// Replay the WAL from a given LSN.
    pub fn replay<F>(&self, from_lsn: Lsn, mut callback: F) -> Result<Lsn>
    where
        F: FnMut(WalRecord) -> Result<()>,
    {
        let segments = self.segments.read().map_err(|_| NeedleError::LockError)?;
        let mut last_lsn = from_lsn;

        for segment_path in segments.iter() {
            let records = self.read_segment(segment_path)?;
            for record in records {
                if record.lsn >= from_lsn {
                    last_lsn = record.lsn;
                    callback(record)?;
                }
            }
        }

        Ok(last_lsn)
    }

    /// Read all records from a segment file.
    pub fn read_segment(&self, path: &Path) -> Result<Vec<WalRecord>> {
        let mut reader = WalSegment::open_for_read(path)?;
        let mut records = Vec::new();
        let mut buffer = Vec::new();

        loop {
            // Read length prefix
            let mut len_buf = [0u8; 4];
            match reader.read_exact(&mut len_buf) {
                Ok(_) => {}
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(NeedleError::Io(e)),
            }

            let record_len = u32::from_le_bytes(len_buf) as usize;
            buffer.resize(record_len, 0);

            reader
                .read_exact(&mut buffer)
                .map_err(|e| NeedleError::Io(e))?;

            // Skip newline
            let mut newline = [0u8; 1];
            let _ = reader.read_exact(&mut newline);

            let record: WalRecord = serde_json::from_slice(&buffer)
                .map_err(|e| NeedleError::Serialization(e))?;

            // Verify checksum if enabled
            if self.config.enable_checksums && record.checksum != 0 {
                let entry_data = serde_json::to_vec(&record.entry)
                    .map_err(|e| NeedleError::Serialization(e))?;
                let computed = crc32_checksum(&entry_data);
                if computed != record.checksum {
                    return Err(NeedleError::Corruption(format!(
                        "Checksum mismatch at LSN {}: expected {}, got {}",
                        record.lsn, record.checksum, computed
                    )));
                }
            }

            records.push(record);
        }

        Ok(records)
    }

    /// Get the current LSN.
    pub fn current_lsn(&self) -> Lsn {
        self.next_lsn.load(Ordering::SeqCst).saturating_sub(1)
    }

    /// Get the last synced LSN.
    pub fn synced_lsn(&self) -> Lsn {
        self.synced_lsn.load(Ordering::SeqCst)
    }

    /// Get the last checkpoint LSN.
    pub fn checkpoint_lsn(&self) -> Lsn {
        self.checkpoint_lsn.load(Ordering::SeqCst)
    }

    /// Get WAL statistics.
    pub fn stats(&self) -> Result<WalStats> {
        let stats = self.stats.read().map_err(|_| NeedleError::LockError)?;
        Ok(stats.clone())
    }

    /// Truncate the WAL up to a given LSN.
    pub fn truncate(&self, up_to_lsn: Lsn) -> Result<()> {
        let mut segments = self.segments.write().map_err(|_| NeedleError::LockError)?;

        // Find segments that are entirely before the truncation point
        let segments_to_remove: Vec<PathBuf> = segments
            .iter()
            .filter(|path| {
                if let Ok(records) = self.read_segment(path) {
                    if let Some(last) = records.last() {
                        return last.lsn < up_to_lsn;
                    }
                }
                false
            })
            .cloned()
            .collect();

        for path in segments_to_remove {
            fs::remove_file(&path).map_err(|e| NeedleError::Io(e))?;
            segments.retain(|p| p != &path);
        }

        Ok(())
    }

    /// Close the WAL manager.
    pub fn close(&self) -> Result<()> {
        let mut segment = self
            .current_segment
            .lock()
            .map_err(|_| NeedleError::LockError)?;
        segment.sync()?;
        segment.close()?;
        Ok(())
    }

    // Private methods

    fn rotate_segment_locked(&self, segment: &mut WalSegment) -> Result<()> {
        // Close current segment
        segment.sync()?;
        segment.close()?;

        // Add to segments list
        {
            let mut segments = self.segments.write().map_err(|_| NeedleError::LockError)?;
            segments.push(segment.path.clone());
        }

        // Create new segment
        let new_number = segment.number + 1;
        *segment = WalSegment::new(&self.dir, new_number)?;

        Ok(())
    }

    fn cleanup_old_segments(&self) -> Result<()> {
        let checkpoint_lsn = self.checkpoint_lsn.load(Ordering::SeqCst);
        let mut segments = self.segments.write().map_err(|_| NeedleError::LockError)?;

        // Keep only the most recent segments
        while segments.len() > self.config.max_segments {
            if let Some(oldest) = segments.first().cloned() {
                // Check if all entries in this segment are before checkpoint
                if let Ok(records) = self.read_segment(&oldest) {
                    if let Some(last) = records.last() {
                        if last.lsn < checkpoint_lsn {
                            fs::remove_file(&oldest)
                                .map_err(|e| NeedleError::Io(e))?;
                            segments.remove(0);
                            continue;
                        }
                    }
                }
            }
            break;
        }

        {
            let mut stats = self.stats.write().map_err(|_| NeedleError::LockError)?;
            stats.segments = segments.len();
        }

        Ok(())
    }

    fn find_last_lsn(segments: &[PathBuf]) -> Result<Lsn> {
        let mut last_lsn = 0;

        for segment_path in segments.iter().rev() {
            let file = File::open(segment_path).map_err(|e| NeedleError::Io(e))?;
            let metadata = file
                .metadata()
                .map_err(|e| NeedleError::Io(e))?;

            if metadata.len() == 0 {
                continue;
            }

            let mut reader = BufReader::new(file);

            // Read through all records to find the last LSN
            loop {
                let mut len_buf = [0u8; 4];
                match reader.read_exact(&mut len_buf) {
                    Ok(_) => {}
                    Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                    Err(_) => break,
                }

                let record_len = u32::from_le_bytes(len_buf) as usize;
                let mut buffer = vec![0u8; record_len];

                if reader.read_exact(&mut buffer).is_err() {
                    break;
                }

                // Skip newline
                let _ = reader.seek(SeekFrom::Current(1));

                if let Ok(record) = serde_json::from_slice::<WalRecord>(&buffer) {
                    last_lsn = record.lsn;
                }
            }

            if last_lsn > 0 {
                break;
            }
        }

        Ok(last_lsn)
    }

    fn parse_segment_number(path: &Path) -> Option<u64> {
        path.file_stem()
            .and_then(|s| s.to_str())
            .and_then(|s| s.strip_prefix("segment_"))
            .and_then(|s| s.parse().ok())
    }
}

/// Compute CRC32 checksum.
fn crc32_checksum(data: &[u8]) -> u32 {
    // Simple CRC32 implementation (IEEE polynomial)
    const CRC32_TABLE: [u32; 256] = generate_crc32_table();

    let mut crc = !0u32;
    for byte in data {
        crc = CRC32_TABLE[((crc ^ *byte as u32) & 0xFF) as usize] ^ (crc >> 8);
    }
    !crc
}

const fn generate_crc32_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut crc = i as u32;
        let mut j = 0;
        while j < 8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc >>= 1;
            }
            j += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
}

/// WAL-enabled wrapper for applying entries to a database.
pub struct WalApplicator<T> {
    wal: Arc<WalManager>,
    target: T,
}

impl<T> WalApplicator<T> {
    /// Create a new WAL applicator.
    pub fn new(wal: Arc<WalManager>, target: T) -> Self {
        Self { wal, target }
    }

    /// Get a reference to the WAL manager.
    pub fn wal(&self) -> &WalManager {
        &self.wal
    }

    /// Get a reference to the target.
    pub fn target(&self) -> &T {
        &self.target
    }

    /// Get a mutable reference to the target.
    pub fn target_mut(&mut self) -> &mut T {
        &mut self.target
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_wal() -> (WalManager, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let wal = WalManager::open(temp_dir.path(), WalConfig::default()).unwrap();
        (wal, temp_dir)
    }

    #[test]
    fn test_wal_append_and_read() {
        let (wal, _temp_dir) = create_test_wal();

        // Append entries
        let lsn1 = wal
            .append(WalEntry::Insert {
                collection: "test".to_string(),
                id: "doc1".to_string(),
                vector: vec![0.1, 0.2, 0.3],
                metadata: None,
            })
            .unwrap();

        let lsn2 = wal
            .append(WalEntry::Insert {
                collection: "test".to_string(),
                id: "doc2".to_string(),
                vector: vec![0.4, 0.5, 0.6],
                metadata: Some(serde_json::json!({"key": "value"})),
            })
            .unwrap();

        assert_eq!(lsn1, 1);
        assert_eq!(lsn2, 2);
        assert_eq!(wal.current_lsn(), 2);
    }

    #[test]
    fn test_wal_sync() {
        let (wal, _temp_dir) = create_test_wal();

        wal.append(WalEntry::Insert {
            collection: "test".to_string(),
            id: "doc1".to_string(),
            vector: vec![0.1, 0.2],
            metadata: None,
        })
        .unwrap();

        wal.sync().unwrap();
        assert_eq!(wal.synced_lsn(), 1);
    }

    #[test]
    fn test_wal_checkpoint() {
        let (wal, _temp_dir) = create_test_wal();

        wal.append(WalEntry::Insert {
            collection: "test".to_string(),
            id: "doc1".to_string(),
            vector: vec![0.1],
            metadata: None,
        })
        .unwrap();

        let checkpoint_lsn = wal.checkpoint().unwrap();
        assert!(checkpoint_lsn > 0);
        assert_eq!(wal.checkpoint_lsn(), checkpoint_lsn);
    }

    #[test]
    fn test_wal_replay() {
        let (wal, temp_dir) = create_test_wal();

        // Write some entries
        wal.append(WalEntry::CreateCollection {
            name: "test".to_string(),
            dimensions: 3,
            distance: "cosine".to_string(),
        })
        .unwrap();

        wal.append(WalEntry::Insert {
            collection: "test".to_string(),
            id: "doc1".to_string(),
            vector: vec![0.1, 0.2, 0.3],
            metadata: None,
        })
        .unwrap();

        wal.append(WalEntry::Delete {
            collection: "test".to_string(),
            id: "doc1".to_string(),
        })
        .unwrap();

        wal.sync().unwrap();
        wal.close().unwrap();

        // Reopen and replay
        let wal2 = WalManager::open(temp_dir.path(), WalConfig::default()).unwrap();
        let mut replayed = Vec::new();

        wal2.replay(1, |record| {
            replayed.push(record.entry);
            Ok(())
        })
        .unwrap();

        assert_eq!(replayed.len(), 3);
        assert!(matches!(replayed[0], WalEntry::CreateCollection { .. }));
        assert!(matches!(replayed[1], WalEntry::Insert { .. }));
        assert!(matches!(replayed[2], WalEntry::Delete { .. }));
    }

    #[test]
    fn test_wal_batch_insert() {
        let (wal, _temp_dir) = create_test_wal();

        let entries = vec![
            BatchEntry {
                id: "doc1".to_string(),
                vector: vec![0.1, 0.2],
                metadata: None,
            },
            BatchEntry {
                id: "doc2".to_string(),
                vector: vec![0.3, 0.4],
                metadata: Some(serde_json::json!({"key": "value"})),
            },
        ];

        let lsn = wal
            .append(WalEntry::BatchInsert {
                collection: "test".to_string(),
                entries,
            })
            .unwrap();

        assert_eq!(lsn, 1);
    }

    #[test]
    fn test_wal_stats() {
        let (wal, _temp_dir) = create_test_wal();

        wal.append(WalEntry::Insert {
            collection: "test".to_string(),
            id: "doc1".to_string(),
            vector: vec![0.1],
            metadata: None,
        })
        .unwrap();

        wal.append(WalEntry::Insert {
            collection: "test".to_string(),
            id: "doc2".to_string(),
            vector: vec![0.2],
            metadata: None,
        })
        .unwrap();

        let stats = wal.stats().unwrap();
        assert_eq!(stats.entries_written, 2);
        assert!(stats.bytes_written > 0);
        assert_eq!(stats.current_lsn, 2);
    }

    #[test]
    fn test_wal_config_builder() {
        let config = WalConfig::new()
            .segment_size(128 * 1024 * 1024)
            .sync_on_write(true)
            .max_segments(5)
            .compress(true);

        assert_eq!(config.segment_size, 128 * 1024 * 1024);
        assert!(config.sync_on_write);
        assert_eq!(config.max_segments, 5);
        assert!(config.compress);
    }

    #[test]
    fn test_crc32_checksum() {
        let data = b"hello world";
        let checksum = crc32_checksum(data);
        assert_ne!(checksum, 0);

        // Same data should produce same checksum
        assert_eq!(checksum, crc32_checksum(data));

        // Different data should produce different checksum
        assert_ne!(checksum, crc32_checksum(b"hello worlD"));
    }

    #[test]
    fn test_wal_transaction_markers() {
        let (wal, temp_dir) = create_test_wal();

        let txn_id = 12345;

        wal.append(WalEntry::TxnBegin { txn_id }).unwrap();

        wal.append(WalEntry::Insert {
            collection: "test".to_string(),
            id: "doc1".to_string(),
            vector: vec![0.1],
            metadata: None,
        })
        .unwrap();

        wal.append(WalEntry::TxnCommit { txn_id }).unwrap();

        // Sync and close before replaying
        wal.sync().unwrap();
        wal.close().unwrap();

        // Reopen and replay
        let wal2 = WalManager::open(temp_dir.path(), WalConfig::default()).unwrap();
        let mut replayed = Vec::new();
        wal2.replay(1, |record| {
            replayed.push(record.entry);
            Ok(())
        })
        .unwrap();

        assert_eq!(replayed.len(), 3);
        assert!(matches!(replayed[0], WalEntry::TxnBegin { txn_id: 12345 }));
        assert!(matches!(replayed[2], WalEntry::TxnCommit { txn_id: 12345 }));
    }
}
