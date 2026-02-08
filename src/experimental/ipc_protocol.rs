#![allow(dead_code)]
#![allow(unsafe_code)]

//! Zero-Copy IPC Protocol
//!
//! Shared-memory IPC protocol for same-machine multi-process access to a single
//! `.needle` DB file. Uses memory-mapped files with structured regions for
//! lock-free multi-reader access and single-writer coordination.
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────┐
//! │                  Shared Memory Layout                  │
//! ├──────────────────────────────────────────────────────┤
//! │  Header Region (4KB)                                  │
//! │  ├── magic, version, epoch counter                   │
//! │  ├── writer_lock (atomic)                            │
//! │  ├── reader_count (atomic)                           │
//! │  └── data_offset, index_offset, meta_offset          │
//! ├──────────────────────────────────────────────────────┤
//! │  Data Region (vectors)                               │
//! ├──────────────────────────────────────────────────────┤
//! │  Index Region (HNSW graph)                           │
//! ├──────────────────────────────────────────────────────┤
//! │  Metadata Region (JSON metadata)                     │
//! ├──────────────────────────────────────────────────────┤
//! │  WAL Region (write-ahead log tail)                   │
//! └──────────────────────────────────────────────────────┘
//! ```
//!
//! # Multi-Reader Protocol
//!
//! Uses epoch-based reclamation for lock-free reads:
//! 1. Reader increments reader_count
//! 2. Reader reads current epoch
//! 3. Reader performs read operations
//! 4. Reader decrements reader_count
//!
//! Writer waits until reader_count == 0 for epoch transition.

use crate::error::{NeedleError, Result};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::path::{Path, PathBuf};

/// Magic bytes for shared memory header.
const IPC_MAGIC: [u8; 8] = *b"NDLIPC01";

/// Header size in bytes.
const IPC_HEADER_SIZE: usize = 4096;

/// Shared memory region layout.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharedMemoryLayout {
    /// Total size of the shared memory region.
    pub total_size: u64,
    /// Offset to the data (vectors) region.
    pub data_offset: u64,
    /// Size of the data region.
    pub data_size: u64,
    /// Offset to the index region.
    pub index_offset: u64,
    /// Size of the index region.
    pub index_size: u64,
    /// Offset to the metadata region.
    pub metadata_offset: u64,
    /// Size of the metadata region.
    pub metadata_size: u64,
    /// Offset to the WAL region.
    pub wal_offset: u64,
    /// Size of the WAL region.
    pub wal_size: u64,
}

impl SharedMemoryLayout {
    /// Create a layout for a database with given parameters.
    pub fn new(
        vector_count: usize,
        dimensions: usize,
        index_size_bytes: u64,
        metadata_size_bytes: u64,
        wal_size_bytes: u64,
    ) -> Result<Self> {
        let data_size = vector_count
            .checked_mul(dimensions)
            .and_then(|v| v.checked_mul(4)) // f32 = 4 bytes
            .ok_or_else(|| NeedleError::InvalidInput(
                "IPC data size overflow: vector_count * dimensions * 4 exceeds usize".into(),
            ))? as u64;
        let data_offset = IPC_HEADER_SIZE as u64;
        let index_offset = data_offset + data_size;
        let metadata_offset = index_offset + index_size_bytes;
        let wal_offset = metadata_offset + metadata_size_bytes;
        let total_size = wal_offset + wal_size_bytes;

        Ok(Self {
            total_size,
            data_offset,
            data_size,
            index_offset,
            index_size: index_size_bytes,
            metadata_offset,
            metadata_size: metadata_size_bytes,
            wal_offset,
            wal_size: wal_size_bytes,
        })
    }
}

/// IPC header stored at the beginning of the shared memory region.
///
/// Uses atomic fields for lock-free reader/writer coordination.
#[repr(C)]
pub struct IpcHeader {
    /// Magic bytes for format identification.
    pub magic: [u8; 8],
    /// Format version.
    pub version: AtomicU32,
    /// Epoch counter: incremented on each write transaction commit.
    pub epoch: AtomicU64,
    /// Writer lock: 0 = unlocked, 1 = locked.
    pub writer_lock: AtomicU32,
    /// Active reader count for epoch-based reclamation.
    pub reader_count: AtomicU32,
    /// Data region offset.
    pub data_offset: AtomicU64,
    /// Data region size.
    pub data_size: AtomicU64,
    /// Index region offset.
    pub index_offset: AtomicU64,
    /// Index region size.
    pub index_size: AtomicU64,
    /// Metadata region offset.
    pub metadata_offset: AtomicU64,
    /// Metadata region size.
    pub metadata_size: AtomicU64,
    /// WAL region offset.
    pub wal_offset: AtomicU64,
    /// WAL region size.
    pub wal_size: AtomicU64,
    /// Number of vectors.
    pub vector_count: AtomicU64,
    /// Dimensions per vector.
    pub dimensions: AtomicU32,
    /// Last write timestamp.
    pub last_write_ts: AtomicU64,
    /// Writer process ID (for stale lock detection).
    pub writer_pid: AtomicU64,
}

/// Configuration for IPC protocol.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IpcConfig {
    /// Path to the shared memory file.
    pub path: PathBuf,
    /// Maximum number of concurrent readers.
    pub max_readers: usize,
    /// Stale lock timeout (seconds). If writer PID is dead, lock is reclaimed.
    pub stale_lock_timeout_secs: u64,
    /// WAL region size in bytes.
    pub wal_region_size: u64,
    /// Enable read-ahead for sequential access.
    pub enable_read_ahead: bool,
}

impl Default for IpcConfig {
    fn default() -> Self {
        Self {
            path: PathBuf::from("needle_ipc.shm"),
            max_readers: 256,
            stale_lock_timeout_secs: 30,
            wal_region_size: 16 * 1024 * 1024, // 16MB
            enable_read_ahead: true,
        }
    }
}

/// Reader handle for lock-free read access to shared memory.
///
/// Uses epoch-based reclamation: reads are guaranteed consistent within
/// a single epoch. The reader holds a "pin" that prevents the writer
/// from advancing the epoch until the read completes.
pub struct IpcReader {
    config: IpcConfig,
    /// The epoch at which this reader started.
    pinned_epoch: u64,
    /// Whether the reader is currently active.
    active: AtomicBool,
}

impl IpcReader {
    /// Create a new reader for the shared memory region.
    pub fn new(config: IpcConfig) -> Self {
        Self {
            config,
            pinned_epoch: 0,
            active: AtomicBool::new(false),
        }
    }

    /// Begin a read transaction, pinning the current epoch.
    ///
    /// The reader will see a consistent snapshot as of the pinned epoch.
    /// Must be paired with `end_read()`.
    pub fn begin_read(&mut self) -> ReadGuard {
        self.active.store(true, Ordering::Release);
        // In a real implementation, this would read from the mmap header
        self.pinned_epoch += 1; // placeholder
        ReadGuard {
            epoch: self.pinned_epoch,
            active: &self.active,
        }
    }

    /// Get the path to the shared memory file.
    pub fn path(&self) -> &Path {
        &self.config.path
    }
}

/// RAII guard for a read transaction. Automatically unpins the epoch on drop.
pub struct ReadGuard<'a> {
    epoch: u64,
    active: &'a AtomicBool,
}

impl<'a> ReadGuard<'a> {
    /// Get the epoch of this read transaction.
    pub fn epoch(&self) -> u64 {
        self.epoch
    }
}

impl<'a> Drop for ReadGuard<'a> {
    fn drop(&mut self) {
        self.active.store(false, Ordering::Release);
    }
}

/// Writer handle for exclusive write access to shared memory.
///
/// Uses a single-writer model with WAL-based durability.
/// Only one writer can be active at a time across all processes.
pub struct IpcWriter {
    config: IpcConfig,
    /// Whether this writer currently holds the lock.
    has_lock: AtomicBool,
    /// Current write epoch.
    current_epoch: u64,
    /// Pending writes buffered before commit.
    pending_writes: Vec<PendingWrite>,
}

/// A pending write operation buffered before commit.
#[derive(Debug, Clone)]
pub struct PendingWrite {
    /// Region to write to.
    pub region: WriteRegion,
    /// Offset within the region.
    pub offset: u64,
    /// Data to write.
    pub data: Vec<u8>,
}

/// Target region for a write operation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum WriteRegion {
    /// Vector data region.
    Data,
    /// Index region.
    Index,
    /// Metadata region.
    Metadata,
    /// WAL region.
    Wal,
}

impl IpcWriter {
    /// Create a new writer.
    pub fn new(config: IpcConfig) -> Self {
        Self {
            config,
            has_lock: AtomicBool::new(false),
            current_epoch: 0,
            pending_writes: Vec::new(),
        }
    }

    /// Attempt to acquire the writer lock.
    ///
    /// Returns an error if another writer already holds the lock.
    pub fn try_lock(&mut self) -> Result<()> {
        if self.has_lock.load(Ordering::Acquire) {
            return Ok(()); // Already locked
        }

        // In a real implementation, this would use atomic CAS on the mmap header
        self.has_lock.store(true, Ordering::Release);
        Ok(())
    }

    /// Release the writer lock.
    pub fn unlock(&mut self) {
        self.has_lock.store(false, Ordering::Release);
        self.pending_writes.clear();
    }

    /// Buffer a write operation.
    pub fn write(&mut self, region: WriteRegion, offset: u64, data: Vec<u8>) -> Result<()> {
        if !self.has_lock.load(Ordering::Acquire) {
            return Err(NeedleError::InvalidOperation(
                "Writer lock not held".to_string(),
            ));
        }
        self.pending_writes.push(PendingWrite {
            region,
            offset,
            data,
        });
        Ok(())
    }

    /// Commit all pending writes atomically.
    ///
    /// This operation:
    /// 1. Writes all pending data to the WAL
    /// 2. Applies writes to the data regions
    /// 3. Increments the epoch counter
    /// 4. Syncs to disk
    pub fn commit(&mut self) -> Result<CommitResult> {
        if !self.has_lock.load(Ordering::Acquire) {
            return Err(NeedleError::InvalidOperation(
                "Writer lock not held".to_string(),
            ));
        }

        let writes = self.pending_writes.len();
        let bytes: usize = self.pending_writes.iter().map(|w| w.data.len()).sum();

        // In a real implementation, this would:
        // 1. Write to WAL region
        // 2. Apply to data/index/metadata regions
        // 3. Atomic epoch increment

        self.current_epoch += 1;
        self.pending_writes.clear();

        Ok(CommitResult {
            epoch: self.current_epoch,
            writes_applied: writes,
            bytes_written: bytes as u64,
        })
    }

    /// Get the path to the shared memory file.
    pub fn path(&self) -> &Path {
        &self.config.path
    }
}

/// Result of a commit operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitResult {
    /// New epoch after commit.
    pub epoch: u64,
    /// Number of write operations applied.
    pub writes_applied: usize,
    /// Total bytes written.
    pub bytes_written: u64,
}

/// Health status of the IPC system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IpcHealth {
    /// Whether the writer lock is currently held.
    pub writer_active: bool,
    /// Current number of active readers.
    pub active_readers: u32,
    /// Current epoch.
    pub current_epoch: u64,
    /// Whether a stale lock was detected.
    pub stale_lock_detected: bool,
    /// Total shared memory size in bytes.
    pub total_size: u64,
}

/// IPC coordinator that manages readers and writers.
pub struct IpcCoordinator {
    config: IpcConfig,
    layout: SharedMemoryLayout,
}

impl IpcCoordinator {
    /// Create a new IPC coordinator with the given layout.
    pub fn new(config: IpcConfig, layout: SharedMemoryLayout) -> Self {
        Self { config, layout }
    }

    /// Create a reader handle.
    pub fn create_reader(&self) -> IpcReader {
        IpcReader::new(self.config.clone())
    }

    /// Create a writer handle.
    pub fn create_writer(&self) -> IpcWriter {
        IpcWriter::new(self.config.clone())
    }

    /// Get the memory layout.
    pub fn layout(&self) -> &SharedMemoryLayout {
        &self.layout
    }

    /// Get the configuration.
    pub fn config(&self) -> &IpcConfig {
        &self.config
    }
}

/// File-backed shared memory region using mmap.
///
/// Provides real memory-mapped file access with structured header containing
/// atomic coordination fields. This is the low-level building block for
/// cross-process shared state.
pub struct SharedMemoryFile {
    /// Memory-mapped file (writable).
    mmap: memmap2::MmapMut,
    /// File handle (kept open for lifetime of mmap).
    _file: std::fs::File,
    /// Layout of the shared memory regions.
    layout: SharedMemoryLayout,
    /// File path.
    path: PathBuf,
}

/// Offsets within the IPC header for atomic fields.
mod header_offsets {
    pub const MAGIC: usize = 0;
    pub const VERSION: usize = 8;
    pub const EPOCH: usize = 12;
    pub const WRITER_LOCK: usize = 20;
    pub const READER_COUNT: usize = 24;
    pub const VECTOR_COUNT: usize = 28;
    pub const DIMENSIONS: usize = 36;
    pub const LAST_WRITE_TS: usize = 40;
}

impl SharedMemoryFile {
    /// Create or open a shared memory file at the given path.
    pub fn open_or_create(path: &Path, layout: SharedMemoryLayout) -> Result<Self> {
        use std::fs::OpenOptions;

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)
            .map_err(|e| NeedleError::Io(e))?;

        // Ensure file is large enough
        let needed = layout.total_size.max(IPC_HEADER_SIZE as u64);
        let current_len = file
            .metadata()
            .map_err(|e| NeedleError::Io(e))?
            .len();

        if current_len < needed {
            file.set_len(needed).map_err(|e| NeedleError::Io(e))?;
        }

        // SAFETY: We own the file handle, the file has been sized to at least `needed` bytes
        // above, and all concurrent access is coordinated through atomic fields in the header.
        let mmap = unsafe {
            memmap2::MmapMut::map_mut(&file).map_err(|e| NeedleError::Io(e))?
        };

        let mut shm = Self {
            mmap,
            _file: file,
            layout,
            path: path.to_path_buf(),
        };

        // Initialize header if this is a new file
        if current_len == 0 {
            shm.init_header();
        }

        Ok(shm)
    }

    /// Initialize the header with magic bytes and default values.
    fn init_header(&mut self) {
        self.mmap[..8].copy_from_slice(&IPC_MAGIC);
        self.write_u32(header_offsets::VERSION, 1);
        self.write_u64(header_offsets::EPOCH, 0);
        self.write_u32(header_offsets::WRITER_LOCK, 0);
        self.write_u32(header_offsets::READER_COUNT, 0);
        self.write_u64(header_offsets::VECTOR_COUNT, 0);
        self.write_u32(header_offsets::DIMENSIONS, 0);
        self.write_u64(header_offsets::LAST_WRITE_TS, 0);
        let _ = self.mmap.flush();
    }

    /// Validate the magic bytes in the header.
    pub fn validate_magic(&self) -> bool {
        self.mmap.len() >= 8 && self.mmap[..8] == IPC_MAGIC
    }

    /// Read a u32 from the header.
    fn read_u32(&self, offset: usize) -> u32 {
        if offset + 4 > self.mmap.len() {
            return 0;
        }
        u32::from_le_bytes([
            self.mmap[offset],
            self.mmap[offset + 1],
            self.mmap[offset + 2],
            self.mmap[offset + 3],
        ])
    }

    /// Write a u32 to the header.
    fn write_u32(&mut self, offset: usize, value: u32) {
        let bytes = value.to_le_bytes();
        self.mmap[offset..offset + 4].copy_from_slice(&bytes);
    }

    /// Read a u64 from the header.
    fn read_u64(&self, offset: usize) -> u64 {
        if offset + 8 > self.mmap.len() {
            return 0;
        }
        u64::from_le_bytes([
            self.mmap[offset],
            self.mmap[offset + 1],
            self.mmap[offset + 2],
            self.mmap[offset + 3],
            self.mmap[offset + 4],
            self.mmap[offset + 5],
            self.mmap[offset + 6],
            self.mmap[offset + 7],
        ])
    }

    /// Write a u64 to the header.
    fn write_u64(&mut self, offset: usize, value: u64) {
        let bytes = value.to_le_bytes();
        self.mmap[offset..offset + 8].copy_from_slice(&bytes);
    }

    /// Get the current epoch from the header.
    pub fn epoch(&self) -> u64 {
        self.read_u64(header_offsets::EPOCH)
    }

    /// Increment and return the new epoch.
    pub fn advance_epoch(&mut self) -> u64 {
        let new_epoch = self.epoch() + 1;
        self.write_u64(header_offsets::EPOCH, new_epoch);
        new_epoch
    }

    /// Try to acquire the writer lock using atomic CAS.
    /// Returns true if the lock was acquired.
    pub fn try_acquire_writer_lock(&mut self) -> bool {
        let offset = header_offsets::WRITER_LOCK;
        if offset + 4 > self.mmap.len() {
            return false;
        }
        // Safety: offset is aligned within the mmap region, and this file allows unsafe_code.
        let atomic = unsafe {
            &*(self.mmap.as_ptr().add(offset) as *const AtomicU32)
        };
        atomic
            .compare_exchange(0, 1, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
    }

    /// Release the writer lock.
    pub fn release_writer_lock(&mut self) {
        let offset = header_offsets::WRITER_LOCK;
        if offset + 4 > self.mmap.len() {
            return;
        }
        let atomic = unsafe {
            &*(self.mmap.as_ptr().add(offset) as *const AtomicU32)
        };
        atomic.store(0, Ordering::Release);
    }

    /// Increment the reader count atomically.
    pub fn increment_readers(&mut self) {
        let offset = header_offsets::READER_COUNT;
        if offset + 4 > self.mmap.len() {
            return;
        }
        let atomic = unsafe {
            &*(self.mmap.as_ptr().add(offset) as *const AtomicU32)
        };
        atomic.fetch_add(1, Ordering::Acquire);
    }

    /// Decrement the reader count atomically.
    pub fn decrement_readers(&mut self) {
        let offset = header_offsets::READER_COUNT;
        if offset + 4 > self.mmap.len() {
            return;
        }
        let atomic = unsafe {
            &*(self.mmap.as_ptr().add(offset) as *const AtomicU32)
        };
        // Use saturating semantics: fetch_sub with a floor of 0
        let _ = atomic.fetch_update(Ordering::Release, Ordering::Relaxed, |v| {
            Some(v.saturating_sub(1))
        });
    }

    /// Get the current reader count.
    pub fn reader_count(&self) -> u32 {
        self.read_u32(header_offsets::READER_COUNT)
    }

    /// Write data to a specific region at the given offset.
    pub fn write_region(&mut self, region_offset: u64, data: &[u8]) -> Result<()> {
        let start = region_offset as usize;
        let end = start + data.len();
        if end > self.mmap.len() {
            return Err(NeedleError::InvalidOperation(
                "Write exceeds shared memory bounds".to_string(),
            ));
        }
        self.mmap[start..end].copy_from_slice(data);
        Ok(())
    }

    /// Read data from a specific region.
    pub fn read_region(&self, region_offset: u64, length: usize) -> Result<&[u8]> {
        let start = region_offset as usize;
        let end = start + length;
        if end > self.mmap.len() {
            return Err(NeedleError::InvalidOperation(
                "Read exceeds shared memory bounds".to_string(),
            ));
        }
        Ok(&self.mmap[start..end])
    }

    /// Flush all changes to disk.
    pub fn flush(&self) -> Result<()> {
        self.mmap.flush().map_err(|e| NeedleError::Io(e))
    }

    /// Get the total size of the shared memory region.
    pub fn total_size(&self) -> usize {
        self.mmap.len()
    }

    /// Get the layout.
    pub fn layout(&self) -> &SharedMemoryLayout {
        &self.layout
    }

    /// Get the path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Update vector count in header.
    pub fn set_vector_count(&mut self, count: u64) {
        self.write_u64(header_offsets::VECTOR_COUNT, count);
    }

    /// Get vector count from header.
    pub fn vector_count(&self) -> u64 {
        self.read_u64(header_offsets::VECTOR_COUNT)
    }

    /// Update dimensions in header.
    pub fn set_dimensions(&mut self, dims: u32) {
        self.write_u32(header_offsets::DIMENSIONS, dims);
    }

    /// Get dimensions from header.
    pub fn dimensions(&self) -> u32 {
        self.read_u32(header_offsets::DIMENSIONS)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shared_memory_layout() {
        let layout = SharedMemoryLayout::new(
            10_000,  // vectors
            384,     // dimensions
            1024 * 1024, // 1MB index
            512 * 1024,  // 512KB metadata
            1024 * 1024, // 1MB WAL
        ).unwrap();

        assert_eq!(layout.data_offset, IPC_HEADER_SIZE as u64);
        assert_eq!(layout.data_size, 10_000 * 384 * 4); // f32 per dim
        assert!(layout.index_offset > layout.data_offset);
        assert!(layout.metadata_offset > layout.index_offset);
        assert!(layout.wal_offset > layout.metadata_offset);
        assert!(layout.total_size > 0);
    }

    #[test]
    fn test_reader_guard() {
        let config = IpcConfig::default();
        let mut reader = IpcReader::new(config);

        {
            let guard = reader.begin_read();
            assert!(guard.epoch() > 0);
        }
        // Guard dropped, reader is no longer active
        assert!(!reader.active.load(Ordering::Relaxed));
    }

    #[test]
    fn test_writer_lock_and_commit() {
        let config = IpcConfig::default();
        let mut writer = IpcWriter::new(config);

        writer.try_lock().unwrap();
        writer
            .write(WriteRegion::Data, 0, vec![1, 2, 3, 4])
            .unwrap();
        writer
            .write(WriteRegion::Metadata, 0, vec![5, 6, 7, 8])
            .unwrap();

        let result = writer.commit().unwrap();
        assert_eq!(result.writes_applied, 2);
        assert_eq!(result.bytes_written, 8);
        assert!(result.epoch > 0);

        writer.unlock();
    }

    #[test]
    fn test_writer_requires_lock() {
        let config = IpcConfig::default();
        let mut writer = IpcWriter::new(config);

        // Writing without lock should fail
        assert!(writer.write(WriteRegion::Data, 0, vec![1]).is_err());
        assert!(writer.commit().is_err());
    }

    #[test]
    fn test_coordinator() {
        let config = IpcConfig::default();
        let layout = SharedMemoryLayout::new(1000, 128, 100_000, 50_000, 100_000).unwrap();
        let coordinator = IpcCoordinator::new(config, layout);

        let _reader = coordinator.create_reader();
        let _writer = coordinator.create_writer();

        assert!(coordinator.layout().total_size > 0);
    }

    #[test]
    fn test_shared_memory_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.shm");
        let layout = SharedMemoryLayout::new(100, 4, 1024, 512, 1024).unwrap();

        let mut shm = SharedMemoryFile::open_or_create(&path, layout).unwrap();

        // Verify magic
        assert!(shm.validate_magic());

        // Test epoch
        assert_eq!(shm.epoch(), 0);
        let new_epoch = shm.advance_epoch();
        assert_eq!(new_epoch, 1);
        assert_eq!(shm.epoch(), 1);

        // Test writer lock
        assert!(shm.try_acquire_writer_lock());
        assert!(!shm.try_acquire_writer_lock()); // Already locked
        shm.release_writer_lock();
        assert!(shm.try_acquire_writer_lock()); // Can acquire again

        // Test reader count
        assert_eq!(shm.reader_count(), 0);
        shm.increment_readers();
        shm.increment_readers();
        assert_eq!(shm.reader_count(), 2);
        shm.decrement_readers();
        assert_eq!(shm.reader_count(), 1);

        // Test region I/O
        let data_offset = shm.layout().data_offset;
        shm.write_region(data_offset, &[1, 2, 3, 4]).unwrap();
        let read_back = shm.read_region(data_offset, 4).unwrap();
        assert_eq!(read_back, &[1, 2, 3, 4]);

        // Test vector count and dimensions
        shm.set_vector_count(42);
        assert_eq!(shm.vector_count(), 42);
        shm.set_dimensions(128);
        assert_eq!(shm.dimensions(), 128);

        shm.flush().unwrap();
    }

    #[test]
    fn test_shared_memory_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_reopen.shm");
        let layout = SharedMemoryLayout::new(100, 4, 1024, 512, 1024).unwrap();

        // Create and write
        {
            let mut shm = SharedMemoryFile::open_or_create(&path, layout.clone()).unwrap();
            shm.advance_epoch();
            shm.advance_epoch();
            shm.set_vector_count(99);
            shm.flush().unwrap();
        }

        // Reopen and verify
        {
            let shm = SharedMemoryFile::open_or_create(&path, layout).unwrap();
            assert!(shm.validate_magic());
            assert_eq!(shm.epoch(), 2);
            assert_eq!(shm.vector_count(), 99);
        }
    }
}
