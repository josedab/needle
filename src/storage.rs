// Allow dead_code for this internal storage module - some methods used only in specific features
#![allow(dead_code)]

//! Storage Layer - File I/O and Memory-Mapped Storage
//!
//! Low-level storage primitives for the Needle database file format.
//! Implements single-file storage with automatic memory mapping for large files.
//!
//! # File Format
//!
//! Needle uses a custom binary format with a fixed 4KB header:
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────┐
//! │                    Header (4096 bytes)                      │
//! ├────────────────────────────────────────────────────────────┤
//! │  Magic: "NEEDLE01" (8 bytes)                                │
//! │  Version: u32                                               │
//! │  Dimensions: u32                                            │
//! │  Vector Count: u64                                          │
//! │  Index Offset: u64                                          │
//! │  Vector Offset: u64                                         │
//! │  Metadata Offset: u64                                       │
//! │  Checksum: u32 (CRC32)                                      │
//! │  State Size: u64                                            │
//! │  State Checksum: u32                                        │
//! │  [Reserved padding to 4096 bytes]                           │
//! ├────────────────────────────────────────────────────────────┤
//! │                    State Data                               │
//! │  (Serialized collections, indices, metadata)                │
//! └────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Memory Mapping
//!
//! Files larger than `MMAP_THRESHOLD` (10MB) are automatically memory-mapped
//! for efficient random access without loading the entire file into memory.
//!
//! # Constants
//!
//! - `MAGIC`: File magic number "NEEDLE01"
//! - `VERSION`: Current file format version (1)
//! - `HEADER_SIZE`: Fixed header size (4096 bytes)
//! - `MMAP_THRESHOLD`: Size threshold for memory mapping (10MB)
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::storage::{Header, MAGIC, VERSION, HEADER_SIZE};
//!
//! // Create a new header
//! let header = Header::default();
//!
//! // Serialize to bytes
//! let bytes = header.to_bytes();
//! assert_eq!(&bytes[..8], MAGIC);
//!
//! // Parse from bytes
//! let parsed = Header::from_bytes(&bytes)?;
//! assert_eq!(parsed.version, VERSION);
//! ```
//!
//! # Integrity Checks
//!
//! The storage layer includes CRC32 checksums for both the header and state data
//! to detect corruption during reads. Invalid checksums result in a
//! `NeedleError::CorruptedDatabase` error.

use crate::error::{NeedleError, Result};
use memmap2::{Mmap, MmapMut};
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;
use tracing::{debug, info, warn};

/// Magic number for Needle database files
pub const MAGIC: &[u8; 8] = b"NEEDLE01";

/// Current file format version
pub const VERSION: u32 = 1;

/// Header size in bytes
pub const HEADER_SIZE: usize = 4096;

/// Threshold for using memory-mapped I/O (10MB)
pub const MMAP_THRESHOLD: u64 = 10 * 1024 * 1024;

/// Database file header
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Header {
    /// File format version
    pub version: u32,
    /// Default vector dimensions (0 if not set)
    pub dimensions: u32,
    /// Total number of vectors
    pub vector_count: u64,
    /// Offset to index data
    pub index_offset: u64,
    /// Offset to vector data
    pub vector_offset: u64,
    /// Offset to metadata
    pub metadata_offset: u64,
    /// CRC32 checksum of header fields
    pub checksum: u32,
    /// Size of state data in bytes (for integrity check)
    pub state_size: u64,
    /// CRC32 checksum of state data (for integrity check)
    pub state_checksum: u32,
}

impl Default for Header {
    fn default() -> Self {
        Self {
            version: VERSION,
            dimensions: 0,
            vector_count: 0,
            index_offset: HEADER_SIZE as u64,
            vector_offset: 0,
            metadata_offset: 0,
            checksum: 0,
            state_size: 0,
            state_checksum: 0,
        }
    }
}

impl Header {
    /// Serialize header to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(HEADER_SIZE);

        // Magic number
        bytes.extend_from_slice(MAGIC);

        // Version
        bytes.extend_from_slice(&self.version.to_le_bytes());

        // Dimensions
        bytes.extend_from_slice(&self.dimensions.to_le_bytes());

        // Vector count
        bytes.extend_from_slice(&self.vector_count.to_le_bytes());

        // Index offset
        bytes.extend_from_slice(&self.index_offset.to_le_bytes());

        // Vector offset
        bytes.extend_from_slice(&self.vector_offset.to_le_bytes());

        // Metadata offset
        bytes.extend_from_slice(&self.metadata_offset.to_le_bytes());

        // Compute header checksum (over first 48 bytes)
        let checksum = crc32(&bytes);
        bytes.extend_from_slice(&checksum.to_le_bytes());

        // State data integrity fields
        bytes.extend_from_slice(&self.state_size.to_le_bytes());
        bytes.extend_from_slice(&self.state_checksum.to_le_bytes());

        // Pad to HEADER_SIZE
        bytes.resize(HEADER_SIZE, 0);

        bytes
    }

    /// Deserialize header from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 64 {
            return Err(NeedleError::InvalidDatabase("Header too short".into()));
        }

        // Check magic
        if &bytes[0..8] != MAGIC {
            return Err(NeedleError::InvalidDatabase("Invalid magic number".into()));
        }

        // These conversions are infallible given the length check above
        let version = u32::from_le_bytes(
            bytes[8..12]
                .try_into()
                .map_err(|_| NeedleError::Corruption("Invalid header: version bytes".into()))?,
        );
        let dimensions = u32::from_le_bytes(
            bytes[12..16]
                .try_into()
                .map_err(|_| NeedleError::Corruption("Invalid header: dimensions bytes".into()))?,
        );
        let vector_count = u64::from_le_bytes(
            bytes[16..24]
                .try_into()
                .map_err(|_| NeedleError::Corruption("Invalid header: vector_count bytes".into()))?,
        );
        let index_offset = u64::from_le_bytes(
            bytes[24..32]
                .try_into()
                .map_err(|_| NeedleError::Corruption("Invalid header: index_offset bytes".into()))?,
        );
        let vector_offset = u64::from_le_bytes(
            bytes[32..40]
                .try_into()
                .map_err(|_| NeedleError::Corruption("Invalid header: vector_offset bytes".into()))?,
        );
        let metadata_offset = u64::from_le_bytes(
            bytes[40..48]
                .try_into()
                .map_err(|_| NeedleError::Corruption("Invalid header: metadata_offset bytes".into()))?,
        );
        let stored_checksum = u32::from_le_bytes(
            bytes[48..52]
                .try_into()
                .map_err(|_| NeedleError::Corruption("Invalid header: checksum bytes".into()))?,
        );

        // Verify header checksum (covers bytes 0-47)
        let computed_checksum = crc32(&bytes[0..48]);
        if stored_checksum != computed_checksum {
            return Err(NeedleError::Corruption("Header checksum mismatch".into()));
        }

        // Read state integrity fields (bytes 52-63)
        // These may be 0 for older database files (backwards compatible)
        let state_size = u64::from_le_bytes(
            bytes[52..60]
                .try_into()
                .map_err(|_| NeedleError::Corruption("Invalid header: state_size bytes".into()))?,
        );
        let state_checksum = u32::from_le_bytes(
            bytes[60..64]
                .try_into()
                .map_err(|_| NeedleError::Corruption("Invalid header: state_checksum bytes".into()))?,
        );

        Ok(Self {
            version,
            dimensions,
            vector_count,
            index_offset,
            vector_offset,
            metadata_offset,
            checksum: stored_checksum,
            state_size,
            state_checksum,
        })
    }
}

/// CRC32 lookup table (precomputed for 8-16x faster checksums)
const CRC32_TABLE: [u32; 256] = [
    0x00000000, 0x77073096, 0xEE0E612C, 0x990951BA, 0x076DC419, 0x706AF48F, 0xE963A535, 0x9E6495A3,
    0x0EDB8832, 0x79DCB8A4, 0xE0D5E91E, 0x97D2D988, 0x09B64C2B, 0x7EB17CBD, 0xE7B82D07, 0x90BF1D91,
    0x1DB71064, 0x6AB020F2, 0xF3B97148, 0x84BE41DE, 0x1ADAD47D, 0x6DDDE4EB, 0xF4D4B551, 0x83D385C7,
    0x136C9856, 0x646BA8C0, 0xFD62F97A, 0x8A65C9EC, 0x14015C4F, 0x63066CD9, 0xFA0F3D63, 0x8D080DF5,
    0x3B6E20C8, 0x4C69105E, 0xD56041E4, 0xA2677172, 0x3C03E4D1, 0x4B04D447, 0xD20D85FD, 0xA50AB56B,
    0x35B5A8FA, 0x42B2986C, 0xDBBBC9D6, 0xACBCF940, 0x32D86CE3, 0x45DF5C75, 0xDCD60DCF, 0xABD13D59,
    0x26D930AC, 0x51DE003A, 0xC8D75180, 0xBFD06116, 0x21B4F4B5, 0x56B3C423, 0xCFBA9599, 0xB8BDA50F,
    0x2802B89E, 0x5F058808, 0xC60CD9B2, 0xB10BE924, 0x2F6F7C87, 0x58684C11, 0xC1611DAB, 0xB6662D3D,
    0x76DC4190, 0x01DB7106, 0x98D220BC, 0xEFD5102A, 0x71B18589, 0x06B6B51F, 0x9FBFE4A5, 0xE8B8D433,
    0x7807C9A2, 0x0F00F934, 0x9609A88E, 0xE10E9818, 0x7F6A0DBB, 0x086D3D2D, 0x91646C97, 0xE6635C01,
    0x6B6B51F4, 0x1C6C6162, 0x856530D8, 0xF262004E, 0x6C0695ED, 0x1B01A57B, 0x8208F4C1, 0xF50FC457,
    0x65B0D9C6, 0x12B7E950, 0x8BBEB8EA, 0xFCB9887C, 0x62DD1DDF, 0x15DA2D49, 0x8CD37CF3, 0xFBD44C65,
    0x4DB26158, 0x3AB551CE, 0xA3BC0074, 0xD4BB30E2, 0x4ADFA541, 0x3DD895D7, 0xA4D1C46D, 0xD3D6F4FB,
    0x4369E96A, 0x346ED9FC, 0xAD678846, 0xDA60B8D0, 0x44042D73, 0x33031DE5, 0xAA0A4C5F, 0xDD0D7CC9,
    0x5005713C, 0x270241AA, 0xBE0B1010, 0xC90C2086, 0x5768B525, 0x206F85B3, 0xB966D409, 0xCE61E49F,
    0x5EDEF90E, 0x29D9C998, 0xB0D09822, 0xC7D7A8B4, 0x59B33D17, 0x2EB40D81, 0xB7BD5C3B, 0xC0BA6CAD,
    0xEDB88320, 0x9ABFB3B6, 0x03B6E20C, 0x74B1D29A, 0xEAD54739, 0x9DD277AF, 0x04DB2615, 0x73DC1683,
    0xE3630B12, 0x94643B84, 0x0D6D6A3E, 0x7A6A5AA8, 0xE40ECF0B, 0x9309FF9D, 0x0A00AE27, 0x7D079EB1,
    0xF00F9344, 0x8708A3D2, 0x1E01F268, 0x6906C2FE, 0xF762575D, 0x806567CB, 0x196C3671, 0x6E6B06E7,
    0xFED41B76, 0x89D32BE0, 0x10DA7A5A, 0x67DD4ACC, 0xF9B9DF6F, 0x8EBEEFF9, 0x17B7BE43, 0x60B08ED5,
    0xD6D6A3E8, 0xA1D1937E, 0x38D8C2C4, 0x4FDFF252, 0xD1BB67F1, 0xA6BC5767, 0x3FB506DD, 0x48B2364B,
    0xD80D2BDA, 0xAF0A1B4C, 0x36034AF6, 0x41047A60, 0xDF60EFC3, 0xA867DF55, 0x316E8EEF, 0x4669BE79,
    0xCB61B38C, 0xBC66831A, 0x256FD2A0, 0x5268E236, 0xCC0C7795, 0xBB0B4703, 0x220216B9, 0x5505262F,
    0xC5BA3BBE, 0xB2BD0B28, 0x2BB45A92, 0x5CB36A04, 0xC2D7FFA7, 0xB5D0CF31, 0x2CD99E8B, 0x5BDEAE1D,
    0x9B64C2B0, 0xEC63F226, 0x756AA39C, 0x026D930A, 0x9C0906A9, 0xEB0E363F, 0x72076785, 0x05005713,
    0x95BF4A82, 0xE2B87A14, 0x7BB12BAE, 0x0CB61B38, 0x92D28E9B, 0xE5D5BE0D, 0x7CDCEFB7, 0x0BDBDF21,
    0x86D3D2D4, 0xF1D4E242, 0x68DDB3F8, 0x1FDA836E, 0x81BE16CD, 0xF6B9265B, 0x6FB077E1, 0x18B74777,
    0x88085AE6, 0xFF0F6A70, 0x66063BCA, 0x11010B5C, 0x8F659EFF, 0xF862AE69, 0x616BFFD3, 0x166CCF45,
    0xA00AE278, 0xD70DD2EE, 0x4E048354, 0x3903B3C2, 0xA7672661, 0xD06016F7, 0x4969474D, 0x3E6E77DB,
    0xAED16A4A, 0xD9D65ADC, 0x40DF0B66, 0x37D83BF0, 0xA9BCAE53, 0xDEBB9EC5, 0x47B2CF7F, 0x30B5FFE9,
    0xBDBDF21C, 0xCABAC28A, 0x53B39330, 0x24B4A3A6, 0xBAD03605, 0xCDD706B3, 0x54DE5729, 0x23D967BF,
    0xB3667A2E, 0xC4614AB8, 0x5D681B02, 0x2A6F2B94, 0xB40BBE37, 0xC30C8EA1, 0x5A05DF1B, 0x2D02EF8D,
];

/// Compute CRC32 checksum using lookup table (8-16x faster than bit-by-bit)
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFFFFFFu32;
    for &byte in data {
        let index = ((crc ^ byte as u32) & 0xFF) as usize;
        crc = CRC32_TABLE[index] ^ (crc >> 8);
    }
    !crc
}

/// Storage engine for the database file
pub struct StorageEngine {
    /// Database file
    file: File,
    /// Memory-mapped region (if file is large enough)
    mmap: Option<Mmap>,
    /// Mutable memory-mapped region for writes (reserved for future use)
    #[allow(dead_code)]
    mmap_mut: Option<MmapMut>,
    /// File header
    header: Header,
    /// File path for atomic writes
    path: std::path::PathBuf,
}

impl StorageEngine {
    /// Validate and normalize a path to prevent path traversal attacks.
    /// Returns the canonicalized path.
    fn validate_path(path: &Path, must_exist: bool) -> Result<std::path::PathBuf> {
        if must_exist {
            // For existing files, canonicalize the full path (resolves symlinks)
            path.canonicalize().map_err(|e| {
                NeedleError::InvalidDatabase(format!(
                    "Failed to resolve path {:?}: {}",
                    path, e
                ))
            })
        } else {
            // For new files, canonicalize the parent directory and join with filename
            let parent = path.parent().ok_or_else(|| {
                NeedleError::InvalidDatabase("Path has no parent directory".into())
            })?;

            let filename = path.file_name().ok_or_else(|| {
                NeedleError::InvalidDatabase("Path has no filename".into())
            })?;

            // Canonicalize parent (must exist)
            let canonical_parent = if parent.as_os_str().is_empty() {
                // Empty parent means current directory
                std::env::current_dir().map_err(|e| {
                    NeedleError::Io(e)
                })?
            } else {
                parent.canonicalize().map_err(|e| {
                    NeedleError::InvalidDatabase(format!(
                        "Parent directory {:?} does not exist or is inaccessible: {}",
                        parent, e
                    ))
                })?
            };

            // Verify parent is actually a directory
            if !canonical_parent.is_dir() {
                return Err(NeedleError::InvalidDatabase(format!(
                    "Parent path {:?} is not a directory",
                    canonical_parent
                )));
            }

            Ok(canonical_parent.join(filename))
        }
    }

    /// Open or create a database file
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();

        if path.exists() {
            Self::open_existing(path)
        } else {
            Self::create_new(path)
        }
    }

    /// Create a new database file
    fn create_new(path: &Path) -> Result<Self> {
        // Validate path to prevent path traversal attacks
        let canonical_path = Self::validate_path(path, false)?;
        debug!(path = ?canonical_path, "Creating new database file");

        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&canonical_path)?;

        let header = Header::default();
        file.write_all(&header.to_bytes())?;
        file.flush()?;

        info!(path = ?canonical_path, "Created new database file");

        Ok(Self {
            file,
            mmap: None,
            mmap_mut: None,
            header,
            path: canonical_path,
        })
    }

    /// Open an existing database file
    fn open_existing(path: &Path) -> Result<Self> {
        // Validate path to prevent path traversal attacks
        let canonical_path = Self::validate_path(path, true)?;
        debug!(path = ?canonical_path, "Opening existing database file");

        let mut file = OpenOptions::new().read(true).write(true).open(&canonical_path)?;

        // Read header (use stack allocation to avoid 4KB heap allocation)
        let mut header_bytes = [0u8; HEADER_SIZE];
        file.read_exact(&mut header_bytes)?;
        let header = Header::from_bytes(&header_bytes)?;

        // Version check
        if header.version > VERSION {
            warn!(
                file_version = header.version,
                max_version = VERSION,
                "Unsupported database version"
            );
            return Err(NeedleError::InvalidDatabase(format!(
                "Unsupported version: {} (max: {})",
                header.version, VERSION
            )));
        }

        let file_size = file.metadata()?.len();

        // Memory-map if large enough
        let mmap = if file_size > MMAP_THRESHOLD {
            debug!(file_size = file_size, "Using memory-mapped I/O");
            Some(unsafe { Mmap::map(&file)? })
        } else {
            debug!(file_size = file_size, "Using standard file I/O");
            None
        };

        info!(
            path = ?canonical_path,
            vectors = header.vector_count,
            version = header.version,
            "Opened existing database file"
        );

        Ok(Self {
            file,
            mmap,
            mmap_mut: None,
            header,
            path: canonical_path,
        })
    }

    /// Get the header
    pub fn header(&self) -> &Header {
        &self.header
    }

    /// Get mutable header reference
    pub fn header_mut(&mut self) -> &mut Header {
        &mut self.header
    }

    /// Write the header to disk
    pub fn write_header(&mut self) -> Result<()> {
        self.file.seek(SeekFrom::Start(0))?;
        self.file.write_all(&self.header.to_bytes())?;
        self.file.flush()?;
        Ok(())
    }

    /// Write data at a specific offset
    pub fn write_at(&mut self, offset: u64, data: &[u8]) -> Result<()> {
        self.file.seek(SeekFrom::Start(offset))?;
        self.file.write_all(data)?;
        Ok(())
    }

    /// Read data from a specific offset
    pub fn read_at(&mut self, offset: u64, len: usize) -> Result<Vec<u8>> {
        // Use mmap if available
        if let Some(ref mmap) = self.mmap {
            let start = offset as usize;
            let end = start + len;
            if end <= mmap.len() {
                return Ok(mmap[start..end].to_vec());
            }
        }

        // Fall back to file read
        self.file.seek(SeekFrom::Start(offset))?;
        let mut buffer = vec![0u8; len];
        self.file.read_exact(&mut buffer)?;
        Ok(buffer)
    }

    /// Read data from a specific offset as a borrowed slice (zero-copy when mmap available)
    ///
    /// This method provides zero-copy access to memory-mapped file data, avoiding
    /// the allocation and copy overhead of `read_at`. Use this when:
    /// - You only need temporary read access to the data
    /// - You want to avoid memory allocation for large reads
    /// - The data will be processed immediately and not stored
    ///
    /// Returns an error if mmap is not available or if the range is out of bounds.
    /// In those cases, use `read_at` instead which falls back to file I/O.
    pub fn read_at_ref(&self, offset: u64, len: usize) -> Result<&[u8]> {
        if let Some(ref mmap) = self.mmap {
            let start = offset as usize;
            let end = start + len;
            if end <= mmap.len() {
                return Ok(&mmap[start..end]);
            }
            return Err(NeedleError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!("read range [{}, {}) exceeds mmap size {}", start, end, mmap.len()),
            )));
        }
        Err(NeedleError::Io(std::io::Error::other(
            "zero-copy read requires mmap; use read_at instead",
        )))
    }

    /// Check if zero-copy reads are available
    ///
    /// Returns true if the storage has an active memory map, making
    /// `read_at_ref` available for zero-copy reads.
    pub fn has_mmap(&self) -> bool {
        self.mmap.is_some()
    }

    /// Append data to the end of the file
    pub fn append(&mut self, data: &[u8]) -> Result<u64> {
        let offset = self.file.seek(SeekFrom::End(0))?;
        self.file.write_all(data)?;
        Ok(offset)
    }

    /// Sync data to disk
    pub fn sync(&mut self) -> Result<()> {
        self.file.sync_all()?;
        Ok(())
    }

    /// Get file size
    pub fn file_size(&self) -> Result<u64> {
        Ok(self.file.metadata()?.len())
    }

    /// Get the database file path
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Atomically save data using write-to-temp-then-rename pattern.
    /// This ensures the database file is never left in a corrupted state.
    /// The header's state_size and state_checksum fields will be computed automatically.
    pub fn atomic_save(&mut self, header: &Header, state_bytes: &[u8]) -> Result<()> {
        use std::fs;

        debug!(
            state_bytes = state_bytes.len(),
            vectors = header.vector_count,
            "Starting atomic save"
        );

        // Create a header with computed state integrity fields
        let mut final_header = header.clone();
        final_header.state_size = state_bytes.len() as u64;
        final_header.state_checksum = crc32(state_bytes);

        // Create temp file in same directory (same filesystem for atomic rename)
        let temp_path = self.path.with_extension("needle.tmp");

        // Write to temp file
        let mut temp_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&temp_path)?;

        // Write header with computed checksums
        temp_file.write_all(&final_header.to_bytes())?;

        // Write state at header offset
        temp_file.seek(SeekFrom::Start(HEADER_SIZE as u64))?;
        temp_file.write_all(state_bytes)?;

        // Sync to ensure all data is on disk
        temp_file.sync_all()?;

        // Close temp file before rename
        drop(temp_file);

        // Atomic rename (on Unix, rename is atomic if same filesystem)
        fs::rename(&temp_path, &self.path)?;

        // Reopen the file
        let file = OpenOptions::new().read(true).write(true).open(&self.path)?;

        // Invalidate mmap since file changed
        self.mmap = None;
        self.file = file;
        self.header = final_header;

        // Refresh mmap if needed
        self.refresh_mmap()?;

        debug!(path = ?self.path, "Atomic save completed");

        Ok(())
    }

    /// Refresh memory mapping after file growth
    pub fn refresh_mmap(&mut self) -> Result<()> {
        let file_size = self.file.metadata()?.len();
        if file_size > MMAP_THRESHOLD {
            self.mmap = Some(unsafe { Mmap::map(&self.file)? });
        }
        Ok(())
    }
}

/// Vector storage with support for in-memory and memory-mapped storage
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VectorStore {
    /// Stored vectors
    pub vectors: Vec<Vec<f32>>,
    /// Dimensions of vectors
    pub dimensions: usize,
}

impl VectorStore {
    /// Create a new vector store with given dimensions
    pub fn new(dimensions: usize) -> Self {
        Self {
            vectors: Vec::new(),
            dimensions,
        }
    }

    /// Get number of vectors
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Add a vector and return its ID
    pub fn add(&mut self, vector: Vec<f32>) -> Result<usize> {
        if self.dimensions == 0 {
            self.dimensions = vector.len();
        } else if vector.len() != self.dimensions {
            return Err(NeedleError::DimensionMismatch {
                expected: self.dimensions,
                got: vector.len(),
            });
        }

        let id = self.vectors.len();
        self.vectors.push(vector);
        Ok(id)
    }

    /// Get a vector by ID
    pub fn get(&self, id: usize) -> Option<&Vec<f32>> {
        self.vectors.get(id)
    }

    /// Update a vector at a given ID
    pub fn update(&mut self, id: usize, vector: Vec<f32>) -> Result<()> {
        if vector.len() != self.dimensions {
            return Err(NeedleError::DimensionMismatch {
                expected: self.dimensions,
                got: vector.len(),
            });
        }

        if id >= self.vectors.len() {
            return Err(NeedleError::VectorNotFound(id.to_string()));
        }

        self.vectors[id] = vector;
        Ok(())
    }

    /// Get all vectors as a slice
    pub fn as_slice(&self) -> &[Vec<f32>] {
        &self.vectors
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Number of vectors
        bytes.extend_from_slice(&(self.vectors.len() as u64).to_le_bytes());

        // Dimensions
        bytes.extend_from_slice(&(self.dimensions as u32).to_le_bytes());

        // Vectors
        for vec in &self.vectors {
            for &val in vec {
                bytes.extend_from_slice(&val.to_le_bytes());
            }
        }

        bytes
    }

    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 12 {
            return Err(NeedleError::Corruption(
                "Vector store data too short".into(),
            ));
        }

        let count = u64::from_le_bytes(
            bytes[0..8]
                .try_into()
                .map_err(|_| NeedleError::Corruption("Invalid vector store: count bytes".into()))?,
        ) as usize;
        let dimensions = u32::from_le_bytes(
            bytes[8..12]
                .try_into()
                .map_err(|_| NeedleError::Corruption("Invalid vector store: dimensions bytes".into()))?,
        ) as usize;

        let expected_size = 12 + count * dimensions * 4;
        if bytes.len() < expected_size {
            return Err(NeedleError::Corruption("Incomplete vector data".into()));
        }

        let mut vectors = Vec::with_capacity(count);
        let mut offset = 12;

        for _ in 0..count {
            let mut vec = Vec::with_capacity(dimensions);
            for _ in 0..dimensions {
                let val = f32::from_le_bytes(
                    bytes[offset..offset + 4]
                        .try_into()
                        .map_err(|_| NeedleError::Corruption("Invalid vector store: float bytes".into()))?,
                );
                vec.push(val);
                offset += 4;
            }
            vectors.push(vec);
        }

        Ok(Self {
            vectors,
            dimensions,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_header_serialization() {
        let header = Header {
            version: 1,
            dimensions: 128,
            vector_count: 1000,
            index_offset: 4096,
            vector_offset: 8192,
            metadata_offset: 16384,
            checksum: 0,
            state_size: 512,
            state_checksum: 0x12345678,
        };

        let bytes = header.to_bytes();
        let restored = Header::from_bytes(&bytes).unwrap();

        assert_eq!(header.version, restored.version);
        assert_eq!(header.dimensions, restored.dimensions);
        assert_eq!(header.vector_count, restored.vector_count);
        assert_eq!(header.state_size, restored.state_size);
        assert_eq!(header.state_checksum, restored.state_checksum);
    }

    #[test]
    fn test_storage_engine_create() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.needle");

        let storage = StorageEngine::open(&path).unwrap();
        assert_eq!(storage.header().version, VERSION);
        assert_eq!(storage.header().vector_count, 0);
    }

    #[test]
    fn test_storage_engine_write_read() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.needle");

        let mut storage = StorageEngine::open(&path).unwrap();

        let data = b"Hello, Needle!";
        let offset = storage.append(data).unwrap();

        let read_data = storage.read_at(offset, data.len()).unwrap();
        assert_eq!(&read_data, data);
    }

    #[test]
    fn test_vector_store() {
        let mut store = VectorStore::new(3);

        let id1 = store.add(vec![1.0, 2.0, 3.0]).unwrap();
        let id2 = store.add(vec![4.0, 5.0, 6.0]).unwrap();

        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
        assert_eq!(store.len(), 2);

        assert_eq!(store.get(0), Some(&vec![1.0, 2.0, 3.0]));
        assert_eq!(store.get(1), Some(&vec![4.0, 5.0, 6.0]));
    }

    #[test]
    fn test_vector_store_dimension_mismatch() {
        let mut store = VectorStore::new(3);
        store.add(vec![1.0, 2.0, 3.0]).unwrap();

        let result = store.add(vec![1.0, 2.0]); // Wrong dimensions
        assert!(result.is_err());
    }

    #[test]
    fn test_vector_store_serialization() {
        let mut store = VectorStore::new(3);
        store.add(vec![1.0, 2.0, 3.0]).unwrap();
        store.add(vec![4.0, 5.0, 6.0]).unwrap();

        let bytes = store.to_bytes();
        let restored = VectorStore::from_bytes(&bytes).unwrap();

        assert_eq!(store.len(), restored.len());
        assert_eq!(store.dimensions, restored.dimensions);
        assert_eq!(store.get(0), restored.get(0));
        assert_eq!(store.get(1), restored.get(1));
    }
}
