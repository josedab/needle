use crate::error::{NeedleError, Result};
use memmap2::{Mmap, MmapMut};
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

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
    /// CRC32 checksum
    pub checksum: u32,
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

        // Compute checksum
        let checksum = crc32(&bytes);
        bytes.extend_from_slice(&checksum.to_le_bytes());

        // Pad to HEADER_SIZE
        bytes.resize(HEADER_SIZE, 0);

        bytes
    }

    /// Deserialize header from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 56 {
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

        // Verify checksum
        let computed_checksum = crc32(&bytes[0..48]);
        if stored_checksum != computed_checksum {
            return Err(NeedleError::Corruption("Header checksum mismatch".into()));
        }

        Ok(Self {
            version,
            dimensions,
            vector_count,
            index_offset,
            vector_offset,
            metadata_offset,
            checksum: stored_checksum,
        })
    }
}

/// Compute CRC32 checksum
fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFFFFFF;
    for byte in data {
        crc ^= *byte as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc >>= 1;
            }
        }
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
    /// File path (reserved for future use)
    #[allow(dead_code)]
    path: std::path::PathBuf,
}

impl StorageEngine {
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
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;

        let header = Header::default();
        file.write_all(&header.to_bytes())?;
        file.flush()?;

        Ok(Self {
            file,
            mmap: None,
            mmap_mut: None,
            header,
            path: path.to_path_buf(),
        })
    }

    /// Open an existing database file
    fn open_existing(path: &Path) -> Result<Self> {
        let mut file = OpenOptions::new().read(true).write(true).open(path)?;

        // Read header
        let mut header_bytes = vec![0u8; HEADER_SIZE];
        file.read_exact(&mut header_bytes)?;
        let header = Header::from_bytes(&header_bytes)?;

        // Version check
        if header.version > VERSION {
            return Err(NeedleError::InvalidDatabase(format!(
                "Unsupported version: {} (max: {})",
                header.version, VERSION
            )));
        }

        let file_size = file.metadata()?.len();

        // Memory-map if large enough
        let mmap = if file_size > MMAP_THRESHOLD {
            Some(unsafe { Mmap::map(&file)? })
        } else {
            None
        };

        Ok(Self {
            file,
            mmap,
            mmap_mut: None,
            header,
            path: path.to_path_buf(),
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
        };

        let bytes = header.to_bytes();
        let restored = Header::from_bytes(&bytes).unwrap();

        assert_eq!(header.version, restored.version);
        assert_eq!(header.dimensions, restored.dimensions);
        assert_eq!(header.vector_count, restored.vector_count);
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
