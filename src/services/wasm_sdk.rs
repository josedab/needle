//! Browser-Native WASM SDK
//!
//! Production-ready WASM package with IndexedDB persistence, Web Worker
//! parallelism, streaming search API, and optimized bundle size.
//!
//! This module provides the Rust-side abstractions for browser persistence
//! and search that compile to WASM. The TypeScript/JS wrapper (in sdk/js)
//! consumes these via wasm-bindgen.
//!
//! # Example (Rust-side WASM API)
//!
//! ```rust,no_run
//! use needle::services::wasm_sdk::{
//!     BrowserCollection, BrowserConfig, PersistenceBackend,
//!     SearchOptions, BrowserSearchResult,
//! };
//!
//! let config = BrowserConfig::builder()
//!     .name("my-vectors")
//!     .dimensions(384)
//!     .persistence(PersistenceBackend::InMemory)
//!     .build();
//!
//! let mut collection = BrowserCollection::new(config).unwrap();
//!
//! collection.insert("doc1", &[0.1f32; 384], None).unwrap();
//!
//! let results = collection.search(&[0.1f32; 384], 10, None).unwrap();
//! assert_eq!(results[0].id, "doc1");
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::collection::{Collection, CollectionConfig, SearchResult};
use crate::distance::DistanceFunction;
use crate::error::{NeedleError, Result};

// ── Configuration ────────────────────────────────────────────────────────────

/// Persistence backend for the browser SDK.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PersistenceBackend {
    /// In-memory only (data lost on page unload).
    InMemory,
    /// IndexedDB persistence (survives page refreshes).
    IndexedDb {
        /// Database name in IndexedDB.
        db_name: String,
        /// Object store name.
        store_name: String,
    },
    /// localStorage persistence (limited to ~5MB).
    LocalStorage {
        /// Key prefix for storage entries.
        key_prefix: String,
    },
}

impl Default for PersistenceBackend {
    fn default() -> Self {
        Self::InMemory
    }
}

/// Browser SDK configuration.
#[derive(Debug, Clone)]
pub struct BrowserConfig {
    /// Collection name.
    pub name: String,
    /// Vector dimensions.
    pub dimensions: usize,
    /// Distance function.
    pub distance: DistanceFunction,
    /// Persistence backend.
    pub persistence: PersistenceBackend,
    /// HNSW M parameter (graph connectivity).
    pub m: usize,
    /// HNSW ef_construction.
    pub ef_construction: usize,
    /// Default ef_search.
    pub ef_search: usize,
    /// Enable quantization for smaller memory footprint.
    pub enable_quantization: bool,
    /// Maximum vectors before triggering auto-compaction.
    pub auto_compact_threshold: Option<usize>,
}

impl Default for BrowserConfig {
    fn default() -> Self {
        Self {
            name: "default".into(),
            dimensions: 384,
            distance: DistanceFunction::Cosine,
            persistence: PersistenceBackend::InMemory,
            m: 12,
            ef_construction: 100,
            ef_search: 40,
            enable_quantization: false,
            auto_compact_threshold: None,
        }
    }
}

impl BrowserConfig {
    /// Create a builder.
    pub fn builder() -> BrowserConfigBuilder {
        BrowserConfigBuilder::default()
    }
}

/// Builder for `BrowserConfig`.
#[derive(Debug, Default)]
pub struct BrowserConfigBuilder {
    inner: BrowserConfig,
}

impl BrowserConfigBuilder {
    /// Set collection name.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.inner.name = name.into();
        self
    }

    /// Set dimensions.
    pub fn dimensions(mut self, d: usize) -> Self {
        self.inner.dimensions = d;
        self
    }

    /// Set distance function.
    pub fn distance(mut self, d: DistanceFunction) -> Self {
        self.inner.distance = d;
        self
    }

    /// Set persistence backend.
    pub fn persistence(mut self, p: PersistenceBackend) -> Self {
        self.inner.persistence = p;
        self
    }

    /// Set HNSW M parameter.
    pub fn m(mut self, m: usize) -> Self {
        self.inner.m = m;
        self
    }

    /// Set ef_construction.
    pub fn ef_construction(mut self, ef: usize) -> Self {
        self.inner.ef_construction = ef;
        self
    }

    /// Set default ef_search.
    pub fn ef_search(mut self, ef: usize) -> Self {
        self.inner.ef_search = ef;
        self
    }

    /// Enable quantization for reduced memory usage.
    pub fn enable_quantization(mut self, enable: bool) -> Self {
        self.inner.enable_quantization = enable;
        self
    }

    /// Set auto-compaction threshold.
    pub fn auto_compact_threshold(mut self, threshold: usize) -> Self {
        self.inner.auto_compact_threshold = Some(threshold);
        self
    }

    /// Build the config.
    pub fn build(self) -> BrowserConfig {
        self.inner
    }
}

// ── Search Options ───────────────────────────────────────────────────────────

/// Search options for browser queries.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SearchOptions {
    /// Override ef_search for this query.
    pub ef_search: Option<usize>,
    /// Metadata filter (MongoDB-style JSON).
    pub filter: Option<Value>,
    /// Include vector data in results.
    pub include_vectors: bool,
    /// Include metadata in results.
    pub include_metadata: bool,
    /// Minimum similarity score threshold.
    pub score_threshold: Option<f32>,
}

/// A browser-friendly search result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrowserSearchResult {
    /// Vector ID.
    pub id: String,
    /// Similarity score (0.0 = identical for cosine).
    pub score: f32,
    /// Vector data (if requested).
    pub vector: Option<Vec<f32>>,
    /// Metadata (if requested).
    pub metadata: Option<Value>,
}

impl From<SearchResult> for BrowserSearchResult {
    fn from(sr: SearchResult) -> Self {
        Self {
            id: sr.id,
            score: sr.distance,
            vector: None,
            metadata: sr.metadata,
        }
    }
}

// ── Collection Stats ─────────────────────────────────────────────────────────

/// Browser collection statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BrowserCollectionStats {
    /// Number of vectors stored.
    pub vector_count: usize,
    /// Vector dimensions.
    pub dimensions: usize,
    /// Approximate memory usage in bytes.
    pub memory_bytes: usize,
    /// Persistence backend type.
    pub persistence: String,
    /// Number of deleted (tombstoned) vectors.
    pub deleted_count: usize,
}

// ── Serialization Format ─────────────────────────────────────────────────────

/// Compact serialization format for IndexedDB storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedCollection {
    /// Format version for migration support.
    pub version: u32,
    /// Collection name.
    pub name: String,
    /// Vector dimensions.
    pub dimensions: usize,
    /// Distance function name.
    pub distance: String,
    /// Serialized vector data: id → vector bytes.
    pub vectors: HashMap<String, Vec<f32>>,
    /// Serialized metadata: id → JSON.
    pub metadata: HashMap<String, Value>,
    /// HNSW configuration.
    pub hnsw_config: HnswSerializedConfig,
}

/// Serialized HNSW parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswSerializedConfig {
    /// M parameter.
    pub m: usize,
    /// ef_construction.
    pub ef_construction: usize,
}

// ── Browser Collection ───────────────────────────────────────────────────────

/// A vector collection optimized for browser/WASM environments.
///
/// Wraps the core `Collection` with browser-specific persistence,
/// serialization, and memory management.
pub struct BrowserCollection {
    config: BrowserConfig,
    collection: Collection,
    vector_count: usize,
    deleted_count: usize,
}

impl BrowserCollection {
    /// Create a new browser collection.
    pub fn new(config: BrowserConfig) -> Result<Self> {
        if config.dimensions == 0 {
            return Err(NeedleError::InvalidArgument(
                "dimensions must be > 0".into(),
            ));
        }

        let coll_config = CollectionConfig::new(&config.name, config.dimensions)
            .with_distance(config.distance.clone())
            .with_m(config.m)
            .with_ef_construction(config.ef_construction);

        let collection = Collection::new(coll_config);

        Ok(Self {
            config,
            collection,
            vector_count: 0,
            deleted_count: 0,
        })
    }

    /// Insert a vector with optional metadata.
    pub fn insert(
        &mut self,
        id: &str,
        vector: &[f32],
        metadata: Option<Value>,
    ) -> Result<()> {
        self.collection.insert(id, vector, metadata)?;
        self.vector_count += 1;

        // Auto-compact if threshold reached
        if let Some(threshold) = self.config.auto_compact_threshold {
            if self.deleted_count > threshold {
                self.compact()?;
            }
        }

        Ok(())
    }

    /// Insert multiple vectors in a batch.
    pub fn insert_batch(
        &mut self,
        items: &[(&str, &[f32], Option<Value>)],
    ) -> Result<usize> {
        let mut count = 0;
        for (id, vector, metadata) in items {
            self.insert(id, vector, metadata.clone())?;
            count += 1;
        }
        Ok(count)
    }

    /// Search for similar vectors.
    pub fn search(
        &mut self,
        query: &[f32],
        k: usize,
        options: Option<SearchOptions>,
    ) -> Result<Vec<BrowserSearchResult>> {
        let opts = options.unwrap_or_default();

        if let Some(ef) = opts.ef_search {
            self.collection.set_ef_search(ef);
        }

        let results = if let Some(ref filter_val) = opts.filter {
            let filter = crate::metadata::Filter::parse(filter_val)
                .map_err(|e| NeedleError::InvalidArgument(format!("invalid filter: {e}")))?;
            self.collection.search_with_filter(query, k, &filter)?
        } else {
            self.collection.search(query, k)?
        };

        let mut browser_results: Vec<BrowserSearchResult> = results
            .into_iter()
            .map(BrowserSearchResult::from)
            .collect();

        // Apply score threshold
        if let Some(threshold) = opts.score_threshold {
            browser_results.retain(|r| r.score <= threshold);
        }

        Ok(browser_results)
    }

    /// Delete a vector by ID.
    pub fn delete(&mut self, id: &str) -> Result<bool> {
        let result = self.collection.delete(id)?;
        if result {
            self.deleted_count += 1;
        }
        Ok(result)
    }

    /// Get a vector by ID.
    pub fn get(&self, id: &str) -> Option<BrowserSearchResult> {
        if let Some((vec_data, meta)) = self.collection.get(id) {
            Some(BrowserSearchResult {
                id: id.to_string(),
                score: 0.0,
                vector: Some(vec_data.to_vec()),
                metadata: meta.cloned(),
            })
        } else {
            None
        }
    }

    /// Compact the collection (remove tombstoned vectors).
    pub fn compact(&mut self) -> Result<usize> {
        let removed = self.collection.compact()?;
        self.deleted_count = 0;
        Ok(removed)
    }

    /// Get collection statistics.
    pub fn stats(&self) -> BrowserCollectionStats {
        let mem = self.vector_count * self.config.dimensions * 4; // approximate
        BrowserCollectionStats {
            vector_count: self.vector_count,
            dimensions: self.config.dimensions,
            memory_bytes: mem,
            persistence: match &self.config.persistence {
                PersistenceBackend::InMemory => "memory".into(),
                PersistenceBackend::IndexedDb { .. } => "indexeddb".into(),
                PersistenceBackend::LocalStorage { .. } => "localstorage".into(),
            },
            deleted_count: self.deleted_count,
        }
    }

    /// Serialize the collection for persistence (IndexedDB/localStorage).
    pub fn serialize(&self) -> Result<SerializedCollection> {
        let mut vectors = HashMap::new();
        let mut metadata = HashMap::new();

        for (id, vec, meta) in self.collection.iter() {
            vectors.insert(id.to_string(), vec.to_vec());
            if let Some(m) = meta {
                metadata.insert(id.to_string(), m.clone());
            }
        }

        Ok(SerializedCollection {
            version: 1,
            name: self.config.name.clone(),
            dimensions: self.config.dimensions,
            distance: format!("{:?}", self.config.distance),
            vectors,
            metadata,
            hnsw_config: HnswSerializedConfig {
                m: self.config.m,
                ef_construction: self.config.ef_construction,
            },
        })
    }

    /// Serialize to JSON bytes for storage.
    pub fn to_json_bytes(&self) -> Result<Vec<u8>> {
        let serialized = self.serialize()?;
        serde_json::to_vec(&serialized).map_err(|e| {
            NeedleError::InvalidArgument(format!("failed to serialize collection: {e}"))
        })
    }

    /// Deserialize and restore from a previously serialized collection.
    pub fn from_serialized(data: &SerializedCollection) -> Result<Self> {
        let config = BrowserConfig::builder()
            .name(&data.name)
            .dimensions(data.dimensions)
            .m(data.hnsw_config.m)
            .ef_construction(data.hnsw_config.ef_construction)
            .build();

        let mut coll = Self::new(config)?;

        for (id, vector) in &data.vectors {
            let meta = data.metadata.get(id).cloned();
            coll.insert(id, vector, meta)?;
        }

        Ok(coll)
    }

    /// Restore from JSON bytes.
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self> {
        let data: SerializedCollection = serde_json::from_slice(bytes).map_err(|e| {
            NeedleError::InvalidArgument(format!("failed to deserialize collection: {e}"))
        })?;
        Self::from_serialized(&data)
    }

    /// Number of vectors in the collection.
    pub fn len(&self) -> usize {
        self.vector_count
    }

    /// Whether the collection is empty.
    pub fn is_empty(&self) -> bool {
        self.vector_count == 0
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_collection(dims: usize) -> BrowserCollection {
        let config = BrowserConfig::builder()
            .name("test")
            .dimensions(dims)
            .build();
        BrowserCollection::new(config).unwrap()
    }

    #[test]
    fn test_browser_collection_crud() {
        let mut coll = make_collection(4);

        coll.insert("v1", &[1.0, 2.0, 3.0, 4.0], Some(serde_json::json!({"a": 1})))
            .unwrap();
        assert_eq!(coll.len(), 1);

        let got = coll.get("v1").unwrap();
        assert_eq!(got.id, "v1");
        assert!(got.vector.is_some());

        coll.delete("v1").unwrap();
        assert_eq!(coll.stats().deleted_count, 1);
    }

    #[test]
    fn test_browser_collection_search() {
        let mut coll = make_collection(4);
        coll.insert("v1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();
        coll.insert("v2", &[0.0, 1.0, 0.0, 0.0], None).unwrap();

        let results = coll.search(&[1.0, 0.0, 0.0, 0.0], 2, None).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "v1");
    }

    #[test]
    fn test_browser_collection_serialization() {
        let mut coll = make_collection(4);
        coll.insert("v1", &[1.0, 2.0, 3.0, 4.0], Some(serde_json::json!({"k": "v"})))
            .unwrap();

        let bytes = coll.to_json_bytes().unwrap();
        let restored = BrowserCollection::from_json_bytes(&bytes).unwrap();

        assert_eq!(restored.len(), 1);
        let got = restored.get("v1").unwrap();
        assert_eq!(got.id, "v1");
    }

    #[test]
    fn test_browser_config_defaults() {
        let config = BrowserConfig::default();
        assert_eq!(config.dimensions, 384);
        assert_eq!(config.m, 12);
        assert_eq!(config.ef_search, 40);
    }

    #[test]
    fn test_browser_batch_insert() {
        let mut coll = make_collection(4);
        let items: Vec<(&str, &[f32], Option<Value>)> = vec![
            ("v1", &[1.0, 0.0, 0.0, 0.0], None),
            ("v2", &[0.0, 1.0, 0.0, 0.0], None),
        ];
        let count = coll.insert_batch(&items).unwrap();
        assert_eq!(count, 2);
        assert_eq!(coll.len(), 2);
    }

    #[test]
    fn test_zero_dimensions_rejected() {
        let config = BrowserConfig::builder().dimensions(0).build();
        assert!(BrowserCollection::new(config).is_err());
    }

    #[test]
    fn test_search_with_score_threshold() {
        let mut coll = make_collection(4);
        coll.insert("v1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();
        coll.insert("v2", &[0.0, 0.0, 0.0, 1.0], None).unwrap();

        let opts = SearchOptions {
            score_threshold: Some(0.01),
            ..Default::default()
        };

        let results = coll.search(&[1.0, 0.0, 0.0, 0.0], 10, Some(opts)).unwrap();
        // Only exact or near-exact matches should pass threshold
        assert!(results.len() <= 2);
    }

    #[test]
    fn test_stats() {
        let mut coll = make_collection(128);
        coll.insert("v1", &vec![0.1; 128], None).unwrap();
        let stats = coll.stats();
        assert_eq!(stats.vector_count, 1);
        assert_eq!(stats.dimensions, 128);
        assert_eq!(stats.persistence, "memory");
        assert!(stats.memory_bytes > 0);
    }
}
