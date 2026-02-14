//! ⚠️ **Experimental**: This module is under active development. APIs may change without notice.
//!
//! Serverless WASM Runtime
//!
//! Edge runtime adapters for deploying Needle as a Cloudflare Worker,
//! Vercel Edge Function, or Deno Deploy module.

use crate::distance::DistanceFunction;
use crate::error::{NeedleError, Result};
use crate::hnsw::{HnswConfig, HnswIndex};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

/// Target edge platform for deployment.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EdgePlatform {
    CloudflareWorkers,
    VercelEdge,
    DenoEdge,
    BrowserWasm,
    NodeWasm,
    Custom(String),
}

/// Resource limits for a given edge platform.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlatformLimits {
    pub max_memory_bytes: u64,
    pub max_execution_ms: u64,
    pub max_payload_bytes: u64,
    pub supports_persistence: bool,
    pub supports_streaming: bool,
}

impl PlatformLimits {
    /// Returns the default limits for a given platform.
    pub fn for_platform(platform: &EdgePlatform) -> Self {
        match platform {
            EdgePlatform::CloudflareWorkers => PlatformLimits {
                max_memory_bytes: 128 * 1024 * 1024,
                max_execution_ms: 30_000,
                max_payload_bytes: 100 * 1024 * 1024,
                supports_persistence: true,
                supports_streaming: true,
            },
            EdgePlatform::VercelEdge => PlatformLimits {
                max_memory_bytes: 256 * 1024 * 1024,
                max_execution_ms: 25_000,
                max_payload_bytes: 4 * 1024 * 1024,
                supports_persistence: false,
                supports_streaming: true,
            },
            EdgePlatform::DenoEdge => PlatformLimits {
                max_memory_bytes: 512 * 1024 * 1024,
                max_execution_ms: u64::MAX,
                max_payload_bytes: u64::MAX,
                supports_persistence: true,
                supports_streaming: true,
            },
            EdgePlatform::BrowserWasm => PlatformLimits {
                max_memory_bytes: 256 * 1024 * 1024,
                max_execution_ms: u64::MAX,
                max_payload_bytes: u64::MAX,
                supports_persistence: false,
                supports_streaming: false,
            },
            EdgePlatform::NodeWasm | EdgePlatform::Custom(_) => PlatformLimits {
                max_memory_bytes: 512 * 1024 * 1024,
                max_execution_ms: u64::MAX,
                max_payload_bytes: u64::MAX,
                supports_persistence: true,
                supports_streaming: true,
            },
        }
    }
}

/// Key-value storage adapter for serverless persistence.
pub trait KVAdapter {
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>>;
    fn set(&self, key: &str, value: &[u8]) -> Result<()>;
    fn delete(&self, key: &str) -> Result<()>;
    fn list_keys(&self, prefix: &str) -> Result<Vec<String>>;
}

/// Simple in-memory KV store backed by a HashMap.
pub struct InMemoryKV {
    store: RwLock<HashMap<String, Vec<u8>>>,
}

impl InMemoryKV {
    pub fn new() -> Self {
        Self {
            store: RwLock::new(HashMap::new()),
        }
    }
}

impl Default for InMemoryKV {
    fn default() -> Self {
        Self::new()
    }
}

impl KVAdapter for InMemoryKV {
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        Ok(self.store.read().get(key).cloned())
    }

    fn set(&self, key: &str, value: &[u8]) -> Result<()> {
        self.store.write().insert(key.to_string(), value.to_vec());
        Ok(())
    }

    fn delete(&self, key: &str) -> Result<()> {
        self.store.write().remove(key);
        Ok(())
    }

    fn list_keys(&self, prefix: &str) -> Result<Vec<String>> {
        let store = self.store.read();
        let keys: Vec<String> = store
            .keys()
            .filter(|k| k.starts_with(prefix))
            .cloned()
            .collect();
        Ok(keys)
    }
}

/// Compression mode for snapshot data.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum CompressionMode {
    None,
    LZ4,
    Snappy,
}

impl Default for CompressionMode {
    fn default() -> Self {
        CompressionMode::None
    }
}

/// Configuration for the serverless runtime.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ServerlessConfig {
    pub platform: EdgePlatform,
    pub max_collection_size: usize,
    pub enable_binary_quantization: bool,
    pub snapshot_enabled: bool,
    pub cold_start_budget_ms: u64,
    pub compression: CompressionMode,
}

impl Default for ServerlessConfig {
    fn default() -> Self {
        Self {
            platform: EdgePlatform::CloudflareWorkers,
            max_collection_size: 100_000,
            enable_binary_quantization: true,
            snapshot_enabled: false,
            cold_start_budget_ms: 500,
            compression: CompressionMode::None,
        }
    }
}

/// Metrics captured during cold start initialization.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ColdStartMetrics {
    pub init_time_ms: u64,
    pub index_load_time_ms: u64,
    pub first_query_time_ms: u64,
    pub total_cold_start_ms: u64,
}

/// Information about a persisted snapshot.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SnapshotInfo {
    pub id: String,
    pub created_at: String,
    pub size_bytes: u64,
    pub vector_count: usize,
    pub dimensions: usize,
    pub checksum: String,
}

/// Search result returned by the serverless runtime.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ServerlessSearchResult {
    pub id: String,
    pub distance: f32,
    pub metadata: Option<serde_json::Value>,
}

/// Aggregate statistics for the runtime.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RuntimeStats {
    pub collections: usize,
    pub total_vectors: usize,
    pub estimated_memory_bytes: u64,
    pub platform: EdgePlatform,
}

/// A single collection managed by the serverless runtime.
pub struct ServerlessCollection {
    pub name: String,
    pub dimensions: usize,
    pub vectors: HashMap<String, (Vec<f32>, Option<serde_json::Value>)>,
    pub index: HnswIndex,
}

impl ServerlessCollection {
    fn new(name: String, dimensions: usize) -> Self {
        let config = HnswConfig::default();
        let index = HnswIndex::new(config, DistanceFunction::Cosine);
        Self {
            name,
            dimensions,
            vectors: HashMap::new(),
            index,
        }
    }

    fn insert(
        &mut self,
        id: String,
        vector: Vec<f32>,
        metadata: Option<serde_json::Value>,
    ) -> Result<()> {
        if vector.len() != self.dimensions {
            return Err(NeedleError::InvalidVector(format!(
                "expected {} dimensions, got {}",
                self.dimensions,
                vector.len()
            )));
        }
        let vector_id = self.vectors.len();
        // Collect all raw vectors for the HNSW index
        let all_vectors: Vec<Vec<f32>> = self
            .vectors
            .values()
            .map(|(v, _)| v.clone())
            .chain(std::iter::once(vector.clone()))
            .collect();
        self.index.insert(vector_id, &vector, &all_vectors)?;
        self.vectors.insert(id, (vector, metadata));
        Ok(())
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<ServerlessSearchResult>> {
        if query.len() != self.dimensions {
            return Err(NeedleError::InvalidVector(format!(
                "expected {} dimensions, got {}",
                self.dimensions,
                query.len()
            )));
        }
        let all_vectors: Vec<Vec<f32>> = self.vectors.values().map(|(v, _)| v.clone()).collect();
        let id_map: Vec<String> = self.vectors.keys().cloned().collect();
        let raw_results = self.index.search(query, k, &all_vectors);
        let results = raw_results
            .into_iter()
            .filter_map(|(vid, dist)| {
                id_map.get(vid).map(|string_id| {
                    let meta = self.vectors.get(string_id).and_then(|(_, m)| m.clone());
                    ServerlessSearchResult {
                        id: string_id.clone(),
                        distance: dist,
                        metadata: meta,
                    }
                })
            })
            .collect();
        Ok(results)
    }
}

/// Serializable representation of a collection for snapshots.
#[derive(Serialize, Deserialize)]
struct CollectionSnapshot {
    name: String,
    dimensions: usize,
    vectors: Vec<(String, Vec<f32>, Option<serde_json::Value>)>,
}

/// The main serverless runtime that manages collections and persistence.
pub struct ServerlessRuntime {
    pub config: ServerlessConfig,
    kv: Box<dyn KVAdapter + Send + Sync>,
    collections: HashMap<String, ServerlessCollection>,
    cold_start_metrics: Option<ColdStartMetrics>,
    init_instant: Instant,
}

impl ServerlessRuntime {
    /// Create a new runtime with the given config and KV adapter.
    pub fn new(config: ServerlessConfig, kv: Box<dyn KVAdapter + Send + Sync>) -> Self {
        Self {
            config,
            kv,
            collections: HashMap::new(),
            cold_start_metrics: None,
            init_instant: Instant::now(),
        }
    }

    /// Create a new collection, validating against platform limits.
    pub fn create_collection(&mut self, name: &str, dimensions: usize) -> Result<()> {
        if self.collections.contains_key(name) {
            return Err(NeedleError::InvalidInput(format!(
                "collection '{}' already exists",
                name
            )));
        }
        let limits = self.platform_limits();
        // Rough estimate: each dimension is 4 bytes per vector, with max_collection_size vectors
        let estimated_bytes = (self.config.max_collection_size as u64) * (dimensions as u64) * 4;
        if estimated_bytes > limits.max_memory_bytes {
            return Err(NeedleError::InvalidConfig(format!(
                "collection would exceed platform memory limit of {} bytes",
                limits.max_memory_bytes
            )));
        }
        self.collections.insert(
            name.to_string(),
            ServerlessCollection::new(name.to_string(), dimensions),
        );
        Ok(())
    }

    /// Insert a vector into a collection.
    pub fn insert(
        &mut self,
        collection: &str,
        id: &str,
        vector: Vec<f32>,
        metadata: Option<serde_json::Value>,
    ) -> Result<()> {
        let max_size = self.config.max_collection_size;
        let col = self
            .collections
            .get_mut(collection)
            .ok_or_else(|| NeedleError::CollectionNotFound(collection.to_string()))?;
        if col.vectors.len() >= max_size {
            return Err(NeedleError::InvalidOperation(format!(
                "collection '{}' has reached max size of {}",
                collection, max_size
            )));
        }
        col.insert(id.to_string(), vector, metadata)
    }

    /// Search a collection for the k nearest neighbors.
    pub fn search(
        &self,
        collection: &str,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<ServerlessSearchResult>> {
        let col = self
            .collections
            .get(collection)
            .ok_or_else(|| NeedleError::CollectionNotFound(collection.to_string()))?;
        col.search(query, k)
    }

    /// Save a snapshot of a collection to the KV store.
    pub fn save_snapshot(&self, collection: &str) -> Result<SnapshotInfo> {
        let col = self
            .collections
            .get(collection)
            .ok_or_else(|| NeedleError::CollectionNotFound(collection.to_string()))?;

        let snapshot = CollectionSnapshot {
            name: col.name.clone(),
            dimensions: col.dimensions,
            vectors: col
                .vectors
                .iter()
                .map(|(id, (v, m))| (id.clone(), v.clone(), m.clone()))
                .collect(),
        };

        let data = serde_json::to_vec(&snapshot)
            .map_err(|e| NeedleError::InvalidInput(format!("serialization error: {}", e)))?;

        let snapshot_id = format!("{}_{}", collection, self.init_instant.elapsed().as_millis());
        let key = format!("snapshots/{}/{}", collection, snapshot_id);
        self.kv.set(&key, &data)?;

        // Simple checksum: sum of all bytes mod u64
        let checksum: u64 = data.iter().map(|b| *b as u64).sum();

        Ok(SnapshotInfo {
            id: snapshot_id,
            created_at: format!("{:?}", std::time::SystemTime::now()),
            size_bytes: data.len() as u64,
            vector_count: col.vectors.len(),
            dimensions: col.dimensions,
            checksum: format!("{:016x}", checksum),
        })
    }

    /// Load a collection from a snapshot stored in the KV store.
    pub fn load_snapshot(&mut self, collection: &str, snapshot_id: &str) -> Result<()> {
        let key = format!("snapshots/{}/{}", collection, snapshot_id);
        let data = self.kv.get(&key)?.ok_or_else(|| {
            NeedleError::NotFound(format!("snapshot '{}' not found", snapshot_id))
        })?;

        let snapshot: CollectionSnapshot = serde_json::from_slice(&data)
            .map_err(|e| NeedleError::InvalidInput(format!("deserialization error: {}", e)))?;

        let mut col = ServerlessCollection::new(snapshot.name.clone(), snapshot.dimensions);
        for (id, vector, metadata) in snapshot.vectors {
            col.insert(id, vector, metadata)?;
        }
        self.collections.insert(collection.to_string(), col);
        Ok(())
    }

    /// Returns the platform limits for the configured platform.
    pub fn platform_limits(&self) -> PlatformLimits {
        PlatformLimits::for_platform(&self.config.platform)
    }

    /// Returns cold start metrics if available.
    pub fn cold_start_metrics(&self) -> Option<&ColdStartMetrics> {
        self.cold_start_metrics.as_ref()
    }

    /// Set cold start metrics (e.g. after initial loading).
    pub fn set_cold_start_metrics(&mut self, metrics: ColdStartMetrics) {
        self.cold_start_metrics = Some(metrics);
    }

    /// Returns the number of collections.
    pub fn collection_count(&self) -> usize {
        self.collections.len()
    }

    /// Returns aggregate runtime statistics.
    pub fn stats(&self) -> RuntimeStats {
        let total_vectors: usize = self.collections.values().map(|c| c.vectors.len()).sum();
        let estimated_memory: u64 = self
            .collections
            .values()
            .map(|c| (c.vectors.len() as u64) * (c.dimensions as u64) * 4)
            .sum();
        RuntimeStats {
            collections: self.collections.len(),
            total_vectors,
            estimated_memory_bytes: estimated_memory,
            platform: self.config.platform.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_runtime() -> ServerlessRuntime {
        let config = ServerlessConfig::default();
        let kv = Box::new(InMemoryKV::new());
        ServerlessRuntime::new(config, kv)
    }

    fn random_vector(dim: usize) -> Vec<f32> {
        (0..dim).map(|i| (i as f32 * 0.1).sin()).collect()
    }

    #[test]
    fn test_create_collection() {
        let mut rt = make_runtime();
        assert!(rt.create_collection("docs", 128).is_ok());
        assert_eq!(rt.collection_count(), 1);
        // Duplicate should fail
        assert!(rt.create_collection("docs", 128).is_err());
    }

    #[test]
    fn test_insert_and_search() {
        let mut rt = make_runtime();
        rt.create_collection("test", 4).unwrap();
        rt.insert("test", "v1", vec![1.0, 0.0, 0.0, 0.0], None)
            .unwrap();
        rt.insert("test", "v2", vec![0.0, 1.0, 0.0, 0.0], None)
            .unwrap();
        rt.insert("test", "v3", vec![1.0, 0.1, 0.0, 0.0], None)
            .unwrap();

        let results = rt.search("test", &[1.0, 0.0, 0.0, 0.0], 2).unwrap();
        assert!(!results.is_empty());
        assert!(results.len() <= 2);
    }

    #[test]
    fn test_platform_limits() {
        let cf = PlatformLimits::for_platform(&EdgePlatform::CloudflareWorkers);
        assert_eq!(cf.max_memory_bytes, 128 * 1024 * 1024);
        assert_eq!(cf.max_execution_ms, 30_000);
        assert!(cf.supports_persistence);

        let vercel = PlatformLimits::for_platform(&EdgePlatform::VercelEdge);
        assert_eq!(vercel.max_memory_bytes, 256 * 1024 * 1024);
        assert_eq!(vercel.max_execution_ms, 25_000);
        assert!(!vercel.supports_persistence);

        let deno = PlatformLimits::for_platform(&EdgePlatform::DenoEdge);
        assert_eq!(deno.max_memory_bytes, 512 * 1024 * 1024);
        assert_eq!(deno.max_execution_ms, u64::MAX);

        let browser = PlatformLimits::for_platform(&EdgePlatform::BrowserWasm);
        assert!(!browser.supports_streaming);
    }

    #[test]
    fn test_collection_size_limit() {
        let config = ServerlessConfig {
            max_collection_size: 2,
            ..Default::default()
        };
        let kv = Box::new(InMemoryKV::new());
        let mut rt = ServerlessRuntime::new(config, kv);
        rt.create_collection("small", 3).unwrap();
        rt.insert("small", "a", vec![1.0, 0.0, 0.0], None).unwrap();
        rt.insert("small", "b", vec![0.0, 1.0, 0.0], None).unwrap();
        let err = rt.insert("small", "c", vec![0.0, 0.0, 1.0], None);
        assert!(err.is_err());
    }

    #[test]
    fn test_in_memory_kv() {
        let kv = InMemoryKV::new();
        assert!(kv.get("missing").unwrap().is_none());

        kv.set("key1", b"hello").unwrap();
        assert_eq!(kv.get("key1").unwrap(), Some(b"hello".to_vec()));

        kv.set("key2", b"world").unwrap();
        let mut keys = kv.list_keys("key").unwrap();
        keys.sort();
        assert_eq!(keys, vec!["key1", "key2"]);

        kv.delete("key1").unwrap();
        assert!(kv.get("key1").unwrap().is_none());
    }

    #[test]
    fn test_save_load_snapshot() {
        let mut rt = make_runtime();
        rt.create_collection("snap", 3).unwrap();
        rt.insert("snap", "v1", vec![1.0, 2.0, 3.0], None).unwrap();
        rt.insert(
            "snap",
            "v2",
            vec![4.0, 5.0, 6.0],
            Some(serde_json::json!({"tag": "test"})),
        )
        .unwrap();

        let info = rt.save_snapshot("snap").unwrap();
        assert_eq!(info.vector_count, 2);
        assert_eq!(info.dimensions, 3);
        assert!(info.size_bytes > 0);

        // Remove the collection and reload from snapshot
        rt.collections.remove("snap");
        assert_eq!(rt.collection_count(), 0);

        rt.load_snapshot("snap", &info.id).unwrap();
        assert_eq!(rt.collection_count(), 1);
        let stats = rt.stats();
        assert_eq!(stats.total_vectors, 2);
    }

    #[test]
    fn test_cold_start_metrics() {
        let mut rt = make_runtime();
        assert!(rt.cold_start_metrics().is_none());

        let metrics = ColdStartMetrics {
            init_time_ms: 50,
            index_load_time_ms: 100,
            first_query_time_ms: 10,
            total_cold_start_ms: 160,
        };
        rt.set_cold_start_metrics(metrics.clone());

        let m = rt.cold_start_metrics().unwrap();
        assert_eq!(m.total_cold_start_ms, 160);
        assert_eq!(m.init_time_ms, 50);
    }

    #[test]
    fn test_runtime_stats() {
        let mut rt = make_runtime();
        rt.create_collection("a", 4).unwrap();
        rt.create_collection("b", 8).unwrap();
        rt.insert("a", "v1", vec![1.0, 0.0, 0.0, 0.0], None)
            .unwrap();
        rt.insert("b", "v1", vec![1.0; 8], None).unwrap();
        rt.insert("b", "v2", vec![0.0; 8], None).unwrap();

        let stats = rt.stats();
        assert_eq!(stats.collections, 2);
        assert_eq!(stats.total_vectors, 3);
        // a: 1*4*4=16, b: 2*8*4=64 => 80
        assert_eq!(stats.estimated_memory_bytes, 80);
        assert_eq!(stats.platform, EdgePlatform::CloudflareWorkers);
    }

    #[test]
    fn test_multiple_collections() {
        let mut rt = make_runtime();
        rt.create_collection("c1", 3).unwrap();
        rt.create_collection("c2", 5).unwrap();
        rt.create_collection("c3", 10).unwrap();
        assert_eq!(rt.collection_count(), 3);

        rt.insert("c1", "a", vec![1.0, 0.0, 0.0], None).unwrap();
        rt.insert("c2", "a", vec![1.0; 5], None).unwrap();
        rt.insert("c3", "a", vec![1.0; 10], None).unwrap();

        // Searching wrong collection dimensions should fail
        assert!(rt.search("c1", &[1.0; 5], 1).is_err());
        // Correct dimensions should succeed
        assert!(rt.search("c1", &[1.0, 0.0, 0.0], 1).is_ok());
        // Nonexistent collection
        assert!(rt.search("nope", &[1.0], 1).is_err());
    }

    #[test]
    fn test_config_defaults() {
        let config = ServerlessConfig::default();
        assert_eq!(config.platform, EdgePlatform::CloudflareWorkers);
        assert_eq!(config.max_collection_size, 100_000);
        assert!(config.enable_binary_quantization);
        assert!(!config.snapshot_enabled);
        assert_eq!(config.cold_start_budget_ms, 500);
        assert!(matches!(config.compression, CompressionMode::None));
    }
}
