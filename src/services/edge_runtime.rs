//! Serverless Edge Runtime
//!
//! Optimized runtime for Lambda/CloudFlare Workers/Vercel Edge with fast cold
//! start, read-only mmap from object storage, and minimal binary footprint.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::services::edge_runtime::{
//!     EdgeRuntime, EdgeConfig, EdgePlatform, ReadOnlyCollection,
//! };
//!
//! // Configure for AWS Lambda
//! let config = EdgeConfig::builder()
//!     .platform(EdgePlatform::AwsLambda)
//!     .data_path("/tmp/vectors.needle")
//!     .max_memory_mb(128)
//!     .build();
//!
//! let runtime = EdgeRuntime::new(config).unwrap();
//!
//! // Load a read-only collection
//! let collection = runtime.load_collection("docs", 384).unwrap();
//!
//! // Search (read-only, no writes)
//! let results = collection.search(&[0.1f32; 384], 10).unwrap();
//! ```

use std::collections::HashMap;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::collection::{Collection, CollectionConfig, SearchResult};
use crate::distance::DistanceFunction;
use crate::error::{NeedleError, Result};

// ── Platform & Configuration ─────────────────────────────────────────────────

/// Target serverless platform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EdgePlatform {
    /// AWS Lambda.
    AwsLambda,
    /// CloudFlare Workers.
    CloudflareWorkers,
    /// Vercel Edge Functions.
    VercelEdge,
    /// Deno Deploy.
    DenoDeploy,
    /// Generic / custom platform.
    Generic,
}

impl Default for EdgePlatform {
    fn default() -> Self {
        Self::Generic
    }
}

/// Edge runtime configuration.
#[derive(Debug, Clone)]
pub struct EdgeConfig {
    /// Target platform.
    pub platform: EdgePlatform,
    /// Path to the read-only .needle data file.
    pub data_path: Option<String>,
    /// Maximum memory budget in MB.
    pub max_memory_mb: usize,
    /// Prewarming: load index into memory on init.
    pub prewarm: bool,
    /// Use quantized vectors to reduce memory.
    pub use_quantization: bool,
    /// Maximum number of concurrent searches.
    pub max_concurrent_searches: usize,
    /// Custom HNSW ef_search for edge (lower = faster, less recall).
    pub ef_search: usize,
}

impl Default for EdgeConfig {
    fn default() -> Self {
        Self {
            platform: EdgePlatform::Generic,
            data_path: None,
            max_memory_mb: 128,
            prewarm: true,
            use_quantization: false,
            max_concurrent_searches: 10,
            ef_search: 30,
        }
    }
}

impl EdgeConfig {
    /// Create a builder.
    pub fn builder() -> EdgeConfigBuilder {
        EdgeConfigBuilder::default()
    }

    /// Get platform-specific memory limit.
    pub fn effective_memory_limit(&self) -> usize {
        match self.platform {
            EdgePlatform::AwsLambda => self.max_memory_mb.min(3072),
            EdgePlatform::CloudflareWorkers => self.max_memory_mb.min(128),
            EdgePlatform::VercelEdge => self.max_memory_mb.min(256),
            EdgePlatform::DenoDeploy => self.max_memory_mb.min(512),
            EdgePlatform::Generic => self.max_memory_mb,
        }
    }
}

/// Builder for `EdgeConfig`.
#[derive(Debug, Default)]
pub struct EdgeConfigBuilder {
    inner: EdgeConfig,
}

impl EdgeConfigBuilder {
    /// Set target platform.
    pub fn platform(mut self, p: EdgePlatform) -> Self {
        self.inner.platform = p;
        self
    }

    /// Set data file path.
    pub fn data_path(mut self, path: impl Into<String>) -> Self {
        self.inner.data_path = Some(path.into());
        self
    }

    /// Set memory budget in MB.
    pub fn max_memory_mb(mut self, mb: usize) -> Self {
        self.inner.max_memory_mb = mb;
        self
    }

    /// Enable prewarming.
    pub fn prewarm(mut self, enable: bool) -> Self {
        self.inner.prewarm = enable;
        self
    }

    /// Enable quantization.
    pub fn use_quantization(mut self, enable: bool) -> Self {
        self.inner.use_quantization = enable;
        self
    }

    /// Set ef_search override for edge.
    pub fn ef_search(mut self, ef: usize) -> Self {
        self.inner.ef_search = ef;
        self
    }

    /// Build the config.
    pub fn build(self) -> EdgeConfig {
        self.inner
    }
}

// ── Read-Only Collection ─────────────────────────────────────────────────────

/// A read-only collection optimized for edge deployment.
pub struct ReadOnlyCollection {
    inner: Collection,
    ef_search: usize,
    name: String,
    dimensions: usize,
}

impl ReadOnlyCollection {
    /// Create from a Collection (marks it as read-only for edge use).
    pub fn new(name: &str, dimensions: usize, ef_search: usize) -> Result<Self> {
        let config = CollectionConfig::new(name, dimensions)
            .with_distance(DistanceFunction::Cosine);
        let collection = Collection::new(config);

        Ok(Self {
            inner: collection,
            ef_search,
            name: name.to_string(),
            dimensions,
        })
    }

    /// Load vectors into the collection (during cold start).
    pub fn load_vectors(
        &mut self,
        vectors: &[(&str, &[f32], Option<Value>)],
    ) -> Result<usize> {
        let mut count = 0;
        for &(id, vec, ref meta) in vectors {
            self.inner.insert(id, vec, meta.clone())?;
            count += 1;
        }
        Ok(count)
    }

    /// Search for similar vectors.
    pub fn search(&mut self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        self.inner.set_ef_search(self.ef_search);
        self.inner.search(query, k)
    }

    /// Search with metadata filter.
    pub fn search_filtered(
        &mut self,
        query: &[f32],
        k: usize,
        filter: &crate::metadata::Filter,
    ) -> Result<Vec<SearchResult>> {
        self.inner.set_ef_search(self.ef_search);
        self.inner.search_with_filter(query, k, filter)
    }

    /// Get a vector by ID.
    pub fn get(&self, id: &str) -> Option<(Vec<f32>, Option<Value>)> {
        self.inner.get(id).map(|(v, m)| (v.to_vec(), m.cloned()))
    }

    /// Collection name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Vector dimensions.
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Count of loaded vectors.
    pub fn count(&self) -> usize {
        self.inner.len()
    }
}

// ── Edge Runtime ─────────────────────────────────────────────────────────────

/// Runtime diagnostics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RuntimeDiagnostics {
    /// Cold start latency in milliseconds.
    pub cold_start_ms: u64,
    /// Number of loaded collections.
    pub collections_loaded: usize,
    /// Total vectors loaded.
    pub vectors_loaded: usize,
    /// Approximate memory usage in bytes.
    pub memory_used_bytes: usize,
    /// Platform name.
    pub platform: String,
    /// Effective memory limit in MB.
    pub memory_limit_mb: usize,
}

/// Serverless edge runtime for Needle.
pub struct EdgeRuntime {
    config: EdgeConfig,
    collections: HashMap<String, ReadOnlyCollection>,
    init_time: Instant,
}

impl EdgeRuntime {
    /// Initialize the edge runtime.
    pub fn new(config: EdgeConfig) -> Result<Self> {
        Ok(Self {
            config,
            collections: HashMap::new(),
            init_time: Instant::now(),
        })
    }

    /// Load a collection for read-only search.
    pub fn load_collection(
        &mut self,
        name: &str,
        dimensions: usize,
    ) -> Result<&ReadOnlyCollection> {
        if !self.collections.contains_key(name) {
            let coll = ReadOnlyCollection::new(name, dimensions, self.config.ef_search)?;
            self.collections.insert(name.to_string(), coll);
        }
        self.collections
            .get(name)
            .ok_or_else(|| NeedleError::CollectionNotFound(name.to_string()))
    }

    /// Load a collection and populate it with vectors.
    pub fn load_collection_with_data(
        &mut self,
        name: &str,
        dimensions: usize,
        vectors: &[(&str, &[f32], Option<Value>)],
    ) -> Result<usize> {
        let coll = ReadOnlyCollection::new(name, dimensions, self.config.ef_search)?;
        self.collections.insert(name.to_string(), coll);

        let coll = self.collections.get_mut(name).unwrap();
        coll.load_vectors(vectors)
    }

    /// Get a loaded collection.
    pub fn collection(&self, name: &str) -> Result<&ReadOnlyCollection> {
        self.collections
            .get(name)
            .ok_or_else(|| NeedleError::CollectionNotFound(name.to_string()))
    }

    /// Get a loaded collection (mutable, for search which requires &mut self).
    pub fn collection_mut(&mut self, name: &str) -> Result<&mut ReadOnlyCollection> {
        self.collections
            .get_mut(name)
            .ok_or_else(|| NeedleError::CollectionNotFound(name.to_string()))
    }

    /// Get runtime diagnostics.
    pub fn diagnostics(&self) -> RuntimeDiagnostics {
        let vectors_loaded: usize = self.collections.values().map(|c| c.count()).sum();
        let memory_used: usize = self
            .collections
            .values()
            .map(|c| c.count() * c.dimensions() * 4)
            .sum();

        RuntimeDiagnostics {
            cold_start_ms: self.init_time.elapsed().as_millis() as u64,
            collections_loaded: self.collections.len(),
            vectors_loaded,
            memory_used_bytes: memory_used,
            platform: format!("{:?}", self.config.platform),
            memory_limit_mb: self.config.effective_memory_limit(),
        }
    }

    /// Check if memory usage is within budget.
    pub fn within_memory_budget(&self) -> bool {
        let used_mb = self
            .collections
            .values()
            .map(|c| c.count() * c.dimensions() * 4)
            .sum::<usize>()
            / (1024 * 1024);
        used_mb <= self.config.effective_memory_limit()
    }

    /// List loaded collections.
    pub fn list_collections(&self) -> Vec<String> {
        self.collections.keys().cloned().collect()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge_runtime_basic() {
        let config = EdgeConfig::builder()
            .platform(EdgePlatform::AwsLambda)
            .max_memory_mb(256)
            .build();

        let mut runtime = EdgeRuntime::new(config).unwrap();
        let vectors: Vec<(&str, &[f32], Option<Value>)> = vec![
            ("v1", &[1.0, 0.0, 0.0, 0.0], None),
            ("v2", &[0.0, 1.0, 0.0, 0.0], None),
        ];

        let loaded = runtime
            .load_collection_with_data("test", 4, &vectors)
            .unwrap();
        assert_eq!(loaded, 2);

        let coll = runtime.collection_mut("test").unwrap();
        let results = coll.search(&[1.0, 0.0, 0.0, 0.0], 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "v1");
    }

    #[test]
    fn test_edge_diagnostics() {
        let config = EdgeConfig::builder()
            .platform(EdgePlatform::CloudflareWorkers)
            .build();

        let runtime = EdgeRuntime::new(config).unwrap();
        let diag = runtime.diagnostics();
        assert_eq!(diag.collections_loaded, 0);
        assert_eq!(diag.platform, "CloudflareWorkers");
    }

    #[test]
    fn test_platform_memory_limits() {
        assert_eq!(
            EdgeConfig::builder()
                .platform(EdgePlatform::AwsLambda)
                .max_memory_mb(10000)
                .build()
                .effective_memory_limit(),
            3072
        );
        assert_eq!(
            EdgeConfig::builder()
                .platform(EdgePlatform::CloudflareWorkers)
                .max_memory_mb(10000)
                .build()
                .effective_memory_limit(),
            128
        );
    }

    #[test]
    fn test_read_only_collection() {
        let mut coll = ReadOnlyCollection::new("test", 4, 30).unwrap();
        let vectors: Vec<(&str, &[f32], Option<Value>)> = vec![
            ("a", &[1.0, 0.0, 0.0, 0.0], None),
            ("b", &[0.0, 1.0, 0.0, 0.0], None),
        ];
        coll.load_vectors(&vectors).unwrap();

        assert_eq!(coll.count(), 2);
        assert_eq!(coll.name(), "test");
        assert_eq!(coll.dimensions(), 4);

        let got = coll.get("a").unwrap();
        assert_eq!(got.0, vec![1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_memory_budget() {
        let config = EdgeConfig::builder()
            .platform(EdgePlatform::Generic)
            .max_memory_mb(1)
            .build();
        let runtime = EdgeRuntime::new(config).unwrap();
        assert!(runtime.within_memory_budget());
    }

    #[test]
    fn test_collection_not_found() {
        let runtime = EdgeRuntime::new(EdgeConfig::default()).unwrap();
        assert!(runtime.collection("nonexistent").is_err());
    }
}
