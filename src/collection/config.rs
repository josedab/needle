use serde::{Deserialize, Serialize};
use std::fmt;

use crate::distance::DistanceFunction;
use crate::error::{NeedleError, Result};
use crate::hnsw::HnswConfig;

/// Collection statistics
#[must_use]
#[derive(Debug, Clone)]
pub struct CollectionStats {
    /// Collection name
    pub name: String,
    /// Number of vectors
    pub vector_count: usize,
    /// Vector dimensions
    pub dimensions: usize,
    /// Distance function used
    pub distance_function: DistanceFunction,
    /// Estimated memory for vectors (bytes)
    pub vector_memory_bytes: usize,
    /// Estimated memory for metadata (bytes)
    pub metadata_memory_bytes: usize,
    /// Estimated memory for index (bytes)
    pub index_memory_bytes: usize,
    /// Total estimated memory (bytes)
    pub total_memory_bytes: usize,
    /// HNSW index statistics
    pub index_stats: crate::hnsw::HnswStats,
}

impl fmt::Display for CollectionStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn format_bytes(bytes: usize) -> String {
            if bytes >= 1_073_741_824 {
                format!("{:.1} GB", bytes as f64 / 1_073_741_824.0)
            } else if bytes >= 1_048_576 {
                format!("{:.1} MB", bytes as f64 / 1_048_576.0)
            } else if bytes >= 1024 {
                format!("{:.1} KB", bytes as f64 / 1024.0)
            } else {
                format!("{bytes} B")
            }
        }

        write!(
            f,
            "Collection '{}': {} vectors, {} dims, {:?} distance, {} total memory \
             (vectors: {}, metadata: {}, index: {})",
            self.name,
            self.vector_count,
            self.dimensions,
            self.distance_function,
            format_bytes(self.total_memory_bytes),
            format_bytes(self.vector_memory_bytes),
            format_bytes(self.metadata_memory_bytes),
            format_bytes(self.index_memory_bytes),
        )
    }
}

/// Query cache statistics
#[derive(Debug, Clone, Default)]
pub struct QueryCacheStats {
    /// Number of cache hits
    pub hits: u64,
    /// Number of cache misses
    pub misses: u64,
    /// Current number of cached entries
    pub size: usize,
    /// Maximum cache capacity
    pub capacity: usize,
    /// Semantic cache hits (similarity-based matches)
    pub semantic_hits: u64,
    /// Semantic cache misses
    pub semantic_misses: u64,
}

/// Configuration for semantic query caching (similarity-based cache lookups)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticQueryCacheConfig {
    /// Maximum number of cached query results
    pub capacity: usize,
    /// Similarity threshold (0.0-1.0) for cache hits. Higher = stricter matching.
    pub similarity_threshold: f32,
    /// TTL in seconds for cache entries (None = no expiration)
    pub ttl_seconds: Option<u64>,
}

impl Default for SemanticQueryCacheConfig {
    fn default() -> Self {
        Self {
            capacity: 100,
            similarity_threshold: 0.95,
            ttl_seconds: None,
        }
    }
}

impl SemanticQueryCacheConfig {
    /// Create a new semantic query cache configuration.
    pub fn new(capacity: usize, similarity_threshold: f32) -> Self {
        Self {
            capacity,
            similarity_threshold,
            ttl_seconds: None,
        }
    }

    /// Set TTL in seconds for cache entries.
    #[must_use]
    pub fn with_ttl_seconds(mut self, ttl: u64) -> Self {
        self.ttl_seconds = Some(ttl);
        self
    }
}

/// Policy to apply when a near-duplicate vector is detected on insert.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DedupPolicy {
    /// Reject the insert entirely.
    Reject,
    /// Keep the existing vector but merge metadata from the new insert.
    MergeMetadata,
    /// Store the new vector as a versioned variant (appends "-vN" to ID).
    Version,
}

impl Default for DedupPolicy {
    fn default() -> Self {
        Self::Reject
    }
}

/// Configuration for semantic deduplication on insert.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticDedupConfig {
    /// Enable semantic deduplication.
    pub enabled: bool,
    /// Distance threshold below which vectors are considered duplicates.
    /// Uses cosine distance by default (0.0 = identical).
    pub distance_threshold: f32,
    /// Policy to apply when a duplicate is detected.
    pub policy: DedupPolicy,
}

impl Default for SemanticDedupConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            distance_threshold: 0.02,
            policy: DedupPolicy::Reject,
        }
    }
}

impl SemanticDedupConfig {
    /// Strict preset: threshold=0.01, rejects near-duplicates.
    pub fn strict() -> Self {
        Self { enabled: true, distance_threshold: 0.01, policy: DedupPolicy::Reject }
    }

    /// Moderate preset: threshold=0.05, rejects near-duplicates.
    pub fn moderate() -> Self {
        Self { enabled: true, distance_threshold: 0.05, policy: DedupPolicy::Reject }
    }

    /// Relaxed preset: threshold=0.1, rejects near-duplicates.
    pub fn relaxed() -> Self {
        Self { enabled: true, distance_threshold: 0.1, policy: DedupPolicy::Reject }
    }

    /// Create config with specified threshold and policy.
    pub fn new(threshold: f32, policy: DedupPolicy) -> Self {
        Self { enabled: true, distance_threshold: threshold, policy }
    }
}

impl QueryCacheStats {
    /// Returns the cache hit ratio (0.0 to 1.0)
    pub fn hit_ratio(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

/// Configuration for query result caching.
///
/// Query caching stores search results to avoid redundant HNSW traversals
/// for identical queries. This is beneficial when the same queries are
/// executed repeatedly, such as in benchmarking or when serving repeated
/// user requests.
///
/// # Example
///
/// ```
/// use needle::{CollectionConfig, QueryCacheConfig};
///
/// // Enable caching with 1000 entries
/// let cache_config = QueryCacheConfig::new(1000);
///
/// let config = CollectionConfig::new("embeddings", 128)
///     .with_query_cache(cache_config);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryCacheConfig {
    /// Maximum number of query results to cache.
    /// Set to 0 to disable caching.
    pub capacity: usize,
}

impl QueryCacheConfig {
    /// Create a new query cache configuration.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Maximum number of query results to cache
    pub fn new(capacity: usize) -> Self {
        Self { capacity }
    }

    /// Create a disabled cache configuration.
    pub fn disabled() -> Self {
        Self { capacity: 0 }
    }

    /// Check if caching is enabled.
    pub fn is_enabled(&self) -> bool {
        self.capacity > 0
    }
}

impl Default for QueryCacheConfig {
    fn default() -> Self {
        Self::disabled()
    }
}

/// Collection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionConfig {
    /// Name of the collection
    pub name: String,
    /// Vector dimensions
    pub dimensions: usize,
    /// Distance function
    pub distance: DistanceFunction,
    /// HNSW configuration
    pub hnsw: HnswConfig,
    /// Slow query threshold in microseconds.
    /// If set, queries exceeding this time will be logged at warn level.
    #[serde(default)]
    pub slow_query_threshold_us: Option<u64>,
    /// Query cache configuration
    #[serde(default)]
    pub query_cache: QueryCacheConfig,
    /// Default TTL (time-to-live) for vectors in seconds.
    /// If set, vectors without an explicit TTL will expire after this duration.
    /// Expired vectors are automatically removed during search (lazy) or sweep operations.
    #[serde(default)]
    pub default_ttl_seconds: Option<u64>,
    /// Enable lazy expiration during search operations (default: true).
    /// When true, expired vectors are filtered out during search results.
    /// When false, expired vectors remain until explicitly swept or compacted.
    #[serde(default = "default_lazy_expiration")]
    pub lazy_expiration: bool,
    /// Semantic query cache configuration (similarity-based cache lookups)
    #[serde(default)]
    pub semantic_cache: Option<SemanticQueryCacheConfig>,
    /// Semantic deduplication configuration
    #[serde(default)]
    pub dedup: Option<SemanticDedupConfig>,
}

fn default_lazy_expiration() -> bool {
    true
}

/// Maximum allowed length for collection names.
const MAX_COLLECTION_NAME_LEN: usize = 256;

/// Validate a collection name.
///
/// Names must be non-empty, at most 256 characters, and contain only
/// alphanumeric characters, underscores, or hyphens.
fn validate_collection_name(name: &str) -> Result<()> {
    if name.is_empty() {
        return Err(NeedleError::InvalidConfig(
            "Collection name must not be empty".to_string(),
        ));
    }
    if name.len() > MAX_COLLECTION_NAME_LEN {
        return Err(NeedleError::InvalidConfig(format!(
            "Collection name exceeds maximum length of {} characters",
            MAX_COLLECTION_NAME_LEN
        )));
    }
    if !name
        .chars()
        .all(|c| c.is_alphanumeric() || c == '_' || c == '-')
    {
        return Err(NeedleError::InvalidConfig(
            "Collection name must contain only alphanumeric characters, underscores, or hyphens"
                .to_string(),
        ));
    }
    Ok(())
}

impl CollectionConfig {
    /// Create a new collection config with default settings
    ///
    /// # Panics
    /// Panics if dimensions is 0 or if the name is invalid.
    #[must_use]
    pub fn new(name: impl Into<String>, dimensions: usize) -> Self {
        let name = name.into();
        assert!(dimensions > 0, "Vector dimensions must be greater than 0");
        assert!(
            validate_collection_name(&name).is_ok(),
            "Invalid collection name: must be non-empty, max {} chars, alphanumeric/underscore/hyphen only",
            MAX_COLLECTION_NAME_LEN
        );
        Self {
            name,
            dimensions,
            distance: DistanceFunction::Cosine,
            hnsw: HnswConfig::default(),
            slow_query_threshold_us: None,
            query_cache: QueryCacheConfig::default(),
            default_ttl_seconds: None,
            lazy_expiration: true,
            semantic_cache: None,
            dedup: None,
        }
    }

    /// Create a new collection config with validation.
    pub fn try_new(name: impl Into<String>, dimensions: usize) -> Result<Self> {
        let name = name.into();
        validate_collection_name(&name)?;
        if dimensions == 0 {
            return Err(NeedleError::InvalidConfig(
                "Vector dimensions must be greater than 0".to_string(),
            ));
        }
        Ok(Self::new(name, dimensions))
    }

    /// Set the distance function
    #[must_use]
    pub fn with_distance(mut self, distance: DistanceFunction) -> Self {
        self.distance = distance;
        self
    }

    /// Set the HNSW M parameter
    #[must_use]
    pub fn with_m(mut self, m: usize) -> Self {
        self.hnsw = HnswConfig::with_m(m);
        self
    }

    /// Set ef_construction
    #[must_use]
    pub fn with_ef_construction(mut self, ef: usize) -> Self {
        self.hnsw.ef_construction = ef;
        self
    }

    /// Set the full HNSW configuration
    #[must_use]
    pub fn with_hnsw_config(mut self, config: HnswConfig) -> Self {
        self.hnsw = config;
        self
    }

    /// Set the slow query threshold in microseconds.
    ///
    /// When set, search queries that exceed this duration will be logged
    /// at the warn level with query details.
    ///
    /// # Example
    ///
    /// ```
    /// use needle::CollectionConfig;
    ///
    /// // Log queries slower than 100ms
    /// let config = CollectionConfig::new("embeddings", 128)
    ///     .with_slow_query_threshold_us(100_000);
    /// ```
    #[must_use]
    pub fn with_slow_query_threshold_us(mut self, threshold_us: u64) -> Self {
        self.slow_query_threshold_us = Some(threshold_us);
        self
    }

    /// Enable query result caching with the specified configuration.
    ///
    /// Query caching stores search results to avoid redundant HNSW traversals
    /// for identical queries. The cache is automatically invalidated when
    /// vectors are inserted, updated, or deleted.
    ///
    /// # Example
    ///
    /// ```
    /// use needle::{CollectionConfig, QueryCacheConfig};
    ///
    /// // Enable caching with 1000 entries
    /// let config = CollectionConfig::new("embeddings", 128)
    ///     .with_query_cache(QueryCacheConfig::new(1000));
    /// ```
    #[must_use]
    pub fn with_query_cache(mut self, cache_config: QueryCacheConfig) -> Self {
        self.query_cache = cache_config;
        self
    }

    /// Enable query result caching with a specified capacity.
    ///
    /// Shorthand for `with_query_cache(QueryCacheConfig::new(capacity))`.
    ///
    /// # Example
    ///
    /// ```
    /// use needle::CollectionConfig;
    ///
    /// // Enable caching with 500 entries
    /// let config = CollectionConfig::new("embeddings", 128)
    ///     .with_query_cache_capacity(500);
    /// ```
    #[must_use]
    pub fn with_query_cache_capacity(mut self, capacity: usize) -> Self {
        self.query_cache = QueryCacheConfig::new(capacity);
        self
    }

    /// Set the default TTL (time-to-live) for vectors in seconds.
    ///
    /// When set, vectors inserted without an explicit TTL will automatically
    /// expire after this duration. Expired vectors are filtered out during
    /// search (if lazy_expiration is enabled) or removed during sweep operations.
    ///
    /// # Example
    ///
    /// ```
    /// use needle::CollectionConfig;
    ///
    /// // Vectors expire after 1 hour by default
    /// let config = CollectionConfig::new("ephemeral", 128)
    ///     .with_default_ttl_seconds(3600);
    /// ```
    #[must_use]
    pub fn with_default_ttl_seconds(mut self, ttl_seconds: u64) -> Self {
        self.default_ttl_seconds = Some(ttl_seconds);
        self
    }

    /// Configure lazy expiration behavior.
    ///
    /// When enabled (default), expired vectors are automatically filtered out
    /// from search results without requiring an explicit sweep operation.
    ///
    /// When disabled, expired vectors remain visible in search results until
    /// explicitly removed via `expire_vectors()` or `compact()`.
    ///
    /// # Example
    ///
    /// ```
    /// use needle::CollectionConfig;
    ///
    /// // Disable lazy expiration - vectors remain until sweep
    /// let config = CollectionConfig::new("manual-cleanup", 128)
    ///     .with_default_ttl_seconds(3600)
    ///     .with_lazy_expiration(false);
    /// ```
    #[must_use]
    pub fn with_lazy_expiration(mut self, enabled: bool) -> Self {
        self.lazy_expiration = enabled;
        self
    }

    /// Enable semantic query caching with the specified configuration.
    ///
    /// Semantic caching uses similarity-based lookup to return cached results
    /// for queries that are similar (but not necessarily identical) to previous queries.
    ///
    /// # Example
    ///
    /// ```
    /// use needle::{CollectionConfig, SemanticQueryCacheConfig};
    ///
    /// let config = CollectionConfig::new("embeddings", 128)
    ///     .with_semantic_cache(SemanticQueryCacheConfig::new(100, 0.95));
    /// ```
    #[must_use]
    pub fn with_semantic_cache(mut self, config: SemanticQueryCacheConfig) -> Self {
        self.semantic_cache = Some(config);
        self
    }

    /// Set semantic deduplication configuration
    #[must_use]
    pub fn with_dedup(mut self, dedup: SemanticDedupConfig) -> Self {
        self.dedup = Some(dedup);
        self
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::distance::DistanceFunction;

    // ── try_new tests ────────────────────────────────────────────────────

    #[test]
    fn test_try_new_valid() {
        let config = CollectionConfig::try_new("test", 128).unwrap();
        assert_eq!(config.name, "test");
        assert_eq!(config.dimensions, 128);
        assert_eq!(config.distance, DistanceFunction::Cosine);
    }

    #[test]
    fn test_try_new_zero_dimensions_returns_error() {
        let result = CollectionConfig::try_new("test", 0);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, NeedleError::InvalidConfig(_)));
    }

    #[test]
    fn test_try_new_single_dimension() {
        let config = CollectionConfig::try_new("test", 1).unwrap();
        assert_eq!(config.dimensions, 1);
    }

    #[test]
    fn test_try_new_large_dimension() {
        let config = CollectionConfig::try_new("test", 65536).unwrap();
        assert_eq!(config.dimensions, 65536);
    }

    #[test]
    fn test_try_new_empty_name() {
        let result = CollectionConfig::try_new("", 128);
        assert!(result.is_err());
    }

    // ── new panics ───────────────────────────────────────────────────────

    #[test]
    #[should_panic(expected = "Vector dimensions must be greater than 0")]
    fn test_new_zero_dimensions_panics() {
        CollectionConfig::new("test", 0);
    }

    // ── Builder methods ──────────────────────────────────────────────────

    #[test]
    fn test_with_distance() {
        let config = CollectionConfig::new("test", 128)
            .with_distance(DistanceFunction::Euclidean);
        assert_eq!(config.distance, DistanceFunction::Euclidean);
    }

    #[test]
    fn test_with_m() {
        let config = CollectionConfig::new("test", 128).with_m(32);
        assert_eq!(config.hnsw.m, 32);
    }

    #[test]
    fn test_with_ef_construction() {
        let config = CollectionConfig::new("test", 128).with_ef_construction(400);
        assert_eq!(config.hnsw.ef_construction, 400);
    }

    #[test]
    fn test_with_slow_query_threshold() {
        let config = CollectionConfig::new("test", 128)
            .with_slow_query_threshold_us(100_000);
        assert_eq!(config.slow_query_threshold_us, Some(100_000));
    }

    // ── TTL configuration ────────────────────────────────────────────────

    #[test]
    fn test_default_ttl_is_none() {
        let config = CollectionConfig::new("test", 128);
        assert!(config.default_ttl_seconds.is_none());
    }

    #[test]
    fn test_with_ttl_seconds() {
        let config = CollectionConfig::new("test", 128)
            .with_default_ttl_seconds(3600);
        assert_eq!(config.default_ttl_seconds, Some(3600));
    }

    #[test]
    fn test_with_ttl_zero() {
        let config = CollectionConfig::new("test", 128)
            .with_default_ttl_seconds(0);
        assert_eq!(config.default_ttl_seconds, Some(0));
    }

    #[test]
    fn test_with_ttl_max() {
        let config = CollectionConfig::new("test", 128)
            .with_default_ttl_seconds(u64::MAX);
        assert_eq!(config.default_ttl_seconds, Some(u64::MAX));
    }

    // ── Lazy expiration ──────────────────────────────────────────────────

    #[test]
    fn test_lazy_expiration_default_true() {
        let config = CollectionConfig::new("test", 128);
        assert!(config.lazy_expiration);
    }

    #[test]
    fn test_with_lazy_expiration_disabled() {
        let config = CollectionConfig::new("test", 128)
            .with_lazy_expiration(false);
        assert!(!config.lazy_expiration);
    }

    // ── Query cache ──────────────────────────────────────────────────────

    #[test]
    fn test_query_cache_default_disabled() {
        let config = CollectionConfig::new("test", 128);
        assert!(!config.query_cache.is_enabled());
        assert_eq!(config.query_cache.capacity, 0);
    }

    #[test]
    fn test_with_query_cache() {
        let config = CollectionConfig::new("test", 128)
            .with_query_cache(QueryCacheConfig::new(1000));
        assert!(config.query_cache.is_enabled());
        assert_eq!(config.query_cache.capacity, 1000);
    }

    #[test]
    fn test_with_query_cache_capacity() {
        let config = CollectionConfig::new("test", 128)
            .with_query_cache_capacity(500);
        assert!(config.query_cache.is_enabled());
        assert_eq!(config.query_cache.capacity, 500);
    }

    #[test]
    fn test_query_cache_disabled() {
        let cache = QueryCacheConfig::disabled();
        assert!(!cache.is_enabled());
        assert_eq!(cache.capacity, 0);
    }

    // ── Semantic cache ───────────────────────────────────────────────────

    #[test]
    fn test_semantic_cache_default_none() {
        let config = CollectionConfig::new("test", 128);
        assert!(config.semantic_cache.is_none());
    }

    #[test]
    fn test_with_semantic_cache() {
        let config = CollectionConfig::new("test", 128)
            .with_semantic_cache(SemanticQueryCacheConfig::new(100, 0.95));
        let sc = config.semantic_cache.unwrap();
        assert_eq!(sc.capacity, 100);
        assert!((sc.similarity_threshold - 0.95).abs() < f32::EPSILON);
        assert!(sc.ttl_seconds.is_none());
    }

    #[test]
    fn test_semantic_cache_with_ttl() {
        let sc = SemanticQueryCacheConfig::new(50, 0.9)
            .with_ttl_seconds(600);
        assert_eq!(sc.ttl_seconds, Some(600));
    }

    #[test]
    fn test_semantic_cache_defaults() {
        let sc = SemanticQueryCacheConfig::default();
        assert_eq!(sc.capacity, 100);
        assert!((sc.similarity_threshold - 0.95).abs() < f32::EPSILON);
        assert!(sc.ttl_seconds.is_none());
    }

    // ── QueryCacheStats ──────────────────────────────────────────────────

    #[test]
    fn test_hit_ratio_no_requests() {
        let stats = QueryCacheStats::default();
        assert!((stats.hit_ratio() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_hit_ratio_all_hits() {
        let stats = QueryCacheStats {
            hits: 100,
            misses: 0,
            ..Default::default()
        };
        assert!((stats.hit_ratio() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_hit_ratio_half() {
        let stats = QueryCacheStats {
            hits: 50,
            misses: 50,
            ..Default::default()
        };
        assert!((stats.hit_ratio() - 0.5).abs() < f64::EPSILON);
    }

    // ── Serialization round-trip ─────────────────────────────────────────

    #[test]
    fn test_config_serde_roundtrip() {
        let config = CollectionConfig::new("test", 128)
            .with_distance(DistanceFunction::DotProduct)
            .with_default_ttl_seconds(3600)
            .with_lazy_expiration(false)
            .with_query_cache_capacity(500)
            .with_semantic_cache(SemanticQueryCacheConfig::new(100, 0.9));

        let json = serde_json::to_string(&config).unwrap();
        let restored: CollectionConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.name, "test");
        assert_eq!(restored.dimensions, 128);
        assert_eq!(restored.default_ttl_seconds, Some(3600));
        assert!(!restored.lazy_expiration);
        assert_eq!(restored.query_cache.capacity, 500);
        assert!(restored.semantic_cache.is_some());
    }

    // ── CollectionStats Display ──────────────────────────────────────────

    fn make_hnsw_stats() -> crate::hnsw::HnswStats {
        crate::hnsw::HnswStats {
            num_vectors: 0,
            num_deleted: 0,
            num_layers: 0,
            total_edges: 0,
            avg_connections_per_node: 0.0,
            entry_point: None,
            entry_level: 0,
            m: 16,
            ef_construction: 200,
            ef_search: 50,
        }
    }

    #[test]
    fn test_collection_stats_display_bytes() {
        let stats = CollectionStats {
            name: "test".to_string(),
            vector_count: 10,
            dimensions: 4,
            distance_function: DistanceFunction::Cosine,
            vector_memory_bytes: 500,
            metadata_memory_bytes: 100,
            index_memory_bytes: 200,
            total_memory_bytes: 800,
            index_stats: make_hnsw_stats(),
        };
        let display = format!("{}", stats);
        assert!(display.contains("800 B"));
    }

    // ── Name validation edge cases ──────────────────────────────────────

    #[test]
    fn test_try_new_name_with_spaces() {
        let result = CollectionConfig::try_new("hello world", 128);
        assert!(result.is_err());
    }

    #[test]
    fn test_try_new_name_with_special_chars() {
        let result = CollectionConfig::try_new("test@#$!", 128);
        assert!(result.is_err());
    }

    #[test]
    fn test_try_new_name_with_dots() {
        let result = CollectionConfig::try_new("my.collection", 128);
        assert!(result.is_err());
    }

    #[test]
    fn test_try_new_name_too_long() {
        let long_name = "a".repeat(MAX_COLLECTION_NAME_LEN + 1);
        let result = CollectionConfig::try_new(&long_name, 128);
        assert!(result.is_err());
    }

    #[test]
    fn test_try_new_name_at_max_length() {
        let name = "a".repeat(MAX_COLLECTION_NAME_LEN);
        let config = CollectionConfig::try_new(&name, 128).unwrap();
        assert_eq!(config.name.len(), MAX_COLLECTION_NAME_LEN);
    }

    #[test]
    fn test_try_new_name_with_underscores_and_hyphens() {
        let config = CollectionConfig::try_new("my_collection-v2", 128).unwrap();
        assert_eq!(config.name, "my_collection-v2");
    }

    #[test]
    #[should_panic]
    fn test_new_invalid_name_panics() {
        CollectionConfig::new("bad name!", 128);
    }

    // ── Dedup config ────────────────────────────────────────────────────

    #[test]
    fn test_dedup_default_none() {
        let config = CollectionConfig::new("test", 128);
        assert!(config.dedup.is_none());
    }

    #[test]
    fn test_with_dedup_strict() {
        let config = CollectionConfig::new("test", 128)
            .with_dedup(SemanticDedupConfig::strict());
        let dedup = config.dedup.unwrap();
        assert!(dedup.enabled);
        assert!((dedup.distance_threshold - 0.01).abs() < f32::EPSILON);
        assert_eq!(dedup.policy, DedupPolicy::Reject);
    }

    #[test]
    fn test_with_dedup_moderate() {
        let dedup = SemanticDedupConfig::moderate();
        assert!(dedup.enabled);
        assert!((dedup.distance_threshold - 0.05).abs() < f32::EPSILON);
    }

    #[test]
    fn test_with_dedup_relaxed() {
        let dedup = SemanticDedupConfig::relaxed();
        assert!(dedup.enabled);
        assert!((dedup.distance_threshold - 0.1).abs() < f32::EPSILON);
    }

    #[test]
    fn test_dedup_custom_policy() {
        let dedup = SemanticDedupConfig::new(0.03, DedupPolicy::MergeMetadata);
        assert!(dedup.enabled);
        assert_eq!(dedup.policy, DedupPolicy::MergeMetadata);
    }

    #[test]
    fn test_dedup_version_policy() {
        let dedup = SemanticDedupConfig::new(0.05, DedupPolicy::Version);
        assert_eq!(dedup.policy, DedupPolicy::Version);
    }

    #[test]
    fn test_dedup_default_disabled() {
        let dedup = SemanticDedupConfig::default();
        assert!(!dedup.enabled);
    }

    // ── Display formatting ──────────────────────────────────────────────

    #[test]
    fn test_collection_stats_display_mb() {
        let stats = CollectionStats {
            name: "big".to_string(),
            vector_count: 100_000,
            dimensions: 384,
            distance_function: DistanceFunction::Cosine,
            vector_memory_bytes: 100 * 1_048_576,
            metadata_memory_bytes: 10 * 1_048_576,
            index_memory_bytes: 50 * 1_048_576,
            total_memory_bytes: 160 * 1_048_576,
            index_stats: make_hnsw_stats(),
        };
        let display = format!("{}", stats);
        assert!(display.contains("MB"));
    }

    #[test]
    fn test_collection_stats_display_gb() {
        let stats = CollectionStats {
            name: "huge".to_string(),
            vector_count: 10_000_000,
            dimensions: 768,
            distance_function: DistanceFunction::DotProduct,
            vector_memory_bytes: 2 * 1_073_741_824,
            metadata_memory_bytes: 1_073_741_824,
            index_memory_bytes: 1_073_741_824,
            total_memory_bytes: 4 * 1_073_741_824,
            index_stats: make_hnsw_stats(),
        };
        let display = format!("{}", stats);
        assert!(display.contains("GB"));
    }

    // ── Builder chaining ────────────────────────────────────────────────

    #[test]
    fn test_full_builder_chain() {
        let config = CollectionConfig::new("test", 256)
            .with_distance(DistanceFunction::DotProduct)
            .with_m(32)
            .with_ef_construction(400)
            .with_slow_query_threshold_us(50_000)
            .with_query_cache_capacity(1000)
            .with_default_ttl_seconds(7200)
            .with_lazy_expiration(false)
            .with_semantic_cache(SemanticQueryCacheConfig::new(200, 0.9))
            .with_dedup(SemanticDedupConfig::strict());

        assert_eq!(config.name, "test");
        assert_eq!(config.dimensions, 256);
        assert_eq!(config.distance, DistanceFunction::DotProduct);
        assert_eq!(config.hnsw.m, 32);
        assert_eq!(config.hnsw.ef_construction, 400);
        assert_eq!(config.slow_query_threshold_us, Some(50_000));
        assert_eq!(config.query_cache.capacity, 1000);
        assert_eq!(config.default_ttl_seconds, Some(7200));
        assert!(!config.lazy_expiration);
        assert!(config.semantic_cache.is_some());
        assert!(config.dedup.is_some());
    }

    #[test]
    fn test_collection_stats_display_kb() {
        let stats = CollectionStats {
            name: "test".to_string(),
            vector_count: 100,
            dimensions: 4,
            distance_function: DistanceFunction::Cosine,
            vector_memory_bytes: 0,
            metadata_memory_bytes: 0,
            index_memory_bytes: 0,
            total_memory_bytes: 2048,
            index_stats: make_hnsw_stats(),
        };
        let display = format!("{}", stats);
        assert!(display.contains("KB"));
    }
}
