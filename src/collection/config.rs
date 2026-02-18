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
}

fn default_lazy_expiration() -> bool {
    true
}

impl CollectionConfig {
    /// Create a new collection config with default settings
    ///
    /// # Panics
    /// Panics if dimensions is 0.
    #[must_use]
    pub fn new(name: impl Into<String>, dimensions: usize) -> Self {
        assert!(dimensions > 0, "Vector dimensions must be greater than 0");
        Self {
            name: name.into(),
            dimensions,
            distance: DistanceFunction::Cosine,
            hnsw: HnswConfig::default(),
            slow_query_threshold_us: None,
            query_cache: QueryCacheConfig::default(),
            default_ttl_seconds: None,
            lazy_expiration: true,
        }
    }

    /// Create a new collection config with validation.
    pub fn try_new(name: impl Into<String>, dimensions: usize) -> Result<Self> {
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
}
