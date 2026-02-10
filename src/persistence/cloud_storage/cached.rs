//! Caching layers: simple cache and 3-tier smart cache.

use crate::error::{NeedleError, Result};
use super::config::{CacheConfig, StorageBackend};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

// ============================================================================
// 3-Tier Smart Cache (Memory -> SSD -> Cloud)
// ============================================================================

/// Configuration for the 3-tier smart cache.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TieredCacheConfig {
    /// Maximum memory cache size in bytes.
    pub memory_max_size: usize,
    /// Maximum SSD cache size in bytes.
    pub ssd_max_size: usize,
    /// SSD cache directory path.
    pub ssd_cache_path: PathBuf,
    /// Default TTL for memory-cached items.
    pub memory_ttl: Duration,
    /// Default TTL for SSD-cached items.
    pub ssd_ttl: Duration,
    /// Enable prefetching based on access patterns.
    pub enable_prefetch: bool,
    /// Maximum number of prefetch items.
    pub max_prefetch_items: usize,
    /// Promote items from SSD to memory after N accesses.
    pub promotion_threshold: u32,
    /// Enable access pattern tracking for analytics.
    pub enable_access_tracking: bool,
}

impl Default for TieredCacheConfig {
    fn default() -> Self {
        Self {
            memory_max_size: 100 * 1024 * 1024, // 100MB
            ssd_max_size: 1024 * 1024 * 1024,   // 1GB
            ssd_cache_path: PathBuf::from("/tmp/needle_cache"),
            memory_ttl: Duration::from_secs(300),  // 5 minutes
            ssd_ttl: Duration::from_secs(3600),    // 1 hour
            enable_prefetch: true,
            max_prefetch_items: 10,
            promotion_threshold: 3,
            enable_access_tracking: true,
        }
    }
}

/// Entry in the tiered cache with metadata.
#[derive(Clone)]
struct TieredCacheEntry {
    /// Cached data (present for memory tier, None for SSD tier entries in memory index).
    data: Option<Vec<u8>>,
    /// Which tier this entry resides in.
    tier: CacheTier,
    /// Expiration time.
    expires_at: Instant,
    /// Last access time.
    last_accessed: Instant,
    /// Access count (for promotion decisions).
    access_count: u32,
    /// Size in bytes.
    size: usize,
    /// SSD file path (if stored on SSD).
    ssd_path: Option<PathBuf>,
}

/// Cache tier enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheTier {
    /// In-memory cache (fastest).
    Memory,
    /// SSD/disk cache (fast).
    Ssd,
    /// Origin storage (cloud backend).
    Origin,
}

/// Statistics for the tiered cache.
#[derive(Debug, Clone, Default)]
pub struct TieredCacheStats {
    /// Memory tier hits.
    pub memory_hits: Arc<AtomicU64>,
    /// SSD tier hits.
    pub ssd_hits: Arc<AtomicU64>,
    /// Origin (cloud) fetches.
    pub origin_fetches: Arc<AtomicU64>,
    /// Memory tier evictions.
    pub memory_evictions: Arc<AtomicU64>,
    /// SSD tier evictions.
    pub ssd_evictions: Arc<AtomicU64>,
    /// Promotions from SSD to memory.
    pub promotions: Arc<AtomicU64>,
    /// Demotions from memory to SSD.
    pub demotions: Arc<AtomicU64>,
    /// Prefetch hits.
    pub prefetch_hits: Arc<AtomicU64>,
    /// Total bytes in memory.
    pub memory_bytes: Arc<AtomicU64>,
    /// Total bytes on SSD.
    pub ssd_bytes: Arc<AtomicU64>,
}

impl TieredCacheStats {
    /// Calculate overall hit rate (memory + SSD hits vs total requests).
    pub fn hit_rate(&self) -> f64 {
        let memory_hits = self.memory_hits.load(Ordering::Relaxed) as f64;
        let ssd_hits = self.ssd_hits.load(Ordering::Relaxed) as f64;
        let origin_fetches = self.origin_fetches.load(Ordering::Relaxed) as f64;
        let total = memory_hits + ssd_hits + origin_fetches;
        if total > 0.0 {
            (memory_hits + ssd_hits) / total
        } else {
            0.0
        }
    }

    /// Calculate memory hit rate.
    pub fn memory_hit_rate(&self) -> f64 {
        let memory_hits = self.memory_hits.load(Ordering::Relaxed) as f64;
        let ssd_hits = self.ssd_hits.load(Ordering::Relaxed) as f64;
        let origin_fetches = self.origin_fetches.load(Ordering::Relaxed) as f64;
        let total = memory_hits + ssd_hits + origin_fetches;
        if total > 0.0 {
            memory_hits / total
        } else {
            0.0
        }
    }
}

/// Access pattern tracking for prefetching.
#[derive(Debug, Clone)]
struct AccessPattern {
    /// Keys accessed in sequence.
    recent_keys: Vec<String>,
    /// Maximum keys to track.
    max_keys: usize,
    /// Detected sequential patterns (key prefix -> next likely key).
    sequential_patterns: HashMap<String, Vec<String>>,
}

impl AccessPattern {
    fn new(max_keys: usize) -> Self {
        Self {
            recent_keys: Vec::with_capacity(max_keys),
            max_keys,
            sequential_patterns: HashMap::new(),
        }
    }

    fn record_access(&mut self, key: &str) {
        // Record the key
        if self.recent_keys.len() >= self.max_keys {
            self.recent_keys.remove(0);
        }
        self.recent_keys.push(key.to_string());

        // Detect sequential patterns
        if self.recent_keys.len() >= 2 {
            let prev_key = &self.recent_keys[self.recent_keys.len() - 2];
            let patterns = self.sequential_patterns
                .entry(prev_key.clone())
                .or_default();
            if !patterns.contains(&key.to_string()) && patterns.len() < 5 {
                patterns.push(key.to_string());
            }
        }
    }

    fn predict_next(&self, key: &str) -> Vec<String> {
        self.sequential_patterns
            .get(key)
            .cloned()
            .unwrap_or_default()
    }
}

/// 3-tier smart cache backend wrapper.
///
/// Provides intelligent caching with:
/// - Memory tier: fastest access, limited size
/// - SSD tier: fast access, larger capacity
/// - Cloud tier: origin storage (slowest)
///
/// Features:
/// - Automatic promotion/demotion between tiers
/// - Access pattern tracking for prefetching
/// - LRU eviction within each tier
pub struct TieredCacheBackend<B: StorageBackend> {
    /// Inner (origin) backend.
    inner: B,
    /// Configuration.
    config: TieredCacheConfig,
    /// Cache index (tracks all entries across tiers).
    cache_index: parking_lot::RwLock<HashMap<String, TieredCacheEntry>>,
    /// Statistics.
    stats: TieredCacheStats,
    /// Access pattern tracker.
    access_patterns: parking_lot::Mutex<AccessPattern>,
    /// Current memory usage.
    memory_usage: AtomicU64,
    /// Current SSD usage.
    ssd_usage: AtomicU64,
}

impl<B: StorageBackend> TieredCacheBackend<B> {
    /// Create a new tiered cache backend.
    pub fn new(inner: B, config: TieredCacheConfig) -> Result<Self> {
        // Ensure SSD cache directory exists
        std::fs::create_dir_all(&config.ssd_cache_path)?;

        Ok(Self {
            inner,
            config: config.clone(),
            cache_index: parking_lot::RwLock::new(HashMap::new()),
            stats: TieredCacheStats::default(),
            access_patterns: parking_lot::Mutex::new(AccessPattern::new(100)),
            memory_usage: AtomicU64::new(0),
            ssd_usage: AtomicU64::new(0),
        })
    }

    /// Get cache statistics.
    pub fn stats(&self) -> &TieredCacheStats {
        &self.stats
    }

    /// Clear all caches.
    pub fn clear_all(&self) -> Result<()> {
        // Clear memory cache
        let mut index = self.cache_index.write();

        // Delete SSD files
        for entry in index.values() {
            if let Some(ref path) = entry.ssd_path {
                let _ = std::fs::remove_file(path);
            }
        }

        index.clear();
        self.memory_usage.store(0, Ordering::Relaxed);
        self.ssd_usage.store(0, Ordering::Relaxed);
        self.stats.memory_bytes.store(0, Ordering::Relaxed);
        self.stats.ssd_bytes.store(0, Ordering::Relaxed);

        Ok(())
    }

    /// Clear only memory tier (demote to SSD).
    pub fn clear_memory(&self) -> Result<()> {
        let mut index = self.cache_index.write();

        for (key, entry) in index.iter_mut() {
            if entry.tier == CacheTier::Memory {
                // Demote to SSD
                if let Some(ref data) = entry.data {
                    let ssd_path = self.config.ssd_cache_path.join(key_to_filename(key));
                    if std::fs::write(&ssd_path, data).is_ok() {
                        entry.tier = CacheTier::Ssd;
                        entry.ssd_path = Some(ssd_path);
                        entry.data = None;
                        self.stats.demotions.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        }

        self.memory_usage.store(0, Ordering::Relaxed);
        self.stats.memory_bytes.store(0, Ordering::Relaxed);

        Ok(())
    }

    /// Evict expired entries from all tiers.
    pub fn evict_expired(&self) {
        let mut index = self.cache_index.write();
        let now = Instant::now();

        let expired_keys: Vec<String> = index
            .iter()
            .filter(|(_, entry)| entry.expires_at <= now)
            .map(|(k, _)| k.clone())
            .collect();

        for key in expired_keys {
            if let Some(entry) = index.remove(&key) {
                match entry.tier {
                    CacheTier::Memory => {
                        self.memory_usage.fetch_sub(entry.size as u64, Ordering::Relaxed);
                        self.stats.memory_bytes.fetch_sub(entry.size as u64, Ordering::Relaxed);
                        self.stats.memory_evictions.fetch_add(1, Ordering::Relaxed);
                    }
                    CacheTier::Ssd => {
                        if let Some(ref path) = entry.ssd_path {
                            let _ = std::fs::remove_file(path);
                        }
                        self.ssd_usage.fetch_sub(entry.size as u64, Ordering::Relaxed);
                        self.stats.ssd_bytes.fetch_sub(entry.size as u64, Ordering::Relaxed);
                        self.stats.ssd_evictions.fetch_add(1, Ordering::Relaxed);
                    }
                    CacheTier::Origin => {}
                }
            }
        }
    }

    /// Evict from memory to make room (LRU-based).
    fn evict_memory(&self, needed_space: usize, index: &mut HashMap<String, TieredCacheEntry>) {
        let current_usage = self.memory_usage.load(Ordering::Relaxed) as usize;
        if current_usage + needed_space <= self.config.memory_max_size {
            return;
        }

        // Collect memory entries sorted by last access time (LRU)
        let mut memory_entries: Vec<_> = index
            .iter()
            .filter(|(_, e)| e.tier == CacheTier::Memory)
            .map(|(k, e)| (k.clone(), e.last_accessed, e.size))
            .collect();

        memory_entries.sort_by_key(|(_, accessed, _)| *accessed);

        let target_size = self.config.memory_max_size.saturating_sub(needed_space);
        let mut freed = 0usize;

        for (key, _, size) in memory_entries {
            if current_usage - freed <= target_size {
                break;
            }

            if let Some(entry) = index.get_mut(&key) {
                // Try to demote to SSD
                if let Some(ref data) = entry.data {
                    let ssd_path = self.config.ssd_cache_path.join(key_to_filename(&key));
                    if std::fs::write(&ssd_path, data).is_ok() {
                        entry.tier = CacheTier::Ssd;
                        entry.ssd_path = Some(ssd_path);
                        entry.data = None;
                        entry.expires_at = Instant::now() + self.config.ssd_ttl;
                        self.ssd_usage.fetch_add(size as u64, Ordering::Relaxed);
                        self.stats.ssd_bytes.fetch_add(size as u64, Ordering::Relaxed);
                        self.stats.demotions.fetch_add(1, Ordering::Relaxed);
                    }
                }

                self.memory_usage.fetch_sub(size as u64, Ordering::Relaxed);
                self.stats.memory_bytes.fetch_sub(size as u64, Ordering::Relaxed);
                self.stats.memory_evictions.fetch_add(1, Ordering::Relaxed);
                freed += size;
            }
        }
    }

    /// Evict from SSD to make room (LRU-based).
    fn evict_ssd(&self, needed_space: usize, index: &mut HashMap<String, TieredCacheEntry>) {
        let current_usage = self.ssd_usage.load(Ordering::Relaxed) as usize;
        if current_usage + needed_space <= self.config.ssd_max_size {
            return;
        }

        // Collect SSD entries sorted by last access time (LRU)
        let mut ssd_entries: Vec<_> = index
            .iter()
            .filter(|(_, e)| e.tier == CacheTier::Ssd)
            .map(|(k, e)| (k.clone(), e.last_accessed, e.size))
            .collect();

        ssd_entries.sort_by_key(|(_, accessed, _)| *accessed);

        let target_size = self.config.ssd_max_size.saturating_sub(needed_space);
        let mut freed = 0usize;

        for (key, _, size) in ssd_entries {
            if current_usage - freed <= target_size {
                break;
            }

            if let Some(entry) = index.remove(&key) {
                if let Some(ref path) = entry.ssd_path {
                    let _ = std::fs::remove_file(path);
                }
                self.ssd_usage.fetch_sub(size as u64, Ordering::Relaxed);
                self.stats.ssd_bytes.fetch_sub(size as u64, Ordering::Relaxed);
                self.stats.ssd_evictions.fetch_add(1, Ordering::Relaxed);
                freed += size;
            }
        }
    }

    /// Promote entry from SSD to memory.
    #[allow(dead_code)]
    fn promote_to_memory(&self, _key: &str, entry: &mut TieredCacheEntry) -> Result<()> {
        if entry.tier != CacheTier::Ssd {
            return Ok(());
        }

        // Read from SSD
        let ssd_path = entry.ssd_path.as_ref().ok_or_else(|| {
            NeedleError::Io(std::io::Error::other("SSD path not found for entry"))
        })?;

        let data = std::fs::read(ssd_path)?;
        let size = data.len();

        // Evict memory if needed
        let mut index = self.cache_index.write();
        self.evict_memory(size, &mut index);

        // Update entry
        entry.data = Some(data);
        entry.tier = CacheTier::Memory;
        entry.expires_at = Instant::now() + self.config.memory_ttl;

        // Clean up SSD file
        let _ = std::fs::remove_file(ssd_path);
        entry.ssd_path = None;

        // Update stats
        self.ssd_usage.fetch_sub(size as u64, Ordering::Relaxed);
        self.stats.ssd_bytes.fetch_sub(size as u64, Ordering::Relaxed);
        self.memory_usage.fetch_add(size as u64, Ordering::Relaxed);
        self.stats.memory_bytes.fetch_add(size as u64, Ordering::Relaxed);
        self.stats.promotions.fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    /// Cache data at the appropriate tier.
    fn cache_data(&self, key: &str, data: &[u8]) {
        let size = data.len();
        let now = Instant::now();

        let mut index = self.cache_index.write();

        // Determine which tier to use based on size and current usage
        if size <= self.config.memory_max_size / 4 {
            // Small enough for memory
            self.evict_memory(size, &mut index);

            let entry = TieredCacheEntry {
                data: Some(data.to_vec()),
                tier: CacheTier::Memory,
                expires_at: now + self.config.memory_ttl,
                last_accessed: now,
                access_count: 1,
                size,
                ssd_path: None,
            };

            index.insert(key.to_string(), entry);
            self.memory_usage.fetch_add(size as u64, Ordering::Relaxed);
            self.stats.memory_bytes.fetch_add(size as u64, Ordering::Relaxed);
        } else {
            // Write to SSD
            self.evict_ssd(size, &mut index);

            let ssd_path = self.config.ssd_cache_path.join(key_to_filename(key));
            if let Some(parent) = ssd_path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }

            if std::fs::write(&ssd_path, data).is_ok() {
                let entry = TieredCacheEntry {
                    data: None,
                    tier: CacheTier::Ssd,
                    expires_at: now + self.config.ssd_ttl,
                    last_accessed: now,
                    access_count: 1,
                    size,
                    ssd_path: Some(ssd_path),
                };

                index.insert(key.to_string(), entry);
                self.ssd_usage.fetch_add(size as u64, Ordering::Relaxed);
                self.stats.ssd_bytes.fetch_add(size as u64, Ordering::Relaxed);
            }
        }
    }

    /// Prefetch predicted keys in the background.
    async fn prefetch(&self, key: &str) {
        if !self.config.enable_prefetch {
            return;
        }

        let predictions = {
            let patterns = self.access_patterns.lock();
            patterns.predict_next(key)
        };

        for predicted_key in predictions.into_iter().take(self.config.max_prefetch_items) {
            // Check if already cached
            {
                let index = self.cache_index.read();
                if index.contains_key(&predicted_key) {
                    continue;
                }
            }

            // Fetch and cache
            if let Ok(data) = self.inner.read(&predicted_key).await {
                self.cache_data(&predicted_key, &data);
                self.stats.prefetch_hits.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

impl<B: StorageBackend> StorageBackend for TieredCacheBackend<B> {
    fn read<'a>(&'a self, key: &'a str) -> Pin<Box<dyn Future<Output = Result<Vec<u8>>> + Send + 'a>> {
        Box::pin(async move {
            let now = Instant::now();

            // Track access pattern (outside of cache lock)
            if self.config.enable_access_tracking {
                let mut patterns = self.access_patterns.lock();
                patterns.record_access(key);
            }

            // Check cache - do all work synchronously, store result to use after lock is released
            enum CacheResult {
                MemoryHit(Vec<u8>),
                SsdHit(Vec<u8>),
                Miss,
            }

            let cache_result = {
                let mut index = self.cache_index.write();
                if let Some(entry) = index.get_mut(key) {
                    if entry.expires_at > now {
                        entry.last_accessed = now;
                        entry.access_count += 1;

                        match entry.tier {
                            CacheTier::Memory => {
                                self.stats.memory_hits.fetch_add(1, Ordering::Relaxed);
                                if let Some(ref data) = entry.data {
                                    CacheResult::MemoryHit(data.clone())
                                } else {
                                    CacheResult::Miss
                                }
                            }
                            CacheTier::Ssd => {
                                self.stats.ssd_hits.fetch_add(1, Ordering::Relaxed);

                                // Read from SSD
                                if let Some(ref ssd_path) = entry.ssd_path {
                                    if let Ok(data) = std::fs::read(ssd_path) {
                                        // Check if should promote to memory
                                        if entry.access_count >= self.config.promotion_threshold {
                                            let ssd_path_clone = ssd_path.clone();
                                            let size = entry.size;

                                            // Update entry for promotion
                                            entry.data = Some(data.clone());
                                            entry.tier = CacheTier::Memory;
                                            entry.expires_at = now + self.config.memory_ttl;
                                            entry.ssd_path = None;

                                            // Clean up SSD file
                                            let _ = std::fs::remove_file(&ssd_path_clone);

                                            // Update stats
                                            self.ssd_usage.fetch_sub(size as u64, Ordering::Relaxed);
                                            self.stats.ssd_bytes.fetch_sub(size as u64, Ordering::Relaxed);
                                            self.memory_usage.fetch_add(size as u64, Ordering::Relaxed);
                                            self.stats.memory_bytes.fetch_add(size as u64, Ordering::Relaxed);
                                            self.stats.promotions.fetch_add(1, Ordering::Relaxed);
                                        }

                                        CacheResult::SsdHit(data)
                                    } else {
                                        CacheResult::Miss
                                    }
                                } else {
                                    CacheResult::Miss
                                }
                            }
                            CacheTier::Origin => CacheResult::Miss,
                        }
                    } else {
                        CacheResult::Miss
                    }
                } else {
                    CacheResult::Miss
                }
            }; // Lock released here

            // Now handle the result without holding the lock
            match cache_result {
                CacheResult::MemoryHit(data) | CacheResult::SsdHit(data) => {
                    // Prefetch can now safely await
                    self.prefetch(key).await;
                    return Ok(data);
                }
                CacheResult::Miss => {}
            }

            // Cache miss - fetch from origin
            self.stats.origin_fetches.fetch_add(1, Ordering::Relaxed);
            let data = self.inner.read(key).await?;

            // Cache the data
            self.cache_data(key, &data);

            // Trigger prefetch
            self.prefetch(key).await;

            Ok(data)
        })
    }

    fn write<'a>(&'a self, key: &'a str, data: &'a [u8]) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            // Write to origin
            self.inner.write(key, data).await?;

            // Update cache
            self.cache_data(key, data);

            Ok(())
        })
    }

    fn delete<'a>(&'a self, key: &'a str) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            // Delete from origin
            self.inner.delete(key).await?;

            // Remove from cache
            let mut index = self.cache_index.write();
            if let Some(entry) = index.remove(key) {
                match entry.tier {
                    CacheTier::Memory => {
                        self.memory_usage.fetch_sub(entry.size as u64, Ordering::Relaxed);
                        self.stats.memory_bytes.fetch_sub(entry.size as u64, Ordering::Relaxed);
                    }
                    CacheTier::Ssd => {
                        if let Some(ref path) = entry.ssd_path {
                            let _ = std::fs::remove_file(path);
                        }
                        self.ssd_usage.fetch_sub(entry.size as u64, Ordering::Relaxed);
                        self.stats.ssd_bytes.fetch_sub(entry.size as u64, Ordering::Relaxed);
                    }
                    CacheTier::Origin => {}
                }
            }

            Ok(())
        })
    }

    fn list<'a>(&'a self, prefix: &'a str) -> Pin<Box<dyn Future<Output = Result<Vec<String>>> + Send + 'a>> {
        // List always goes to origin (cache may be incomplete)
        self.inner.list(prefix)
    }

    fn exists<'a>(&'a self, key: &'a str) -> Pin<Box<dyn Future<Output = Result<bool>> + Send + 'a>> {
        Box::pin(async move {
            // Check cache first
            {
                let index = self.cache_index.read();
                if let Some(entry) = index.get(key) {
                    if entry.expires_at > Instant::now() {
                        return Ok(true);
                    }
                }
            }

            // Check origin
            self.inner.exists(key).await
        })
    }
}

/// Convert a key to a valid filename for SSD caching.
pub(super) fn key_to_filename(key: &str) -> String {
    // Replace path separators and other problematic characters
    // Using a loop for clarity on which characters are replaced
    let mut result = key.to_string();
    for c in ['/', '\\', ':', '*', '?', '"', '<', '>', '|'] {
        result = result.replace(c, "_");
    }
    result
}

// ============================================================================
// Simple Caching Layer
// ============================================================================

/// Cached storage backend wrapper.
pub struct CachedBackend<B: StorageBackend> {
    /// Inner backend.
    inner: B,
    /// Cache configuration.
    config: CacheConfig,
    /// Cache storage.
    cache: parking_lot::RwLock<HashMap<String, CacheEntry>>,
    /// Cache statistics.
    stats: CacheStats,
}

/// Cache entry.
#[derive(Clone)]
#[allow(dead_code)]
struct CacheEntry {
    data: Vec<u8>,
    expires_at: Instant,
    last_accessed: Instant,
}

/// Cache statistics.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    hits: Arc<AtomicU64>,
    misses: Arc<AtomicU64>,
    evictions: Arc<AtomicU64>,
    bytes_cached: Arc<AtomicU64>,
}

impl CacheStats {
    /// Get cache hits.
    pub fn hits(&self) -> u64 {
        self.hits.load(Ordering::Relaxed)
    }

    /// Get cache misses.
    pub fn misses(&self) -> u64 {
        self.misses.load(Ordering::Relaxed)
    }

    /// Get evictions.
    pub fn evictions(&self) -> u64 {
        self.evictions.load(Ordering::Relaxed)
    }

    /// Get bytes cached.
    pub fn bytes_cached(&self) -> u64 {
        self.bytes_cached.load(Ordering::Relaxed)
    }

    /// Get hit rate.
    pub fn hit_rate(&self) -> f64 {
        let hits = self.hits() as f64;
        let total = hits + self.misses() as f64;
        if total > 0.0 {
            hits / total
        } else {
            0.0
        }
    }
}

impl<B: StorageBackend> CachedBackend<B> {
    /// Create a new cached backend.
    pub fn new(inner: B, config: CacheConfig) -> Self {
        Self {
            inner,
            config,
            cache: parking_lot::RwLock::new(HashMap::new()),
            stats: CacheStats::default(),
        }
    }

    /// Get cache statistics.
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Clear the cache.
    pub fn clear_cache(&self) {
        let mut cache = self.cache.write();
        cache.clear();
        self.stats.bytes_cached.store(0, Ordering::Relaxed);
    }

    /// Invalidate a specific key.
    pub fn invalidate(&self, key: &str) {
        let mut cache = self.cache.write();
        if let Some(entry) = cache.remove(key) {
            self.stats
                .bytes_cached
                .fetch_sub(entry.data.len() as u64, Ordering::Relaxed);
        }
    }

    /// Evict expired entries.
    pub fn evict_expired(&self) {
        let mut cache = self.cache.write();
        let now = Instant::now();

        let expired_keys: Vec<String> = cache
            .iter()
            .filter(|(_, entry)| entry.expires_at <= now)
            .map(|(k, _)| k.clone())
            .collect();

        for key in expired_keys {
            if let Some(entry) = cache.remove(&key) {
                self.stats.evictions.fetch_add(1, Ordering::Relaxed);
                self.stats
                    .bytes_cached
                    .fetch_sub(entry.data.len() as u64, Ordering::Relaxed);
            }
        }
    }

    /// Check if key is cached.
    pub fn is_cached(&self, key: &str) -> bool {
        let cache = self.cache.read();
        if let Some(entry) = cache.get(key) {
            entry.expires_at > Instant::now()
        } else {
            false
        }
    }
}

impl<B: StorageBackend> StorageBackend for CachedBackend<B> {
    fn read<'a>(&'a self, key: &'a str) -> Pin<Box<dyn Future<Output = Result<Vec<u8>>> + Send + 'a>> {
        Box::pin(async move {
            // Check cache first
            {
                let cache = self.cache.read();
                if let Some(entry) = cache.get(key) {
                    if entry.expires_at > Instant::now() {
                        self.stats.hits.fetch_add(1, Ordering::Relaxed);
                        return Ok(entry.data.clone());
                    }
                }
            }

            // Cache miss - fetch from backend
            self.stats.misses.fetch_add(1, Ordering::Relaxed);
            let data = self.inner.read(key).await?;

            // Add to cache
            {
                let mut cache = self.cache.write();
                let entry = CacheEntry {
                    data: data.clone(),
                    expires_at: Instant::now() + self.config.default_ttl,
                    last_accessed: Instant::now(),
                };
                self.stats
                    .bytes_cached
                    .fetch_add(entry.data.len() as u64, Ordering::Relaxed);
                cache.insert(key.to_string(), entry);
            }

            Ok(data)
        })
    }

    fn write<'a>(&'a self, key: &'a str, data: &'a [u8]) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            // Write to backend
            self.inner.write(key, data).await?;

            // Update cache
            {
                let mut cache = self.cache.write();
                let entry = CacheEntry {
                    data: data.to_vec(),
                    expires_at: Instant::now() + self.config.default_ttl,
                    last_accessed: Instant::now(),
                };

                // Remove old entry if exists
                if let Some(old) = cache.remove(key) {
                    self.stats
                        .bytes_cached
                        .fetch_sub(old.data.len() as u64, Ordering::Relaxed);
                }

                self.stats
                    .bytes_cached
                    .fetch_add(entry.data.len() as u64, Ordering::Relaxed);
                cache.insert(key.to_string(), entry);
            }

            Ok(())
        })
    }

    fn delete<'a>(&'a self, key: &'a str) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            // Delete from backend
            self.inner.delete(key).await?;

            // Remove from cache
            self.invalidate(key);

            Ok(())
        })
    }

    fn list<'a>(&'a self, prefix: &'a str) -> Pin<Box<dyn Future<Output = Result<Vec<String>>> + Send + 'a>> {
        // List is not cached - always go to backend
        self.inner.list(prefix)
    }

    fn exists<'a>(&'a self, key: &'a str) -> Pin<Box<dyn Future<Output = Result<bool>> + Send + 'a>> {
        Box::pin(async move {
            // Check cache first
            if self.is_cached(key) {
                return Ok(true);
            }

            // Check backend
            self.inner.exists(key).await
        })
    }
}
