//! Tiered Storage - Hot/Warm/Cold data management for vector databases.
//!
//! Implements automatic data tiering based on access patterns and age,
//! optimizing cost and performance for large-scale deployments.
//!
//! # Features
//!
//! - **Automatic tiering**: Vectors move between tiers based on policies
//! - **Access-based promotion**: Frequently accessed vectors stay hot
//! - **Age-based demotion**: Old vectors move to cheaper storage
//! - **Configurable policies**: Custom rules for tier transitions
//! - **Compression**: Warm/cold tiers use compression
//!
//! # Example
//!
//! ```ignore
//! use needle::tiered::{TieredStorage, TierPolicy, StorageTier};
//!
//! let mut storage = TieredStorage::new(TierPolicy::default());
//!
//! // Add vector (starts in hot tier)
//! storage.put("vec1", &embedding, metadata)?;
//!
//! // Access bumps to hot tier
//! let vec = storage.get("vec1")?;
//!
//! // Run maintenance to apply tiering
//! storage.run_maintenance()?;
//! ```

use crate::error::{NeedleError, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, BTreeMap};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Storage tier levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StorageTier {
    /// Hot tier: In-memory, fastest access, highest cost.
    Hot,
    /// Warm tier: Compressed in-memory, balanced performance/cost.
    Warm,
    /// Cold tier: On-disk, slowest access, lowest cost.
    Cold,
    /// Archive tier: Highly compressed, rarely accessed.
    Archive,
}

impl StorageTier {
    /// Get relative access latency (lower is faster).
    pub fn latency_factor(&self) -> f64 {
        match self {
            StorageTier::Hot => 1.0,
            StorageTier::Warm => 2.0,
            StorageTier::Cold => 10.0,
            StorageTier::Archive => 100.0,
        }
    }

    /// Get relative storage cost (lower is cheaper).
    pub fn cost_factor(&self) -> f64 {
        match self {
            StorageTier::Hot => 10.0,
            StorageTier::Warm => 5.0,
            StorageTier::Cold => 1.0,
            StorageTier::Archive => 0.1,
        }
    }

    /// Get compression ratio for this tier.
    pub fn compression_ratio(&self) -> f32 {
        match self {
            StorageTier::Hot => 1.0,
            StorageTier::Warm => 0.5,
            StorageTier::Cold => 0.25,
            StorageTier::Archive => 0.1,
        }
    }
}

/// Policy for tier transitions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TierPolicy {
    /// Time in hot tier before considering demotion.
    pub hot_retention: Duration,
    /// Time in warm tier before considering demotion.
    pub warm_retention: Duration,
    /// Time in cold tier before considering archive.
    pub cold_retention: Duration,
    /// Access count threshold for hot tier promotion.
    pub hot_access_threshold: u32,
    /// Access count threshold for warm tier promotion.
    pub warm_access_threshold: u32,
    /// Maximum vectors in hot tier.
    pub hot_tier_limit: Option<usize>,
    /// Maximum vectors in warm tier.
    pub warm_tier_limit: Option<usize>,
    /// Enable automatic tiering.
    pub auto_tier: bool,
    /// Interval between maintenance runs.
    pub maintenance_interval: Duration,
}

impl Default for TierPolicy {
    fn default() -> Self {
        Self {
            hot_retention: Duration::from_secs(3600),      // 1 hour
            warm_retention: Duration::from_secs(86400),    // 1 day
            cold_retention: Duration::from_secs(604800),   // 1 week
            hot_access_threshold: 10,
            warm_access_threshold: 3,
            hot_tier_limit: Some(100_000),
            warm_tier_limit: Some(1_000_000),
            auto_tier: true,
            maintenance_interval: Duration::from_secs(300), // 5 minutes
        }
    }
}

/// Metadata for a stored vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorMetadata {
    /// Vector ID.
    pub id: String,
    /// Current storage tier.
    pub tier: StorageTier,
    /// Creation timestamp.
    pub created_at: u64,
    /// Last access timestamp.
    pub last_accessed: u64,
    /// Last tier change timestamp.
    pub tier_changed_at: u64,
    /// Access count since last tier change.
    pub access_count: u32,
    /// Total access count.
    pub total_access_count: u64,
    /// Original size in bytes.
    pub original_size: usize,
    /// Compressed size in bytes.
    pub compressed_size: usize,
    /// User-provided metadata.
    pub user_metadata: HashMap<String, String>,
}

/// Stored vector data.
#[derive(Debug, Clone)]
struct StoredVector {
    /// Vector data (possibly compressed).
    data: Vec<u8>,
    /// Whether data is compressed.
    compressed: bool,
    /// Original dimensions.
    dimensions: usize,
}

/// Tiered storage system.
pub struct TieredStorage {
    /// Tiering policy.
    policy: TierPolicy,
    /// Vector metadata.
    metadata: HashMap<String, VectorMetadata>,
    /// Hot tier storage.
    hot_tier: HashMap<String, StoredVector>,
    /// Warm tier storage.
    warm_tier: HashMap<String, StoredVector>,
    /// Cold tier storage (simulated on-disk).
    cold_tier: HashMap<String, StoredVector>,
    /// Archive tier storage.
    archive_tier: HashMap<String, StoredVector>,
    /// Time-based index for maintenance.
    time_index: BTreeMap<u64, Vec<String>>,
    /// Last maintenance run.
    last_maintenance: u64,
    /// Statistics.
    stats: TierStats,
}

/// Statistics about tiered storage.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TierStats {
    /// Vectors in hot tier.
    pub hot_count: usize,
    /// Vectors in warm tier.
    pub warm_count: usize,
    /// Vectors in cold tier.
    pub cold_count: usize,
    /// Vectors in archive tier.
    pub archive_count: usize,
    /// Total bytes in hot tier.
    pub hot_bytes: usize,
    /// Total bytes in warm tier.
    pub warm_bytes: usize,
    /// Total bytes in cold tier.
    pub cold_bytes: usize,
    /// Total bytes in archive tier.
    pub archive_bytes: usize,
    /// Promotions performed.
    pub promotions: u64,
    /// Demotions performed.
    pub demotions: u64,
    /// Cache hits.
    pub cache_hits: u64,
    /// Cache misses.
    pub cache_misses: u64,
}

impl TieredStorage {
    /// Create new tiered storage.
    pub fn new(policy: TierPolicy) -> Self {
        Self {
            policy,
            metadata: HashMap::new(),
            hot_tier: HashMap::new(),
            warm_tier: HashMap::new(),
            cold_tier: HashMap::new(),
            archive_tier: HashMap::new(),
            time_index: BTreeMap::new(),
            last_maintenance: Self::now(),
            stats: TierStats::default(),
        }
    }

    /// Get current timestamp.
    fn now() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }

    /// Store a vector.
    pub fn put(
        &mut self,
        id: &str,
        vector: &[f32],
        user_metadata: HashMap<String, String>,
    ) -> Result<()> {
        let now = Self::now();
        let original_size = vector.len() * 4;

        // Store in hot tier (uncompressed)
        let data = Self::vector_to_bytes(vector);
        let stored = StoredVector {
            data: data.clone(),
            compressed: false,
            dimensions: vector.len(),
        };

        // Create metadata
        let meta = VectorMetadata {
            id: id.to_string(),
            tier: StorageTier::Hot,
            created_at: now,
            last_accessed: now,
            tier_changed_at: now,
            access_count: 0,
            total_access_count: 0,
            original_size,
            compressed_size: data.len(),
            user_metadata,
        };

        // Remove from any existing tier
        self.remove_from_tiers(id);

        // Add to hot tier
        self.hot_tier.insert(id.to_string(), stored);
        self.metadata.insert(id.to_string(), meta);
        self.stats.hot_count += 1;
        self.stats.hot_bytes += original_size;

        // Add to time index
        self.time_index
            .entry(now)
            .or_default()
            .push(id.to_string());

        Ok(())
    }

    /// Get a vector.
    pub fn get(&mut self, id: &str) -> Result<Vec<f32>> {
        // Get the tier before borrowing metadata mutably
        let tier = {
            let meta = self.metadata.get_mut(id)
                .ok_or_else(|| NeedleError::NotFound(format!("Vector '{}' not found", id)))?;

            // Update access stats
            meta.last_accessed = Self::now();
            meta.access_count += 1;
            meta.total_access_count += 1;
            meta.tier
        };

        // Find and return the vector - clone the data we need
        let (data, compressed, dimensions) = {
            let stored = self.find_stored_vector(id)?;
            (stored.data.clone(), stored.compressed, stored.dimensions)
        };

        // Track cache hit/miss (using saved tier value)
        if tier == StorageTier::Hot {
            self.stats.cache_hits += 1;
        } else {
            self.stats.cache_misses += 1;

            // Promote to hotter tier if access threshold met
            if self.policy.auto_tier {
                self.consider_promotion(id);
            }
        }

        // Decompress if needed
        let vector = if compressed {
            Self::decompress_vector(&data, dimensions)?
        } else {
            Self::bytes_to_vector(&data)
        };

        Ok(vector)
    }

    /// Check if vector exists.
    pub fn contains(&self, id: &str) -> bool {
        self.metadata.contains_key(id)
    }

    /// Delete a vector.
    pub fn delete(&mut self, id: &str) -> Result<()> {
        if !self.metadata.contains_key(id) {
            return Err(NeedleError::NotFound(format!("Vector '{}' not found", id)));
        }

        self.remove_from_tiers(id);
        self.metadata.remove(id);

        Ok(())
    }

    /// Get vector metadata.
    pub fn get_metadata(&self, id: &str) -> Option<&VectorMetadata> {
        self.metadata.get(id)
    }

    /// Get current tier for a vector.
    pub fn get_tier(&self, id: &str) -> Option<StorageTier> {
        self.metadata.get(id).map(|m| m.tier)
    }

    /// Manually move vector to a specific tier.
    pub fn move_to_tier(&mut self, id: &str, target_tier: StorageTier) -> Result<()> {
        let current_tier = self.metadata.get(id)
            .ok_or_else(|| NeedleError::NotFound(format!("Vector '{}' not found", id)))?
            .tier;

        if current_tier == target_tier {
            return Ok(());
        }

        // Get the vector data
        let stored = self.find_stored_vector(id)?.clone();
        let dimensions = stored.dimensions;

        // Get uncompressed data
        let raw_data = if stored.compressed {
            let vector = Self::decompress_vector(&stored.data, dimensions)?;
            Self::vector_to_bytes(&vector)
        } else {
            stored.data
        };

        // Remove from current tier
        self.remove_from_tiers(id);

        // Prepare data for target tier
        let new_stored = match target_tier {
            StorageTier::Hot => StoredVector {
                data: raw_data,
                compressed: false,
                dimensions,
            },
            StorageTier::Warm => StoredVector {
                data: Self::compress_data(&raw_data, target_tier.compression_ratio()),
                compressed: true,
                dimensions,
            },
            StorageTier::Cold | StorageTier::Archive => StoredVector {
                data: Self::compress_data(&raw_data, target_tier.compression_ratio()),
                compressed: true,
                dimensions,
            },
        };

        // Insert into target tier
        let compressed_size = new_stored.data.len();
        match target_tier {
            StorageTier::Hot => {
                self.hot_tier.insert(id.to_string(), new_stored);
                self.stats.hot_count += 1;
                self.stats.hot_bytes += dimensions * 4;
            }
            StorageTier::Warm => {
                self.warm_tier.insert(id.to_string(), new_stored);
                self.stats.warm_count += 1;
                self.stats.warm_bytes += compressed_size;
            }
            StorageTier::Cold => {
                self.cold_tier.insert(id.to_string(), new_stored);
                self.stats.cold_count += 1;
                self.stats.cold_bytes += compressed_size;
            }
            StorageTier::Archive => {
                self.archive_tier.insert(id.to_string(), new_stored);
                self.stats.archive_count += 1;
                self.stats.archive_bytes += compressed_size;
            }
        }

        // Update metadata
        if let Some(meta) = self.metadata.get_mut(id) {
            meta.tier = target_tier;
            meta.tier_changed_at = Self::now();
            meta.access_count = 0;
            meta.compressed_size = compressed_size;
        }

        // Track promotion/demotion
        if target_tier.cost_factor() > current_tier.cost_factor() {
            self.stats.promotions += 1;
        } else {
            self.stats.demotions += 1;
        }

        Ok(())
    }

    /// Run maintenance to apply tiering policies.
    pub fn run_maintenance(&mut self) -> Result<MaintenanceReport> {
        let now = Self::now();
        let mut report = MaintenanceReport::default();

        // Collect vectors to process
        let vector_ids: Vec<String> = self.metadata.keys().cloned().collect();

        for id in vector_ids {
            let meta = match self.metadata.get(&id) {
                Some(m) => m.clone(),
                None => continue,
            };

            let age_in_tier = now - meta.tier_changed_at;

            // Check demotion rules
            let should_demote = match meta.tier {
                StorageTier::Hot => {
                    age_in_tier > self.policy.hot_retention.as_secs()
                        && meta.access_count < self.policy.hot_access_threshold
                }
                StorageTier::Warm => {
                    age_in_tier > self.policy.warm_retention.as_secs()
                        && meta.access_count < self.policy.warm_access_threshold
                }
                StorageTier::Cold => {
                    age_in_tier > self.policy.cold_retention.as_secs()
                }
                StorageTier::Archive => false,
            };

            if should_demote {
                let target = match meta.tier {
                    StorageTier::Hot => StorageTier::Warm,
                    StorageTier::Warm => StorageTier::Cold,
                    StorageTier::Cold => StorageTier::Archive,
                    StorageTier::Archive => StorageTier::Archive,
                };

                if target != meta.tier {
                    self.move_to_tier(&id, target)?;
                    report.demotions += 1;
                }
            }
        }

        // Check tier limits
        self.enforce_tier_limits(&mut report)?;

        self.last_maintenance = now;
        report.duration_ms = (Self::now() - now) * 1000;

        Ok(report)
    }

    /// Enforce tier size limits.
    fn enforce_tier_limits(&mut self, report: &mut MaintenanceReport) -> Result<()> {
        // Check hot tier limit
        if let Some(limit) = self.policy.hot_tier_limit {
            while self.stats.hot_count > limit {
                // Find least recently accessed vector in hot tier
                if let Some(id) = self.find_lru_vector(StorageTier::Hot) {
                    self.move_to_tier(&id, StorageTier::Warm)?;
                    report.demotions += 1;
                } else {
                    break;
                }
            }
        }

        // Check warm tier limit
        if let Some(limit) = self.policy.warm_tier_limit {
            while self.stats.warm_count > limit {
                if let Some(id) = self.find_lru_vector(StorageTier::Warm) {
                    self.move_to_tier(&id, StorageTier::Cold)?;
                    report.demotions += 1;
                } else {
                    break;
                }
            }
        }

        Ok(())
    }

    /// Find least recently used vector in a tier.
    fn find_lru_vector(&self, tier: StorageTier) -> Option<String> {
        self.metadata
            .iter()
            .filter(|(_, m)| m.tier == tier)
            .min_by_key(|(_, m)| m.last_accessed)
            .map(|(id, _)| id.clone())
    }

    /// Consider promoting a vector to a hotter tier.
    fn consider_promotion(&mut self, id: &str) {
        let meta = match self.metadata.get(id) {
            Some(m) => m.clone(),
            None => return,
        };

        let should_promote = match meta.tier {
            StorageTier::Archive | StorageTier::Cold => {
                meta.access_count >= self.policy.warm_access_threshold
            }
            StorageTier::Warm => {
                meta.access_count >= self.policy.hot_access_threshold
            }
            StorageTier::Hot => false,
        };

        if should_promote {
            let target = match meta.tier {
                StorageTier::Archive => StorageTier::Cold,
                StorageTier::Cold => StorageTier::Warm,
                StorageTier::Warm => StorageTier::Hot,
                StorageTier::Hot => StorageTier::Hot,
            };

            if target != meta.tier {
                let _ = self.move_to_tier(id, target);
            }
        }
    }

    /// Remove vector from all tiers.
    fn remove_from_tiers(&mut self, id: &str) {
        if let Some(stored) = self.hot_tier.remove(id) {
            self.stats.hot_count = self.stats.hot_count.saturating_sub(1);
            self.stats.hot_bytes = self.stats.hot_bytes.saturating_sub(stored.dimensions * 4);
        }
        if let Some(stored) = self.warm_tier.remove(id) {
            self.stats.warm_count = self.stats.warm_count.saturating_sub(1);
            self.stats.warm_bytes = self.stats.warm_bytes.saturating_sub(stored.data.len());
        }
        if let Some(stored) = self.cold_tier.remove(id) {
            self.stats.cold_count = self.stats.cold_count.saturating_sub(1);
            self.stats.cold_bytes = self.stats.cold_bytes.saturating_sub(stored.data.len());
        }
        if let Some(stored) = self.archive_tier.remove(id) {
            self.stats.archive_count = self.stats.archive_count.saturating_sub(1);
            self.stats.archive_bytes = self.stats.archive_bytes.saturating_sub(stored.data.len());
        }
    }

    /// Find stored vector across tiers.
    fn find_stored_vector(&self, id: &str) -> Result<&StoredVector> {
        if let Some(stored) = self.hot_tier.get(id) {
            return Ok(stored);
        }
        if let Some(stored) = self.warm_tier.get(id) {
            return Ok(stored);
        }
        if let Some(stored) = self.cold_tier.get(id) {
            return Ok(stored);
        }
        if let Some(stored) = self.archive_tier.get(id) {
            return Ok(stored);
        }
        Err(NeedleError::NotFound(format!("Vector '{}' not found in any tier", id)))
    }

    /// Convert vector to bytes.
    fn vector_to_bytes(vector: &[f32]) -> Vec<u8> {
        vector.iter().flat_map(|f| f.to_le_bytes()).collect()
    }

    /// Convert bytes to vector.
    fn bytes_to_vector(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()
    }

    /// Simple compression (simulated - in production use real compression).
    fn compress_data(data: &[u8], _ratio: f32) -> Vec<u8> {
        // In production, use lz4, zstd, or snappy
        // For now, just return the data as-is
        data.to_vec()
    }

    /// Decompress vector data.
    fn decompress_vector(data: &[u8], _dimensions: usize) -> Result<Vec<f32>> {
        // In production, decompress first
        Ok(Self::bytes_to_vector(data))
    }

    /// Get storage statistics.
    pub fn stats(&self) -> &TierStats {
        &self.stats
    }

    /// Get total vector count.
    pub fn len(&self) -> usize {
        self.metadata.len()
    }

    /// Check if storage is empty.
    pub fn is_empty(&self) -> bool {
        self.metadata.is_empty()
    }

    /// List vectors in a specific tier.
    pub fn list_tier(&self, tier: StorageTier) -> Vec<String> {
        self.metadata
            .iter()
            .filter(|(_, m)| m.tier == tier)
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Get cost estimate for current storage.
    pub fn estimate_cost(&self) -> TierCostEstimate {
        let hot_cost = self.stats.hot_bytes as f64 * StorageTier::Hot.cost_factor();
        let warm_cost = self.stats.warm_bytes as f64 * StorageTier::Warm.cost_factor();
        let cold_cost = self.stats.cold_bytes as f64 * StorageTier::Cold.cost_factor();
        let archive_cost = self.stats.archive_bytes as f64 * StorageTier::Archive.cost_factor();

        TierCostEstimate {
            hot_cost,
            warm_cost,
            cold_cost,
            archive_cost,
            total_cost: hot_cost + warm_cost + cold_cost + archive_cost,
            potential_savings: self.calculate_potential_savings(),
        }
    }

    /// Calculate potential savings from optimal tiering.
    fn calculate_potential_savings(&self) -> f64 {
        // Estimate savings if all cold-eligible data moved to cold tier
        let hot_to_warm_savings = self.stats.hot_bytes as f64
            * (StorageTier::Hot.cost_factor() - StorageTier::Warm.cost_factor())
            * 0.3; // Assume 30% could be demoted

        let warm_to_cold_savings = self.stats.warm_bytes as f64
            * (StorageTier::Warm.cost_factor() - StorageTier::Cold.cost_factor())
            * 0.2; // Assume 20% could be demoted

        hot_to_warm_savings + warm_to_cold_savings
    }
}

/// Report from maintenance run.
#[derive(Debug, Clone, Default)]
pub struct MaintenanceReport {
    /// Number of promotions.
    pub promotions: usize,
    /// Number of demotions.
    pub demotions: usize,
    /// Vectors processed.
    pub vectors_processed: usize,
    /// Duration in milliseconds.
    pub duration_ms: u64,
}

/// Cost estimate for tiered storage.
#[derive(Debug, Clone)]
pub struct TierCostEstimate {
    /// Cost for hot tier.
    pub hot_cost: f64,
    /// Cost for warm tier.
    pub warm_cost: f64,
    /// Cost for cold tier.
    pub cold_cost: f64,
    /// Cost for archive tier.
    pub archive_cost: f64,
    /// Total cost.
    pub total_cost: f64,
    /// Potential savings from better tiering.
    pub potential_savings: f64,
}

/// Builder for tiered storage policies.
pub struct TierPolicyBuilder {
    policy: TierPolicy,
}

impl TierPolicyBuilder {
    /// Create new builder with defaults.
    pub fn new() -> Self {
        Self {
            policy: TierPolicy::default(),
        }
    }

    /// Set hot tier retention time.
    pub fn hot_retention(mut self, duration: Duration) -> Self {
        self.policy.hot_retention = duration;
        self
    }

    /// Set warm tier retention time.
    pub fn warm_retention(mut self, duration: Duration) -> Self {
        self.policy.warm_retention = duration;
        self
    }

    /// Set cold tier retention time.
    pub fn cold_retention(mut self, duration: Duration) -> Self {
        self.policy.cold_retention = duration;
        self
    }

    /// Set hot tier access threshold.
    pub fn hot_access_threshold(mut self, count: u32) -> Self {
        self.policy.hot_access_threshold = count;
        self
    }

    /// Set warm tier access threshold.
    pub fn warm_access_threshold(mut self, count: u32) -> Self {
        self.policy.warm_access_threshold = count;
        self
    }

    /// Set hot tier limit.
    pub fn hot_tier_limit(mut self, limit: usize) -> Self {
        self.policy.hot_tier_limit = Some(limit);
        self
    }

    /// Set warm tier limit.
    pub fn warm_tier_limit(mut self, limit: usize) -> Self {
        self.policy.warm_tier_limit = Some(limit);
        self
    }

    /// Enable/disable auto tiering.
    pub fn auto_tier(mut self, enabled: bool) -> Self {
        self.policy.auto_tier = enabled;
        self
    }

    /// Build the policy.
    pub fn build(self) -> TierPolicy {
        self.policy
    }
}

impl Default for TierPolicyBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_storage_tier_properties() {
        assert!(StorageTier::Hot.latency_factor() < StorageTier::Cold.latency_factor());
        assert!(StorageTier::Hot.cost_factor() > StorageTier::Cold.cost_factor());
        assert!(StorageTier::Archive.compression_ratio() < StorageTier::Hot.compression_ratio());
    }

    #[test]
    fn test_put_and_get() {
        let mut storage = TieredStorage::new(TierPolicy::default());

        let vector = vec![1.0, 2.0, 3.0, 4.0];
        storage.put("vec1", &vector, HashMap::new()).unwrap();

        let retrieved = storage.get("vec1").unwrap();
        assert_eq!(retrieved, vector);
    }

    #[test]
    fn test_initial_tier_is_hot() {
        let mut storage = TieredStorage::new(TierPolicy::default());

        storage.put("vec1", &[1.0, 2.0, 3.0], HashMap::new()).unwrap();

        assert_eq!(storage.get_tier("vec1"), Some(StorageTier::Hot));
    }

    #[test]
    fn test_manual_tier_move() {
        let mut storage = TieredStorage::new(TierPolicy::default());

        storage.put("vec1", &[1.0, 2.0, 3.0, 4.0], HashMap::new()).unwrap();
        assert_eq!(storage.get_tier("vec1"), Some(StorageTier::Hot));

        storage.move_to_tier("vec1", StorageTier::Warm).unwrap();
        assert_eq!(storage.get_tier("vec1"), Some(StorageTier::Warm));

        storage.move_to_tier("vec1", StorageTier::Cold).unwrap();
        assert_eq!(storage.get_tier("vec1"), Some(StorageTier::Cold));

        // Vector should still be retrievable
        let retrieved = storage.get("vec1").unwrap();
        assert_eq!(retrieved, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_delete() {
        let mut storage = TieredStorage::new(TierPolicy::default());

        storage.put("vec1", &[1.0, 2.0], HashMap::new()).unwrap();
        assert!(storage.contains("vec1"));

        storage.delete("vec1").unwrap();
        assert!(!storage.contains("vec1"));
    }

    #[test]
    fn test_stats_tracking() {
        let mut storage = TieredStorage::new(TierPolicy::default());

        storage.put("vec1", &[1.0, 2.0, 3.0, 4.0], HashMap::new()).unwrap();
        storage.put("vec2", &[5.0, 6.0, 7.0, 8.0], HashMap::new()).unwrap();

        let stats = storage.stats();
        assert_eq!(stats.hot_count, 2);
        assert_eq!(stats.hot_bytes, 32); // 2 vectors * 4 floats * 4 bytes
    }

    #[test]
    fn test_access_tracking() {
        let mut storage = TieredStorage::new(TierPolicy::default());

        storage.put("vec1", &[1.0, 2.0], HashMap::new()).unwrap();

        // Access multiple times
        for _ in 0..5 {
            let _ = storage.get("vec1");
        }

        let meta = storage.get_metadata("vec1").unwrap();
        assert_eq!(meta.total_access_count, 5);
    }

    #[test]
    fn test_user_metadata() {
        let mut storage = TieredStorage::new(TierPolicy::default());

        let mut user_meta = HashMap::new();
        user_meta.insert("source".to_string(), "test".to_string());
        user_meta.insert("category".to_string(), "example".to_string());

        storage.put("vec1", &[1.0, 2.0], user_meta).unwrap();

        let meta = storage.get_metadata("vec1").unwrap();
        assert_eq!(meta.user_metadata.get("source"), Some(&"test".to_string()));
    }

    #[test]
    fn test_list_tier() {
        let mut storage = TieredStorage::new(TierPolicy::default());

        storage.put("vec1", &[1.0, 2.0], HashMap::new()).unwrap();
        storage.put("vec2", &[3.0, 4.0], HashMap::new()).unwrap();
        storage.put("vec3", &[5.0, 6.0], HashMap::new()).unwrap();

        // Move some to warm tier
        storage.move_to_tier("vec2", StorageTier::Warm).unwrap();

        let hot_list = storage.list_tier(StorageTier::Hot);
        let warm_list = storage.list_tier(StorageTier::Warm);

        assert_eq!(hot_list.len(), 2);
        assert_eq!(warm_list.len(), 1);
        assert!(warm_list.contains(&"vec2".to_string()));
    }

    #[test]
    fn test_cost_estimate() {
        let mut storage = TieredStorage::new(TierPolicy::default());

        storage.put("vec1", &[1.0, 2.0, 3.0, 4.0], HashMap::new()).unwrap();
        storage.put("vec2", &[1.0, 2.0, 3.0, 4.0], HashMap::new()).unwrap();

        let cost = storage.estimate_cost();
        assert!(cost.hot_cost > 0.0);
        assert!(cost.total_cost > 0.0);
    }

    #[test]
    fn test_policy_builder() {
        let policy = TierPolicyBuilder::new()
            .hot_retention(Duration::from_secs(7200))
            .warm_retention(Duration::from_secs(172800))
            .hot_access_threshold(20)
            .hot_tier_limit(50000)
            .auto_tier(true)
            .build();

        assert_eq!(policy.hot_retention.as_secs(), 7200);
        assert_eq!(policy.warm_retention.as_secs(), 172800);
        assert_eq!(policy.hot_access_threshold, 20);
        assert_eq!(policy.hot_tier_limit, Some(50000));
    }

    #[test]
    fn test_tier_limit_enforcement() {
        let policy = TierPolicyBuilder::new()
            .hot_tier_limit(2)
            .build();

        let mut storage = TieredStorage::new(policy);

        // Add more than limit
        storage.put("vec1", &[1.0, 2.0], HashMap::new()).unwrap();
        storage.put("vec2", &[3.0, 4.0], HashMap::new()).unwrap();
        storage.put("vec3", &[5.0, 6.0], HashMap::new()).unwrap();
        storage.put("vec4", &[7.0, 8.0], HashMap::new()).unwrap();

        // Run maintenance to enforce limits
        let report = storage.run_maintenance().unwrap();

        // Some vectors should have been demoted
        assert!(storage.stats().hot_count <= 2);
        assert!(report.demotions > 0);
    }

    #[test]
    fn test_cache_hit_tracking() {
        let mut storage = TieredStorage::new(TierPolicy::default());

        storage.put("vec1", &[1.0, 2.0], HashMap::new()).unwrap();

        // Hot tier access = cache hit
        let _ = storage.get("vec1");
        assert_eq!(storage.stats().cache_hits, 1);

        // Move to cold and access = cache miss
        storage.move_to_tier("vec1", StorageTier::Cold).unwrap();
        let _ = storage.get("vec1");
        assert_eq!(storage.stats().cache_misses, 1);
    }

    #[test]
    fn test_promotion_on_access() {
        let policy = TierPolicyBuilder::new()
            .warm_access_threshold(2)
            .auto_tier(true)
            .build();

        let mut storage = TieredStorage::new(policy);

        storage.put("vec1", &[1.0, 2.0], HashMap::new()).unwrap();
        storage.move_to_tier("vec1", StorageTier::Cold).unwrap();

        assert_eq!(storage.get_tier("vec1"), Some(StorageTier::Cold));

        // Access enough times to trigger promotion
        for _ in 0..3 {
            let _ = storage.get("vec1");
        }

        // Should have been promoted to warm
        assert_eq!(storage.get_tier("vec1"), Some(StorageTier::Warm));
    }

    #[test]
    fn test_len_and_is_empty() {
        let mut storage = TieredStorage::new(TierPolicy::default());

        assert!(storage.is_empty());
        assert_eq!(storage.len(), 0);

        storage.put("vec1", &[1.0, 2.0], HashMap::new()).unwrap();

        assert!(!storage.is_empty());
        assert_eq!(storage.len(), 1);
    }
}
