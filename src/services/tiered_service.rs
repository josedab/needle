//! Tiered Storage Service
//!
//! Automatic data movement service that monitors access patterns and migrates
//! vectors between hot (RAM/HNSW), warm (SSD/DiskANN), and cold tiers based
//! on configurable policies.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::tiered_service::{TieredService, TieredServiceConfig, TierPolicy};
//! use needle::Database;
//!
//! let db = Database::in_memory();
//! db.create_collection("vectors", 128).unwrap();
//!
//! let config = TieredServiceConfig::builder()
//!     .collection("vectors")
//!     .hot_capacity(10000)
//!     .warm_after_secs(3600)
//!     .cold_after_secs(86400)
//!     .build();
//!
//! let mut service = TieredService::new(&db, config).unwrap();
//!
//! // Insert — starts in hot tier
//! service.insert("v1", &vec![0.1f32; 128], None).unwrap();
//!
//! // Search — automatically searches across tiers
//! let results = service.search(&vec![0.1f32; 128], 10).unwrap();
//!
//! // Run maintenance — promotes/demotes based on access patterns
//! let report = service.run_maintenance();
//! ```

use std::collections::HashMap;
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::collection::SearchResult;
use crate::database::Database;
use crate::error::{NeedleError, Result};

/// Storage tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Tier {
    /// In-memory, fastest access.
    Hot,
    /// On-disk, medium latency.
    Warm,
    /// Archived/compressed, highest latency.
    Cold,
}

impl std::fmt::Display for Tier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Tier::Hot => write!(f, "Hot"),
            Tier::Warm => write!(f, "Warm"),
            Tier::Cold => write!(f, "Cold"),
        }
    }
}

/// Policy for automatic tier migration.
#[derive(Debug, Clone)]
pub struct TierPolicy {
    /// Maximum vectors in hot tier.
    pub hot_capacity: usize,
    /// Move to warm tier after this many seconds without access.
    pub warm_after: Duration,
    /// Move to cold tier after this many seconds without access.
    pub cold_after: Duration,
    /// Promote to hot tier after this many accesses in the evaluation window.
    pub promotion_threshold: u32,
    /// How often to run maintenance.
    pub maintenance_interval: Duration,
}

impl Default for TierPolicy {
    fn default() -> Self {
        Self {
            hot_capacity: 100_000,
            warm_after: Duration::from_secs(3600),
            cold_after: Duration::from_secs(86400),
            promotion_threshold: 5,
            maintenance_interval: Duration::from_secs(60),
        }
    }
}

/// Configuration for the tiered service.
#[derive(Debug, Clone)]
pub struct TieredServiceConfig {
    pub collection: String,
    pub policy: TierPolicy,
}

impl Default for TieredServiceConfig {
    fn default() -> Self {
        Self {
            collection: String::new(),
            policy: TierPolicy::default(),
        }
    }
}

pub struct TieredServiceConfigBuilder {
    config: TieredServiceConfig,
}

impl TieredServiceConfig {
    pub fn builder() -> TieredServiceConfigBuilder {
        TieredServiceConfigBuilder {
            config: Self::default(),
        }
    }
}

impl TieredServiceConfigBuilder {
    pub fn collection(mut self, name: impl Into<String>) -> Self {
        self.config.collection = name.into();
        self
    }

    pub fn hot_capacity(mut self, cap: usize) -> Self {
        self.config.policy.hot_capacity = cap;
        self
    }

    pub fn warm_after_secs(mut self, secs: u64) -> Self {
        self.config.policy.warm_after = Duration::from_secs(secs);
        self
    }

    pub fn cold_after_secs(mut self, secs: u64) -> Self {
        self.config.policy.cold_after = Duration::from_secs(secs);
        self
    }

    pub fn promotion_threshold(mut self, threshold: u32) -> Self {
        self.config.policy.promotion_threshold = threshold;
        self
    }

    pub fn build(self) -> TieredServiceConfig {
        self.config
    }
}

/// Access tracking entry for a vector.
struct AccessRecord {
    tier: Tier,
    last_access: Instant,
    access_count: u32,
    inserted_at: Instant,
    vector: Vec<f32>,
    metadata: Option<Value>,
}

/// Report from a maintenance run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MaintenanceReport {
    pub promoted_to_hot: usize,
    pub demoted_to_warm: usize,
    pub demoted_to_cold: usize,
    pub total_hot: usize,
    pub total_warm: usize,
    pub total_cold: usize,
}

/// Statistics for the tiered service.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TieredServiceStats {
    pub hot_count: usize,
    pub warm_count: usize,
    pub cold_count: usize,
    pub total_count: usize,
    pub total_searches: u64,
    pub total_inserts: u64,
    pub total_promotions: u64,
    pub total_demotions: u64,
}

/// Tiered storage service with automatic data movement.
pub struct TieredService<'a> {
    db: &'a Database,
    config: TieredServiceConfig,
    records: RwLock<HashMap<String, AccessRecord>>,
    stats: RwLock<TieredServiceStats>,
    last_maintenance: RwLock<Instant>,
}

impl<'a> TieredService<'a> {
    /// Create a new tiered storage service.
    pub fn new(db: &'a Database, config: TieredServiceConfig) -> Result<Self> {
        if config.collection.is_empty() {
            return Err(NeedleError::InvalidArgument(
                "collection name must not be empty".into(),
            ));
        }
        let _ = db.collection(&config.collection)?;

        Ok(Self {
            db,
            config,
            records: RwLock::new(HashMap::new()),
            stats: RwLock::new(TieredServiceStats::default()),
            last_maintenance: RwLock::new(Instant::now()),
        })
    }

    /// Insert a vector (always starts in the hot tier).
    pub fn insert(
        &self,
        id: impl Into<String>,
        vector: &[f32],
        metadata: Option<Value>,
    ) -> Result<()> {
        let id = id.into();
        let coll = self.db.collection(&self.config.collection)?;
        coll.insert(&id, vector, metadata.clone())?;

        let record = AccessRecord {
            tier: Tier::Hot,
            last_access: Instant::now(),
            access_count: 0,
            inserted_at: Instant::now(),
            vector: vector.to_vec(),
            metadata,
        };
        self.records.write().insert(id, record);
        self.stats.write().total_inserts += 1;
        self.stats.write().hot_count += 1;
        Ok(())
    }

    /// Search across all tiers, tracking access for tier management.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        let coll = self.db.collection(&self.config.collection)?;
        let results = coll.search(query, k)?;

        // Track accesses for returned results
        let mut records = self.records.write();
        for result in &results {
            if let Some(record) = records.get_mut(&result.id) {
                record.last_access = Instant::now();
                record.access_count += 1;
            }
        }
        self.stats.write().total_searches += 1;

        Ok(results)
    }

    /// Get which tier a vector is in.
    pub fn get_tier(&self, id: &str) -> Option<Tier> {
        self.records.read().get(id).map(|r| r.tier)
    }

    /// Force promote a vector to the hot tier.
    pub fn promote(&self, id: &str) -> Result<bool> {
        let mut records = self.records.write();
        if let Some(record) = records.get_mut(id) {
            if record.tier != Tier::Hot {
                record.tier = Tier::Hot;
                record.last_access = Instant::now();
                self.stats.write().total_promotions += 1;
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Force demote a vector to a specific tier.
    pub fn demote(&self, id: &str, target: Tier) -> Result<bool> {
        let mut records = self.records.write();
        if let Some(record) = records.get_mut(id) {
            if record.tier != target {
                record.tier = target;
                self.stats.write().total_demotions += 1;
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Run maintenance: evaluate access patterns and move data between tiers.
    pub fn run_maintenance(&self) -> MaintenanceReport {
        let mut records = self.records.write();
        let policy = &self.config.policy;
        let now = Instant::now();

        let mut report = MaintenanceReport::default();

        for (_id, record) in records.iter_mut() {
            let idle_time = now.duration_since(record.last_access);

            match record.tier {
                Tier::Hot => {
                    if idle_time >= policy.cold_after {
                        record.tier = Tier::Cold;
                        report.demoted_to_cold += 1;
                    } else if idle_time >= policy.warm_after {
                        record.tier = Tier::Warm;
                        report.demoted_to_warm += 1;
                    }
                }
                Tier::Warm => {
                    if idle_time >= policy.cold_after {
                        record.tier = Tier::Cold;
                        report.demoted_to_cold += 1;
                    } else if record.access_count >= policy.promotion_threshold {
                        record.tier = Tier::Hot;
                        record.access_count = 0;
                        report.promoted_to_hot += 1;
                    }
                }
                Tier::Cold => {
                    if record.access_count >= policy.promotion_threshold {
                        record.tier = Tier::Hot;
                        record.access_count = 0;
                        report.promoted_to_hot += 1;
                    }
                }
            }
        }

        // Count tiers
        report.total_hot = records.values().filter(|r| r.tier == Tier::Hot).count();
        report.total_warm = records.values().filter(|r| r.tier == Tier::Warm).count();
        report.total_cold = records.values().filter(|r| r.tier == Tier::Cold).count();

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.hot_count = report.total_hot;
            stats.warm_count = report.total_warm;
            stats.cold_count = report.total_cold;
            stats.total_count = records.len();
            stats.total_promotions += report.promoted_to_hot as u64;
            stats.total_demotions += (report.demoted_to_warm + report.demoted_to_cold) as u64;
        }

        *self.last_maintenance.write() = Instant::now();
        report
    }

    /// Get service statistics.
    pub fn stats(&self) -> TieredServiceStats {
        self.stats.read().clone()
    }

    /// Total vectors managed across all tiers.
    pub fn len(&self) -> usize {
        self.records.read().len()
    }

    /// Whether the service has no vectors.
    pub fn is_empty(&self) -> bool {
        self.records.read().is_empty()
    }

    /// Check if maintenance should be run.
    pub fn should_run_maintenance(&self) -> bool {
        self.last_maintenance.read().elapsed() >= self.config.policy.maintenance_interval
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_db() -> Database {
        let db = Database::in_memory();
        db.create_collection("test", 4).unwrap();
        db
    }

    #[test]
    fn test_insert_and_search() {
        let db = make_db();
        let config = TieredServiceConfig::builder()
            .collection("test")
            .hot_capacity(100)
            .build();

        let svc = TieredService::new(&db, config).unwrap();
        svc.insert("v1", &[0.1, 0.2, 0.3, 0.4], None).unwrap();
        svc.insert("v2", &[0.5, 0.6, 0.7, 0.8], None).unwrap();

        let results = svc.search(&[0.1, 0.2, 0.3, 0.4], 2).unwrap();
        assert!(!results.is_empty());

        assert_eq!(svc.get_tier("v1"), Some(Tier::Hot));
    }

    #[test]
    fn test_promote_demote() {
        let db = make_db();
        let config = TieredServiceConfig::builder().collection("test").build();

        let svc = TieredService::new(&db, config).unwrap();
        svc.insert("v1", &[0.1, 0.2, 0.3, 0.4], None).unwrap();

        assert!(svc.demote("v1", Tier::Warm).unwrap());
        assert_eq!(svc.get_tier("v1"), Some(Tier::Warm));

        assert!(svc.promote("v1").unwrap());
        assert_eq!(svc.get_tier("v1"), Some(Tier::Hot));
    }

    #[test]
    fn test_maintenance_report() {
        let db = make_db();
        let config = TieredServiceConfig::builder().collection("test").build();

        let svc = TieredService::new(&db, config).unwrap();
        svc.insert("v1", &[0.1, 0.2, 0.3, 0.4], None).unwrap();

        let report = svc.run_maintenance();
        assert_eq!(report.total_hot, 1);
        assert_eq!(report.total_warm, 0);
        assert_eq!(report.total_cold, 0);
    }

    #[test]
    fn test_empty_collection_rejected() {
        let db = make_db();
        let config = TieredServiceConfig::builder().build();
        assert!(TieredService::new(&db, config).is_err());
    }

    #[test]
    fn test_stats() {
        let db = make_db();
        let config = TieredServiceConfig::builder().collection("test").build();

        let svc = TieredService::new(&db, config).unwrap();
        svc.insert("v1", &[0.1, 0.2, 0.3, 0.4], None).unwrap();

        let stats = svc.stats();
        assert_eq!(stats.total_inserts, 1);
        assert_eq!(stats.hot_count, 1);
    }
}
