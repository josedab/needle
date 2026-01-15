//! Adaptive Index Service
//!
//! Database-level integration that wraps a collection with the adaptive runtime,
//! enabling automatic index selection, zero-downtime migration, and continuous
//! workload learning via a simple builder API.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::adaptive_service::{AdaptiveIndexService, AdaptiveServiceConfig};
//! use needle::Database;
//!
//! let db = Database::in_memory();
//! db.create_collection("docs", 128).unwrap();
//!
//! let config = AdaptiveServiceConfig::builder()
//!     .collection("docs")
//!     .memory_budget_mb(512)
//!     .build();
//!
//! let mut service = AdaptiveIndexService::new(&db, config).unwrap();
//!
//! // Insert — service tracks workload automatically
//! service.insert("v1", &vec![0.1f32; 128], None).unwrap();
//!
//! // Search — uses the best index for current workload
//! let results = service.search(&vec![0.1f32; 128], 10).unwrap();
//!
//! // Check what the system recommends
//! let report = service.status_report();
//! ```

use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::adaptive_runtime::{
    ActiveIndexType, AdaptiveRuntime, AdaptiveRuntimeConfig, MigrationPolicy, RuntimeHealth,
};
use crate::collection::SearchResult;
use crate::database::Database;
use crate::error::{NeedleError, Result};
use crate::metadata::Filter;

/// Configuration for the adaptive index service.
#[derive(Debug, Clone)]
pub struct AdaptiveServiceConfig {
    pub collection: String,
    pub memory_budget_bytes: usize,
    pub enable_auto_migration: bool,
    pub migration_cooldown_secs: u64,
}

impl Default for AdaptiveServiceConfig {
    fn default() -> Self {
        Self {
            collection: String::new(),
            memory_budget_bytes: 512 * 1024 * 1024,
            enable_auto_migration: true,
            migration_cooldown_secs: 300,
        }
    }
}

pub struct AdaptiveServiceConfigBuilder {
    config: AdaptiveServiceConfig,
}

impl AdaptiveServiceConfig {
    pub fn builder() -> AdaptiveServiceConfigBuilder {
        AdaptiveServiceConfigBuilder {
            config: Self::default(),
        }
    }
}

impl AdaptiveServiceConfigBuilder {
    pub fn collection(mut self, name: impl Into<String>) -> Self {
        self.config.collection = name.into();
        self
    }

    pub fn memory_budget_mb(mut self, mb: usize) -> Self {
        self.config.memory_budget_bytes = mb * 1024 * 1024;
        self
    }

    pub fn enable_auto_migration(mut self, enable: bool) -> Self {
        self.config.enable_auto_migration = enable;
        self
    }

    pub fn migration_cooldown_secs(mut self, secs: u64) -> Self {
        self.config.migration_cooldown_secs = secs;
        self
    }

    pub fn build(self) -> AdaptiveServiceConfig {
        self.config
    }
}

/// Status report from the adaptive index service.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveStatusReport {
    pub active_index: String,
    pub migration_phase: String,
    pub total_vectors: usize,
    pub total_searches: u64,
    pub total_inserts: u64,
    pub healthy: bool,
}

/// Database-level adaptive index management service.
pub struct AdaptiveIndexService<'a> {
    db: &'a Database,
    config: AdaptiveServiceConfig,
    runtime: AdaptiveRuntime,
    search_count: u64,
    insert_count: u64,
}

impl<'a> AdaptiveIndexService<'a> {
    /// Create a new adaptive index service for a collection.
    pub fn new(db: &'a Database, config: AdaptiveServiceConfig) -> Result<Self> {
        if config.collection.is_empty() {
            return Err(NeedleError::InvalidArgument(
                "collection name must not be empty".into(),
            ));
        }
        let coll_ref = db.collection(&config.collection)?;
        let dimension = coll_ref.dimensions().unwrap_or(128);

        let runtime_config = AdaptiveRuntimeConfig {
            memory_budget: config.memory_budget_bytes,
            auto_migrate: config.enable_auto_migration,
            migration_policy: MigrationPolicy {
                cooldown: Duration::from_secs(config.migration_cooldown_secs),
                ..MigrationPolicy::default()
            },
            ..AdaptiveRuntimeConfig::default()
        };

        let runtime = AdaptiveRuntime::new(dimension, runtime_config);

        Ok(Self {
            db,
            config,
            runtime,
            search_count: 0,
            insert_count: 0,
        })
    }

    /// Insert a vector, tracked by the adaptive runtime.
    pub fn insert(
        &mut self,
        id: impl Into<String>,
        vector: &[f32],
        metadata: Option<Value>,
    ) -> Result<()> {
        let id = id.into();
        // Insert into the real collection
        let coll = self.db.collection(&self.config.collection)?;
        coll.insert(&id, vector, metadata.clone())?;
        // Also track in the adaptive runtime for workload learning
        self.runtime.insert(id, vector, metadata)?;
        self.insert_count += 1;
        Ok(())
    }

    /// Search using the adaptive runtime's currently recommended index.
    pub fn search(&mut self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        let results = self.runtime.search(query, k)?;
        self.search_count += 1;
        Ok(results)
    }

    /// Search with a metadata filter.
    pub fn search_with_filter(
        &mut self,
        query: &[f32],
        k: usize,
        filter: &Filter,
    ) -> Result<Vec<SearchResult>> {
        let results = self.runtime.search_with_filter(query, k, filter)?;
        self.search_count += 1;
        Ok(results)
    }

    /// Get the currently active index type.
    pub fn active_index(&self) -> ActiveIndexType {
        self.runtime.active_index()
    }

    /// Force migration to a specific index type.
    pub fn force_migrate(&self, target: ActiveIndexType) -> Result<()> {
        self.runtime.force_migrate(target)
    }

    /// Get a comprehensive status report.
    pub fn status_report(&self) -> AdaptiveStatusReport {
        let health = self.runtime.health();
        AdaptiveStatusReport {
            active_index: format!("{}", self.runtime.active_index()),
            migration_phase: format!("{:?}", self.runtime.migration_status()),
            total_vectors: self.runtime.len(),
            total_searches: self.search_count,
            total_inserts: self.insert_count,
            healthy: health.vector_count > 0 || self.insert_count == 0,
        }
    }

    /// Get runtime health metrics.
    pub fn health(&self) -> RuntimeHealth {
        self.runtime.health()
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
    fn test_adaptive_service_basic() {
        let db = make_db();
        let config = AdaptiveServiceConfig::builder()
            .collection("test")
            .memory_budget_mb(256)
            .build();
        let mut svc = AdaptiveIndexService::new(&db, config).unwrap();

        svc.insert("v1", &[0.1, 0.2, 0.3, 0.4], None).unwrap();
        svc.insert("v2", &[0.5, 0.6, 0.7, 0.8], None).unwrap();

        let results = svc.search(&[0.1, 0.2, 0.3, 0.4], 2).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_adaptive_service_status() {
        let db = make_db();
        let config = AdaptiveServiceConfig::builder()
            .collection("test")
            .build();
        let svc = AdaptiveIndexService::new(&db, config).unwrap();

        let report = svc.status_report();
        assert_eq!(report.active_index, "HNSW");
    }

    #[test]
    fn test_empty_collection_rejected() {
        let db = make_db();
        let config = AdaptiveServiceConfig::builder().build();
        assert!(AdaptiveIndexService::new(&db, config).is_err());
    }
}
