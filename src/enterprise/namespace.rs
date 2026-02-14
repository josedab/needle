//! Namespace and Multi-Tenancy
//!
//! Provides isolation and multi-tenant support for Needle:
//! - Namespace-based collection isolation
//! - Tenant management and quotas
//! - Access control primitives
//! - Resource limits per namespace
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::namespace::{NamespaceManager, Namespace, TenantConfig};
//!
//! let manager = NamespaceManager::new();
//!
//! // Create namespace for a tenant
//! let namespace = manager.create_namespace("tenant_123", TenantConfig::default())?;
//!
//! // Create collection within namespace
//! namespace.create_collection("documents", 384)?;
//!
//! // Access collections with isolation
//! let collection = namespace.collection("documents")?;
//! ```

use crate::database::Database;
use crate::error::{NeedleError, Result};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Configuration for a tenant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TenantConfig {
    /// Maximum number of collections
    pub max_collections: Option<usize>,
    /// Maximum total vectors across all collections
    pub max_vectors: Option<usize>,
    /// Maximum storage size in bytes
    pub max_storage_bytes: Option<u64>,
    /// Maximum vector dimensions
    pub max_dimensions: Option<usize>,
    /// Rate limit: operations per second
    pub rate_limit_ops: Option<u32>,
    /// Enable read-only mode
    pub read_only: bool,
    /// Custom metadata
    pub metadata: Option<serde_json::Value>,
}

impl Default for TenantConfig {
    fn default() -> Self {
        Self {
            max_collections: Some(100),
            max_vectors: Some(1_000_000),
            max_storage_bytes: Some(10 * 1024 * 1024 * 1024), // 10 GB
            max_dimensions: Some(4096),
            rate_limit_ops: None,
            read_only: false,
            metadata: None,
        }
    }
}

impl TenantConfig {
    /// Create unlimited config (no quotas)
    pub fn unlimited() -> Self {
        Self {
            max_collections: None,
            max_vectors: None,
            max_storage_bytes: None,
            max_dimensions: None,
            rate_limit_ops: None,
            read_only: false,
            metadata: None,
        }
    }

    /// Create read-only config
    pub fn read_only() -> Self {
        Self {
            read_only: true,
            ..Default::default()
        }
    }
}

/// Namespace for tenant isolation
pub struct Namespace {
    /// Namespace ID
    id: String,
    /// Configuration
    config: TenantConfig,
    /// Underlying database
    db: Arc<Database>,
    /// Collection prefix for isolation
    prefix: String,
    /// Usage statistics
    stats: Arc<NamespaceStats>,
}

/// Namespace usage statistics
#[derive(Debug, Default)]
pub struct NamespaceStats {
    /// Total vectors stored
    pub total_vectors: AtomicU64,
    /// Total collections
    pub total_collections: AtomicU64,
    /// Operations count
    pub operations: AtomicU64,
    /// Read operations
    pub reads: AtomicU64,
    /// Write operations
    pub writes: AtomicU64,
    /// Search operations
    pub searches: AtomicU64,
    /// Estimated storage bytes
    pub storage_bytes: AtomicU64,
}

impl NamespaceStats {
    pub fn snapshot(&self) -> NamespaceStatsSnapshot {
        NamespaceStatsSnapshot {
            total_vectors: self.total_vectors.load(Ordering::Relaxed),
            total_collections: self.total_collections.load(Ordering::Relaxed),
            operations: self.operations.load(Ordering::Relaxed),
            reads: self.reads.load(Ordering::Relaxed),
            writes: self.writes.load(Ordering::Relaxed),
            searches: self.searches.load(Ordering::Relaxed),
            storage_bytes: self.storage_bytes.load(Ordering::Relaxed),
        }
    }
}

/// Snapshot of namespace statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamespaceStatsSnapshot {
    pub total_vectors: u64,
    pub total_collections: u64,
    pub operations: u64,
    pub reads: u64,
    pub writes: u64,
    pub searches: u64,
    pub storage_bytes: u64,
}

impl Namespace {
    /// Create a new namespace
    fn new(id: String, config: TenantConfig, db: Arc<Database>) -> Self {
        let prefix = format!("ns_{}__", id);
        Self {
            id,
            config,
            db,
            prefix,
            stats: Arc::new(NamespaceStats::default()),
        }
    }

    /// Get namespace ID
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Get configuration
    pub fn config(&self) -> &TenantConfig {
        &self.config
    }

    /// Get statistics
    pub fn stats(&self) -> NamespaceStatsSnapshot {
        self.stats.snapshot()
    }

    /// Create a collection in this namespace
    pub fn create_collection(&self, name: &str, dimensions: usize) -> Result<()> {
        self.check_write_access()?;

        // Check dimension limit
        if let Some(max_dims) = self.config.max_dimensions {
            if dimensions > max_dims {
                return Err(NeedleError::InvalidInput(format!(
                    "Dimensions {} exceeds maximum {}",
                    dimensions, max_dims
                )));
            }
        }

        // Check collection limit
        if let Some(max_colls) = self.config.max_collections {
            let current = self.stats.total_collections.load(Ordering::Relaxed) as usize;
            if current >= max_colls {
                return Err(NeedleError::QuotaExceeded(format!(
                    "Maximum collections ({}) reached",
                    max_colls
                )));
            }
        }

        let prefixed_name = self.prefixed_name(name);
        self.db.create_collection(&prefixed_name, dimensions)?;

        self.stats.total_collections.fetch_add(1, Ordering::Relaxed);
        self.stats.operations.fetch_add(1, Ordering::Relaxed);
        self.stats.writes.fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    /// Get a collection from this namespace
    pub fn collection(&self, name: &str) -> Result<NamespaceCollection<'_>> {
        let prefixed_name = self.prefixed_name(name);
        let coll_ref = self.db.collection(&prefixed_name)?;

        self.stats.operations.fetch_add(1, Ordering::Relaxed);
        self.stats.reads.fetch_add(1, Ordering::Relaxed);

        Ok(NamespaceCollection {
            inner: coll_ref,
            stats: Arc::clone(&self.stats),
            config: self.config.clone(),
        })
    }

    /// Delete a collection from this namespace
    pub fn delete_collection(&self, name: &str) -> Result<bool> {
        self.check_write_access()?;

        let prefixed_name = self.prefixed_name(name);
        let deleted = self.db.delete_collection(&prefixed_name)?;

        if deleted {
            self.stats.total_collections.fetch_sub(1, Ordering::Relaxed);
        }

        self.stats.operations.fetch_add(1, Ordering::Relaxed);
        self.stats.writes.fetch_add(1, Ordering::Relaxed);

        Ok(deleted)
    }

    /// List collections in this namespace
    pub fn list_collections(&self) -> Vec<String> {
        let all_collections = self.db.list_collections();

        self.stats.operations.fetch_add(1, Ordering::Relaxed);
        self.stats.reads.fetch_add(1, Ordering::Relaxed);

        // Filter and remove prefix
        all_collections
            .into_iter()
            .filter_map(|name| {
                if name.starts_with(&self.prefix) {
                    Some(name[self.prefix.len()..].to_string())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Check if collection exists
    pub fn has_collection(&self, name: &str) -> bool {
        let prefixed_name = self.prefixed_name(name);
        self.db.collection(&prefixed_name).is_ok()
    }

    /// Prefixed collection name
    fn prefixed_name(&self, name: &str) -> String {
        format!("{}{}", self.prefix, name)
    }

    /// Check write access
    fn check_write_access(&self) -> Result<()> {
        if self.config.read_only {
            Err(NeedleError::InvalidInput(
                "Namespace is read-only".to_string(),
            ))
        } else {
            Ok(())
        }
    }
}

/// Collection within a namespace (with quotas)
pub struct NamespaceCollection<'a> {
    inner: crate::database::CollectionRef<'a>,
    stats: Arc<NamespaceStats>,
    config: TenantConfig,
}

impl<'a> NamespaceCollection<'a> {
    /// Insert a vector
    pub fn insert(
        &self,
        id: impl Into<String>,
        vector: &[f32],
        metadata: Option<serde_json::Value>,
    ) -> Result<()> {
        if self.config.read_only {
            return Err(NeedleError::InvalidInput(
                "Namespace is read-only".to_string(),
            ));
        }

        // Check vector limit
        if let Some(max_vectors) = self.config.max_vectors {
            let current = self.stats.total_vectors.load(Ordering::Relaxed) as usize;
            if current >= max_vectors {
                return Err(NeedleError::QuotaExceeded(format!(
                    "Maximum vectors ({}) reached",
                    max_vectors
                )));
            }
        }

        self.inner.insert(id, vector, metadata)?;

        self.stats.total_vectors.fetch_add(1, Ordering::Relaxed);
        self.stats.operations.fetch_add(1, Ordering::Relaxed);
        self.stats.writes.fetch_add(1, Ordering::Relaxed);

        // Estimate storage increase
        let size_estimate = (vector.len() * 4 + 100) as u64; // 4 bytes per float + overhead
        self.stats
            .storage_bytes
            .fetch_add(size_estimate, Ordering::Relaxed);

        Ok(())
    }

    /// Search vectors
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<crate::collection::SearchResult>> {
        self.stats.operations.fetch_add(1, Ordering::Relaxed);
        self.stats.searches.fetch_add(1, Ordering::Relaxed);

        self.inner.search(query, k)
    }

    /// Search with filter
    pub fn search_with_filter(
        &self,
        query: &[f32],
        k: usize,
        filter: &crate::metadata::Filter,
    ) -> Result<Vec<crate::collection::SearchResult>> {
        self.stats.operations.fetch_add(1, Ordering::Relaxed);
        self.stats.searches.fetch_add(1, Ordering::Relaxed);

        self.inner.search_with_filter(query, k, filter)
    }

    /// Get a vector by ID
    pub fn get(&self, id: &str) -> Option<(Vec<f32>, Option<serde_json::Value>)> {
        self.stats.operations.fetch_add(1, Ordering::Relaxed);
        self.stats.reads.fetch_add(1, Ordering::Relaxed);

        self.inner.get(id)
    }

    /// Delete a vector
    pub fn delete(&self, id: &str) -> Result<bool> {
        if self.config.read_only {
            return Err(NeedleError::InvalidInput(
                "Namespace is read-only".to_string(),
            ));
        }

        let deleted = self.inner.delete(id)?;

        if deleted {
            self.stats.total_vectors.fetch_sub(1, Ordering::Relaxed);
        }

        self.stats.operations.fetch_add(1, Ordering::Relaxed);
        self.stats.writes.fetch_add(1, Ordering::Relaxed);

        Ok(deleted)
    }

    /// Count vectors
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.inner.len() == 0
    }

    /// Get dimensions
    pub fn dimensions(&self) -> usize {
        self.inner.dimensions().unwrap_or(0)
    }
}

/// Manager for namespaces
pub struct NamespaceManager {
    /// Shared database
    db: Arc<Database>,
    /// Registered namespaces
    namespaces: RwLock<HashMap<String, Arc<Namespace>>>,
    /// Default config for new namespaces
    default_config: TenantConfig,
}

impl NamespaceManager {
    /// Create a new namespace manager
    pub fn new() -> Self {
        Self::with_database(Database::in_memory())
    }

    /// Create with existing database
    pub fn with_database(db: Database) -> Self {
        Self {
            db: Arc::new(db),
            namespaces: RwLock::new(HashMap::new()),
            default_config: TenantConfig::default(),
        }
    }

    /// Set default tenant config
    pub fn set_default_config(&mut self, config: TenantConfig) {
        self.default_config = config;
    }

    /// Create a new namespace
    pub fn create_namespace(&self, id: &str, config: TenantConfig) -> Result<Arc<Namespace>> {
        let mut namespaces = self.namespaces.write();

        if namespaces.contains_key(id) {
            return Err(NeedleError::InvalidInput(format!(
                "Namespace {} already exists",
                id
            )));
        }

        let namespace = Arc::new(Namespace::new(id.to_string(), config, Arc::clone(&self.db)));

        namespaces.insert(id.to_string(), Arc::clone(&namespace));

        Ok(namespace)
    }

    /// Create namespace with default config
    pub fn create_namespace_default(&self, id: &str) -> Result<Arc<Namespace>> {
        self.create_namespace(id, self.default_config.clone())
    }

    /// Get an existing namespace
    pub fn namespace(&self, id: &str) -> Option<Arc<Namespace>> {
        self.namespaces.read().get(id).cloned()
    }

    /// Delete a namespace (and all its collections)
    pub fn delete_namespace(&self, id: &str) -> Result<bool> {
        let mut namespaces = self.namespaces.write();

        if let Some(ns) = namespaces.remove(id) {
            // Delete all collections in namespace
            let collections = ns.list_collections();
            for coll_name in collections {
                let _ = ns.delete_collection(&coll_name);
            }
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// List all namespaces
    pub fn list_namespaces(&self) -> Vec<String> {
        self.namespaces.read().keys().cloned().collect()
    }

    /// Check if namespace exists
    pub fn has_namespace(&self, id: &str) -> bool {
        self.namespaces.read().contains_key(id)
    }

    /// Get statistics for all namespaces
    pub fn all_stats(&self) -> HashMap<String, NamespaceStatsSnapshot> {
        self.namespaces
            .read()
            .iter()
            .map(|(id, ns)| (id.clone(), ns.stats()))
            .collect()
    }

    /// Get underlying database (for advanced use)
    pub fn database(&self) -> &Database {
        &self.db
    }
}

impl Default for NamespaceManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Access control level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AccessLevel {
    /// No access
    None,
    /// Read-only access
    Read,
    /// Read and write access
    Write,
    /// Full access including admin operations
    Admin,
}

/// Access control entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessControl {
    /// Namespace access levels by namespace ID
    namespace_access: HashMap<String, AccessLevel>,
    /// Default access level
    default_access: AccessLevel,
}

impl AccessControl {
    /// Create new access control
    pub fn new(default_access: AccessLevel) -> Self {
        Self {
            namespace_access: HashMap::new(),
            default_access,
        }
    }

    /// Grant access to a namespace
    pub fn grant(&mut self, namespace_id: &str, level: AccessLevel) {
        self.namespace_access
            .insert(namespace_id.to_string(), level);
    }

    /// Revoke access to a namespace
    pub fn revoke(&mut self, namespace_id: &str) {
        self.namespace_access.remove(namespace_id);
    }

    /// Get access level for a namespace
    pub fn level(&self, namespace_id: &str) -> AccessLevel {
        self.namespace_access
            .get(namespace_id)
            .copied()
            .unwrap_or(self.default_access)
    }

    /// Check if can read
    pub fn can_read(&self, namespace_id: &str) -> bool {
        matches!(
            self.level(namespace_id),
            AccessLevel::Read | AccessLevel::Write | AccessLevel::Admin
        )
    }

    /// Check if can write
    pub fn can_write(&self, namespace_id: &str) -> bool {
        matches!(
            self.level(namespace_id),
            AccessLevel::Write | AccessLevel::Admin
        )
    }

    /// Check if has admin access
    pub fn is_admin(&self, namespace_id: &str) -> bool {
        matches!(self.level(namespace_id), AccessLevel::Admin)
    }
}

impl Default for AccessControl {
    fn default() -> Self {
        Self::new(AccessLevel::None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_namespace() {
        let manager = NamespaceManager::new();
        let ns = manager
            .create_namespace("tenant1", TenantConfig::default())
            .unwrap();

        assert_eq!(ns.id(), "tenant1");
    }

    #[test]
    fn test_namespace_isolation() {
        let manager = NamespaceManager::new();

        let ns1 = manager
            .create_namespace("tenant1", TenantConfig::default())
            .unwrap();
        let ns2 = manager
            .create_namespace("tenant2", TenantConfig::default())
            .unwrap();

        // Create collection in ns1
        ns1.create_collection("docs", 8).unwrap();

        // ns1 should see it
        assert!(ns1.has_collection("docs"));
        assert_eq!(ns1.list_collections().len(), 1);

        // ns2 should NOT see it
        assert!(!ns2.has_collection("docs"));
        assert_eq!(ns2.list_collections().len(), 0);
    }

    #[test]
    fn test_collection_quota() {
        let manager = NamespaceManager::new();

        let config = TenantConfig {
            max_collections: Some(2),
            ..Default::default()
        };
        let ns = manager.create_namespace("tenant1", config).unwrap();

        ns.create_collection("coll1", 8).unwrap();
        ns.create_collection("coll2", 8).unwrap();

        // Third should fail
        let result = ns.create_collection("coll3", 8);
        assert!(result.is_err());
    }

    #[test]
    fn test_vector_quota() {
        let manager = NamespaceManager::new();

        let config = TenantConfig {
            max_vectors: Some(5),
            ..Default::default()
        };
        let ns = manager.create_namespace("tenant1", config).unwrap();

        ns.create_collection("docs", 4).unwrap();
        let coll = ns.collection("docs").unwrap();

        // Insert up to quota
        for i in 0..5 {
            coll.insert(format!("doc{}", i), &[1.0, 2.0, 3.0, 4.0], None)
                .unwrap();
        }

        // Next should fail
        let result = coll.insert("doc5", &[1.0, 2.0, 3.0, 4.0], None);
        assert!(result.is_err());
    }

    #[test]
    fn test_read_only_namespace() {
        let manager = NamespaceManager::new();

        let config = TenantConfig::read_only();
        let ns = manager.create_namespace("readonly", config).unwrap();

        // Should fail to create collection
        let result = ns.create_collection("docs", 8);
        assert!(result.is_err());
    }

    #[test]
    fn test_dimension_limit() {
        let manager = NamespaceManager::new();

        let config = TenantConfig {
            max_dimensions: Some(100),
            ..Default::default()
        };
        let ns = manager.create_namespace("tenant1", config).unwrap();

        // Small dimension should work
        ns.create_collection("small", 50).unwrap();

        // Large dimension should fail
        let result = ns.create_collection("large", 200);
        assert!(result.is_err());
    }

    #[test]
    fn test_namespace_stats() {
        let manager = NamespaceManager::new();
        let ns = manager
            .create_namespace("tenant1", TenantConfig::default())
            .unwrap();

        ns.create_collection("docs", 4).unwrap();
        let coll = ns.collection("docs").unwrap();

        coll.insert("doc1", &[1.0, 2.0, 3.0, 4.0], None).unwrap();
        coll.insert("doc2", &[1.0, 2.0, 3.0, 4.0], None).unwrap();
        let _ = coll.search(&[1.0, 2.0, 3.0, 4.0], 1);

        let stats = ns.stats();
        assert_eq!(stats.total_vectors, 2);
        assert_eq!(stats.total_collections, 1);
        assert!(stats.operations > 0);
        assert!(stats.writes > 0);
        assert!(stats.searches > 0);
    }

    #[test]
    fn test_delete_namespace() {
        let manager = NamespaceManager::new();

        let ns = manager
            .create_namespace("tenant1", TenantConfig::default())
            .unwrap();
        ns.create_collection("docs", 8).unwrap();

        assert!(manager.has_namespace("tenant1"));
        manager.delete_namespace("tenant1").unwrap();
        assert!(!manager.has_namespace("tenant1"));
    }

    #[test]
    fn test_access_control() {
        let mut acl = AccessControl::new(AccessLevel::None);

        // No access by default
        assert!(!acl.can_read("tenant1"));
        assert!(!acl.can_write("tenant1"));

        // Grant read access
        acl.grant("tenant1", AccessLevel::Read);
        assert!(acl.can_read("tenant1"));
        assert!(!acl.can_write("tenant1"));

        // Grant write access
        acl.grant("tenant1", AccessLevel::Write);
        assert!(acl.can_read("tenant1"));
        assert!(acl.can_write("tenant1"));

        // Revoke
        acl.revoke("tenant1");
        assert!(!acl.can_read("tenant1"));
    }

    #[test]
    fn test_list_namespaces() {
        let manager = NamespaceManager::new();

        manager
            .create_namespace("tenant1", TenantConfig::default())
            .unwrap();
        manager
            .create_namespace("tenant2", TenantConfig::default())
            .unwrap();
        manager
            .create_namespace("tenant3", TenantConfig::default())
            .unwrap();

        let namespaces = manager.list_namespaces();
        assert_eq!(namespaces.len(), 3);
        assert!(namespaces.contains(&"tenant1".to_string()));
        assert!(namespaces.contains(&"tenant2".to_string()));
        assert!(namespaces.contains(&"tenant3".to_string()));
    }
}
