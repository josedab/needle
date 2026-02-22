#![allow(clippy::unwrap_used)]
//! Multi-Tenant Query Isolation
//!
//! Per-tenant routing with resource limits, namespace enforcement, and audit
//! logging. Tenants are identified by header (X-Tenant-ID) and isolated at
//! the query level.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::services::tenant_router::{
//!     TenantRouter, TenantConfig, TenantLimits, QueryContext, AuditEntry,
//! };
//! use needle::Database;
//!
//! let db = Database::in_memory();
//! let mut router = TenantRouter::new();
//!
//! router.register_tenant("acme", TenantConfig {
//!     namespace_prefix: "acme_".into(),
//!     limits: TenantLimits { max_qps: 100, max_ef_search: 200, max_results: 1000 },
//!     enabled: true,
//! });
//!
//! // Tenant search is namespace-isolated
//! let ctx = router.authorize("acme").unwrap();
//! assert_eq!(ctx.collection_prefix, "acme_");
//! ```

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::error::{NeedleError, Result};

/// Resource limits for a tenant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TenantLimits {
    pub max_qps: u32,
    pub max_ef_search: usize,
    pub max_results: usize,
}

impl Default for TenantLimits {
    fn default() -> Self {
        Self { max_qps: 100, max_ef_search: 500, max_results: 1000 }
    }
}

/// Per-tenant configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TenantConfig {
    pub namespace_prefix: String,
    pub limits: TenantLimits,
    pub enabled: bool,
}

/// Query context after tenant authorization.
#[derive(Debug, Clone)]
pub struct QueryContext {
    pub tenant_id: String,
    pub collection_prefix: String,
    pub limits: TenantLimits,
}

impl QueryContext {
    /// Resolve a collection name with tenant namespace prefix.
    pub fn resolve_collection(&self, name: &str) -> String {
        format!("{}{}", self.collection_prefix, name)
    }

    /// Check if a result count is within limits.
    pub fn check_k(&self, k: usize) -> Result<()> {
        if k > self.limits.max_results {
            return Err(NeedleError::InvalidArgument(format!(
                "k={} exceeds tenant limit {}", k, self.limits.max_results
            )));
        }
        Ok(())
    }
}

/// Audit log entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    pub tenant_id: String,
    pub operation: String,
    pub collection: String,
    pub timestamp: u64,
    pub success: bool,
    pub latency_us: u64,
}

/// Tenant usage statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TenantUsage {
    pub total_queries: u64,
    pub total_inserts: u64,
    pub total_deletes: u64,
    pub last_active: u64,
}

/// Multi-tenant query router.
pub struct TenantRouter {
    tenants: HashMap<String, TenantConfig>,
    usage: HashMap<String, TenantUsage>,
    audit_log: Vec<AuditEntry>,
    max_audit_entries: usize,
}

impl TenantRouter {
    pub fn new() -> Self {
        Self {
            tenants: HashMap::new(),
            usage: HashMap::new(),
            audit_log: Vec::new(),
            max_audit_entries: 100_000,
        }
    }

    /// Register a tenant.
    pub fn register_tenant(&mut self, tenant_id: &str, config: TenantConfig) {
        self.tenants.insert(tenant_id.to_string(), config);
        self.usage.insert(tenant_id.to_string(), TenantUsage::default());
    }

    /// Remove a tenant.
    pub fn remove_tenant(&mut self, tenant_id: &str) -> bool {
        self.usage.remove(tenant_id);
        self.tenants.remove(tenant_id).is_some()
    }

    /// Authorize a request for a tenant. Returns QueryContext on success.
    pub fn authorize(&self, tenant_id: &str) -> Result<QueryContext> {
        let config = self.tenants.get(tenant_id)
            .ok_or_else(|| NeedleError::NotFound(format!("Tenant '{tenant_id}'")))?;
        if !config.enabled {
            return Err(NeedleError::InvalidArgument(format!("Tenant '{tenant_id}' is disabled")));
        }
        Ok(QueryContext {
            tenant_id: tenant_id.to_string(),
            collection_prefix: config.namespace_prefix.clone(),
            limits: config.limits.clone(),
        })
    }

    /// Record an operation in the audit log.
    pub fn audit(&mut self, tenant_id: &str, operation: &str, collection: &str, success: bool, latency_us: u64) {
        let entry = AuditEntry {
            tenant_id: tenant_id.to_string(),
            operation: operation.to_string(),
            collection: collection.to_string(),
            timestamp: now_secs(),
            success,
            latency_us,
        };
        self.audit_log.push(entry);
        if self.audit_log.len() > self.max_audit_entries {
            self.audit_log.remove(0);
        }

        if let Some(usage) = self.usage.get_mut(tenant_id) {
            match operation {
                "search" => usage.total_queries += 1,
                "insert" => usage.total_inserts += 1,
                "delete" => usage.total_deletes += 1,
                _ => {}
            }
            usage.last_active = now_secs();
        }
    }

    /// Get audit log for a tenant.
    pub fn audit_log(&self, tenant_id: &str) -> Vec<&AuditEntry> {
        self.audit_log.iter().filter(|e| e.tenant_id == tenant_id).collect()
    }

    /// Get usage stats for a tenant.
    pub fn usage(&self, tenant_id: &str) -> Option<&TenantUsage> {
        self.usage.get(tenant_id)
    }

    /// List all tenants.
    pub fn list_tenants(&self) -> Vec<(&str, &TenantConfig)> {
        self.tenants.iter().map(|(k, v)| (k.as_str(), v)).collect()
    }

    /// Number of registered tenants.
    pub fn tenant_count(&self) -> usize { self.tenants.len() }
}

impl Default for TenantRouter {
    fn default() -> Self { Self::new() }
}

fn now_secs() -> u64 { SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs() }

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(prefix: &str) -> TenantConfig {
        TenantConfig {
            namespace_prefix: prefix.into(),
            limits: TenantLimits::default(),
            enabled: true,
        }
    }

    #[test]
    fn test_register_and_authorize() {
        let mut router = TenantRouter::new();
        router.register_tenant("acme", make_config("acme_"));

        let ctx = router.authorize("acme").unwrap();
        assert_eq!(ctx.tenant_id, "acme");
        assert_eq!(ctx.collection_prefix, "acme_");
    }

    #[test]
    fn test_unknown_tenant_rejected() {
        let router = TenantRouter::new();
        assert!(router.authorize("unknown").is_err());
    }

    #[test]
    fn test_disabled_tenant_rejected() {
        let mut router = TenantRouter::new();
        router.register_tenant("acme", TenantConfig {
            namespace_prefix: "acme_".into(),
            limits: TenantLimits::default(),
            enabled: false,
        });
        assert!(router.authorize("acme").is_err());
    }

    #[test]
    fn test_collection_resolution() {
        let mut router = TenantRouter::new();
        router.register_tenant("acme", make_config("acme_"));
        let ctx = router.authorize("acme").unwrap();
        assert_eq!(ctx.resolve_collection("docs"), "acme_docs");
    }

    #[test]
    fn test_k_limit_check() {
        let mut router = TenantRouter::new();
        router.register_tenant("acme", TenantConfig {
            namespace_prefix: "acme_".into(),
            limits: TenantLimits { max_results: 50, ..Default::default() },
            enabled: true,
        });
        let ctx = router.authorize("acme").unwrap();
        assert!(ctx.check_k(50).is_ok());
        assert!(ctx.check_k(51).is_err());
    }

    #[test]
    fn test_audit_logging() {
        let mut router = TenantRouter::new();
        router.register_tenant("acme", make_config("acme_"));

        router.audit("acme", "search", "docs", true, 500);
        router.audit("acme", "insert", "docs", true, 100);

        let log = router.audit_log("acme");
        assert_eq!(log.len(), 2);

        let usage = router.usage("acme").unwrap();
        assert_eq!(usage.total_queries, 1);
        assert_eq!(usage.total_inserts, 1);
    }

    #[test]
    fn test_remove_tenant() {
        let mut router = TenantRouter::new();
        router.register_tenant("acme", make_config("acme_"));
        assert!(router.remove_tenant("acme"));
        assert_eq!(router.tenant_count(), 0);
    }

    #[test]
    fn test_list_tenants() {
        let mut router = TenantRouter::new();
        router.register_tenant("a", make_config("a_"));
        router.register_tenant("b", make_config("b_"));
        assert_eq!(router.list_tenants().len(), 2);
    }
}
