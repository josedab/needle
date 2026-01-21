//! Managed Cloud Service MVP
//!
//! Multi-tenant provisioning, usage metering, control plane API, and tenant
//! lifecycle management for hosting Needle as a managed service.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::services::managed_cloud::{
//!     CloudControlPlane, TenantConfig, Tenant, UsageMetrics,
//!     ProvisionRequest, ProvisionResult,
//! };
//!
//! let mut cp = CloudControlPlane::new();
//!
//! // Provision a new tenant
//! let result = cp.provision(ProvisionRequest {
//!     tenant_id: "acme-corp".into(),
//!     plan: Plan::Pro,
//!     region: "us-east-1".into(),
//!     config: TenantConfig::default(),
//! }).unwrap();
//!
//! // Track usage
//! cp.record_usage("acme-corp", UsageEvent::Search { vectors_scanned: 10_000 });
//! let metrics = cp.usage("acme-corp").unwrap();
//! println!("Searches: {}", metrics.total_searches);
//! ```

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::error::{NeedleError, Result};

/// Service plan tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Plan {
    Free,
    Starter,
    Pro,
    Enterprise,
}

/// Tenant status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TenantStatus {
    Provisioning,
    Active,
    Suspended,
    Deprovisioned,
}

/// Per-tenant configuration limits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TenantConfig {
    pub max_collections: usize,
    pub max_vectors: usize,
    pub max_dimensions: usize,
    pub max_qps: u32,
    pub storage_mb: usize,
}

impl Default for TenantConfig {
    fn default() -> Self {
        Self { max_collections: 10, max_vectors: 1_000_000, max_dimensions: 2048, max_qps: 100, storage_mb: 512 }
    }
}

impl TenantConfig {
    /// Limits for a given plan.
    pub fn for_plan(plan: Plan) -> Self {
        match plan {
            Plan::Free => Self { max_collections: 3, max_vectors: 100_000, max_dimensions: 768, max_qps: 10, storage_mb: 64 },
            Plan::Starter => Self::default(),
            Plan::Pro => Self { max_collections: 50, max_vectors: 10_000_000, max_dimensions: 4096, max_qps: 1000, storage_mb: 10_240 },
            Plan::Enterprise => Self { max_collections: 500, max_vectors: 100_000_000, max_dimensions: 8192, max_qps: 10_000, storage_mb: 102_400 },
        }
    }
}

/// A provisioned tenant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tenant {
    pub id: String,
    pub plan: Plan,
    pub region: String,
    pub status: TenantStatus,
    pub config: TenantConfig,
    pub created_at: u64,
    pub api_key: String,
}

/// Provision request.
#[derive(Debug, Clone)]
pub struct ProvisionRequest {
    pub tenant_id: String,
    pub plan: Plan,
    pub region: String,
    pub config: TenantConfig,
}

/// Provision result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvisionResult {
    pub tenant_id: String,
    pub api_key: String,
    pub endpoint: String,
    pub status: TenantStatus,
}

/// Usage event types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UsageEvent {
    Search { vectors_scanned: usize },
    Insert { count: usize },
    Delete { count: usize },
    StorageBytes { delta: i64 },
}

/// Aggregated usage metrics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UsageMetrics {
    pub total_searches: u64,
    pub total_inserts: u64,
    pub total_deletes: u64,
    pub vectors_scanned: u64,
    pub storage_bytes: i64,
    pub api_calls: u64,
}

/// Cloud control plane.
pub struct CloudControlPlane {
    tenants: HashMap<String, Tenant>,
    usage: HashMap<String, UsageMetrics>,
}

impl CloudControlPlane {
    pub fn new() -> Self {
        Self { tenants: HashMap::new(), usage: HashMap::new() }
    }

    pub fn provision(&mut self, req: ProvisionRequest) -> Result<ProvisionResult> {
        if self.tenants.contains_key(&req.tenant_id) {
            return Err(NeedleError::Conflict(format!("Tenant '{}' exists", req.tenant_id)));
        }
        let api_key = format!("nk_{}_{}", req.tenant_id, now_secs());
        let tenant = Tenant {
            id: req.tenant_id.clone(), plan: req.plan, region: req.region.clone(),
            status: TenantStatus::Active, config: req.config,
            created_at: now_secs(), api_key: api_key.clone(),
        };
        self.tenants.insert(req.tenant_id.clone(), tenant);
        self.usage.insert(req.tenant_id.clone(), UsageMetrics::default());
        Ok(ProvisionResult {
            tenant_id: req.tenant_id, api_key,
            endpoint: format!("https://{}.needle.dev", req.region),
            status: TenantStatus::Active,
        })
    }

    pub fn deprovision(&mut self, tenant_id: &str) -> Result<()> {
        let t = self.tenants.get_mut(tenant_id)
            .ok_or_else(|| NeedleError::NotFound(format!("Tenant '{tenant_id}'")))?;
        t.status = TenantStatus::Deprovisioned;
        Ok(())
    }

    pub fn record_usage(&mut self, tenant_id: &str, event: UsageEvent) {
        let m = self.usage.entry(tenant_id.into()).or_default();
        m.api_calls += 1;
        match event {
            UsageEvent::Search { vectors_scanned } => { m.total_searches += 1; m.vectors_scanned += vectors_scanned as u64; }
            UsageEvent::Insert { count } => { m.total_inserts += count as u64; }
            UsageEvent::Delete { count } => { m.total_deletes += count as u64; }
            UsageEvent::StorageBytes { delta } => { m.storage_bytes += delta; }
        }
    }

    pub fn usage(&self, tenant_id: &str) -> Option<&UsageMetrics> { self.usage.get(tenant_id) }
    pub fn tenant(&self, tenant_id: &str) -> Option<&Tenant> { self.tenants.get(tenant_id) }
    pub fn tenant_count(&self) -> usize { self.tenants.len() }
    pub fn active_tenants(&self) -> Vec<&Tenant> {
        self.tenants.values().filter(|t| t.status == TenantStatus::Active).collect()
    }
}

impl Default for CloudControlPlane {
    fn default() -> Self { Self::new() }
}

fn now_secs() -> u64 { SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs() }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provision() {
        let mut cp = CloudControlPlane::new();
        let r = cp.provision(ProvisionRequest {
            tenant_id: "t1".into(), plan: Plan::Pro, region: "us-east-1".into(),
            config: TenantConfig::for_plan(Plan::Pro),
        }).unwrap();
        assert_eq!(r.status, TenantStatus::Active);
        assert!(r.api_key.starts_with("nk_"));
        assert_eq!(cp.tenant_count(), 1);
    }

    #[test]
    fn test_duplicate_tenant() {
        let mut cp = CloudControlPlane::new();
        cp.provision(ProvisionRequest { tenant_id: "t1".into(), plan: Plan::Free, region: "eu".into(), config: TenantConfig::default() }).unwrap();
        assert!(cp.provision(ProvisionRequest { tenant_id: "t1".into(), plan: Plan::Free, region: "eu".into(), config: TenantConfig::default() }).is_err());
    }

    #[test]
    fn test_usage() {
        let mut cp = CloudControlPlane::new();
        cp.provision(ProvisionRequest { tenant_id: "t1".into(), plan: Plan::Starter, region: "us".into(), config: TenantConfig::default() }).unwrap();
        cp.record_usage("t1", UsageEvent::Search { vectors_scanned: 5000 });
        cp.record_usage("t1", UsageEvent::Insert { count: 10 });
        let m = cp.usage("t1").unwrap();
        assert_eq!(m.total_searches, 1);
        assert_eq!(m.total_inserts, 10);
        assert_eq!(m.api_calls, 2);
    }

    #[test]
    fn test_deprovision() {
        let mut cp = CloudControlPlane::new();
        cp.provision(ProvisionRequest { tenant_id: "t1".into(), plan: Plan::Free, region: "us".into(), config: TenantConfig::default() }).unwrap();
        cp.deprovision("t1").unwrap();
        assert_eq!(cp.tenant("t1").unwrap().status, TenantStatus::Deprovisioned);
        assert!(cp.active_tenants().is_empty());
    }

    #[test]
    fn test_plan_configs() {
        let free = TenantConfig::for_plan(Plan::Free);
        let ent = TenantConfig::for_plan(Plan::Enterprise);
        assert!(ent.max_vectors > free.max_vectors);
        assert!(ent.max_qps > free.max_qps);
    }
}
