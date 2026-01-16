//! Multi-Tenant Namespace Isolation
//!
//! Provides complete tenant isolation with per-namespace resource quotas,
//! independent encryption keys, access control policies, and GDPR compliance features.

use crate::error::{NeedleError, Result};
use chrono::Utc;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;

fn sha256_hex(data: &[u8]) -> String {
    let hash = Sha256::digest(data);
    let mut s = String::with_capacity(hash.len() * 2);
    for byte in hash {
        let _ = write!(s, "{:02x}", byte);
    }
    s
}

// ── Core Types ──────────────────────────────────────────────────────────────

/// Unique tenant identifier.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TenantId(pub String);

/// Per-tenant resource configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TenantConfig {
    pub max_vectors: u64,
    pub max_collections: u32,
    pub max_storage_bytes: u64,
    pub rate_limit_qps: u32,
    pub encryption_enabled: bool,
    pub audit_logging: bool,
}

impl Default for TenantConfig {
    fn default() -> Self {
        Self {
            max_vectors: 1_000_000,
            max_collections: 100,
            max_storage_bytes: 1_073_741_824, // 1 GB
            rate_limit_qps: 1_000,
            encryption_enabled: false,
            audit_logging: false,
        }
    }
}

/// Lifecycle status of a tenant.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum TenantStatus {
    Active,
    Suspended,
    PendingDeletion,
    Deleted,
}

/// A tenant with its configuration, status, and resource tracking.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Tenant {
    pub id: TenantId,
    pub name: String,
    pub config: TenantConfig,
    pub status: TenantStatus,
    pub created_at: String,
    pub updated_at: String,
    pub encryption_key_hash: Option<String>,
    pub collections: HashSet<String>,
    pub current_vectors: u64,
    pub current_storage_bytes: u64,
}

// ── Access Control ──────────────────────────────────────────────────────────

/// Access policy binding a tenant to a role and permissions.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AccessPolicy {
    pub tenant_id: TenantId,
    pub role: TenantRole,
    pub permissions: HashSet<Permission>,
}

/// Role assigned to a tenant.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TenantRole {
    Admin,
    ReadWrite,
    ReadOnly,
    Custom(String),
}

/// Granular permission for tenant operations.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Permission {
    CreateCollection,
    DeleteCollection,
    InsertVector,
    DeleteVector,
    SearchVector,
    ExportData,
    ManageTenant,
}

// ── Audit & Observability ───────────────────────────────────────────────────

/// An immutable audit log entry.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuditLogEntry {
    pub timestamp: String,
    pub tenant_id: TenantId,
    pub action: String,
    pub resource: String,
    pub success: bool,
    pub details: Option<String>,
}

/// Snapshot of a tenant's current resource usage.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResourceUsage {
    pub vectors_used: u64,
    pub vectors_limit: u64,
    pub storage_used_bytes: u64,
    pub storage_limit_bytes: u64,
    pub collections_used: u32,
    pub collections_limit: u32,
    pub queries_today: u64,
    pub utilization_pct: f32,
}

// ── GDPR ────────────────────────────────────────────────────────────────────

/// Manifest produced by a GDPR data-export request.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GdprExport {
    pub tenant_id: TenantId,
    pub exported_at: String,
    pub collections: Vec<String>,
    pub total_vectors: u64,
    pub format: ExportFormat,
    pub checksum: String,
}

/// Supported export formats for GDPR data portability.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ExportFormat {
    Json,
    Csv,
    Parquet,
}

// ── Tenant Manager ──────────────────────────────────────────────────────────

/// Thread-safe manager for multi-tenant isolation.
pub struct TenantManager {
    inner: RwLock<TenantManagerInner>,
}

struct TenantManagerInner {
    tenants: HashMap<TenantId, Tenant>,
    policies: HashMap<TenantId, AccessPolicy>,
    audit_log: Vec<AuditLogEntry>,
    max_audit_log_size: usize,
}

impl TenantManager {
    /// Create a new, empty `TenantManager`.
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(TenantManagerInner {
                tenants: HashMap::new(),
                policies: HashMap::new(),
                audit_log: Vec::new(),
                max_audit_log_size: 100_000,
            }),
        }
    }

    /// Register a new tenant. Returns an error if the id already exists.
    pub fn create_tenant(&self, id: TenantId, name: String, config: TenantConfig) -> Result<()> {
        let mut inner = self.inner.write();
        if inner.tenants.contains_key(&id) {
            return Err(NeedleError::Conflict(format!(
                "Tenant '{}' already exists",
                id.0
            )));
        }
        let now = Utc::now().to_rfc3339();
        let tenant = Tenant {
            id: id.clone(),
            name,
            config,
            status: TenantStatus::Active,
            created_at: now.clone(),
            updated_at: now,
            encryption_key_hash: None,
            collections: HashSet::new(),
            current_vectors: 0,
            current_storage_bytes: 0,
        };
        inner.tenants.insert(id, tenant);
        Ok(())
    }

    /// Return a clone of the tenant, if it exists.
    pub fn get_tenant(&self, id: &TenantId) -> Option<Tenant> {
        self.inner.read().tenants.get(id).cloned()
    }

    /// Update the configuration for an existing tenant.
    pub fn update_tenant_config(&self, id: &TenantId, config: TenantConfig) -> Result<()> {
        let mut inner = self.inner.write();
        let tenant = inner
            .tenants
            .get_mut(id)
            .ok_or_else(|| NeedleError::NotFound(format!("Tenant '{}' not found", id.0)))?;
        tenant.config = config;
        tenant.updated_at = Utc::now().to_rfc3339();
        Ok(())
    }

    /// Suspend a tenant, preventing further operations.
    pub fn suspend_tenant(&self, id: &TenantId) -> Result<()> {
        let mut inner = self.inner.write();
        let tenant = inner
            .tenants
            .get_mut(id)
            .ok_or_else(|| NeedleError::NotFound(format!("Tenant '{}' not found", id.0)))?;
        tenant.status = TenantStatus::Suspended;
        tenant.updated_at = Utc::now().to_rfc3339();
        Ok(())
    }

    /// Mark a tenant for deletion (sets status to `PendingDeletion`).
    pub fn delete_tenant(&self, id: &TenantId) -> Result<()> {
        let mut inner = self.inner.write();
        let tenant = inner
            .tenants
            .get_mut(id)
            .ok_or_else(|| NeedleError::NotFound(format!("Tenant '{}' not found", id.0)))?;
        tenant.status = TenantStatus::PendingDeletion;
        tenant.updated_at = Utc::now().to_rfc3339();
        Ok(())
    }

    /// List all registered tenants (cloned).
    pub fn list_tenants(&self) -> Vec<Tenant> {
        self.inner.read().tenants.values().cloned().collect()
    }

    /// Store the SHA-256 hash of an encryption key for the given tenant.
    pub fn set_encryption_key(&self, tenant_id: &TenantId, key: &[u8]) -> Result<()> {
        let mut inner = self.inner.write();
        let tenant = inner
            .tenants
            .get_mut(tenant_id)
            .ok_or_else(|| NeedleError::NotFound(format!("Tenant '{}' not found", tenant_id.0)))?;
        let hash = sha256_hex(key);
        tenant.encryption_key_hash = Some(hash);
        tenant.updated_at = Utc::now().to_rfc3339();
        Ok(())
    }

    /// Verify that the provided key matches the stored hash.
    pub fn verify_encryption_key(&self, tenant_id: &TenantId, key: &[u8]) -> Result<bool> {
        let inner = self.inner.read();
        let tenant = inner
            .tenants
            .get(tenant_id)
            .ok_or_else(|| NeedleError::NotFound(format!("Tenant '{}' not found", tenant_id.0)))?;
        match &tenant.encryption_key_hash {
            Some(stored) => {
                let hash = sha256_hex(key);
                Ok(hash == *stored)
            }
            None => Ok(false),
        }
    }

    /// Check whether adding `vectors_to_add` would exceed the tenant's quota.
    pub fn check_quota(&self, tenant_id: &TenantId, vectors_to_add: u64) -> Result<bool> {
        let inner = self.inner.read();
        let tenant = inner
            .tenants
            .get(tenant_id)
            .ok_or_else(|| NeedleError::NotFound(format!("Tenant '{}' not found", tenant_id.0)))?;
        Ok(tenant.current_vectors + vectors_to_add <= tenant.config.max_vectors)
    }

    /// Record additional usage for a tenant.
    pub fn record_usage(
        &self,
        tenant_id: &TenantId,
        vectors_added: u64,
        storage_added: u64,
    ) -> Result<()> {
        let mut inner = self.inner.write();
        let tenant = inner
            .tenants
            .get_mut(tenant_id)
            .ok_or_else(|| NeedleError::NotFound(format!("Tenant '{}' not found", tenant_id.0)))?;
        tenant.current_vectors += vectors_added;
        tenant.current_storage_bytes += storage_added;
        tenant.updated_at = Utc::now().to_rfc3339();
        Ok(())
    }

    /// Build a `ResourceUsage` snapshot for the tenant.
    pub fn get_usage(&self, tenant_id: &TenantId) -> Result<ResourceUsage> {
        let inner = self.inner.read();
        let tenant = inner
            .tenants
            .get(tenant_id)
            .ok_or_else(|| NeedleError::NotFound(format!("Tenant '{}' not found", tenant_id.0)))?;
        let utilization = if tenant.config.max_vectors > 0 {
            (tenant.current_vectors as f32 / tenant.config.max_vectors as f32) * 100.0
        } else {
            0.0
        };
        Ok(ResourceUsage {
            vectors_used: tenant.current_vectors,
            vectors_limit: tenant.config.max_vectors,
            storage_used_bytes: tenant.current_storage_bytes,
            storage_limit_bytes: tenant.config.max_storage_bytes,
            collections_used: tenant.collections.len() as u32,
            collections_limit: tenant.config.max_collections,
            queries_today: 0,
            utilization_pct: utilization,
        })
    }

    /// Set (or replace) the access policy for a tenant.
    pub fn set_access_policy(&self, tenant_id: &TenantId, policy: AccessPolicy) -> Result<()> {
        let inner = self.inner.read();
        if !inner.tenants.contains_key(tenant_id) {
            return Err(NeedleError::NotFound(format!(
                "Tenant '{}' not found",
                tenant_id.0
            )));
        }
        drop(inner);

        let mut inner = self.inner.write();
        inner.policies.insert(tenant_id.clone(), policy);
        Ok(())
    }

    /// Check whether the tenant's policy includes the given permission.
    pub fn check_permission(&self, tenant_id: &TenantId, permission: &Permission) -> Result<bool> {
        let inner = self.inner.read();
        if !inner.tenants.contains_key(tenant_id) {
            return Err(NeedleError::NotFound(format!(
                "Tenant '{}' not found",
                tenant_id.0
            )));
        }
        match inner.policies.get(tenant_id) {
            Some(policy) => Ok(policy.permissions.contains(permission)),
            None => Ok(false),
        }
    }

    /// Append an audit log entry (bounded by `max_audit_log_size`).
    pub fn log_audit(&self, entry: AuditLogEntry) {
        let mut inner = self.inner.write();
        if inner.audit_log.len() >= inner.max_audit_log_size {
            inner.audit_log.remove(0);
        }
        inner.audit_log.push(entry);
    }

    /// Return references to all audit entries for a given tenant.
    pub fn get_audit_log(&self, tenant_id: &TenantId) -> Vec<AuditLogEntry> {
        let inner = self.inner.read();
        inner
            .audit_log
            .iter()
            .filter(|e| e.tenant_id == *tenant_id)
            .cloned()
            .collect()
    }

    /// Build a GDPR export manifest for the tenant.
    pub fn prepare_gdpr_export(&self, tenant_id: &TenantId) -> Result<GdprExport> {
        let inner = self.inner.read();
        let tenant = inner
            .tenants
            .get(tenant_id)
            .ok_or_else(|| NeedleError::NotFound(format!("Tenant '{}' not found", tenant_id.0)))?;

        let collections: Vec<String> = tenant.collections.iter().cloned().collect();
        let export = GdprExport {
            tenant_id: tenant_id.clone(),
            exported_at: Utc::now().to_rfc3339(),
            total_vectors: tenant.current_vectors,
            collections,
            format: ExportFormat::Json,
            checksum: sha256_hex(format!("{}:{}", tenant_id.0, tenant.current_vectors).as_bytes()),
        };
        Ok(export)
    }

    /// Mark a tenant for complete GDPR deletion and log the event.
    pub fn gdpr_delete(&self, tenant_id: &TenantId) -> Result<()> {
        {
            let mut inner = self.inner.write();
            let tenant = inner.tenants.get_mut(tenant_id).ok_or_else(|| {
                NeedleError::NotFound(format!("Tenant '{}' not found", tenant_id.0))
            })?;
            tenant.status = TenantStatus::Deleted;
            tenant.updated_at = Utc::now().to_rfc3339();
        }

        self.log_audit(AuditLogEntry {
            timestamp: Utc::now().to_rfc3339(),
            tenant_id: tenant_id.clone(),
            action: "gdpr_delete".to_string(),
            resource: "tenant".to_string(),
            success: true,
            details: Some("Tenant marked for GDPR deletion".to_string()),
        });

        Ok(())
    }

    /// Return the number of registered tenants.
    pub fn tenant_count(&self) -> usize {
        self.inner.read().tenants.len()
    }

    /// Enforce quota before an operation. Returns error if quota exceeded.
    pub fn enforce_quota(&self, tenant_id: &TenantId, vectors_to_add: u64) -> Result<()> {
        let inner = self.inner.read();
        let tenant = inner
            .tenants
            .get(tenant_id)
            .ok_or_else(|| NeedleError::NotFound(format!("Tenant '{}' not found", tenant_id.0)))?;

        if tenant.status != TenantStatus::Active {
            return Err(NeedleError::InvalidOperation(format!(
                "Tenant '{}' is not active (status: {:?})",
                tenant_id.0, tenant.status
            )));
        }

        if tenant.current_vectors + vectors_to_add > tenant.config.max_vectors {
            return Err(NeedleError::QuotaExceeded(format!(
                "Tenant '{}': adding {} vectors would exceed limit of {} (current: {})",
                tenant_id.0, vectors_to_add, tenant.config.max_vectors, tenant.current_vectors
            )));
        }

        Ok(())
    }

    /// Enforce storage quota.
    pub fn enforce_storage_quota(&self, tenant_id: &TenantId, bytes_to_add: u64) -> Result<()> {
        let inner = self.inner.read();
        let tenant = inner
            .tenants
            .get(tenant_id)
            .ok_or_else(|| NeedleError::NotFound(format!("Tenant '{}' not found", tenant_id.0)))?;

        if tenant.current_storage_bytes + bytes_to_add > tenant.config.max_storage_bytes {
            return Err(NeedleError::QuotaExceeded(format!(
                "Tenant '{}': adding {} bytes would exceed storage limit of {} (current: {})",
                tenant_id.0,
                bytes_to_add,
                tenant.config.max_storage_bytes,
                tenant.current_storage_bytes
            )));
        }

        Ok(())
    }

    /// Enforce collection count quota.
    pub fn enforce_collection_quota(&self, tenant_id: &TenantId) -> Result<()> {
        let inner = self.inner.read();
        let tenant = inner
            .tenants
            .get(tenant_id)
            .ok_or_else(|| NeedleError::NotFound(format!("Tenant '{}' not found", tenant_id.0)))?;

        if tenant.collections.len() as u32 >= tenant.config.max_collections {
            return Err(NeedleError::QuotaExceeded(format!(
                "Tenant '{}': collection limit of {} reached",
                tenant_id.0, tenant.config.max_collections
            )));
        }

        Ok(())
    }

    /// Register a collection under a tenant's namespace.
    pub fn add_collection(&self, tenant_id: &TenantId, collection_name: String) -> Result<()> {
        self.enforce_collection_quota(tenant_id)?;
        let mut inner = self.inner.write();
        let tenant = inner
            .tenants
            .get_mut(tenant_id)
            .ok_or_else(|| NeedleError::NotFound(format!("Tenant '{}' not found", tenant_id.0)))?;
        tenant.collections.insert(collection_name);
        tenant.updated_at = Utc::now().to_rfc3339();
        Ok(())
    }

    /// Remove a collection from a tenant's namespace.
    pub fn remove_collection(&self, tenant_id: &TenantId, collection_name: &str) -> Result<()> {
        let mut inner = self.inner.write();
        let tenant = inner
            .tenants
            .get_mut(tenant_id)
            .ok_or_else(|| NeedleError::NotFound(format!("Tenant '{}' not found", tenant_id.0)))?;
        tenant.collections.remove(collection_name);
        tenant.updated_at = Utc::now().to_rfc3339();
        Ok(())
    }
}

// ── Usage Metering ──────────────────────────────────────────────────────────

/// Types of metered usage events.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum UsageEventKind {
    VectorInsert,
    VectorDelete,
    SearchQuery,
    FilteredSearchQuery,
    BatchSearch,
    CollectionCreate,
    CollectionDelete,
    Export,
    Import,
}

/// A single metered usage event for billing purposes.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UsageEvent {
    pub tenant_id: TenantId,
    pub event_kind: UsageEventKind,
    pub timestamp: String,
    /// Number of units consumed (e.g., vectors inserted, queries executed).
    pub quantity: u64,
    /// Estimated cost in micro-cents (1/10,000 of a dollar).
    pub cost_micro_cents: u64,
    pub metadata: Option<serde_json::Value>,
}

/// Configuration for usage metering pricing.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MeteringConfig {
    /// Cost per vector insert in micro-cents.
    pub cost_per_insert_mc: u64,
    /// Cost per search query in micro-cents.
    pub cost_per_query_mc: u64,
    /// Cost per GB-month of storage in micro-cents.
    pub cost_per_gb_month_mc: u64,
    /// Cost per batch search in micro-cents.
    pub cost_per_batch_mc: u64,
}

impl Default for MeteringConfig {
    fn default() -> Self {
        Self {
            cost_per_insert_mc: 1,         // $0.0001 per insert
            cost_per_query_mc: 5,          // $0.0005 per query
            cost_per_gb_month_mc: 250_000, // $25 per GB-month
            cost_per_batch_mc: 25,         // $0.0025 per batch search
        }
    }
}

/// Aggregated billing summary for a tenant.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BillingSummary {
    pub tenant_id: TenantId,
    pub period_start: String,
    pub period_end: String,
    pub total_inserts: u64,
    pub total_queries: u64,
    pub total_storage_gb_hours: f64,
    pub total_cost_micro_cents: u64,
    /// Formatted total cost in dollars.
    pub total_cost_usd: String,
    pub events_count: usize,
}

/// Usage metering engine that tracks events and generates billing summaries.
pub struct UsageMeter {
    config: MeteringConfig,
    events: RwLock<Vec<UsageEvent>>,
    max_events: usize,
}

impl UsageMeter {
    pub fn new(config: MeteringConfig) -> Self {
        Self {
            config,
            events: RwLock::new(Vec::new()),
            max_events: 1_000_000,
        }
    }

    /// Record a usage event with automatic cost calculation.
    pub fn record(&self, tenant_id: TenantId, event_kind: UsageEventKind, quantity: u64) {
        let cost_mc = match event_kind {
            UsageEventKind::VectorInsert => quantity * self.config.cost_per_insert_mc,
            UsageEventKind::SearchQuery | UsageEventKind::FilteredSearchQuery => {
                quantity * self.config.cost_per_query_mc
            }
            UsageEventKind::BatchSearch => quantity * self.config.cost_per_batch_mc,
            _ => 0,
        };

        let event = UsageEvent {
            tenant_id,
            event_kind,
            timestamp: Utc::now().to_rfc3339(),
            quantity,
            cost_micro_cents: cost_mc,
            metadata: None,
        };

        let mut events = self.events.write();
        if events.len() >= self.max_events {
            // Drop oldest 10%
            let drain_count = self.max_events / 10;
            events.drain(..drain_count);
        }
        events.push(event);
    }

    /// Generate a billing summary for a tenant over a time period.
    pub fn billing_summary(
        &self,
        tenant_id: &TenantId,
        period_start: &str,
        period_end: &str,
    ) -> BillingSummary {
        let events = self.events.read();
        let tenant_events: Vec<&UsageEvent> = events
            .iter()
            .filter(|e| {
                e.tenant_id == *tenant_id
                    && e.timestamp.as_str() >= period_start
                    && e.timestamp.as_str() <= period_end
            })
            .collect();

        let total_inserts: u64 = tenant_events
            .iter()
            .filter(|e| e.event_kind == UsageEventKind::VectorInsert)
            .map(|e| e.quantity)
            .sum();

        let total_queries: u64 = tenant_events
            .iter()
            .filter(|e| {
                matches!(
                    e.event_kind,
                    UsageEventKind::SearchQuery
                        | UsageEventKind::FilteredSearchQuery
                        | UsageEventKind::BatchSearch
                )
            })
            .map(|e| e.quantity)
            .sum();

        let total_cost_mc: u64 = tenant_events.iter().map(|e| e.cost_micro_cents).sum();

        BillingSummary {
            tenant_id: tenant_id.clone(),
            period_start: period_start.to_string(),
            period_end: period_end.to_string(),
            total_inserts,
            total_queries,
            total_storage_gb_hours: 0.0, // computed externally from snapshots
            total_cost_micro_cents: total_cost_mc,
            total_cost_usd: format!("${:.4}", total_cost_mc as f64 / 10_000.0),
            events_count: tenant_events.len(),
        }
    }

    /// Return all events for a tenant.
    pub fn events_for_tenant(&self, tenant_id: &TenantId) -> Vec<UsageEvent> {
        self.events
            .read()
            .iter()
            .filter(|e| e.tenant_id == *tenant_id)
            .cloned()
            .collect()
    }

    /// Total number of events recorded.
    pub fn total_events(&self) -> usize {
        self.events.read().len()
    }
}

// ── Rate Limiter ────────────────────────────────────────────────────────────

/// Simple sliding-window rate limiter for per-tenant QPS enforcement.
pub struct TenantRateLimiter {
    windows: RwLock<HashMap<TenantId, std::collections::VecDeque<std::time::Instant>>>,
}

impl TenantRateLimiter {
    pub fn new() -> Self {
        Self {
            windows: RwLock::new(HashMap::new()),
        }
    }

    /// Check if a request is allowed under the tenant's QPS limit.
    /// Returns `true` if allowed, `false` if rate-limited.
    pub fn check_and_record(&self, tenant_id: &TenantId, max_qps: u32) -> bool {
        let now = std::time::Instant::now();
        let window = std::time::Duration::from_secs(1);

        let mut windows = self.windows.write();
        let deque = windows
            .entry(tenant_id.clone())
            .or_insert_with(std::collections::VecDeque::new);

        // Purge entries older than 1 second.
        while deque
            .front()
            .map_or(false, |t| now.duration_since(*t) > window)
        {
            deque.pop_front();
        }

        if deque.len() as u32 >= max_qps {
            return false;
        }

        deque.push_back(now);
        true
    }

    /// Current request count in the sliding window for a tenant.
    pub fn current_count(&self, tenant_id: &TenantId) -> u32 {
        let now = std::time::Instant::now();
        let window = std::time::Duration::from_secs(1);

        let windows = self.windows.read();
        match windows.get(tenant_id) {
            Some(deque) => deque
                .iter()
                .filter(|t| now.duration_since(**t) <= window)
                .count() as u32,
            None => 0,
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn tid(s: &str) -> TenantId {
        TenantId(s.to_string())
    }

    #[test]
    fn test_create_tenant() {
        let mgr = TenantManager::new();
        mgr.create_tenant(tid("t1"), "Acme".into(), TenantConfig::default())
            .unwrap();
        let t = mgr.get_tenant(&tid("t1")).unwrap();
        assert_eq!(t.name, "Acme");
        assert_eq!(t.status, TenantStatus::Active);
        assert_eq!(mgr.tenant_count(), 1);
    }

    #[test]
    fn test_create_duplicate_tenant() {
        let mgr = TenantManager::new();
        mgr.create_tenant(tid("t1"), "Acme".into(), TenantConfig::default())
            .unwrap();
        let res = mgr.create_tenant(tid("t1"), "Acme2".into(), TenantConfig::default());
        assert!(res.is_err());
    }

    #[test]
    fn test_suspend_tenant() {
        let mgr = TenantManager::new();
        mgr.create_tenant(tid("t1"), "Acme".into(), TenantConfig::default())
            .unwrap();
        mgr.suspend_tenant(&tid("t1")).unwrap();
        let t = mgr.get_tenant(&tid("t1")).unwrap();
        assert_eq!(t.status, TenantStatus::Suspended);
    }

    #[test]
    fn test_delete_tenant() {
        let mgr = TenantManager::new();
        mgr.create_tenant(tid("t1"), "Acme".into(), TenantConfig::default())
            .unwrap();
        mgr.delete_tenant(&tid("t1")).unwrap();
        let t = mgr.get_tenant(&tid("t1")).unwrap();
        assert_eq!(t.status, TenantStatus::PendingDeletion);
    }

    #[test]
    fn test_quota_check() {
        let mgr = TenantManager::new();
        mgr.create_tenant(tid("t1"), "Acme".into(), TenantConfig::default())
            .unwrap();
        assert!(mgr.check_quota(&tid("t1"), 100).unwrap());
    }

    #[test]
    fn test_quota_exceeded() {
        let mgr = TenantManager::new();
        let cfg = TenantConfig {
            max_vectors: 50,
            ..TenantConfig::default()
        };
        mgr.create_tenant(tid("t1"), "Acme".into(), cfg).unwrap();
        mgr.record_usage(&tid("t1"), 40, 0).unwrap();
        assert!(!mgr.check_quota(&tid("t1"), 20).unwrap());
    }

    #[test]
    fn test_record_usage() {
        let mgr = TenantManager::new();
        mgr.create_tenant(tid("t1"), "Acme".into(), TenantConfig::default())
            .unwrap();
        mgr.record_usage(&tid("t1"), 500, 1024).unwrap();
        let t = mgr.get_tenant(&tid("t1")).unwrap();
        assert_eq!(t.current_vectors, 500);
        assert_eq!(t.current_storage_bytes, 1024);
    }

    #[test]
    fn test_encryption_key() {
        let mgr = TenantManager::new();
        mgr.create_tenant(tid("t1"), "Acme".into(), TenantConfig::default())
            .unwrap();
        mgr.set_encryption_key(&tid("t1"), b"secret-key").unwrap();
        assert!(mgr
            .verify_encryption_key(&tid("t1"), b"secret-key")
            .unwrap());
        assert!(!mgr.verify_encryption_key(&tid("t1"), b"wrong-key").unwrap());
    }

    #[test]
    fn test_access_policy() {
        let mgr = TenantManager::new();
        mgr.create_tenant(tid("t1"), "Acme".into(), TenantConfig::default())
            .unwrap();
        let mut perms = HashSet::new();
        perms.insert(Permission::SearchVector);
        perms.insert(Permission::InsertVector);
        let policy = AccessPolicy {
            tenant_id: tid("t1"),
            role: TenantRole::ReadWrite,
            permissions: perms,
        };
        mgr.set_access_policy(&tid("t1"), policy).unwrap();
        assert!(mgr
            .check_permission(&tid("t1"), &Permission::SearchVector)
            .unwrap());
    }

    #[test]
    fn test_permission_check() {
        let mgr = TenantManager::new();
        mgr.create_tenant(tid("t1"), "Acme".into(), TenantConfig::default())
            .unwrap();
        // No policy set → permission denied
        assert!(!mgr
            .check_permission(&tid("t1"), &Permission::ManageTenant)
            .unwrap());
    }

    #[test]
    fn test_audit_log() {
        let mgr = TenantManager::new();
        mgr.create_tenant(tid("t1"), "Acme".into(), TenantConfig::default())
            .unwrap();
        mgr.log_audit(AuditLogEntry {
            timestamp: Utc::now().to_rfc3339(),
            tenant_id: tid("t1"),
            action: "search".into(),
            resource: "collection/docs".into(),
            success: true,
            details: None,
        });
        let log = mgr.get_audit_log(&tid("t1"));
        assert_eq!(log.len(), 1);
        assert_eq!(log[0].action, "search");
    }

    #[test]
    fn test_gdpr_export() {
        let mgr = TenantManager::new();
        mgr.create_tenant(tid("t1"), "Acme".into(), TenantConfig::default())
            .unwrap();
        mgr.record_usage(&tid("t1"), 42, 0).unwrap();
        let export = mgr.prepare_gdpr_export(&tid("t1")).unwrap();
        assert_eq!(export.tenant_id, tid("t1"));
        assert_eq!(export.total_vectors, 42);
        assert!(!export.checksum.is_empty());
    }

    #[test]
    fn test_gdpr_delete() {
        let mgr = TenantManager::new();
        mgr.create_tenant(tid("t1"), "Acme".into(), TenantConfig::default())
            .unwrap();
        mgr.gdpr_delete(&tid("t1")).unwrap();
        let t = mgr.get_tenant(&tid("t1")).unwrap();
        assert_eq!(t.status, TenantStatus::Deleted);
        let log = mgr.get_audit_log(&tid("t1"));
        assert!(log.iter().any(|e| e.action == "gdpr_delete"));
    }

    #[test]
    fn test_resource_usage() {
        let mgr = TenantManager::new();
        let cfg = TenantConfig {
            max_vectors: 1000,
            max_collections: 10,
            max_storage_bytes: 1_048_576,
            ..TenantConfig::default()
        };
        mgr.create_tenant(tid("t1"), "Acme".into(), cfg).unwrap();
        mgr.record_usage(&tid("t1"), 250, 4096).unwrap();
        let usage = mgr.get_usage(&tid("t1")).unwrap();
        assert_eq!(usage.vectors_used, 250);
        assert_eq!(usage.vectors_limit, 1000);
        assert_eq!(usage.storage_used_bytes, 4096);
        assert_eq!(usage.storage_limit_bytes, 1_048_576);
        assert!((usage.utilization_pct - 25.0).abs() < 0.01);
    }

    #[test]
    fn test_enforce_quota() {
        let mgr = TenantManager::new();
        let cfg = TenantConfig {
            max_vectors: 100,
            ..TenantConfig::default()
        };
        mgr.create_tenant(tid("t1"), "Acme".into(), cfg).unwrap();
        mgr.record_usage(&tid("t1"), 90, 0).unwrap();

        // Should pass: 90 + 5 = 95 <= 100
        mgr.enforce_quota(&tid("t1"), 5).unwrap();
        // Should fail: 90 + 20 = 110 > 100
        assert!(mgr.enforce_quota(&tid("t1"), 20).is_err());
    }

    #[test]
    fn test_enforce_quota_suspended_tenant() {
        let mgr = TenantManager::new();
        mgr.create_tenant(tid("t1"), "Acme".into(), TenantConfig::default())
            .unwrap();
        mgr.suspend_tenant(&tid("t1")).unwrap();
        assert!(mgr.enforce_quota(&tid("t1"), 1).is_err());
    }

    #[test]
    fn test_enforce_storage_quota() {
        let mgr = TenantManager::new();
        let cfg = TenantConfig {
            max_storage_bytes: 1000,
            ..TenantConfig::default()
        };
        mgr.create_tenant(tid("t1"), "Acme".into(), cfg).unwrap();
        mgr.record_usage(&tid("t1"), 0, 900).unwrap();

        mgr.enforce_storage_quota(&tid("t1"), 50).unwrap();
        assert!(mgr.enforce_storage_quota(&tid("t1"), 200).is_err());
    }

    #[test]
    fn test_enforce_collection_quota() {
        let mgr = TenantManager::new();
        let cfg = TenantConfig {
            max_collections: 2,
            ..TenantConfig::default()
        };
        mgr.create_tenant(tid("t1"), "Acme".into(), cfg).unwrap();
        mgr.add_collection(&tid("t1"), "col1".into()).unwrap();
        mgr.add_collection(&tid("t1"), "col2".into()).unwrap();
        assert!(mgr.enforce_collection_quota(&tid("t1")).is_err());
    }

    #[test]
    fn test_add_remove_collection() {
        let mgr = TenantManager::new();
        mgr.create_tenant(tid("t1"), "Acme".into(), TenantConfig::default())
            .unwrap();
        mgr.add_collection(&tid("t1"), "docs".into()).unwrap();

        let t = mgr.get_tenant(&tid("t1")).unwrap();
        assert!(t.collections.contains("docs"));

        mgr.remove_collection(&tid("t1"), "docs").unwrap();
        let t = mgr.get_tenant(&tid("t1")).unwrap();
        assert!(!t.collections.contains("docs"));
    }

    #[test]
    fn test_usage_meter_record_and_summary() {
        let meter = UsageMeter::new(MeteringConfig::default());
        let t = tid("t1");

        meter.record(t.clone(), UsageEventKind::VectorInsert, 100);
        meter.record(t.clone(), UsageEventKind::SearchQuery, 50);
        meter.record(t.clone(), UsageEventKind::BatchSearch, 5);

        assert_eq!(meter.total_events(), 3);
        let events = meter.events_for_tenant(&t);
        assert_eq!(events.len(), 3);

        // Cost checks
        assert_eq!(events[0].cost_micro_cents, 100); // 100 inserts * 1 mc
        assert_eq!(events[1].cost_micro_cents, 250); // 50 queries * 5 mc
        assert_eq!(events[2].cost_micro_cents, 125); // 5 batch * 25 mc
    }

    #[test]
    fn test_billing_summary() {
        let meter = UsageMeter::new(MeteringConfig::default());
        let t = tid("t1");

        meter.record(t.clone(), UsageEventKind::VectorInsert, 1000);
        meter.record(t.clone(), UsageEventKind::SearchQuery, 200);

        let summary = meter.billing_summary(&t, "2020-01-01T00:00:00Z", "2030-12-31T23:59:59Z");
        assert_eq!(summary.total_inserts, 1000);
        assert_eq!(summary.total_queries, 200);
        assert_eq!(summary.events_count, 2);
        assert!(summary.total_cost_micro_cents > 0);
        assert!(summary.total_cost_usd.starts_with('$'));
    }

    #[test]
    fn test_rate_limiter() {
        let limiter = TenantRateLimiter::new();
        let t = tid("t1");

        // Allow 3 QPS
        assert!(limiter.check_and_record(&t, 3));
        assert!(limiter.check_and_record(&t, 3));
        assert!(limiter.check_and_record(&t, 3));
        // 4th should be rejected
        assert!(!limiter.check_and_record(&t, 3));

        assert_eq!(limiter.current_count(&t), 3);
    }
}
