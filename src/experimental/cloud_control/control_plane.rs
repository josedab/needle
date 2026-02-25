#![allow(clippy::unwrap_used)]

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::error::{NeedleError, Result};

use super::tenant::{
    ApiKey, BillingInfo, OverageCharge, PaymentStatus, Permission, Tenant, TenantStatus,
    TenantUsage, UsageEvent, UsageEventType,
};
use super::tiers::{ResourceTier, TenantConfig, TierLimits};

/// Control plane configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlPlaneConfig {
    /// Enable billing
    pub billing_enabled: bool,
    /// Enforce rate limits
    pub rate_limiting_enabled: bool,
    /// Auto-suspend on limit breach
    pub auto_suspend_enabled: bool,
    /// Grace period for overages (seconds)
    pub overage_grace_period_secs: u64,
}

impl Default for ControlPlaneConfig {
    fn default() -> Self {
        Self {
            billing_enabled: true,
            rate_limiting_enabled: true,
            auto_suspend_enabled: true,
            overage_grace_period_secs: 86400, // 24 hours
        }
    }
}

/// Cloud control plane for managed service
pub struct ControlPlane {
    #[allow(dead_code)]
    config: ControlPlaneConfig,
    tenants: RwLock<HashMap<String, Tenant>>,
    api_keys: RwLock<HashMap<String, ApiKey>>,
    usage_events: RwLock<Vec<UsageEvent>>,
    next_tenant_id: AtomicU64,
    next_key_id: AtomicU64,
}

impl ControlPlane {
    /// Create a new control plane
    pub fn new(config: ControlPlaneConfig) -> Self {
        Self {
            config,
            tenants: RwLock::new(HashMap::new()),
            api_keys: RwLock::new(HashMap::new()),
            usage_events: RwLock::new(Vec::new()),
            next_tenant_id: AtomicU64::new(1),
            next_key_id: AtomicU64::new(1),
        }
    }

    /// Create a new tenant
    pub fn create_tenant(&self, config: TenantConfig) -> Result<Tenant> {
        if config.name.is_empty() {
            return Err(NeedleError::InvalidInput(
                "Tenant name required".to_string(),
            ));
        }

        // Check for duplicate name
        let tenants = self.tenants.read();
        if tenants.values().any(|t| t.config.name == config.name) {
            return Err(NeedleError::InvalidInput(format!(
                "Tenant '{}' already exists",
                config.name
            )));
        }
        drop(tenants);

        let id = format!(
            "tenant_{}",
            self.next_tenant_id.fetch_add(1, Ordering::SeqCst)
        );
        let now = current_timestamp();

        let tenant = Tenant {
            id: id.clone(),
            config,
            status: TenantStatus::Provisioning,
            created_at: now,
            updated_at: now,
            usage: TenantUsage::default(),
        };

        self.tenants.write().insert(id.clone(), tenant.clone());

        Ok(tenant)
    }

    /// Get tenant by ID
    pub fn get_tenant(&self, tenant_id: &str) -> Option<Tenant> {
        self.tenants.read().get(tenant_id).cloned()
    }

    /// Update tenant configuration
    pub fn update_tenant(&self, tenant_id: &str, config: TenantConfig) -> Result<Tenant> {
        let mut tenants = self.tenants.write();
        let tenant = tenants
            .get_mut(tenant_id)
            .ok_or_else(|| NeedleError::NotFound(format!("Tenant not found: {}", tenant_id)))?;

        tenant.config = config;
        tenant.updated_at = current_timestamp();

        Ok(tenant.clone())
    }

    /// Activate a tenant (after provisioning)
    pub fn activate_tenant(&self, tenant_id: &str) -> Result<()> {
        let mut tenants = self.tenants.write();
        let tenant = tenants
            .get_mut(tenant_id)
            .ok_or_else(|| NeedleError::NotFound(format!("Tenant not found: {}", tenant_id)))?;

        tenant.status = TenantStatus::Active;
        tenant.updated_at = current_timestamp();

        Ok(())
    }

    /// Suspend a tenant
    pub fn suspend_tenant(&self, tenant_id: &str) -> Result<()> {
        let mut tenants = self.tenants.write();
        let tenant = tenants
            .get_mut(tenant_id)
            .ok_or_else(|| NeedleError::NotFound(format!("Tenant not found: {}", tenant_id)))?;

        tenant.status = TenantStatus::Suspended;
        tenant.updated_at = current_timestamp();

        Ok(())
    }

    /// Delete a tenant
    pub fn delete_tenant(&self, tenant_id: &str) -> Result<()> {
        let mut tenants = self.tenants.write();
        let tenant = tenants
            .get_mut(tenant_id)
            .ok_or_else(|| NeedleError::NotFound(format!("Tenant not found: {}", tenant_id)))?;

        tenant.status = TenantStatus::PendingDeletion;
        tenant.updated_at = current_timestamp();

        // Delete associated API keys
        self.api_keys
            .write()
            .retain(|_, k| k.tenant_id != tenant_id);

        Ok(())
    }

    /// List all tenants
    pub fn list_tenants(&self) -> Vec<Tenant> {
        self.tenants
            .read()
            .values()
            .filter(|t| t.status != TenantStatus::Deleted)
            .cloned()
            .collect()
    }

    /// Create an API key for a tenant
    pub fn create_api_key(
        &self,
        tenant_id: &str,
        name: &str,
        permissions: Vec<Permission>,
    ) -> Result<(String, ApiKey)> {
        // Verify tenant exists
        let tenants = self.tenants.read();
        if !tenants.contains_key(tenant_id) {
            return Err(NeedleError::NotFound(format!(
                "Tenant not found: {}",
                tenant_id
            )));
        }
        drop(tenants);

        let key_id = format!("key_{}", self.next_key_id.fetch_add(1, Ordering::SeqCst));
        let raw_key = generate_api_key();
        let key_hash = hash_api_key(&raw_key);

        let api_key = ApiKey {
            id: key_id.clone(),
            key_hash,
            name: name.to_string(),
            tenant_id: tenant_id.to_string(),
            permissions,
            created_at: current_timestamp(),
            expires_at: None,
            last_used_at: None,
            active: true,
        };

        self.api_keys
            .write()
            .insert(key_id.clone(), api_key.clone());

        // Return the raw key (only time it's available)
        Ok((raw_key, api_key))
    }

    /// Validate an API key and return tenant info
    pub fn validate_api_key(&self, raw_key: &str) -> Option<(Tenant, ApiKey)> {
        let key_hash = hash_api_key(raw_key);

        let api_keys = self.api_keys.read();
        let api_key = api_keys
            .values()
            .find(|k| k.key_hash == key_hash && k.active)?;

        // Update last used
        let tenant_id = api_key.tenant_id.clone();
        let key_id = api_key.id.clone();
        let api_key = api_key.clone();
        drop(api_keys);

        // Update last used timestamp
        if let Some(key) = self.api_keys.write().get_mut(&key_id) {
            key.last_used_at = Some(current_timestamp());
        }

        let tenant = self.get_tenant(&tenant_id)?;

        // Check tenant is active
        if tenant.status != TenantStatus::Active {
            return None;
        }

        Some((tenant, api_key))
    }

    /// Revoke an API key
    pub fn revoke_api_key(&self, key_id: &str) -> Result<()> {
        let mut keys = self.api_keys.write();
        let key = keys
            .get_mut(key_id)
            .ok_or_else(|| NeedleError::NotFound(format!("API key not found: {}", key_id)))?;

        key.active = false;
        Ok(())
    }

    /// List API keys for a tenant
    pub fn list_api_keys(&self, tenant_id: &str) -> Vec<ApiKey> {
        self.api_keys
            .read()
            .values()
            .filter(|k| k.tenant_id == tenant_id)
            .cloned()
            .collect()
    }

    /// Record a usage event
    pub fn record_usage(&self, tenant_id: &str, event_type: UsageEventType, quantity: u64) {
        let event = UsageEvent {
            id: format!("evt_{}_{}", tenant_id, current_timestamp()),
            tenant_id: tenant_id.to_string(),
            event_type,
            quantity,
            timestamp: current_timestamp(),
            metadata: HashMap::new(),
        };

        self.usage_events.write().push(event);

        // Update tenant usage
        if let Some(tenant) = self.tenants.write().get_mut(tenant_id) {
            match event_type {
                UsageEventType::Query => {
                    tenant.usage.queries_this_minute += quantity;
                    tenant.usage.total_queries += quantity;
                }
                UsageEventType::Insert => {
                    tenant.usage.vectors_count += quantity;
                    tenant.usage.total_inserts += quantity;
                }
                UsageEventType::Delete => {
                    tenant.usage.vectors_count =
                        tenant.usage.vectors_count.saturating_sub(quantity);
                    tenant.usage.total_deletes += quantity;
                }
                UsageEventType::StorageIncrease => {
                    tenant.usage.storage_bytes += quantity;
                }
                UsageEventType::StorageDecrease => {
                    tenant.usage.storage_bytes =
                        tenant.usage.storage_bytes.saturating_sub(quantity);
                }
                UsageEventType::ApiCall => {
                    tenant.usage.requests_today += quantity;
                }
            }
        }
    }

    /// Check if a tenant is within limits
    pub fn check_limits(&self, tenant_id: &str) -> Result<LimitCheckResult> {
        let tenant = self
            .get_tenant(tenant_id)
            .ok_or_else(|| NeedleError::NotFound(format!("Tenant not found: {}", tenant_id)))?;

        let limits = tenant
            .config
            .custom_limits
            .as_ref()
            .cloned()
            .unwrap_or_else(|| tenant.config.tier.limits());

        let mut violations = Vec::new();

        if tenant.usage.collections_count > limits.max_collections {
            violations.push(LimitViolation {
                resource: "collections".to_string(),
                current: tenant.usage.collections_count,
                limit: limits.max_collections,
            });
        }

        if tenant.usage.vectors_count > limits.max_vectors {
            violations.push(LimitViolation {
                resource: "vectors".to_string(),
                current: tenant.usage.vectors_count,
                limit: limits.max_vectors,
            });
        }

        if tenant.usage.storage_bytes > limits.max_storage_bytes {
            violations.push(LimitViolation {
                resource: "storage".to_string(),
                current: tenant.usage.storage_bytes,
                limit: limits.max_storage_bytes,
            });
        }

        if tenant.usage.queries_this_minute > limits.queries_per_minute {
            violations.push(LimitViolation {
                resource: "queries_per_minute".to_string(),
                current: tenant.usage.queries_this_minute,
                limit: limits.queries_per_minute,
            });
        }

        if tenant.usage.requests_today > limits.requests_per_day {
            violations.push(LimitViolation {
                resource: "requests_per_day".to_string(),
                current: tenant.usage.requests_today,
                limit: limits.requests_per_day,
            });
        }

        Ok(LimitCheckResult {
            within_limits: violations.is_empty(),
            violations,
        })
    }

    /// Generate billing for a tenant
    pub fn generate_billing(&self, tenant_id: &str) -> Result<BillingInfo> {
        let tenant = self
            .get_tenant(tenant_id)
            .ok_or_else(|| NeedleError::NotFound(format!("Tenant not found: {}", tenant_id)))?;

        let now = current_timestamp();
        let period_start = now - (now % (30 * 24 * 3600)); // Start of current 30-day period
        let period_end = period_start + (30 * 24 * 3600);

        let base_charge = tenant.config.tier.price_cents();
        let limits = tenant.config.tier.limits();

        let mut overage_charges = Vec::new();

        // Calculate overage for vectors
        if tenant.usage.vectors_count > limits.max_vectors {
            let overage = tenant.usage.vectors_count - limits.max_vectors;
            let charge = (overage / 1000) * 10; // $0.10 per 1000 vectors
            overage_charges.push(OverageCharge {
                resource: "vectors".to_string(),
                overage_amount: overage,
                unit_price_cents: 10,
                charge_cents: charge,
            });
        }

        // Calculate overage for storage
        if tenant.usage.storage_bytes > limits.max_storage_bytes {
            let overage_gb =
                (tenant.usage.storage_bytes - limits.max_storage_bytes) / (1024 * 1024 * 1024);
            let charge = overage_gb * 100; // $1.00 per GB
            overage_charges.push(OverageCharge {
                resource: "storage".to_string(),
                overage_amount: overage_gb,
                unit_price_cents: 100,
                charge_cents: charge,
            });
        }

        let total_overage: u64 = overage_charges.iter().map(|c| c.charge_cents).sum();

        Ok(BillingInfo {
            tenant_id: tenant_id.to_string(),
            period_start,
            period_end,
            base_charge_cents: base_charge,
            overage_charges,
            total_cents: base_charge + total_overage,
            status: PaymentStatus::Pending,
        })
    }

    /// Get usage summary for a tenant
    pub fn get_usage_summary(&self, tenant_id: &str) -> Result<UsageSummary> {
        let tenant = self
            .get_tenant(tenant_id)
            .ok_or_else(|| NeedleError::NotFound(format!("Tenant not found: {}", tenant_id)))?;

        let limits = tenant
            .config
            .custom_limits
            .as_ref()
            .cloned()
            .unwrap_or_else(|| tenant.config.tier.limits());

        Ok(UsageSummary {
            tenant_id: tenant_id.to_string(),
            tier: tenant.config.tier,
            current: tenant.usage.clone(),
            utilization: UsageUtilization {
                collections_pct: percent(tenant.usage.collections_count, limits.max_collections),
                vectors_pct: percent(tenant.usage.vectors_count, limits.max_vectors),
                storage_pct: percent(tenant.usage.storage_bytes, limits.max_storage_bytes),
            },
            limits,
        })
    }
}

impl Default for ControlPlane {
    fn default() -> Self {
        Self::new(ControlPlaneConfig::default())
    }
}

/// Result of limit check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LimitCheckResult {
    pub within_limits: bool,
    pub violations: Vec<LimitViolation>,
}

/// A limit violation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LimitViolation {
    pub resource: String,
    pub current: u64,
    pub limit: u64,
}

/// Usage summary for a tenant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageSummary {
    pub tenant_id: String,
    pub tier: ResourceTier,
    pub current: TenantUsage,
    pub limits: TierLimits,
    pub utilization: UsageUtilization,
}

/// Usage utilization percentages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageUtilization {
    pub collections_pct: f64,
    pub vectors_pct: f64,
    pub storage_pct: f64,
}

/// Get current timestamp
pub(crate) fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Generate a random API key
pub(crate) fn generate_api_key() -> String {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};

    let state = RandomState::new();
    let mut hasher = state.build_hasher();
    hasher.write_u64(current_timestamp());
    hasher.write_u64(rand_u64());

    format!("ndk_{:016x}{:016x}", hasher.finish(), rand_u64())
}

/// Simple pseudo-random u64
pub(crate) fn rand_u64() -> u64 {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};

    let state = RandomState::new();
    let mut hasher = state.build_hasher();
    hasher.write_u64(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64,
    );
    hasher.finish()
}

/// Hash an API key for storage
pub(crate) fn hash_api_key(key: &str) -> String {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};

    let state = RandomState::new();
    let mut hasher = state.build_hasher();
    hasher.write(key.as_bytes());
    format!("{:016x}", hasher.finish())
}

/// Calculate percentage
pub(crate) fn percent(current: u64, limit: u64) -> f64 {
    if limit == 0 || limit == u64::MAX {
        0.0
    } else {
        (current as f64 / limit as f64) * 100.0
    }
}
