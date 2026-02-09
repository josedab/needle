//! Cloud Control Plane for Managed Needle Service
//!
//! Provides management APIs for a cloud-hosted Needle service:
//! - Tenant management and isolation
//! - Resource provisioning and scaling
//! - Usage metering and billing
//! - API key management
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::cloud_control::{ControlPlane, TenantConfig, ResourceTier};
//!
//! let control_plane = ControlPlane::new(ControlPlaneConfig::default());
//!
//! // Create a new tenant
//! let tenant = control_plane.create_tenant(TenantConfig {
//!     name: "acme-corp".to_string(),
//!     tier: ResourceTier::Professional,
//!     ..Default::default()
//! })?;
//!
//! // Provision resources
//! control_plane.provision_database(&tenant.id)?;
//!
//! // Generate API key
//! let api_key = control_plane.create_api_key(&tenant.id, "production")?;
//! ```

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::error::{NeedleError, Result};

/// Resource tier for tenant plans
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResourceTier {
    /// Free tier - limited resources
    Free,
    /// Developer tier - for small projects
    Developer,
    /// Professional tier - for production use
    Professional,
    /// Enterprise tier - custom limits
    Enterprise,
}

impl Default for ResourceTier {
    fn default() -> Self {
        ResourceTier::Free
    }
}

impl ResourceTier {
    /// Get tier limits
    pub fn limits(&self) -> TierLimits {
        match self {
            ResourceTier::Free => TierLimits {
                max_collections: 3,
                max_vectors: 10_000,
                max_dimensions: 768,
                max_storage_bytes: 100 * 1024 * 1024, // 100MB
                queries_per_minute: 60,
                requests_per_day: 10_000,
                support_level: SupportLevel::Community,
            },
            ResourceTier::Developer => TierLimits {
                max_collections: 10,
                max_vectors: 100_000,
                max_dimensions: 1536,
                max_storage_bytes: 1024 * 1024 * 1024, // 1GB
                queries_per_minute: 300,
                requests_per_day: 100_000,
                support_level: SupportLevel::Email,
            },
            ResourceTier::Professional => TierLimits {
                max_collections: 50,
                max_vectors: 1_000_000,
                max_dimensions: 4096,
                max_storage_bytes: 10 * 1024 * 1024 * 1024, // 10GB
                queries_per_minute: 1000,
                requests_per_day: 1_000_000,
                support_level: SupportLevel::Priority,
            },
            ResourceTier::Enterprise => TierLimits {
                max_collections: u64::MAX,
                max_vectors: u64::MAX,
                max_dimensions: 8192,
                max_storage_bytes: u64::MAX,
                queries_per_minute: u64::MAX,
                requests_per_day: u64::MAX,
                support_level: SupportLevel::Dedicated,
            },
        }
    }

    /// Get monthly price in cents
    pub fn price_cents(&self) -> u64 {
        match self {
            ResourceTier::Free => 0,
            ResourceTier::Developer => 2900,     // $29/month
            ResourceTier::Professional => 9900,  // $99/month
            ResourceTier::Enterprise => 0,       // Custom pricing
        }
    }
}

/// Tier resource limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TierLimits {
    pub max_collections: u64,
    pub max_vectors: u64,
    pub max_dimensions: usize,
    pub max_storage_bytes: u64,
    pub queries_per_minute: u64,
    pub requests_per_day: u64,
    pub support_level: SupportLevel,
}

/// Support level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SupportLevel {
    Community,
    Email,
    Priority,
    Dedicated,
}

/// Tenant configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TenantConfig {
    /// Tenant name
    pub name: String,
    /// Resource tier
    pub tier: ResourceTier,
    /// Organization email
    pub email: String,
    /// Custom limits override (for Enterprise)
    pub custom_limits: Option<TierLimits>,
    /// Enabled features
    pub features: Vec<String>,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

impl Default for TenantConfig {
    fn default() -> Self {
        Self {
            name: String::new(),
            tier: ResourceTier::Free,
            email: String::new(),
            custom_limits: None,
            features: Vec::new(),
            metadata: HashMap::new(),
        }
    }
}

/// Tenant information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tenant {
    /// Unique tenant ID
    pub id: String,
    /// Configuration
    pub config: TenantConfig,
    /// Status
    pub status: TenantStatus,
    /// Created timestamp
    pub created_at: u64,
    /// Updated timestamp
    pub updated_at: u64,
    /// Current usage
    pub usage: TenantUsage,
}

/// Tenant status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TenantStatus {
    /// Active and operational
    Active,
    /// Provisioning in progress
    Provisioning,
    /// Suspended (e.g., payment issue)
    Suspended,
    /// Scheduled for deletion
    PendingDeletion,
    /// Deleted
    Deleted,
}

/// Current tenant usage
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TenantUsage {
    pub collections_count: u64,
    pub vectors_count: u64,
    pub storage_bytes: u64,
    pub queries_this_minute: u64,
    pub requests_today: u64,
    pub total_queries: u64,
    pub total_inserts: u64,
    pub total_deletes: u64,
}

/// API key for tenant access
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKey {
    /// Key ID (public)
    pub id: String,
    /// Key hash (stored, never exposed)
    pub key_hash: String,
    /// Display name
    pub name: String,
    /// Tenant ID this key belongs to
    pub tenant_id: String,
    /// Permissions
    pub permissions: Vec<Permission>,
    /// Created timestamp
    pub created_at: u64,
    /// Expires timestamp (if any)
    pub expires_at: Option<u64>,
    /// Last used timestamp
    pub last_used_at: Option<u64>,
    /// Is active
    pub active: bool,
}

/// API permissions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Permission {
    /// Read vectors
    Read,
    /// Write vectors
    Write,
    /// Delete vectors
    Delete,
    /// Manage collections
    ManageCollections,
    /// View metrics
    ViewMetrics,
    /// Admin access
    Admin,
}

/// Billing information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BillingInfo {
    /// Tenant ID
    pub tenant_id: String,
    /// Current billing period start
    pub period_start: u64,
    /// Current billing period end
    pub period_end: u64,
    /// Base charge (tier subscription)
    pub base_charge_cents: u64,
    /// Overage charges
    pub overage_charges: Vec<OverageCharge>,
    /// Total amount due
    pub total_cents: u64,
    /// Payment status
    pub status: PaymentStatus,
}

/// Overage charge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverageCharge {
    /// Resource type
    pub resource: String,
    /// Amount over limit
    pub overage_amount: u64,
    /// Unit price in cents
    pub unit_price_cents: u64,
    /// Total charge
    pub charge_cents: u64,
}

/// Payment status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PaymentStatus {
    Pending,
    Paid,
    Failed,
    Waived,
}

/// Usage event for metering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageEvent {
    /// Event ID
    pub id: String,
    /// Tenant ID
    pub tenant_id: String,
    /// Event type
    pub event_type: UsageEventType,
    /// Quantity
    pub quantity: u64,
    /// Timestamp
    pub timestamp: u64,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

/// Usage event types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UsageEventType {
    Query,
    Insert,
    Delete,
    StorageIncrease,
    StorageDecrease,
    ApiCall,
}

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
            return Err(NeedleError::InvalidInput("Tenant name required".to_string()));
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

        let id = format!("tenant_{}", self.next_tenant_id.fetch_add(1, Ordering::SeqCst));
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

        self.api_keys.write().insert(key_id.clone(), api_key.clone());

        // Return the raw key (only time it's available)
        Ok((raw_key, api_key))
    }

    /// Validate an API key and return tenant info
    pub fn validate_api_key(&self, raw_key: &str) -> Option<(Tenant, ApiKey)> {
        let key_hash = hash_api_key(raw_key);

        let api_keys = self.api_keys.read();
        let api_key = api_keys.values().find(|k| k.key_hash == key_hash && k.active)?;

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
            id: format!(
                "evt_{}_{}",
                tenant_id,
                current_timestamp()
            ),
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
                    tenant.usage.vectors_count = tenant.usage.vectors_count.saturating_sub(quantity);
                    tenant.usage.total_deletes += quantity;
                }
                UsageEventType::StorageIncrease => {
                    tenant.usage.storage_bytes += quantity;
                }
                UsageEventType::StorageDecrease => {
                    tenant.usage.storage_bytes = tenant.usage.storage_bytes.saturating_sub(quantity);
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
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Generate a random API key
fn generate_api_key() -> String {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};

    let state = RandomState::new();
    let mut hasher = state.build_hasher();
    hasher.write_u64(current_timestamp());
    hasher.write_u64(rand_u64());

    format!("ndk_{:016x}{:016x}", hasher.finish(), rand_u64())
}

/// Simple pseudo-random u64
fn rand_u64() -> u64 {
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
fn hash_api_key(key: &str) -> String {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};

    let state = RandomState::new();
    let mut hasher = state.build_hasher();
    hasher.write(key.as_bytes());
    format!("{:016x}", hasher.finish())
}

/// Calculate percentage
fn percent(current: u64, limit: u64) -> f64 {
    if limit == 0 || limit == u64::MAX {
        0.0
    } else {
        (current as f64 / limit as f64) * 100.0
    }
}

// ---------------------------------------------------------------------------
// Multi-Region Routing
// ---------------------------------------------------------------------------

/// Identifier for a geographic region.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Region {
    UsEast1,
    UsWest2,
    EuWest1,
    EuCentral1,
    ApSoutheast1,
    ApNortheast1,
    Custom(String),
}

impl std::fmt::Display for Region {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Region::UsEast1 => write!(f, "us-east-1"),
            Region::UsWest2 => write!(f, "us-west-2"),
            Region::EuWest1 => write!(f, "eu-west-1"),
            Region::EuCentral1 => write!(f, "eu-central-1"),
            Region::ApSoutheast1 => write!(f, "ap-southeast-1"),
            Region::ApNortheast1 => write!(f, "ap-northeast-1"),
            Region::Custom(s) => write!(f, "{}", s),
        }
    }
}

/// Health status of a regional deployment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RegionHealth {
    Healthy,
    Degraded,
    Unhealthy,
    Unknown,
}

/// A regional Needle deployment endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionalEndpoint {
    pub region: Region,
    pub endpoint_url: String,
    pub health: RegionHealth,
    pub latency_ms: f64,
    pub capacity_pct: f64,
    pub last_health_check: u64,
}

/// Strategy for routing requests across regions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RoutingStrategy {
    /// Route to nearest healthy region by latency
    LatencyBased,
    /// Round-robin across healthy regions
    RoundRobin,
    /// Route to the region with most available capacity
    CapacityBased,
    /// Always use primary, failover on unhealthy
    PrimaryWithFailover,
}

impl Default for RoutingStrategy {
    fn default() -> Self {
        RoutingStrategy::LatencyBased
    }
}

/// Multi-region router that directs traffic to the best endpoint.
pub struct RegionRouter {
    endpoints: RwLock<Vec<RegionalEndpoint>>,
    strategy: RoutingStrategy,
    primary_region: Region,
    round_robin_counter: AtomicU64,
}

impl RegionRouter {
    pub fn new(primary_region: Region, strategy: RoutingStrategy) -> Self {
        Self {
            endpoints: RwLock::new(Vec::new()),
            strategy,
            primary_region,
            round_robin_counter: AtomicU64::new(0),
        }
    }

    /// Register a regional endpoint.
    pub fn add_endpoint(&self, endpoint: RegionalEndpoint) {
        self.endpoints.write().push(endpoint);
    }

    /// Update the health status of a region.
    pub fn update_health(&self, region: &Region, health: RegionHealth, latency_ms: f64) {
        let mut eps = self.endpoints.write();
        if let Some(ep) = eps.iter_mut().find(|e| &e.region == region) {
            ep.health = health;
            ep.latency_ms = latency_ms;
            ep.last_health_check = current_timestamp();
        }
    }

    /// Select the best endpoint for a request based on the routing strategy.
    pub fn route(&self) -> Option<RegionalEndpoint> {
        let eps = self.endpoints.read();
        let healthy: Vec<&RegionalEndpoint> = eps
            .iter()
            .filter(|e| e.health == RegionHealth::Healthy || e.health == RegionHealth::Degraded)
            .collect();

        if healthy.is_empty() {
            return None;
        }

        match self.strategy {
            RoutingStrategy::LatencyBased => healthy
                .iter()
                .min_by(|a, b| a.latency_ms.partial_cmp(&b.latency_ms).unwrap_or(std::cmp::Ordering::Equal))
                .map(|e| (*e).clone()),

            RoutingStrategy::RoundRobin => {
                let idx = self.round_robin_counter.fetch_add(1, Ordering::Relaxed) as usize;
                Some(healthy[idx % healthy.len()].clone())
            }

            RoutingStrategy::CapacityBased => healthy
                .iter()
                .min_by(|a, b| a.capacity_pct.partial_cmp(&b.capacity_pct).unwrap_or(std::cmp::Ordering::Equal))
                .map(|e| (*e).clone()),

            RoutingStrategy::PrimaryWithFailover => {
                if let Some(primary) = healthy.iter().find(|e| e.region == self.primary_region) {
                    Some((*primary).clone())
                } else {
                    healthy.first().map(|e| (*e).clone())
                }
            }
        }
    }

    /// Return all endpoints with their health status.
    pub fn list_endpoints(&self) -> Vec<RegionalEndpoint> {
        self.endpoints.read().clone()
    }
}

// ---------------------------------------------------------------------------
// SLA Monitoring
// ---------------------------------------------------------------------------

/// Tracks SLA compliance for a tenant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlaPolicy {
    /// Target availability percentage (e.g. 99.95)
    pub target_availability_pct: f64,
    /// Maximum p99 query latency in milliseconds
    pub max_p99_latency_ms: f64,
    /// Maximum data loss window in seconds (RPO)
    pub max_data_loss_seconds: u64,
    /// Maximum recovery time in seconds (RTO)
    pub max_recovery_seconds: u64,
}

impl Default for SlaPolicy {
    fn default() -> Self {
        Self {
            target_availability_pct: 99.9,
            max_p99_latency_ms: 50.0,
            max_data_loss_seconds: 60,
            max_recovery_seconds: 300,
        }
    }
}

/// A recorded SLA breach event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlaBreach {
    pub timestamp: u64,
    pub breach_type: SlaBreachType,
    pub actual_value: f64,
    pub threshold: f64,
    pub duration_seconds: u64,
}

/// Types of SLA breaches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SlaBreachType {
    Availability,
    Latency,
    DataLoss,
    RecoveryTime,
}

/// Sliding-window SLA health report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlaReport {
    pub window_seconds: u64,
    pub availability_pct: f64,
    pub p99_latency_ms: f64,
    pub total_requests: u64,
    pub failed_requests: u64,
    pub breaches: Vec<SlaBreach>,
    pub compliant: bool,
}

/// Monitors SLA compliance using a rolling window of request outcomes.
pub struct SlaMonitor {
    policy: SlaPolicy,
    window_seconds: u64,
    request_log: RwLock<Vec<RequestOutcome>>,
    breaches: RwLock<Vec<SlaBreach>>,
}

#[derive(Debug, Clone)]
struct RequestOutcome {
    timestamp: u64,
    success: bool,
    latency_ms: f64,
}

impl SlaMonitor {
    pub fn new(policy: SlaPolicy, window_seconds: u64) -> Self {
        Self {
            policy,
            window_seconds,
            request_log: RwLock::new(Vec::new()),
            breaches: RwLock::new(Vec::new()),
        }
    }

    /// Record a request outcome for SLA tracking.
    pub fn record_request(&self, success: bool, latency_ms: f64) {
        let outcome = RequestOutcome {
            timestamp: current_timestamp(),
            success,
            latency_ms,
        };
        self.request_log.write().push(outcome);
    }

    /// Generate an SLA compliance report for the current window.
    pub fn report(&self) -> SlaReport {
        let now = current_timestamp();
        let cutoff = now.saturating_sub(self.window_seconds);

        let log = self.request_log.read();
        let windowed: Vec<&RequestOutcome> = log.iter().filter(|r| r.timestamp >= cutoff).collect();

        let total = windowed.len() as u64;
        let failed = windowed.iter().filter(|r| !r.success).count() as u64;
        let availability_pct = if total > 0 {
            ((total - failed) as f64 / total as f64) * 100.0
        } else {
            100.0
        };

        let p99_latency_ms = {
            let mut latencies: Vec<f64> = windowed.iter().map(|r| r.latency_ms).collect();
            latencies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            if latencies.is_empty() {
                0.0
            } else {
                let idx = ((latencies.len() as f64) * 0.99).ceil() as usize;
                latencies[idx.min(latencies.len() - 1)]
            }
        };

        let mut new_breaches = Vec::new();
        if availability_pct < self.policy.target_availability_pct && total > 0 {
            new_breaches.push(SlaBreach {
                timestamp: now,
                breach_type: SlaBreachType::Availability,
                actual_value: availability_pct,
                threshold: self.policy.target_availability_pct,
                duration_seconds: self.window_seconds,
            });
        }
        if p99_latency_ms > self.policy.max_p99_latency_ms && total > 0 {
            new_breaches.push(SlaBreach {
                timestamp: now,
                breach_type: SlaBreachType::Latency,
                actual_value: p99_latency_ms,
                threshold: self.policy.max_p99_latency_ms,
                duration_seconds: self.window_seconds,
            });
        }

        if !new_breaches.is_empty() {
            self.breaches.write().extend(new_breaches.clone());
        }

        let all_breaches = self.breaches.read().clone();
        SlaReport {
            window_seconds: self.window_seconds,
            availability_pct,
            p99_latency_ms,
            total_requests: total,
            failed_requests: failed,
            breaches: all_breaches,
            compliant: availability_pct >= self.policy.target_availability_pct
                && p99_latency_ms <= self.policy.max_p99_latency_ms,
        }
    }

    /// Prune old entries outside the monitoring window.
    pub fn prune(&self) {
        let cutoff = current_timestamp().saturating_sub(self.window_seconds * 2);
        self.request_log.write().retain(|r| r.timestamp >= cutoff);
    }
}

// ---------------------------------------------------------------------------
// Service Orchestrator
// ---------------------------------------------------------------------------

/// Represents a managed Needle database instance for a tenant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManagedInstance {
    pub instance_id: String,
    pub tenant_id: String,
    pub region: Region,
    pub status: InstanceStatus,
    pub database_path: String,
    pub allocated_memory_bytes: u64,
    pub allocated_storage_bytes: u64,
    pub created_at: u64,
    pub last_heartbeat: u64,
}

/// Status of a managed instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InstanceStatus {
    Provisioning,
    Running,
    Stopping,
    Stopped,
    Failed,
    Migrating,
}

/// Configuration for the service orchestrator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestratorConfig {
    pub default_region: Region,
    pub heartbeat_timeout_seconds: u64,
    pub max_instances_per_tenant: usize,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            default_region: Region::UsEast1,
            heartbeat_timeout_seconds: 60,
            max_instances_per_tenant: 5,
        }
    }
}

/// Orchestrates provisioning, scaling, and lifecycle of managed Needle instances.
pub struct ServiceOrchestrator {
    config: OrchestratorConfig,
    instances: RwLock<HashMap<String, ManagedInstance>>,
    instance_counter: AtomicU64,
}

impl ServiceOrchestrator {
    pub fn new(config: OrchestratorConfig) -> Self {
        Self {
            config,
            instances: RwLock::new(HashMap::new()),
            instance_counter: AtomicU64::new(0),
        }
    }

    /// Provision a new managed instance for a tenant in a given region.
    pub fn provision(
        &self,
        tenant_id: &str,
        region: Option<Region>,
        memory_bytes: u64,
        storage_bytes: u64,
    ) -> Result<ManagedInstance> {
        let tenant_instances: Vec<_> = self
            .instances
            .read()
            .values()
            .filter(|i| i.tenant_id == tenant_id && i.status != InstanceStatus::Stopped)
            .cloned()
            .collect();

        if tenant_instances.len() >= self.config.max_instances_per_tenant {
            return Err(NeedleError::InvalidOperation(format!(
                "Tenant {} already has {} instances (max {})",
                tenant_id,
                tenant_instances.len(),
                self.config.max_instances_per_tenant
            )));
        }

        let seq = self.instance_counter.fetch_add(1, Ordering::Relaxed);
        let region = region.unwrap_or_else(|| self.config.default_region.clone());
        let instance_id = format!("inst_{:012x}", seq);
        let database_path = format!("/data/{}/{}.needle", tenant_id, instance_id);

        let instance = ManagedInstance {
            instance_id: instance_id.clone(),
            tenant_id: tenant_id.to_string(),
            region,
            status: InstanceStatus::Provisioning,
            database_path,
            allocated_memory_bytes: memory_bytes,
            allocated_storage_bytes: storage_bytes,
            created_at: current_timestamp(),
            last_heartbeat: current_timestamp(),
        };

        self.instances
            .write()
            .insert(instance_id.clone(), instance.clone());
        Ok(instance)
    }

    /// Mark an instance as running after provisioning completes.
    pub fn mark_running(&self, instance_id: &str) -> Result<()> {
        let mut instances = self.instances.write();
        let inst = instances.get_mut(instance_id).ok_or_else(|| {
            NeedleError::NotFound(format!("Instance {} not found", instance_id))
        })?;
        inst.status = InstanceStatus::Running;
        inst.last_heartbeat = current_timestamp();
        Ok(())
    }

    /// Record a heartbeat from a running instance.
    pub fn heartbeat(&self, instance_id: &str) -> Result<()> {
        let mut instances = self.instances.write();
        let inst = instances.get_mut(instance_id).ok_or_else(|| {
            NeedleError::NotFound(format!("Instance {} not found", instance_id))
        })?;
        inst.last_heartbeat = current_timestamp();
        Ok(())
    }

    /// Stop an instance.
    pub fn stop(&self, instance_id: &str) -> Result<()> {
        let mut instances = self.instances.write();
        let inst = instances.get_mut(instance_id).ok_or_else(|| {
            NeedleError::NotFound(format!("Instance {} not found", instance_id))
        })?;
        inst.status = InstanceStatus::Stopped;
        Ok(())
    }

    /// Detect instances that have missed heartbeats and mark them failed.
    pub fn detect_failures(&self) -> Vec<String> {
        let now = current_timestamp();
        let timeout = self.config.heartbeat_timeout_seconds;
        let mut failed = Vec::new();
        let mut instances = self.instances.write();
        for inst in instances.values_mut() {
            if inst.status == InstanceStatus::Running
                && now.saturating_sub(inst.last_heartbeat) > timeout
            {
                inst.status = InstanceStatus::Failed;
                failed.push(inst.instance_id.clone());
            }
        }
        failed
    }

    /// List all instances for a tenant.
    pub fn list_instances(&self, tenant_id: &str) -> Vec<ManagedInstance> {
        self.instances
            .read()
            .values()
            .filter(|i| i.tenant_id == tenant_id)
            .cloned()
            .collect()
    }

    /// Get a single instance by ID.
    pub fn get_instance(&self, instance_id: &str) -> Option<ManagedInstance> {
        self.instances.read().get(instance_id).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_tenant() {
        let cp = ControlPlane::default();

        let config = TenantConfig {
            name: "acme".to_string(),
            tier: ResourceTier::Developer,
            email: "admin@acme.com".to_string(),
            ..Default::default()
        };

        let tenant = cp.create_tenant(config).unwrap();
        assert!(tenant.id.starts_with("tenant_"));
        assert_eq!(tenant.config.name, "acme");
        assert_eq!(tenant.status, TenantStatus::Provisioning);
    }

    #[test]
    fn test_api_key_creation() {
        let cp = ControlPlane::default();

        let tenant = cp
            .create_tenant(TenantConfig {
                name: "test".to_string(),
                ..Default::default()
            })
            .unwrap();

        let (raw_key, api_key) = cp
            .create_api_key(&tenant.id, "prod-key", vec![Permission::Read, Permission::Write])
            .unwrap();

        assert!(raw_key.starts_with("ndk_"));
        assert_eq!(api_key.name, "prod-key");
        assert_eq!(api_key.permissions.len(), 2);
    }

    #[test]
    fn test_tier_limits() {
        let free = ResourceTier::Free.limits();
        let pro = ResourceTier::Professional.limits();

        assert!(pro.max_vectors > free.max_vectors);
        assert!(pro.max_collections > free.max_collections);
    }

    #[test]
    fn test_usage_tracking() {
        let cp = ControlPlane::default();

        let tenant = cp
            .create_tenant(TenantConfig {
                name: "usage-test".to_string(),
                ..Default::default()
            })
            .unwrap();

        cp.record_usage(&tenant.id, UsageEventType::Insert, 100);
        cp.record_usage(&tenant.id, UsageEventType::Query, 10);

        let updated = cp.get_tenant(&tenant.id).unwrap();
        assert_eq!(updated.usage.vectors_count, 100);
        assert_eq!(updated.usage.total_queries, 10);
    }

    #[test]
    fn test_limit_check() {
        let cp = ControlPlane::default();

        let tenant = cp
            .create_tenant(TenantConfig {
                name: "limit-test".to_string(),
                tier: ResourceTier::Free,
                ..Default::default()
            })
            .unwrap();

        // Insert more than limit
        cp.record_usage(&tenant.id, UsageEventType::Insert, 20000); // Free limit is 10000

        let result = cp.check_limits(&tenant.id).unwrap();
        assert!(!result.within_limits);
        assert!(!result.violations.is_empty());
    }

    #[test]
    fn test_billing_generation() {
        let cp = ControlPlane::default();

        let tenant = cp
            .create_tenant(TenantConfig {
                name: "billing-test".to_string(),
                tier: ResourceTier::Developer,
                ..Default::default()
            })
            .unwrap();

        let billing = cp.generate_billing(&tenant.id).unwrap();
        assert_eq!(billing.base_charge_cents, 2900); // $29
        assert!(billing.total_cents >= billing.base_charge_cents);
    }

    #[test]
    fn test_region_router_latency_based() {
        let router = RegionRouter::new(Region::UsEast1, RoutingStrategy::LatencyBased);
        router.add_endpoint(RegionalEndpoint {
            region: Region::UsEast1,
            endpoint_url: "https://us-east-1.needle.io".into(),
            health: RegionHealth::Healthy,
            latency_ms: 50.0,
            capacity_pct: 60.0,
            last_health_check: 0,
        });
        router.add_endpoint(RegionalEndpoint {
            region: Region::EuWest1,
            endpoint_url: "https://eu-west-1.needle.io".into(),
            health: RegionHealth::Healthy,
            latency_ms: 20.0,
            capacity_pct: 30.0,
            last_health_check: 0,
        });
        let best = router.route().unwrap();
        assert_eq!(best.region, Region::EuWest1); // lower latency
    }

    #[test]
    fn test_region_router_failover() {
        let router = RegionRouter::new(Region::UsEast1, RoutingStrategy::PrimaryWithFailover);
        router.add_endpoint(RegionalEndpoint {
            region: Region::UsEast1,
            endpoint_url: "https://us-east-1.needle.io".into(),
            health: RegionHealth::Unhealthy,
            latency_ms: 100.0,
            capacity_pct: 90.0,
            last_health_check: 0,
        });
        router.add_endpoint(RegionalEndpoint {
            region: Region::UsWest2,
            endpoint_url: "https://us-west-2.needle.io".into(),
            health: RegionHealth::Healthy,
            latency_ms: 30.0,
            capacity_pct: 40.0,
            last_health_check: 0,
        });
        let chosen = router.route().unwrap();
        assert_eq!(chosen.region, Region::UsWest2); // failover
    }

    #[test]
    fn test_sla_monitor_compliant() {
        let policy = SlaPolicy {
            target_availability_pct: 99.0,
            max_p99_latency_ms: 100.0,
            ..Default::default()
        };
        let monitor = SlaMonitor::new(policy, 3600);

        for _ in 0..100 {
            monitor.record_request(true, 10.0);
        }
        let report = monitor.report();
        assert!(report.compliant);
        assert_eq!(report.total_requests, 100);
        assert_eq!(report.failed_requests, 0);
    }

    #[test]
    fn test_sla_monitor_breach() {
        let policy = SlaPolicy {
            target_availability_pct: 99.0,
            max_p99_latency_ms: 50.0,
            ..Default::default()
        };
        let monitor = SlaMonitor::new(policy, 3600);

        for _ in 0..90 {
            monitor.record_request(true, 10.0);
        }
        for _ in 0..10 {
            monitor.record_request(false, 200.0);
        }
        let report = monitor.report();
        assert!(!report.compliant);
        assert!(!report.breaches.is_empty());
    }

    #[test]
    fn test_service_orchestrator_provision() {
        let orch = ServiceOrchestrator::new(OrchestratorConfig::default());
        let inst = orch
            .provision("tenant_001", Some(Region::UsEast1), 1024 * 1024 * 512, 1024 * 1024 * 1024)
            .unwrap();
        assert_eq!(inst.status, InstanceStatus::Provisioning);
        assert_eq!(inst.tenant_id, "tenant_001");

        orch.mark_running(&inst.instance_id).unwrap();
        let updated = orch.get_instance(&inst.instance_id).unwrap();
        assert_eq!(updated.status, InstanceStatus::Running);
    }

    #[test]
    fn test_service_orchestrator_max_instances() {
        let config = OrchestratorConfig {
            max_instances_per_tenant: 2,
            ..Default::default()
        };
        let orch = ServiceOrchestrator::new(config);
        orch.provision("t1", None, 1024, 1024).unwrap();
        orch.provision("t1", None, 1024, 1024).unwrap();
        assert!(orch.provision("t1", None, 1024, 1024).is_err());
    }
}
