//! Managed Cloud Service (Needle Cloud)
//!
//! Serverless, pay-per-query hosted Needle with auto-scaling, tiered pricing,
//! and multi-region support. Builds on cloud_control.rs and serverless_runtime.rs.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::services::cloud_service::{
//!     CloudService, CloudConfig, CreateInstanceRequest, CloudTier,
//! };
//!
//! let mut cloud = CloudService::new(CloudConfig::default());
//!
//! // Create an instance for a tenant
//! let instance = cloud.create_instance(CreateInstanceRequest {
//!     tenant_id: "acme".into(),
//!     tier: CloudTier::Pro,
//!     region: "us-east-1".into(),
//!     display_name: "Production".into(),
//! }).unwrap();
//! println!("Endpoint: {}", instance.endpoint);
//! ```

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::error::{NeedleError, Result};

// ── Scaffold imports (experimental feature) ─────────────────────────────────

#[cfg(feature = "experimental")]
#[allow(unused_imports)]
use crate::experimental::cloud_control::{
    InstanceStatus as CoreInstanceStatus, ResourceTier as CoreResourceTier,
    TenantStatus as CoreTenantStatus, TierLimits as CoreTierLimits,
};

#[cfg(feature = "experimental")]
#[allow(unused_imports)]
use crate::experimental::serverless_runtime::EdgePlatform;

// ── Cloud Tiers ─────────────────────────────────────────────────────────────

/// Cloud service tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CloudTier {
    /// Free tier: 100K vectors, 10K queries/month.
    Free,
    /// Pro tier: 10M vectors, auto-scaling.
    Pro,
    /// Enterprise: unlimited, dedicated resources, SLA.
    Enterprise,
}

impl CloudTier {
    /// Convert to the scaffold `ResourceTier` from `cloud_control`.
    #[cfg(feature = "experimental")]
    pub fn to_core_tier(self) -> CoreResourceTier {
        match self {
            CloudTier::Free => CoreResourceTier::Free,
            CloudTier::Pro => CoreResourceTier::Professional,
            CloudTier::Enterprise => CoreResourceTier::Enterprise,
        }
    }

    /// Build a `CloudTier` from the scaffold `ResourceTier`.
    #[cfg(feature = "experimental")]
    pub fn from_core_tier(core: CoreResourceTier) -> Self {
        match core {
            CoreResourceTier::Free | CoreResourceTier::Developer => CloudTier::Free,
            CoreResourceTier::Professional => CloudTier::Pro,
            CoreResourceTier::Enterprise => CloudTier::Enterprise,
        }
    }
}

/// Tier limits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TierLimits {
    pub max_vectors: u64,
    pub max_collections: usize,
    pub max_dimensions: usize,
    pub monthly_queries: u64,
    pub storage_gb: f64,
    pub max_qps: u32,
    pub sla_uptime: f64,
}

impl TierLimits {
    pub fn for_tier(tier: CloudTier) -> Self {
        match tier {
            CloudTier::Free => Self {
                max_vectors: 100_000,
                max_collections: 3,
                max_dimensions: 768,
                monthly_queries: 10_000,
                storage_gb: 0.5,
                max_qps: 5,
                sla_uptime: 0.0,
            },
            CloudTier::Pro => Self {
                max_vectors: 10_000_000,
                max_collections: 50,
                max_dimensions: 4096,
                monthly_queries: 1_000_000,
                storage_gb: 50.0,
                max_qps: 500,
                sla_uptime: 99.9,
            },
            CloudTier::Enterprise => Self {
                max_vectors: u64::MAX,
                max_collections: 500,
                max_dimensions: 8192,
                monthly_queries: u64::MAX,
                storage_gb: 1000.0,
                max_qps: 10_000,
                sla_uptime: 99.99,
            },
        }
    }
}

// ── Pricing ─────────────────────────────────────────────────────────────────

/// Pricing model per tier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TierPricing {
    /// Monthly base price (USD).
    pub base_monthly: f64,
    /// Cost per 1K queries (USD).
    pub per_1k_queries: f64,
    /// Cost per GB storage (USD/month).
    pub per_gb_storage: f64,
    /// Cost per 1K inserts (USD).
    pub per_1k_inserts: f64,
}

impl TierPricing {
    pub fn for_tier(tier: CloudTier) -> Self {
        match tier {
            CloudTier::Free => Self {
                base_monthly: 0.0,
                per_1k_queries: 0.0,
                per_gb_storage: 0.0,
                per_1k_inserts: 0.0,
            },
            CloudTier::Pro => Self {
                base_monthly: 49.0,
                per_1k_queries: 0.10,
                per_gb_storage: 0.25,
                per_1k_inserts: 0.50,
            },
            CloudTier::Enterprise => Self {
                base_monthly: 499.0,
                per_1k_queries: 0.05,
                per_gb_storage: 0.15,
                per_1k_inserts: 0.25,
            },
        }
    }
}

// ── Instances ───────────────────────────────────────────────────────────────

/// Instance status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InstanceStatus {
    Provisioning,
    Running,
    Scaling,
    Suspended,
    Terminated,
}

impl InstanceStatus {
    /// Convert to the scaffold `InstanceStatus` from `cloud_control`.
    #[cfg(feature = "experimental")]
    pub fn to_core_status(self) -> CoreInstanceStatus {
        match self {
            InstanceStatus::Provisioning => CoreInstanceStatus::Provisioning,
            InstanceStatus::Running => CoreInstanceStatus::Running,
            InstanceStatus::Scaling => CoreInstanceStatus::Running,
            InstanceStatus::Suspended => CoreInstanceStatus::Stopped,
            InstanceStatus::Terminated => CoreInstanceStatus::Stopped,
        }
    }

    /// Build an `InstanceStatus` from the scaffold `InstanceStatus`.
    #[cfg(feature = "experimental")]
    pub fn from_core_status(core: CoreInstanceStatus) -> Self {
        match core {
            CoreInstanceStatus::Provisioning => InstanceStatus::Provisioning,
            CoreInstanceStatus::Running => InstanceStatus::Running,
            CoreInstanceStatus::Stopping => InstanceStatus::Suspended,
            CoreInstanceStatus::Stopped => InstanceStatus::Terminated,
            CoreInstanceStatus::Failed => InstanceStatus::Terminated,
            CoreInstanceStatus::Migrating => InstanceStatus::Scaling,
        }
    }
}

/// A cloud instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudInstance {
    pub instance_id: String,
    pub tenant_id: String,
    pub tier: CloudTier,
    pub region: String,
    pub display_name: String,
    pub endpoint: String,
    pub api_key: String,
    pub status: InstanceStatus,
    pub limits: TierLimits,
    pub created_at: u64,
    pub current_vectors: u64,
    pub current_collections: usize,
}

/// Request to create a cloud instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateInstanceRequest {
    pub tenant_id: String,
    pub tier: CloudTier,
    pub region: String,
    pub display_name: String,
}

/// Auto-scaling configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoScaleConfig {
    pub enabled: bool,
    pub min_replicas: u32,
    pub max_replicas: u32,
    pub target_qps_per_replica: u32,
    pub scale_up_threshold: f64,
    pub scale_down_threshold: f64,
    pub cooldown_secs: u64,
}

impl Default for AutoScaleConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_replicas: 1,
            max_replicas: 10,
            target_qps_per_replica: 100,
            scale_up_threshold: 0.8,
            scale_down_threshold: 0.2,
            cooldown_secs: 300,
        }
    }
}

/// Usage metrics for an instance.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InstanceMetrics {
    pub total_queries: u64,
    pub total_inserts: u64,
    pub total_deletes: u64,
    pub storage_bytes: u64,
    pub current_qps: f64,
    pub p99_latency_ms: f64,
    pub current_replicas: u32,
}

/// Invoice for billing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Invoice {
    pub tenant_id: String,
    pub period: String,
    pub base_charge: f64,
    pub query_charge: f64,
    pub storage_charge: f64,
    pub insert_charge: f64,
    pub total: f64,
    pub tier: CloudTier,
}

// ── Cloud Service Config ────────────────────────────────────────────────────

/// Cloud service configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudConfig {
    pub supported_regions: Vec<String>,
    pub max_instances_per_tenant: usize,
    pub auto_scale: AutoScaleConfig,
    pub default_tier: CloudTier,
}

impl Default for CloudConfig {
    fn default() -> Self {
        Self {
            supported_regions: vec![
                "us-east-1".into(),
                "us-west-2".into(),
                "eu-west-1".into(),
                "ap-southeast-1".into(),
            ],
            max_instances_per_tenant: 10,
            auto_scale: AutoScaleConfig::default(),
            default_tier: CloudTier::Free,
        }
    }
}

// ── Cloud Service ───────────────────────────────────────────────────────────

/// Managed cloud service.
pub struct CloudService {
    config: CloudConfig,
    instances: HashMap<String, CloudInstance>,
    metrics: HashMap<String, InstanceMetrics>,
    api_keys: HashMap<String, String>,
    next_id: u64,
}

impl CloudService {
    pub fn new(config: CloudConfig) -> Self {
        Self {
            config,
            instances: HashMap::new(),
            metrics: HashMap::new(),
            api_keys: HashMap::new(),
            next_id: 1,
        }
    }

    /// Create a new cloud instance.
    pub fn create_instance(&mut self, req: CreateInstanceRequest) -> Result<CloudInstance> {
        if !self.config.supported_regions.contains(&req.region) {
            return Err(NeedleError::InvalidArgument(format!(
                "Unsupported region: {}. Supported: {:?}",
                req.region, self.config.supported_regions
            )));
        }

        let tenant_count = self
            .instances
            .values()
            .filter(|i| i.tenant_id == req.tenant_id && i.status != InstanceStatus::Terminated)
            .count();
        if tenant_count >= self.config.max_instances_per_tenant {
            return Err(NeedleError::QuotaExceeded(format!(
                "Maximum instances ({}) per tenant reached",
                self.config.max_instances_per_tenant
            )));
        }

        let id = format!("inst-{:06}", self.next_id);
        self.next_id += 1;
        let api_key = format!("nk_live_{id}_{}", now_secs());
        self.api_keys.insert(api_key.clone(), id.clone());

        let instance = CloudInstance {
            instance_id: id.clone(),
            tenant_id: req.tenant_id,
            tier: req.tier,
            region: req.region.clone(),
            display_name: req.display_name,
            endpoint: format!("https://{id}.{}.needle.cloud", req.region),
            api_key: api_key.clone(),
            status: InstanceStatus::Running,
            limits: TierLimits::for_tier(req.tier),
            created_at: now_secs(),
            current_vectors: 0,
            current_collections: 0,
        };

        self.instances.insert(id.clone(), instance.clone());
        self.metrics.insert(id, InstanceMetrics::default());
        Ok(instance)
    }

    /// Get instance by ID.
    pub fn instance(&self, instance_id: &str) -> Option<&CloudInstance> {
        self.instances.get(instance_id)
    }

    /// List instances for a tenant.
    pub fn list_instances(&self, tenant_id: &str) -> Vec<&CloudInstance> {
        self.instances
            .values()
            .filter(|i| i.tenant_id == tenant_id)
            .collect()
    }

    /// Suspend an instance.
    pub fn suspend_instance(&mut self, instance_id: &str) -> Result<()> {
        let inst = self.instances.get_mut(instance_id).ok_or_else(|| {
            NeedleError::NotFound(format!("Instance '{instance_id}'"))
        })?;
        if inst.status != InstanceStatus::Running {
            return Err(NeedleError::InvalidOperation(format!(
                "Instance '{instance_id}' is not running (status: {:?})",
                inst.status
            )));
        }
        inst.status = InstanceStatus::Suspended;
        Ok(())
    }

    /// Resume a suspended instance.
    pub fn resume_instance(&mut self, instance_id: &str) -> Result<()> {
        let inst = self.instances.get_mut(instance_id).ok_or_else(|| {
            NeedleError::NotFound(format!("Instance '{instance_id}'"))
        })?;
        if inst.status != InstanceStatus::Suspended {
            return Err(NeedleError::InvalidOperation(format!(
                "Instance '{instance_id}' is not suspended"
            )));
        }
        inst.status = InstanceStatus::Running;
        Ok(())
    }

    /// Terminate an instance.
    pub fn terminate_instance(&mut self, instance_id: &str) -> Result<()> {
        let inst = self.instances.get_mut(instance_id).ok_or_else(|| {
            NeedleError::NotFound(format!("Instance '{instance_id}'"))
        })?;
        inst.status = InstanceStatus::Terminated;
        Ok(())
    }

    /// Upgrade/downgrade instance tier.
    pub fn change_tier(&mut self, instance_id: &str, new_tier: CloudTier) -> Result<()> {
        let inst = self.instances.get_mut(instance_id).ok_or_else(|| {
            NeedleError::NotFound(format!("Instance '{instance_id}'"))
        })?;
        inst.tier = new_tier;
        inst.limits = TierLimits::for_tier(new_tier);
        Ok(())
    }

    /// Record a usage event.
    pub fn record_usage(&mut self, instance_id: &str, queries: u64, inserts: u64, storage_delta: i64) {
        let m = self.metrics.entry(instance_id.into()).or_default();
        m.total_queries += queries;
        m.total_inserts += inserts;
        if storage_delta >= 0 {
            m.storage_bytes += storage_delta as u64;
        } else {
            m.storage_bytes = m.storage_bytes.saturating_sub((-storage_delta) as u64);
        }
    }

    /// Get metrics for an instance.
    pub fn metrics(&self, instance_id: &str) -> Option<&InstanceMetrics> {
        self.metrics.get(instance_id)
    }

    /// Check if usage exceeds tier limits.
    pub fn check_limits(&self, instance_id: &str) -> Result<Vec<String>> {
        let inst = self.instances.get(instance_id).ok_or_else(|| {
            NeedleError::NotFound(format!("Instance '{instance_id}'"))
        })?;
        let metrics = self.metrics.get(instance_id);
        let mut warnings = Vec::new();

        if let Some(m) = metrics {
            if m.total_queries > inst.limits.monthly_queries {
                warnings.push(format!(
                    "Monthly query limit exceeded: {}/{}",
                    m.total_queries, inst.limits.monthly_queries
                ));
            }
            let storage_gb = m.storage_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
            if storage_gb > inst.limits.storage_gb {
                warnings.push(format!(
                    "Storage limit exceeded: {:.2}GB/{:.2}GB",
                    storage_gb, inst.limits.storage_gb
                ));
            }
        }
        if inst.current_vectors > inst.limits.max_vectors {
            warnings.push(format!(
                "Vector limit exceeded: {}/{}",
                inst.current_vectors, inst.limits.max_vectors
            ));
        }

        Ok(warnings)
    }

    /// Generate an invoice for a tenant.
    pub fn generate_invoice(&self, tenant_id: &str, period: &str) -> Result<Vec<Invoice>> {
        let instances: Vec<_> = self
            .instances
            .values()
            .filter(|i| i.tenant_id == tenant_id)
            .collect();
        if instances.is_empty() {
            return Err(NeedleError::NotFound(format!("Tenant '{tenant_id}'")));
        }

        let mut invoices = Vec::new();
        for inst in instances {
            let pricing = TierPricing::for_tier(inst.tier);
            let metrics = self.metrics.get(&inst.instance_id);
            let (queries, inserts, storage_bytes) = metrics
                .map_or((0, 0, 0), |m| (m.total_queries, m.total_inserts, m.storage_bytes));

            let query_charge = (queries as f64 / 1000.0) * pricing.per_1k_queries;
            let storage_gb = storage_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
            let storage_charge = storage_gb * pricing.per_gb_storage;
            let insert_charge = (inserts as f64 / 1000.0) * pricing.per_1k_inserts;

            invoices.push(Invoice {
                tenant_id: tenant_id.into(),
                period: period.into(),
                base_charge: pricing.base_monthly,
                query_charge,
                storage_charge,
                insert_charge,
                total: pricing.base_monthly + query_charge + storage_charge + insert_charge,
                tier: inst.tier,
            });
        }
        Ok(invoices)
    }

    /// Evaluate auto-scaling needs and return scaling decisions.
    pub fn evaluate_auto_scale(&self, instance_id: &str) -> Result<ScaleDecision> {
        let inst = self.instances.get(instance_id).ok_or_else(|| {
            NeedleError::NotFound(format!("Instance '{instance_id}'"))
        })?;
        if inst.tier == CloudTier::Free {
            return Ok(ScaleDecision::NoChange);
        }
        let metrics = self.metrics.get(instance_id);
        let current_qps = metrics.map_or(0.0, |m| m.current_qps);
        let current_replicas = metrics.map_or(1, |m| m.current_replicas);
        let target_qps = self.config.auto_scale.target_qps_per_replica as f64;
        let load = current_qps / (current_replicas as f64 * target_qps).max(1.0);

        if load > self.config.auto_scale.scale_up_threshold
            && current_replicas < self.config.auto_scale.max_replicas
        {
            Ok(ScaleDecision::ScaleUp {
                from: current_replicas,
                to: (current_replicas + 1).min(self.config.auto_scale.max_replicas),
            })
        } else if load < self.config.auto_scale.scale_down_threshold
            && current_replicas > self.config.auto_scale.min_replicas
        {
            Ok(ScaleDecision::ScaleDown {
                from: current_replicas,
                to: (current_replicas - 1).max(self.config.auto_scale.min_replicas),
            })
        } else {
            Ok(ScaleDecision::NoChange)
        }
    }

    /// Resolve an API key to an instance.
    pub fn resolve_api_key(&self, api_key: &str) -> Option<&CloudInstance> {
        let instance_id = self.api_keys.get(api_key)?;
        self.instances.get(instance_id)
    }

    /// Rotate API key for an instance.
    pub fn rotate_api_key(&mut self, instance_id: &str) -> Result<String> {
        let inst = self.instances.get_mut(instance_id).ok_or_else(|| {
            NeedleError::NotFound(format!("Instance '{instance_id}'"))
        })?;
        // Remove old key
        self.api_keys.retain(|_, v| v != instance_id);
        // Generate new key
        let new_key = format!("nk_live_{instance_id}_{}", now_secs());
        self.api_keys.insert(new_key.clone(), instance_id.to_string());
        inst.api_key = new_key.clone();
        Ok(new_key)
    }

    /// Get total instance count.
    pub fn total_instances(&self) -> usize {
        self.instances.len()
    }

    /// Get config.
    pub fn config(&self) -> &CloudConfig {
        &self.config
    }
}

/// Auto-scale decision.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScaleDecision {
    NoChange,
    ScaleUp { from: u32, to: u32 },
    ScaleDown { from: u32, to: u32 },
}

impl Default for CloudService {
    fn default() -> Self {
        Self::new(CloudConfig::default())
    }
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_service() -> CloudService {
        CloudService::new(CloudConfig::default())
    }

    fn create_req(tier: CloudTier) -> CreateInstanceRequest {
        CreateInstanceRequest {
            tenant_id: "tenant-1".into(),
            tier,
            region: "us-east-1".into(),
            display_name: "Test".into(),
        }
    }

    #[test]
    fn test_create_instance() {
        let mut svc = make_service();
        let inst = svc.create_instance(create_req(CloudTier::Free)).unwrap();
        assert_eq!(inst.status, InstanceStatus::Running);
        assert!(inst.endpoint.contains("needle.cloud"));
        assert!(inst.api_key.starts_with("nk_live_"));
    }

    #[test]
    fn test_unsupported_region() {
        let mut svc = make_service();
        let req = CreateInstanceRequest {
            region: "invalid-region".into(),
            ..create_req(CloudTier::Free)
        };
        assert!(svc.create_instance(req).is_err());
    }

    #[test]
    fn test_tier_limits() {
        let free = TierLimits::for_tier(CloudTier::Free);
        let pro = TierLimits::for_tier(CloudTier::Pro);
        assert!(pro.max_vectors > free.max_vectors);
        assert!(pro.max_qps > free.max_qps);
    }

    #[test]
    fn test_suspend_resume() {
        let mut svc = make_service();
        let inst = svc.create_instance(create_req(CloudTier::Pro)).unwrap();
        let id = inst.instance_id.clone();

        svc.suspend_instance(&id).unwrap();
        assert_eq!(svc.instance(&id).unwrap().status, InstanceStatus::Suspended);

        svc.resume_instance(&id).unwrap();
        assert_eq!(svc.instance(&id).unwrap().status, InstanceStatus::Running);
    }

    #[test]
    fn test_terminate() {
        let mut svc = make_service();
        let inst = svc.create_instance(create_req(CloudTier::Free)).unwrap();
        svc.terminate_instance(&inst.instance_id).unwrap();
        assert_eq!(
            svc.instance(&inst.instance_id).unwrap().status,
            InstanceStatus::Terminated
        );
    }

    #[test]
    fn test_change_tier() {
        let mut svc = make_service();
        let inst = svc.create_instance(create_req(CloudTier::Free)).unwrap();
        svc.change_tier(&inst.instance_id, CloudTier::Pro).unwrap();
        let updated = svc.instance(&inst.instance_id).unwrap();
        assert_eq!(updated.tier, CloudTier::Pro);
        assert_eq!(updated.limits.max_vectors, 10_000_000);
    }

    #[test]
    fn test_usage_recording() {
        let mut svc = make_service();
        let inst = svc.create_instance(create_req(CloudTier::Pro)).unwrap();
        svc.record_usage(&inst.instance_id, 100, 50, 1024);
        let m = svc.metrics(&inst.instance_id).unwrap();
        assert_eq!(m.total_queries, 100);
        assert_eq!(m.total_inserts, 50);
        assert_eq!(m.storage_bytes, 1024);
    }

    #[test]
    fn test_generate_invoice() {
        let mut svc = make_service();
        let inst = svc.create_instance(create_req(CloudTier::Pro)).unwrap();
        svc.record_usage(&inst.instance_id, 10_000, 5_000, 1_073_741_824);

        let invoices = svc.generate_invoice("tenant-1", "2026-01").unwrap();
        assert_eq!(invoices.len(), 1);
        assert!(invoices[0].total > 49.0); // base + usage
    }

    #[test]
    fn test_free_tier_invoice() {
        let mut svc = make_service();
        svc.create_instance(create_req(CloudTier::Free)).unwrap();
        let invoices = svc.generate_invoice("tenant-1", "2026-01").unwrap();
        assert_eq!(invoices[0].total, 0.0);
    }

    #[test]
    fn test_api_key_resolution() {
        let mut svc = make_service();
        let inst = svc.create_instance(create_req(CloudTier::Pro)).unwrap();
        let resolved = svc.resolve_api_key(&inst.api_key).unwrap();
        assert_eq!(resolved.instance_id, inst.instance_id);
    }

    #[test]
    fn test_api_key_rotation() {
        let mut svc = make_service();
        let inst = svc.create_instance(create_req(CloudTier::Pro)).unwrap();
        let old_key = inst.api_key.clone();
        let new_key = svc.rotate_api_key(&inst.instance_id).unwrap();
        assert_ne!(old_key, new_key);
        assert!(svc.resolve_api_key(&old_key).is_none());
        assert!(svc.resolve_api_key(&new_key).is_some());
    }

    #[test]
    fn test_auto_scale_free_tier() {
        let svc = make_service();
        // Free tier doesn't scale
        let mut svc2 = make_service();
        let inst = svc2.create_instance(create_req(CloudTier::Free)).unwrap();
        assert_eq!(
            svc2.evaluate_auto_scale(&inst.instance_id).unwrap(),
            ScaleDecision::NoChange
        );
        let _ = svc;
    }

    #[test]
    fn test_instance_limit_per_tenant() {
        let mut svc = CloudService::new(CloudConfig {
            max_instances_per_tenant: 2,
            ..Default::default()
        });
        svc.create_instance(create_req(CloudTier::Free)).unwrap();
        svc.create_instance(create_req(CloudTier::Free)).unwrap();
        assert!(svc.create_instance(create_req(CloudTier::Free)).is_err());
    }

    #[test]
    fn test_list_instances() {
        let mut svc = make_service();
        svc.create_instance(create_req(CloudTier::Free)).unwrap();
        svc.create_instance(CreateInstanceRequest {
            tenant_id: "tenant-2".into(),
            ..create_req(CloudTier::Pro)
        })
        .unwrap();
        assert_eq!(svc.list_instances("tenant-1").len(), 1);
        assert_eq!(svc.list_instances("tenant-2").len(), 1);
    }

    #[test]
    fn test_check_limits() {
        let mut svc = make_service();
        let inst = svc.create_instance(create_req(CloudTier::Free)).unwrap();
        svc.record_usage(&inst.instance_id, 20_000, 0, 0);
        let warnings = svc.check_limits(&inst.instance_id).unwrap();
        assert!(!warnings.is_empty());
    }

    #[test]
    fn test_pricing() {
        let free = TierPricing::for_tier(CloudTier::Free);
        let pro = TierPricing::for_tier(CloudTier::Pro);
        assert_eq!(free.base_monthly, 0.0);
        assert!(pro.base_monthly > 0.0);
    }

    // ── Scaffold integration tests (experimental feature only) ──────────

    #[cfg(feature = "experimental")]
    mod scaffold_integration {
        use super::*;

        #[test]
        fn test_cloud_tier_to_core_roundtrip() {
            assert_eq!(
                CloudTier::from_core_tier(CloudTier::Free.to_core_tier()),
                CloudTier::Free
            );
            assert_eq!(
                CloudTier::from_core_tier(CloudTier::Pro.to_core_tier()),
                CloudTier::Pro
            );
            assert_eq!(
                CloudTier::from_core_tier(CloudTier::Enterprise.to_core_tier()),
                CloudTier::Enterprise
            );
        }

        #[test]
        fn test_developer_tier_maps_to_free() {
            assert_eq!(
                CloudTier::from_core_tier(CoreResourceTier::Developer),
                CloudTier::Free
            );
        }

        #[test]
        fn test_instance_status_to_core_roundtrip() {
            assert_eq!(
                InstanceStatus::from_core_status(InstanceStatus::Provisioning.to_core_status()),
                InstanceStatus::Provisioning
            );
            assert_eq!(
                InstanceStatus::from_core_status(InstanceStatus::Running.to_core_status()),
                InstanceStatus::Running
            );
        }

        #[test]
        fn test_instance_status_from_core_failed() {
            assert_eq!(
                InstanceStatus::from_core_status(CoreInstanceStatus::Failed),
                InstanceStatus::Terminated
            );
        }

        #[test]
        fn test_instance_status_from_core_migrating() {
            assert_eq!(
                InstanceStatus::from_core_status(CoreInstanceStatus::Migrating),
                InstanceStatus::Scaling
            );
        }
    }
}
