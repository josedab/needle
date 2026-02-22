#![allow(clippy::unwrap_used)]
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

    /// Suspend a tenant (e.g., for non-payment).
    pub fn suspend(&mut self, tenant_id: &str) -> Result<()> {
        let t = self.tenants.get_mut(tenant_id)
            .ok_or_else(|| NeedleError::NotFound(format!("Tenant '{tenant_id}'")))?;
        if t.status != TenantStatus::Active {
            return Err(NeedleError::InvalidArgument(format!(
                "Tenant '{tenant_id}' is not active (status: {:?})", t.status
            )));
        }
        t.status = TenantStatus::Suspended;
        Ok(())
    }

    /// Reactivate a suspended tenant.
    pub fn reactivate(&mut self, tenant_id: &str) -> Result<()> {
        let t = self.tenants.get_mut(tenant_id)
            .ok_or_else(|| NeedleError::NotFound(format!("Tenant '{tenant_id}'")))?;
        if t.status != TenantStatus::Suspended {
            return Err(NeedleError::InvalidArgument(format!(
                "Tenant '{tenant_id}' is not suspended (status: {:?})", t.status
            )));
        }
        t.status = TenantStatus::Active;
        Ok(())
    }

    /// Upgrade or downgrade a tenant's plan.
    pub fn change_plan(&mut self, tenant_id: &str, new_plan: Plan) -> Result<()> {
        let t = self.tenants.get_mut(tenant_id)
            .ok_or_else(|| NeedleError::NotFound(format!("Tenant '{tenant_id}'")))?;
        t.plan = new_plan;
        t.config = TenantConfig::for_plan(new_plan);
        Ok(())
    }

    /// Estimate the monthly billing for a tenant based on usage metrics.
    pub fn estimate_billing(&self, tenant_id: &str) -> Result<BillingEstimate> {
        let tenant = self.tenants.get(tenant_id)
            .ok_or_else(|| NeedleError::NotFound(format!("Tenant '{tenant_id}'")))?;
        let usage = self.usage.get(tenant_id).cloned().unwrap_or_default();

        let base_price = match tenant.plan {
            Plan::Free => 0.0,
            Plan::Starter => 29.0,
            Plan::Pro => 99.0,
            Plan::Enterprise => 499.0,
        };

        let search_cost = usage.total_searches as f64 * 0.0001; // $0.0001 per search
        let storage_cost = (usage.storage_bytes.max(0) as f64 / (1024.0 * 1024.0 * 1024.0)) * 0.25; // $0.25/GB
        let insert_cost = usage.total_inserts as f64 * 0.0005; // $0.0005 per insert

        Ok(BillingEstimate {
            plan_base: base_price,
            search_cost,
            storage_cost,
            insert_cost,
            total: base_price + search_cost + storage_cost + insert_cost,
        })
    }

    /// List all tenants.
    pub fn list_tenants(&self) -> Vec<&Tenant> {
        self.tenants.values().collect()
    }

    /// Generate a dashboard overview for a tenant.
    pub fn dashboard(&self, tenant_id: &str) -> Result<DashboardOverview> {
        let tenant = self.tenants.get(tenant_id)
            .ok_or_else(|| NeedleError::NotFound(format!("Tenant '{tenant_id}'")))?;
        let usage = self.usage.get(tenant_id).cloned().unwrap_or_default();
        let billing = self.estimate_billing(tenant_id)?;

        Ok(DashboardOverview {
            tenant_id: tenant_id.into(),
            plan: tenant.plan,
            status: tenant.status,
            usage,
            billing,
            collections: Vec::new(),
            recent_activity: Vec::new(),
        })
    }

    /// Calculate per-query billing for a tenant.
    pub fn per_query_billing(&self, tenant_id: &str, pricing: &PerQueryPricing) -> Result<f64> {
        let usage = self.usage.get(tenant_id)
            .ok_or_else(|| NeedleError::NotFound(format!("Tenant '{tenant_id}'")))?;
        Ok(pricing.calculate(usage.total_searches))
    }
}

/// Monthly billing estimate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BillingEstimate {
    /// Base plan price.
    pub plan_base: f64,
    /// Search operation costs.
    pub search_cost: f64,
    /// Storage costs.
    pub storage_cost: f64,
    /// Insert operation costs.
    pub insert_cost: f64,
    /// Total estimated monthly cost.
    pub total: f64,
}

// ── Per-Query Billing ────────────────────────────────────────────────────────

/// Per-query billing model with tiered pricing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerQueryPricing {
    /// Price tiers (sorted by threshold ascending).
    pub tiers: Vec<PricingTier>,
    /// Free tier monthly query allowance.
    pub free_tier_queries: u64,
}

/// A single pricing tier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PricingTier {
    /// Queries up to this count use this tier's price.
    pub up_to_queries: u64,
    /// Cost per query in this tier.
    pub cost_per_query: f64,
}

impl Default for PerQueryPricing {
    fn default() -> Self {
        Self {
            free_tier_queries: 10_000,
            tiers: vec![
                PricingTier { up_to_queries: 100_000, cost_per_query: 0.000_10 },
                PricingTier { up_to_queries: 1_000_000, cost_per_query: 0.000_05 },
                PricingTier { up_to_queries: u64::MAX, cost_per_query: 0.000_02 },
            ],
        }
    }
}

impl PerQueryPricing {
    /// Calculate cost for a given number of queries.
    pub fn calculate(&self, total_queries: u64) -> f64 {
        if total_queries <= self.free_tier_queries {
            return 0.0;
        }
        let billable = total_queries - self.free_tier_queries;
        let mut remaining = billable;
        let mut cost = 0.0;
        let mut prev_boundary = 0u64;

        for tier in &self.tiers {
            let tier_capacity = tier.up_to_queries.saturating_sub(prev_boundary);
            let queries_in_tier = remaining.min(tier_capacity);
            cost += queries_in_tier as f64 * tier.cost_per_query;
            remaining -= queries_in_tier;
            prev_boundary = tier.up_to_queries;
            if remaining == 0 {
                break;
            }
        }
        cost
    }
}

// ── CLI Deploy Command ───────────────────────────────────────────────────────

/// CLI deploy command specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeployCommand {
    /// Target deployment platform.
    pub target: DeployTarget,
    /// Region to deploy in.
    pub region: String,
    /// Instance size.
    pub instance_size: InstanceSize,
    /// Path to the .needle database file.
    pub database_path: String,
    /// Whether to enable auto-scaling.
    pub auto_scale: bool,
    /// Environment variables to set.
    pub env_vars: HashMap<String, String>,
}

/// Supported deployment targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeployTarget {
    /// Docker container on the local machine.
    Docker,
    /// Fly.io serverless.
    FlyIo,
    /// Railway platform.
    Railway,
    /// Render.com.
    Render,
    /// AWS ECS Fargate.
    AwsEcs,
    /// Google Cloud Run.
    GcpCloudRun,
}

/// Instance size for cloud deployments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InstanceSize {
    /// 256MB RAM, 0.25 vCPU.
    Micro,
    /// 512MB RAM, 0.5 vCPU.
    Small,
    /// 1GB RAM, 1 vCPU.
    Medium,
    /// 4GB RAM, 2 vCPU.
    Large,
    /// 16GB RAM, 4 vCPU.
    XLarge,
}

/// Result of a deploy command.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeployResult {
    /// Deployment ID.
    pub deployment_id: String,
    /// Public endpoint URL.
    pub endpoint: String,
    /// Current deployment status.
    pub status: DeployStatus,
    /// Target platform.
    pub target: DeployTarget,
}

/// Deployment status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeployStatus {
    /// Deployment is being prepared.
    Building,
    /// Deployment is starting up.
    Deploying,
    /// Deployment is live and accepting traffic.
    Running,
    /// Deployment failed.
    Failed,
    /// Deployment was stopped.
    Stopped,
}

impl DeployCommand {
    /// Validate the deploy command parameters.
    pub fn validate(&self) -> Result<()> {
        if self.database_path.is_empty() {
            return Err(NeedleError::InvalidArgument(
                "database_path is required".into(),
            ));
        }
        if self.region.is_empty() {
            return Err(NeedleError::InvalidArgument("region is required".into()));
        }
        Ok(())
    }

    /// Generate a deployment manifest for the target platform.
    pub fn generate_manifest(&self) -> Result<DeployManifest> {
        self.validate()?;
        let (cpu_millicores, memory_mb) = match self.instance_size {
            InstanceSize::Micro => (250, 256),
            InstanceSize::Small => (500, 512),
            InstanceSize::Medium => (1000, 1024),
            InstanceSize::Large => (2000, 4096),
            InstanceSize::XLarge => (4000, 16_384),
        };
        Ok(DeployManifest {
            target: self.target,
            region: self.region.clone(),
            cpu_millicores,
            memory_mb,
            auto_scale: self.auto_scale,
            env_vars: self.env_vars.clone(),
            health_check_path: "/health".into(),
            port: 8080,
        })
    }
}

/// Generated deployment manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeployManifest {
    pub target: DeployTarget,
    pub region: String,
    pub cpu_millicores: u32,
    pub memory_mb: u32,
    pub auto_scale: bool,
    pub env_vars: HashMap<String, String>,
    pub health_check_path: String,
    pub port: u16,
}

// ── Dashboard Schema ─────────────────────────────────────────────────────────

/// Dashboard overview for a tenant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardOverview {
    /// Tenant ID.
    pub tenant_id: String,
    /// Current plan.
    pub plan: Plan,
    /// Tenant status.
    pub status: TenantStatus,
    /// Usage summary.
    pub usage: UsageMetrics,
    /// Current billing estimate.
    pub billing: BillingEstimate,
    /// Collection summaries.
    pub collections: Vec<DashboardCollection>,
    /// Recent activity log.
    pub recent_activity: Vec<ActivityEntry>,
}

/// Collection info for the dashboard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardCollection {
    /// Collection name.
    pub name: String,
    /// Number of vectors.
    pub vector_count: u64,
    /// Dimensionality.
    pub dimensions: u32,
    /// Storage size in bytes.
    pub storage_bytes: u64,
    /// Average query latency (ms).
    pub avg_query_latency_ms: f64,
}

/// Activity log entry for audit trail.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivityEntry {
    /// Timestamp (Unix seconds).
    pub timestamp: u64,
    /// Activity type.
    pub action: ActivityAction,
    /// Human-readable description.
    pub description: String,
}

/// Types of dashboard activity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActivityAction {
    CollectionCreated,
    CollectionDeleted,
    VectorsInserted,
    VectorsDeleted,
    PlanChanged,
    ApiKeyRotated,
    DeploymentStarted,
    DeploymentStopped,
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

    #[test]
    fn test_suspend_and_reactivate() {
        let mut cp = CloudControlPlane::new();
        cp.provision(ProvisionRequest { tenant_id: "t1".into(), plan: Plan::Pro, region: "us".into(), config: TenantConfig::default() }).unwrap();

        cp.suspend("t1").unwrap();
        assert_eq!(cp.tenant("t1").unwrap().status, TenantStatus::Suspended);
        assert!(cp.active_tenants().is_empty());

        cp.reactivate("t1").unwrap();
        assert_eq!(cp.tenant("t1").unwrap().status, TenantStatus::Active);
        assert_eq!(cp.active_tenants().len(), 1);
    }

    #[test]
    fn test_change_plan() {
        let mut cp = CloudControlPlane::new();
        cp.provision(ProvisionRequest { tenant_id: "t1".into(), plan: Plan::Starter, region: "us".into(), config: TenantConfig::default() }).unwrap();

        cp.change_plan("t1", Plan::Enterprise).unwrap();
        let t = cp.tenant("t1").unwrap();
        assert_eq!(t.plan, Plan::Enterprise);
        assert_eq!(t.config.max_vectors, 100_000_000);
    }

    #[test]
    fn test_billing_estimate() {
        let mut cp = CloudControlPlane::new();
        cp.provision(ProvisionRequest { tenant_id: "t1".into(), plan: Plan::Pro, region: "us".into(), config: TenantConfig::default() }).unwrap();

        cp.record_usage("t1", UsageEvent::Search { vectors_scanned: 1000 });
        cp.record_usage("t1", UsageEvent::Insert { count: 100 });

        let billing = cp.estimate_billing("t1").unwrap();
        assert_eq!(billing.plan_base, 99.0);
        assert!(billing.total > 99.0); // base + usage
    }

    #[test]
    fn test_list_tenants() {
        let mut cp = CloudControlPlane::new();
        cp.provision(ProvisionRequest { tenant_id: "t1".into(), plan: Plan::Free, region: "us".into(), config: TenantConfig::default() }).unwrap();
        cp.provision(ProvisionRequest { tenant_id: "t2".into(), plan: Plan::Pro, region: "eu".into(), config: TenantConfig::default() }).unwrap();

        assert_eq!(cp.list_tenants().len(), 2);
    }

    #[test]
    fn test_per_query_pricing_free_tier() {
        let pricing = PerQueryPricing::default();
        assert_eq!(pricing.calculate(0), 0.0);
        assert_eq!(pricing.calculate(10_000), 0.0);
    }

    #[test]
    fn test_per_query_pricing_tiered() {
        let pricing = PerQueryPricing::default();
        // 20K queries: 10K free + 10K at $0.0001
        let cost = pricing.calculate(20_000);
        let expected = 10_000.0 * 0.000_10;
        assert!((cost - expected).abs() < 1e-10);
    }

    #[test]
    fn test_per_query_pricing_multi_tier() {
        let pricing = PerQueryPricing::default();
        // 210K queries: 10K free + 100K at tier1($0.0001) + 100K at tier2($0.00005)
        let cost = pricing.calculate(210_000);
        let tier1 = 100_000.0 * 0.000_10;
        let tier2 = 100_000.0 * 0.000_05;
        let expected = tier1 + tier2;
        assert!((cost - expected).abs() < 1e-10);
    }

    #[test]
    fn test_deploy_command_validate() {
        let cmd = DeployCommand {
            target: DeployTarget::FlyIo,
            region: "us-east-1".into(),
            instance_size: InstanceSize::Small,
            database_path: "my.needle".into(),
            auto_scale: true,
            env_vars: HashMap::new(),
        };
        assert!(cmd.validate().is_ok());

        let bad = DeployCommand {
            database_path: String::new(),
            ..cmd.clone()
        };
        assert!(bad.validate().is_err());
    }

    #[test]
    fn test_deploy_manifest_generation() {
        let cmd = DeployCommand {
            target: DeployTarget::AwsEcs,
            region: "eu-west-1".into(),
            instance_size: InstanceSize::Large,
            database_path: "prod.needle".into(),
            auto_scale: false,
            env_vars: [("RUST_LOG".into(), "info".into())].into_iter().collect(),
        };
        let manifest = cmd.generate_manifest().unwrap();
        assert_eq!(manifest.cpu_millicores, 2000);
        assert_eq!(manifest.memory_mb, 4096);
        assert_eq!(manifest.port, 8080);
        assert_eq!(manifest.health_check_path, "/health");
        assert!(!manifest.auto_scale);
    }

    #[test]
    fn test_dashboard_overview() {
        let mut cp = CloudControlPlane::new();
        cp.provision(ProvisionRequest {
            tenant_id: "t1".into(), plan: Plan::Pro, region: "us".into(),
            config: TenantConfig::for_plan(Plan::Pro),
        }).unwrap();
        cp.record_usage("t1", UsageEvent::Search { vectors_scanned: 5000 });

        let dashboard = cp.dashboard("t1").unwrap();
        assert_eq!(dashboard.plan, Plan::Pro);
        assert_eq!(dashboard.status, TenantStatus::Active);
        assert_eq!(dashboard.usage.total_searches, 1);
        assert!(dashboard.billing.total > 0.0);
    }

    #[test]
    fn test_dashboard_not_found() {
        let cp = CloudControlPlane::new();
        assert!(cp.dashboard("nonexistent").is_err());
    }

    #[test]
    fn test_per_query_billing_integration() {
        let mut cp = CloudControlPlane::new();
        cp.provision(ProvisionRequest {
            tenant_id: "t1".into(), plan: Plan::Starter, region: "us".into(),
            config: TenantConfig::default(),
        }).unwrap();
        for _ in 0..50 {
            cp.record_usage("t1", UsageEvent::Search { vectors_scanned: 100 });
        }
        let pricing = PerQueryPricing::default();
        let cost = cp.per_query_billing("t1", &pricing).unwrap();
        assert_eq!(cost, 0.0); // 50 queries < 10K free tier
    }

    #[test]
    fn test_deploy_target_serde() {
        let targets = vec![
            DeployTarget::Docker, DeployTarget::FlyIo, DeployTarget::Railway,
            DeployTarget::Render, DeployTarget::AwsEcs, DeployTarget::GcpCloudRun,
        ];
        for t in targets {
            let s = serde_json::to_string(&t).unwrap();
            let d: DeployTarget = serde_json::from_str(&s).unwrap();
            assert_eq!(t, d);
        }
    }

    #[test]
    fn test_activity_action_variants() {
        let actions = vec![
            ActivityAction::CollectionCreated, ActivityAction::CollectionDeleted,
            ActivityAction::VectorsInserted, ActivityAction::PlanChanged,
            ActivityAction::ApiKeyRotated, ActivityAction::DeploymentStarted,
        ];
        for a in actions {
            let s = serde_json::to_string(&a).unwrap();
            let d: ActivityAction = serde_json::from_str(&s).unwrap();
            assert_eq!(a, d);
        }
    }
}
