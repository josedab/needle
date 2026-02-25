#![allow(clippy::unwrap_used)]
//! ⚠️ **Experimental**: This module is under active development. APIs may change without notice.
//!
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

pub mod api_keys;
pub mod billing;
pub mod control_plane;
pub mod health;
pub mod orchestrator;
pub mod recovery;
pub mod regions;
pub mod sdk;
pub mod sla;
pub mod tenant;
pub mod tiers;

pub use api_keys::*;
pub use billing::*;
pub use control_plane::*;
pub use health::*;
pub use orchestrator::*;
pub use recovery::*;
pub use regions::*;
pub use sdk::*;
pub use sla::*;
pub use tenant::*;
pub use tiers::*;

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_API_KEY: &str = "test-api-key";

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
            .create_api_key(
                &tenant.id,
                "prod-key",
                vec![Permission::Read, Permission::Write],
            )
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
            .provision(
                "tenant_001",
                Some(Region::UsEast1),
                1024 * 1024 * 512,
                1024 * 1024 * 1024,
            )
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

    // -----------------------------------------------------------------------
    // BillingEngine tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_billing_engine_record_usage() {
        let engine = BillingEngine::new();
        engine.record_usage("t1", 500, 1024).unwrap();
        let usage = engine.get_usage("t1").unwrap();
        assert_eq!(usage.queries, 500);
        assert_eq!(usage.storage_bytes, 1024);
        assert_eq!(usage.api_calls, 1);

        engine.record_usage("t1", 200, 2048).unwrap();
        let usage = engine.get_usage("t1").unwrap();
        assert_eq!(usage.queries, 700);
        assert_eq!(usage.storage_bytes, 3072);
        assert_eq!(usage.api_calls, 2);
    }

    #[test]
    fn test_billing_engine_generate_invoice_no_overage() {
        let engine = BillingEngine::new();
        engine.record_usage("t1", 1000, 0).unwrap();
        let invoice = engine
            .generate_invoice("t1", ResourceTier::Developer)
            .unwrap();
        assert_eq!(invoice.base_charge_cents, 2900);
        assert!(invoice.overage_charges.is_empty());
        assert_eq!(invoice.total_cents, 2900);
        assert_eq!(invoice.status, InvoiceStatus::Pending);
    }

    #[test]
    fn test_billing_engine_overage_calculation() {
        let engine = BillingEngine::new();
        // Developer tier includes 100_000 queries; exceed by 50_000
        engine.record_usage("t1", 150_000, 0).unwrap();
        let invoice = engine
            .generate_invoice("t1", ResourceTier::Developer)
            .unwrap();
        // overage = 50_000 queries, charge = (50_000/1000) * 10 = 500 cents
        assert_eq!(invoice.overage_charges.len(), 1);
        assert_eq!(invoice.overage_charges[0].charge_cents, 500);
        assert_eq!(invoice.total_cents, 2900 + 500);
    }

    #[test]
    fn test_billing_engine_mark_paid() {
        let engine = BillingEngine::new();
        engine.record_usage("t1", 10, 0).unwrap();
        let invoice = engine
            .generate_invoice("t1", ResourceTier::Developer)
            .unwrap();
        assert_eq!(invoice.status, InvoiceStatus::Pending);

        engine.mark_paid(&invoice.invoice_id).unwrap();
        let invoices = engine.get_invoices("t1");
        assert_eq!(invoices[0].status, InvoiceStatus::Paid);
    }

    #[test]
    fn test_billing_engine_mark_paid_not_found() {
        let engine = BillingEngine::new();
        assert!(engine.mark_paid("nonexistent").is_err());
    }

    #[test]
    fn test_billing_engine_get_invoices_empty() {
        let engine = BillingEngine::new();
        assert!(engine.get_invoices("t1").is_empty());
    }

    // -----------------------------------------------------------------------
    // HealthDashboard tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_health_dashboard_update_and_get_region() {
        let dash = HealthDashboard::new(std::time::Duration::from_secs(30));
        dash.update_region(RegionHealthStatus {
            region: "us-east-1".to_string(),
            status: HealthIndicator::Healthy,
            latency_p50_ms: 5.0,
            latency_p99_ms: 20.0,
            error_rate: 0.001,
            active_instances: 3,
            cpu_utilization: 0.45,
            memory_utilization: 0.60,
            last_checked: current_timestamp(),
        });

        let status = dash.get_region_status("us-east-1").unwrap();
        assert_eq!(status.status, HealthIndicator::Healthy);
        assert_eq!(status.active_instances, 3);
    }

    #[test]
    fn test_health_dashboard_get_all_regions() {
        let dash = HealthDashboard::new(std::time::Duration::from_secs(30));
        dash.update_region(RegionHealthStatus {
            region: "us-east-1".to_string(),
            status: HealthIndicator::Healthy,
            latency_p50_ms: 5.0,
            latency_p99_ms: 20.0,
            error_rate: 0.0,
            active_instances: 2,
            cpu_utilization: 0.3,
            memory_utilization: 0.4,
            last_checked: 0,
        });
        dash.update_region(RegionHealthStatus {
            region: "eu-west-1".to_string(),
            status: HealthIndicator::Degraded,
            latency_p50_ms: 10.0,
            latency_p99_ms: 50.0,
            error_rate: 0.05,
            active_instances: 1,
            cpu_utilization: 0.8,
            memory_utilization: 0.7,
            last_checked: 0,
        });
        assert_eq!(dash.get_all_regions().len(), 2);
    }

    #[test]
    fn test_health_dashboard_alerts_lifecycle() {
        let dash = HealthDashboard::new(std::time::Duration::from_secs(30));
        dash.add_alert("us-east-1", AlertSeverityLevel::Critical, "High error rate");
        dash.add_alert("eu-west-1", AlertSeverityLevel::Warning, "Elevated latency");

        let active = dash.get_active_alerts();
        assert_eq!(active.len(), 2);

        let alert_id = active[0].alert_id.clone();
        assert!(dash.resolve_alert(&alert_id));
        assert_eq!(dash.get_active_alerts().len(), 1);
    }

    #[test]
    fn test_health_dashboard_resolve_nonexistent() {
        let dash = HealthDashboard::new(std::time::Duration::from_secs(30));
        assert!(!dash.resolve_alert("nonexistent"));
    }

    #[test]
    fn test_health_dashboard_overall_health_empty() {
        let dash = HealthDashboard::new(std::time::Duration::from_secs(30));
        assert_eq!(dash.overall_health(), HealthIndicator::Unknown);
    }

    #[test]
    fn test_health_dashboard_overall_health_worst_wins() {
        let dash = HealthDashboard::new(std::time::Duration::from_secs(30));
        dash.update_region(RegionHealthStatus {
            region: "us-east-1".to_string(),
            status: HealthIndicator::Healthy,
            latency_p50_ms: 5.0,
            latency_p99_ms: 20.0,
            error_rate: 0.0,
            active_instances: 2,
            cpu_utilization: 0.3,
            memory_utilization: 0.4,
            last_checked: 0,
        });
        dash.update_region(RegionHealthStatus {
            region: "eu-west-1".to_string(),
            status: HealthIndicator::Unhealthy,
            latency_p50_ms: 100.0,
            latency_p99_ms: 500.0,
            error_rate: 0.5,
            active_instances: 0,
            cpu_utilization: 0.99,
            memory_utilization: 0.95,
            last_checked: 0,
        });
        assert_eq!(dash.overall_health(), HealthIndicator::Unhealthy);
    }

    // -----------------------------------------------------------------------
    // WebConsole tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_web_console_dashboard_html() {
        let tenants = vec![Tenant {
            id: "t1".to_string(),
            config: TenantConfig {
                name: "acme".to_string(),
                tier: ResourceTier::Developer,
                email: "a@b.com".to_string(),
                ..Default::default()
            },
            status: TenantStatus::Active,
            created_at: 0,
            updated_at: 0,
            usage: TenantUsage::default(),
        }];
        let regions = vec![RegionHealthStatus {
            region: "us-east-1".to_string(),
            status: HealthIndicator::Healthy,
            latency_p50_ms: 5.0,
            latency_p99_ms: 20.0,
            error_rate: 0.001,
            active_instances: 3,
            cpu_utilization: 0.4,
            memory_utilization: 0.5,
            last_checked: 0,
        }];
        let alerts = vec![HealthAlert {
            alert_id: "a1".to_string(),
            region: "us-east-1".to_string(),
            severity: AlertSeverityLevel::Warning,
            message: "High latency".to_string(),
            timestamp: 0,
            resolved: false,
        }];

        let html = WebConsole::generate_dashboard_html(&tenants, &regions, &alerts);
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("Needle Cloud Dashboard"));
        assert!(html.contains("1 total")); // tenant count
        assert!(html.contains("us-east-1"));
        assert!(html.contains("healthy"));
        assert!(html.contains("High latency"));
        assert!(html.contains("Warning"));
    }

    #[test]
    fn test_web_console_tenant_detail_html() {
        let tenant = Tenant {
            id: "t1".to_string(),
            config: TenantConfig {
                name: "acme".to_string(),
                tier: ResourceTier::Developer,
                email: "a@b.com".to_string(),
                ..Default::default()
            },
            status: TenantStatus::Active,
            created_at: 0,
            updated_at: 0,
            usage: TenantUsage::default(),
        };
        let usage = MonthlyUsage {
            tenant_id: "t1".to_string(),
            queries: 5000,
            storage_bytes: 1024,
            vectors_stored: 100,
            api_calls: 42,
            period_start: 0,
        };
        let invoices = vec![Invoice {
            invoice_id: "inv_1".to_string(),
            tenant_id: "t1".to_string(),
            period_start: 0,
            period_end: 100,
            base_charge_cents: 2900,
            overage_charges: vec![],
            total_cents: 2900,
            status: InvoiceStatus::Paid,
        }];

        let html = WebConsole::generate_tenant_detail_html(&tenant, Some(&usage), &invoices);
        assert!(html.contains("Tenant: acme"));
        assert!(html.contains("Queries: 5000"));
        assert!(html.contains("inv_1"));
        assert!(html.contains("$29.00"));
        assert!(html.contains("Paid"));
    }

    #[test]
    fn test_web_console_tenant_detail_no_usage() {
        let tenant = Tenant {
            id: "t1".to_string(),
            config: TenantConfig {
                name: "empty".to_string(),
                ..Default::default()
            },
            status: TenantStatus::Active,
            created_at: 0,
            updated_at: 0,
            usage: TenantUsage::default(),
        };
        let html = WebConsole::generate_tenant_detail_html(&tenant, None, &[]);
        assert!(html.contains("No usage data available"));
        assert!(html.contains("No invoices"));
    }

    // -----------------------------------------------------------------------
    // ApiKeyManager tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_api_key_manager_create_and_validate() {
        let mgr = ApiKeyManager::new();
        let (raw_key, entry) = mgr.create_key("t1", "my-key", vec![Permission::Read]);
        assert!(raw_key.starts_with("ndk_"));
        assert_eq!(entry.name, "my-key");
        assert_eq!(entry.tenant_id, "t1");
        assert!(entry.active);

        let found = mgr.validate_key(&entry.key_hash).unwrap();
        assert_eq!(found.key_id, entry.key_id);
    }

    #[test]
    fn test_api_key_manager_revoke() {
        let mgr = ApiKeyManager::new();
        let (_raw, entry) = mgr.create_key("t1", "revoke-me", vec![Permission::Write]);
        assert!(mgr.revoke_key(&entry.key_id));

        // After revoking, validate should return None
        assert!(mgr.validate_key(&entry.key_hash).is_none());
    }

    #[test]
    fn test_api_key_manager_revoke_nonexistent() {
        let mgr = ApiKeyManager::new();
        assert!(!mgr.revoke_key("no_such_key"));
    }

    #[test]
    fn test_api_key_manager_list_keys() {
        let mgr = ApiKeyManager::new();
        mgr.create_key("t1", "k1", vec![Permission::Read]);
        mgr.create_key("t1", "k2", vec![Permission::Write]);
        mgr.create_key("t2", "k3", vec![Permission::Admin]);

        let t1_keys = mgr.list_keys("t1");
        assert_eq!(t1_keys.len(), 2);
        let t2_keys = mgr.list_keys("t2");
        assert_eq!(t2_keys.len(), 1);
    }

    #[test]
    fn test_api_key_manager_validate_nonexistent() {
        let mgr = ApiKeyManager::new();
        assert!(mgr.validate_key("nonexistent_hash").is_none());
    }

    // ── SDK Client tests ─────────────────────────────────────────────────

    #[test]
    fn test_sdk_client_no_key() {
        let client = NeedleCloudClient::new(SdkConfig::default());
        assert_eq!(client.state(), ConnectionState::Disconnected);
        assert!(!client.is_ready());
        assert!(client.connect().is_err());
    }

    #[test]
    fn test_sdk_client_connect() {
        let config = SdkConfig {
            api_key: TEST_API_KEY.into(),
            ..Default::default()
        };
        let client = NeedleCloudClient::new(config);
        assert_eq!(client.state(), ConnectionState::Connected);
        assert!(client.is_ready());
    }

    #[test]
    fn test_sdk_client_stats() {
        let config = SdkConfig {
            api_key: TEST_API_KEY.into(),
            ..Default::default()
        };
        let client = NeedleCloudClient::new(config);
        client.record_request();
        client.record_request();
        client.record_error();

        let stats = client.stats();
        assert_eq!(stats.total_requests, 2);
        assert_eq!(stats.total_errors, 1);
    }

    #[test]
    fn test_sdk_client_retry_delay() {
        let client = NeedleCloudClient::new(SdkConfig {
            api_key: TEST_API_KEY.into(),
            retry_base_delay_ms: 100,
            ..Default::default()
        });
        assert_eq!(client.retry_delay_ms(0), 100);
        assert_eq!(client.retry_delay_ms(1), 200);
        assert_eq!(client.retry_delay_ms(2), 400);
    }

    #[test]
    fn test_sdk_client_fallback() {
        let config = SdkConfig {
            api_key: TEST_API_KEY.into(),
            local_fallback: true,
            ..Default::default()
        };
        let client = NeedleCloudClient::new(config);

        // Generate lots of requests with high error rate
        for _ in 0..20 {
            client.record_request();
        }
        for _ in 0..15 {
            client.record_error();
        }

        assert_eq!(client.state(), ConnectionState::Fallback);
    }

    // ── Instance Recovery tests ──────────────────────────────────────────

    #[test]
    fn test_recovery_manager_healthy() {
        let mgr = InstanceRecoveryManager::new(AutoRecoveryConfig::default());
        mgr.register("i-001");
        mgr.record_healthy("i-001");

        let health = mgr.get_health("i-001").unwrap();
        assert_eq!(health.status, RecoveryStatus::Healthy);
        assert_eq!(health.consecutive_failures, 0);
    }

    #[test]
    fn test_recovery_manager_failure_threshold() {
        let config = AutoRecoveryConfig {
            failure_threshold: 3,
            restart_cooldown_secs: 0,
            ..Default::default()
        };
        let mgr = InstanceRecoveryManager::new(config);
        mgr.register("i-001");

        // First two failures: degraded, no action
        assert!(mgr.record_failure("i-001").is_none());
        assert!(mgr.record_failure("i-001").is_none());

        let health = mgr.get_health("i-001").unwrap();
        assert_eq!(health.status, RecoveryStatus::Degraded);

        // Third failure: triggers restart
        let action = mgr.record_failure("i-001");
        assert!(matches!(action, Some(RecoveryAction::Restart { .. })));
    }

    #[test]
    fn test_recovery_manager_exhausted() {
        let config = AutoRecoveryConfig {
            failure_threshold: 1,
            max_restarts: 2,
            restart_cooldown_secs: 0,
            ..Default::default()
        };
        let mgr = InstanceRecoveryManager::new(config);
        mgr.register("i-001");

        // Use up all restarts
        mgr.record_failure("i-001"); // restart 1
        mgr.record_failure("i-001"); // restart 2

        // Next failure should escalate
        let action = mgr.record_failure("i-001");
        assert!(matches!(action, Some(RecoveryAction::Escalate { .. })));
    }

    #[test]
    fn test_recovery_manager_all_instances() {
        let mgr = InstanceRecoveryManager::new(AutoRecoveryConfig::default());
        mgr.register("i-001");
        mgr.register("i-002");
        assert_eq!(mgr.all_instances().len(), 2);
    }
}
