#![allow(clippy::unwrap_used)]

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

use super::billing::{Invoice, MonthlyUsage};
use super::control_plane::current_timestamp;
use super::tenant::Tenant;

/// Overall health indicator for a region.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthIndicator {
    Healthy,
    Degraded,
    Unhealthy,
    Unknown,
}

/// Health status of a specific region.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionHealthStatus {
    pub region: String,
    pub status: HealthIndicator,
    pub latency_p50_ms: f64,
    pub latency_p99_ms: f64,
    pub error_rate: f64,
    pub active_instances: usize,
    pub cpu_utilization: f64,
    pub memory_utilization: f64,
    pub last_checked: u64,
}

/// Severity levels for health alerts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertSeverityLevel {
    Info,
    Warning,
    Critical,
}

/// A health alert for a specific region.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthAlert {
    pub alert_id: String,
    pub region: String,
    pub severity: AlertSeverityLevel,
    pub message: String,
    pub timestamp: u64,
    pub resolved: bool,
}

/// Regional health dashboard for monitoring.
pub struct HealthDashboard {
    regions: RwLock<HashMap<String, RegionHealthStatus>>,
    alerts: RwLock<Vec<HealthAlert>>,
    #[allow(dead_code)]
    check_interval: Duration,
}

impl HealthDashboard {
    /// Create a new health dashboard.
    pub fn new(check_interval: Duration) -> Self {
        Self {
            regions: RwLock::new(HashMap::new()),
            alerts: RwLock::new(Vec::new()),
            check_interval,
        }
    }

    /// Update or insert a region's health status.
    pub fn update_region(&self, status: RegionHealthStatus) {
        self.regions.write().insert(status.region.clone(), status);
    }

    /// Get a single region's health status.
    pub fn get_region_status(&self, region: &str) -> Option<RegionHealthStatus> {
        self.regions.read().get(region).cloned()
    }

    /// Get health status for all regions.
    pub fn get_all_regions(&self) -> Vec<RegionHealthStatus> {
        self.regions.read().values().cloned().collect()
    }

    /// Add a health alert for a region.
    pub fn add_alert(&self, region: &str, severity: AlertSeverityLevel, message: &str) {
        let alert = HealthAlert {
            alert_id: format!("alert_{}_{}", region, current_timestamp()),
            region: region.to_string(),
            severity,
            message: message.to_string(),
            timestamp: current_timestamp(),
            resolved: false,
        };
        self.alerts.write().push(alert);
    }

    /// Get all active (unresolved) alerts.
    pub fn get_active_alerts(&self) -> Vec<HealthAlert> {
        self.alerts
            .read()
            .iter()
            .filter(|a| !a.resolved)
            .cloned()
            .collect()
    }

    /// Resolve an alert by ID. Returns `true` if the alert was found and resolved.
    pub fn resolve_alert(&self, alert_id: &str) -> bool {
        let mut alerts = self.alerts.write();
        if let Some(alert) = alerts.iter_mut().find(|a| a.alert_id == alert_id) {
            alert.resolved = true;
            true
        } else {
            false
        }
    }

    /// Compute overall health as the worst indicator across all regions.
    pub fn overall_health(&self) -> HealthIndicator {
        let regions = self.regions.read();
        if regions.is_empty() {
            return HealthIndicator::Unknown;
        }

        let mut worst = HealthIndicator::Healthy;
        for status in regions.values() {
            worst = worse_indicator(worst, status.status);
        }
        worst
    }
}

/// Return the worse of two health indicators.
pub(crate) fn worse_indicator(a: HealthIndicator, b: HealthIndicator) -> HealthIndicator {
    fn severity(h: HealthIndicator) -> u8 {
        match h {
            HealthIndicator::Healthy => 0,
            HealthIndicator::Degraded => 1,
            HealthIndicator::Unhealthy => 2,
            HealthIndicator::Unknown => 3,
        }
    }
    if severity(b) > severity(a) {
        b
    } else {
        a
    }
}

// ---------------------------------------------------------------------------
// Web Console
// ---------------------------------------------------------------------------

/// Generates HTML dashboard pages.
pub struct WebConsole;

impl WebConsole {
    /// Generate a complete HTML dashboard page with tenant count, region
    /// health cards, and an alert list.
    pub fn generate_dashboard_html(
        tenants: &[Tenant],
        regions: &[RegionHealthStatus],
        alerts: &[HealthAlert],
    ) -> String {
        let mut html = String::new();
        html.push_str("<!DOCTYPE html><html><head><meta charset=\"utf-8\">");
        html.push_str("<title>Needle Cloud Dashboard</title>");
        html.push_str("<style>");
        html.push_str("body{font-family:sans-serif;margin:20px;background:#f5f5f5}");
        html.push_str("h1{color:#333}.card{background:#fff;border-radius:8px;padding:16px;margin:8px;box-shadow:0 1px 3px rgba(0,0,0,0.1);display:inline-block;min-width:200px;vertical-align:top}");
        html.push_str(".healthy{border-left:4px solid #4caf50}.degraded{border-left:4px solid #ff9800}.unhealthy{border-left:4px solid #f44336}.unknown{border-left:4px solid #9e9e9e}");
        html.push_str(
            ".alert-critical{color:#f44336}.alert-warning{color:#ff9800}.alert-info{color:#2196f3}",
        );
        html.push_str("table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:8px;border-bottom:1px solid #ddd}");
        html.push_str("</style></head><body>");

        // Header
        html.push_str("<h1>Needle Cloud Dashboard</h1>");

        // Tenant summary
        html.push_str(&format!(
            "<div class=\"card\"><h3>Tenants</h3><p>{} total</p></div>",
            tenants.len()
        ));

        // Region health cards
        html.push_str("<h2>Region Health</h2><div>");
        for r in regions {
            let css_class = match r.status {
                HealthIndicator::Healthy => "healthy",
                HealthIndicator::Degraded => "degraded",
                HealthIndicator::Unhealthy => "unhealthy",
                HealthIndicator::Unknown => "unknown",
            };
            html.push_str(&format!(
                "<div class=\"card {}\"><h3>{}</h3><p>Status: {:?}</p><p>P50: {:.1}ms P99: {:.1}ms</p><p>Error rate: {:.2}%</p><p>Instances: {}</p></div>",
                css_class, r.region, r.status, r.latency_p50_ms, r.latency_p99_ms,
                r.error_rate * 100.0, r.active_instances
            ));
        }
        html.push_str("</div>");

        // Alerts table
        html.push_str("<h2>Active Alerts</h2>");
        if alerts.is_empty() {
            html.push_str("<p>No active alerts.</p>");
        } else {
            html.push_str("<table><tr><th>Region</th><th>Severity</th><th>Message</th></tr>");
            for a in alerts {
                let css = match a.severity {
                    AlertSeverityLevel::Critical => "alert-critical",
                    AlertSeverityLevel::Warning => "alert-warning",
                    AlertSeverityLevel::Info => "alert-info",
                };
                html.push_str(&format!(
                    "<tr class=\"{}\"><td>{}</td><td>{:?}</td><td>{}</td></tr>",
                    css, a.region, a.severity, a.message
                ));
            }
            html.push_str("</table>");
        }

        html.push_str("</body></html>");
        html
    }

    /// Generate a tenant detail HTML page.
    pub fn generate_tenant_detail_html(
        tenant: &Tenant,
        usage: Option<&MonthlyUsage>,
        invoices: &[Invoice],
    ) -> String {
        let mut html = String::new();
        html.push_str("<!DOCTYPE html><html><head><meta charset=\"utf-8\">");
        html.push_str("<title>Tenant Detail</title>");
        html.push_str("<style>");
        html.push_str("body{font-family:sans-serif;margin:20px;background:#f5f5f5}");
        html.push_str("h1{color:#333}.card{background:#fff;border-radius:8px;padding:16px;margin:8px;box-shadow:0 1px 3px rgba(0,0,0,0.1)}");
        html.push_str("table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:8px;border-bottom:1px solid #ddd}");
        html.push_str("</style></head><body>");

        html.push_str(&format!("<h1>Tenant: {}</h1>", tenant.config.name));
        html.push_str(&format!(
            "<div class=\"card\"><p>ID: {}</p><p>Tier: {:?}</p><p>Status: {:?}</p><p>Email: {}</p></div>",
            tenant.id, tenant.config.tier, tenant.status, tenant.config.email
        ));

        // Usage section
        html.push_str("<h2>Current Usage</h2>");
        if let Some(u) = usage {
            html.push_str(&format!(
                "<div class=\"card\"><p>Queries: {}</p><p>Storage: {} bytes</p><p>Vectors: {}</p><p>API Calls: {}</p></div>",
                u.queries, u.storage_bytes, u.vectors_stored, u.api_calls
            ));
        } else {
            html.push_str("<p>No usage data available.</p>");
        }

        // Invoices table
        html.push_str("<h2>Invoices</h2>");
        if invoices.is_empty() {
            html.push_str("<p>No invoices.</p>");
        } else {
            html.push_str("<table><tr><th>Invoice ID</th><th>Total</th><th>Status</th></tr>");
            for inv in invoices {
                html.push_str(&format!(
                    "<tr><td>{}</td><td>${:.2}</td><td>{:?}</td></tr>",
                    inv.invoice_id,
                    inv.total_cents as f64 / 100.0,
                    inv.status
                ));
            }
            html.push_str("</table>");
        }

        html.push_str("</body></html>");
        html
    }
}
