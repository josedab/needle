#![allow(clippy::unwrap_used)]

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::{NeedleError, Result};

use super::control_plane::current_timestamp;
use super::tenant::OverageCharge;
use super::tiers::ResourceTier;

/// Pricing details for a resource tier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanPricing {
    pub tier: ResourceTier,
    pub base_price_cents: u64,
    pub price_per_1k_queries: u64,
    pub price_per_gb_storage: u64,
    pub included_queries: u64,
    pub included_storage_gb: u64,
}

/// Status of an invoice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InvoiceStatus {
    Draft,
    Pending,
    Paid,
    Overdue,
}

/// An invoice generated for a tenant billing period.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Invoice {
    pub invoice_id: String,
    pub tenant_id: String,
    pub period_start: u64,
    pub period_end: u64,
    pub base_charge_cents: u64,
    pub overage_charges: Vec<OverageCharge>,
    pub total_cents: u64,
    pub status: InvoiceStatus,
}

/// Tracks a tenant's usage within a billing period.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonthlyUsage {
    pub tenant_id: String,
    pub queries: u64,
    pub storage_bytes: u64,
    pub vectors_stored: u64,
    pub api_calls: u64,
    pub period_start: u64,
}

/// Usage-based billing engine.
pub struct BillingEngine {
    plans: HashMap<ResourceTier, PlanPricing>,
    invoices: RwLock<HashMap<String, Vec<Invoice>>>,
    usage_tracker: RwLock<HashMap<String, MonthlyUsage>>,
}

impl BillingEngine {
    /// Create a new billing engine with default pricing per tier.
    pub fn new() -> Self {
        let mut plans = HashMap::new();
        plans.insert(
            ResourceTier::Free,
            PlanPricing {
                tier: ResourceTier::Free,
                base_price_cents: 0,
                price_per_1k_queries: 0,
                price_per_gb_storage: 0,
                included_queries: 10_000,
                included_storage_gb: 1,
            },
        );
        plans.insert(
            ResourceTier::Developer,
            PlanPricing {
                tier: ResourceTier::Developer,
                base_price_cents: 2900,
                price_per_1k_queries: 10,
                price_per_gb_storage: 50,
                included_queries: 100_000,
                included_storage_gb: 5,
            },
        );
        plans.insert(
            ResourceTier::Professional,
            PlanPricing {
                tier: ResourceTier::Professional,
                base_price_cents: 9900,
                price_per_1k_queries: 5,
                price_per_gb_storage: 25,
                included_queries: 1_000_000,
                included_storage_gb: 50,
            },
        );
        plans.insert(
            ResourceTier::Enterprise,
            PlanPricing {
                tier: ResourceTier::Enterprise,
                base_price_cents: 49900,
                price_per_1k_queries: 2,
                price_per_gb_storage: 10,
                included_queries: 10_000_000,
                included_storage_gb: 500,
            },
        );
        Self {
            plans,
            invoices: RwLock::new(HashMap::new()),
            usage_tracker: RwLock::new(HashMap::new()),
        }
    }

    /// Record usage for a tenant.
    pub fn record_usage(&self, tenant_id: &str, queries: u64, storage_bytes: u64) -> Result<()> {
        let mut tracker = self.usage_tracker.write();
        let usage = tracker
            .entry(tenant_id.to_string())
            .or_insert_with(|| MonthlyUsage {
                tenant_id: tenant_id.to_string(),
                queries: 0,
                storage_bytes: 0,
                vectors_stored: 0,
                api_calls: 0,
                period_start: current_timestamp(),
            });
        usage.queries += queries;
        usage.storage_bytes += storage_bytes;
        usage.api_calls += 1;
        Ok(())
    }

    /// Get current monthly usage for a tenant.
    pub fn get_usage(&self, tenant_id: &str) -> Option<MonthlyUsage> {
        self.usage_tracker.read().get(tenant_id).cloned()
    }

    /// Generate an invoice for a tenant based on their tier and usage.
    pub fn generate_invoice(&self, tenant_id: &str, tier: ResourceTier) -> Result<Invoice> {
        let pricing = self
            .plans
            .get(&tier)
            .ok_or_else(|| NeedleError::InvalidInput(format!("No pricing for tier {:?}", tier)))?;

        let usage = self
            .usage_tracker
            .read()
            .get(tenant_id)
            .cloned()
            .unwrap_or(MonthlyUsage {
                tenant_id: tenant_id.to_string(),
                queries: 0,
                storage_bytes: 0,
                vectors_stored: 0,
                api_calls: 0,
                period_start: current_timestamp(),
            });

        let now = current_timestamp();
        let period_start = usage.period_start;
        let period_end = now;

        let mut overage_charges = Vec::new();

        // Query overage
        if usage.queries > pricing.included_queries {
            let overage = usage.queries - pricing.included_queries;
            let charge = (overage / 1000) * pricing.price_per_1k_queries;
            overage_charges.push(OverageCharge {
                resource: "queries".to_string(),
                overage_amount: overage,
                unit_price_cents: pricing.price_per_1k_queries,
                charge_cents: charge,
            });
        }

        // Storage overage
        let storage_gb = usage.storage_bytes / (1024 * 1024 * 1024);
        if storage_gb > pricing.included_storage_gb {
            let overage = storage_gb - pricing.included_storage_gb;
            let charge = overage * pricing.price_per_gb_storage;
            overage_charges.push(OverageCharge {
                resource: "storage".to_string(),
                overage_amount: overage,
                unit_price_cents: pricing.price_per_gb_storage,
                charge_cents: charge,
            });
        }

        let total_overage: u64 = overage_charges.iter().map(|c| c.charge_cents).sum();
        let invoice = Invoice {
            invoice_id: format!("inv_{}_{}", tenant_id, now),
            tenant_id: tenant_id.to_string(),
            period_start,
            period_end,
            base_charge_cents: pricing.base_price_cents,
            overage_charges,
            total_cents: pricing.base_price_cents + total_overage,
            status: InvoiceStatus::Pending,
        };

        self.invoices
            .write()
            .entry(tenant_id.to_string())
            .or_default()
            .push(invoice.clone());

        Ok(invoice)
    }

    /// Get all invoices for a tenant.
    pub fn get_invoices(&self, tenant_id: &str) -> Vec<Invoice> {
        self.invoices
            .read()
            .get(tenant_id)
            .cloned()
            .unwrap_or_default()
    }

    /// Mark an invoice as paid.
    pub fn mark_paid(&self, invoice_id: &str) -> Result<()> {
        let mut all = self.invoices.write();
        for invoices in all.values_mut() {
            if let Some(inv) = invoices.iter_mut().find(|i| i.invoice_id == invoice_id) {
                inv.status = InvoiceStatus::Paid;
                return Ok(());
            }
        }
        Err(NeedleError::NotFound(format!(
            "Invoice not found: {}",
            invoice_id
        )))
    }
}

impl Default for BillingEngine {
    fn default() -> Self {
        Self::new()
    }
}
