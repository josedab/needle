//! Cloud Pricing Calculator
//!
//! Usage-based pricing engine with plan tiers and billing estimation
//! for the Needle managed cloud service.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::services::pricing::{PricingEngine, UsageSnapshot, Plan, Invoice};
//!
//! let engine = PricingEngine::new();
//! let usage = UsageSnapshot { vectors_stored: 500_000, searches: 100_000,
//!     storage_gb: 2.5, api_calls: 200_000 };
//! let invoice = engine.calculate(Plan::Pro, &usage);
//! println!("Total: ${:.2}/month", invoice.total);
//! ```

use serde::{Deserialize, Serialize};

/// Pricing plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Plan { Free, Starter, Pro, Enterprise }

/// Usage snapshot for billing.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UsageSnapshot {
    pub vectors_stored: u64,
    pub searches: u64,
    pub storage_gb: f64,
    pub api_calls: u64,
}

/// Plan pricing details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanPricing {
    pub plan: Plan,
    pub base_price: f64,
    pub included_vectors: u64,
    pub included_searches: u64,
    pub included_storage_gb: f64,
    pub price_per_1k_vectors: f64,
    pub price_per_1k_searches: f64,
    pub price_per_gb: f64,
}

/// Generated invoice.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Invoice {
    pub plan: Plan,
    pub base: f64,
    pub vector_overage: f64,
    pub search_overage: f64,
    pub storage_overage: f64,
    pub total: f64,
    pub breakdown: Vec<LineItem>,
}

/// Invoice line item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineItem { pub description: String, pub amount: f64 }

/// Pricing engine.
pub struct PricingEngine { plans: Vec<PlanPricing> }

impl PricingEngine {
    pub fn new() -> Self {
        Self { plans: vec![
            PlanPricing { plan: Plan::Free, base_price: 0.0, included_vectors: 100_000, included_searches: 10_000,
                included_storage_gb: 0.5, price_per_1k_vectors: 0.0, price_per_1k_searches: 0.0, price_per_gb: 0.0 },
            PlanPricing { plan: Plan::Starter, base_price: 29.0, included_vectors: 1_000_000, included_searches: 100_000,
                included_storage_gb: 5.0, price_per_1k_vectors: 0.01, price_per_1k_searches: 0.001, price_per_gb: 0.50 },
            PlanPricing { plan: Plan::Pro, base_price: 99.0, included_vectors: 10_000_000, included_searches: 1_000_000,
                included_storage_gb: 50.0, price_per_1k_vectors: 0.005, price_per_1k_searches: 0.0005, price_per_gb: 0.25 },
            PlanPricing { plan: Plan::Enterprise, base_price: 499.0, included_vectors: 100_000_000, included_searches: 10_000_000,
                included_storage_gb: 500.0, price_per_1k_vectors: 0.002, price_per_1k_searches: 0.0002, price_per_gb: 0.10 },
        ]}
    }

    /// Calculate invoice for a usage snapshot.
    pub fn calculate(&self, plan: Plan, usage: &UsageSnapshot) -> Invoice {
        let pricing = self.plans.iter().find(|p| p.plan == plan).unwrap_or(&self.plans[0]);
        let mut items = Vec::new();

        items.push(LineItem { description: format!("{:?} plan base", plan), amount: pricing.base_price });

        let vec_over = usage.vectors_stored.saturating_sub(pricing.included_vectors);
        let vec_cost = vec_over as f64 / 1000.0 * pricing.price_per_1k_vectors;
        if vec_cost > 0.0 { items.push(LineItem { description: format!("{}K extra vectors", vec_over / 1000), amount: vec_cost }); }

        let search_over = usage.searches.saturating_sub(pricing.included_searches);
        let search_cost = search_over as f64 / 1000.0 * pricing.price_per_1k_searches;
        if search_cost > 0.0 { items.push(LineItem { description: format!("{}K extra searches", search_over / 1000), amount: search_cost }); }

        let storage_over = (usage.storage_gb - pricing.included_storage_gb).max(0.0);
        let storage_cost = storage_over * pricing.price_per_gb;
        if storage_cost > 0.0 { items.push(LineItem { description: format!("{:.1}GB extra storage", storage_over), amount: storage_cost }); }

        let total = pricing.base_price + vec_cost + search_cost + storage_cost;

        Invoice { plan, base: pricing.base_price, vector_overage: vec_cost, search_overage: search_cost, storage_overage: storage_cost, total, breakdown: items }
    }

    /// Get plan pricing details.
    pub fn plan_pricing(&self, plan: Plan) -> Option<&PlanPricing> { self.plans.iter().find(|p| p.plan == plan) }

    /// Compare plans for a usage snapshot.
    pub fn compare(&self, usage: &UsageSnapshot) -> Vec<Invoice> {
        [Plan::Free, Plan::Starter, Plan::Pro, Plan::Enterprise].iter().map(|&p| self.calculate(p, usage)).collect()
    }

    /// Recommend the best plan for a usage snapshot.
    pub fn recommend(&self, usage: &UsageSnapshot) -> Plan {
        let invoices = self.compare(usage);
        invoices.into_iter().min_by(|a, b| a.total.partial_cmp(&b.total).unwrap_or(std::cmp::Ordering::Equal)).map(|i| i.plan).unwrap_or(Plan::Free)
    }
}

impl Default for PricingEngine { fn default() -> Self { Self::new() } }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_free_plan() {
        let engine = PricingEngine::new();
        let invoice = engine.calculate(Plan::Free, &UsageSnapshot { vectors_stored: 50_000, searches: 5_000, storage_gb: 0.3, api_calls: 0 });
        assert_eq!(invoice.total, 0.0);
    }

    #[test]
    fn test_pro_with_overage() {
        let engine = PricingEngine::new();
        let invoice = engine.calculate(Plan::Pro, &UsageSnapshot { vectors_stored: 15_000_000, searches: 500_000, storage_gb: 60.0, api_calls: 0 });
        assert!(invoice.total > 99.0); // base + overage
        assert!(invoice.vector_overage > 0.0);
    }

    #[test]
    fn test_compare() {
        let engine = PricingEngine::new();
        let usage = UsageSnapshot { vectors_stored: 500_000, searches: 50_000, storage_gb: 2.0, api_calls: 0 };
        let invoices = engine.compare(&usage);
        assert_eq!(invoices.len(), 4);
    }

    #[test]
    fn test_recommend() {
        let engine = PricingEngine::new();
        // Small usage → Free
        let small = UsageSnapshot { vectors_stored: 10_000, searches: 1_000, storage_gb: 0.1, api_calls: 0 };
        assert_eq!(engine.recommend(&small), Plan::Free);
    }

    #[test]
    fn test_breakdown() {
        let engine = PricingEngine::new();
        let invoice = engine.calculate(Plan::Starter, &UsageSnapshot { vectors_stored: 2_000_000, searches: 200_000, storage_gb: 10.0, api_calls: 0 });
        assert!(invoice.breakdown.len() >= 2); // base + at least one overage
    }
}
