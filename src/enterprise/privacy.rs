//! Differential Privacy for Vector Queries
//!
//! Provides ε-differential privacy mechanisms for search results:
//! - Laplace and Gaussian noise for distance score perturbation
//! - Per-user/session privacy budget tracking with composition theorems
//! - Configurable epsilon/delta parameters
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::enterprise::privacy::{PrivacyConfig, PrivacyMechanism, PrivacyBudget};
//!
//! let config = PrivacyConfig::new(1.0, 1e-5);
//! let mechanism = PrivacyMechanism::new(config);
//!
//! // Perturb distance scores
//! let noisy_distance = mechanism.perturb_distance(0.5, 1.0);
//! ```

use crate::collection::SearchResult;
use crate::error::{NeedleError, Result};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

/// Noise mechanism type for differential privacy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NoiseMechanism {
    /// Laplace mechanism (pure ε-DP)
    Laplace,
    /// Gaussian mechanism (approximate (ε,δ)-DP)
    Gaussian,
}

impl Default for NoiseMechanism {
    fn default() -> Self {
        Self::Laplace
    }
}

/// Configuration for differential privacy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyConfig {
    /// Privacy parameter epsilon (smaller = more private, typical: 0.1 to 10.0)
    pub epsilon: f64,
    /// Privacy parameter delta (only used for Gaussian mechanism, typical: 1e-5)
    pub delta: f64,
    /// Noise mechanism to use
    pub mechanism: NoiseMechanism,
    /// Sensitivity of the distance function (max change from one record)
    pub sensitivity: f64,
    /// Maximum privacy budget per session before queries are denied
    pub max_budget_per_session: f64,
    /// Whether to enable privacy budget tracking
    pub budget_tracking: bool,
}

impl Default for PrivacyConfig {
    fn default() -> Self {
        Self {
            epsilon: 1.0,
            delta: 1e-5,
            mechanism: NoiseMechanism::Laplace,
            sensitivity: 1.0,
            max_budget_per_session: 10.0,
            budget_tracking: true,
        }
    }
}

impl PrivacyConfig {
    /// Create a new privacy config with given epsilon and delta.
    pub fn new(epsilon: f64, delta: f64) -> Self {
        Self {
            epsilon,
            delta,
            ..Default::default()
        }
    }

    /// Set the noise mechanism.
    #[must_use]
    pub fn with_mechanism(mut self, mechanism: NoiseMechanism) -> Self {
        self.mechanism = mechanism;
        self
    }

    /// Set the sensitivity.
    #[must_use]
    pub fn with_sensitivity(mut self, sensitivity: f64) -> Self {
        self.sensitivity = sensitivity;
        self
    }

    /// Set max budget per session.
    #[must_use]
    pub fn with_max_budget(mut self, budget: f64) -> Self {
        self.max_budget_per_session = budget;
        self
    }
}

/// Tracks accumulated privacy budget per session using basic composition.
#[derive(Debug, Default)]
pub struct PrivacyBudget {
    /// Accumulated epsilon per session ID
    budgets: HashMap<String, f64>,
    /// Total queries processed across all sessions
    total_queries: AtomicU64,
}

impl PrivacyBudget {
    /// Create a new privacy budget tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if a session has remaining budget for a query with given epsilon.
    pub fn has_budget(&self, session_id: &str, epsilon: f64, max_budget: f64) -> bool {
        let used = self.budgets.get(session_id).copied().unwrap_or(0.0);
        used + epsilon <= max_budget
    }

    /// Consume budget for a query. Returns false if budget exceeded.
    pub fn consume(&mut self, session_id: &str, epsilon: f64, max_budget: f64) -> bool {
        let used = self.budgets.entry(session_id.to_string()).or_insert(0.0);
        if *used + epsilon > max_budget {
            return false;
        }
        *used += epsilon;
        self.total_queries.fetch_add(1, Ordering::Relaxed);
        true
    }

    /// Get remaining budget for a session.
    pub fn remaining(&self, session_id: &str, max_budget: f64) -> f64 {
        let used = self.budgets.get(session_id).copied().unwrap_or(0.0);
        (max_budget - used).max(0.0)
    }

    /// Reset budget for a session.
    pub fn reset(&mut self, session_id: &str) {
        self.budgets.remove(session_id);
    }

    /// Get total queries processed.
    pub fn total_queries(&self) -> u64 {
        self.total_queries.load(Ordering::Relaxed)
    }

    /// Get budget summary for all sessions.
    pub fn summary(&self) -> Vec<(String, f64)> {
        self.budgets.iter().map(|(k, v)| (k.clone(), *v)).collect()
    }
}

/// Privacy mechanism for perturbing distance scores.
pub struct PrivacyMechanism {
    config: PrivacyConfig,
}

impl PrivacyMechanism {
    /// Create a new privacy mechanism with the given config.
    pub fn new(config: PrivacyConfig) -> Self {
        Self { config }
    }

    /// Perturb a distance score with calibrated noise.
    ///
    /// Returns the noisy distance value. The noise scale is calibrated
    /// to the sensitivity and epsilon parameters.
    pub fn perturb_distance(&self, distance: f32, sensitivity: f64) -> f32 {
        let noise = match self.config.mechanism {
            NoiseMechanism::Laplace => self.laplace_noise(sensitivity),
            NoiseMechanism::Gaussian => self.gaussian_noise(sensitivity),
        };
        (distance as f64 + noise).max(0.0) as f32
    }

    /// Perturb a vector of distance scores.
    pub fn perturb_distances(&self, distances: &mut [f32], sensitivity: f64) {
        for dist in distances.iter_mut() {
            *dist = self.perturb_distance(*dist, sensitivity);
        }
    }

    /// Generate Laplace noise with scale = sensitivity / epsilon.
    fn laplace_noise(&self, sensitivity: f64) -> f64 {
        let scale = sensitivity / self.config.epsilon;
        let mut rng = rand::thread_rng();
        let u: f64 = rng.gen::<f64>() - 0.5;
        -scale * u.signum() * (1.0 - 2.0 * u.abs()).ln()
    }

    /// Generate Gaussian noise with σ = sensitivity * √(2 ln(1.25/δ)) / ε.
    fn gaussian_noise(&self, sensitivity: f64) -> f64 {
        let sigma = sensitivity * (2.0 * (1.25 / self.config.delta).ln()).sqrt()
            / self.config.epsilon;
        let mut rng = rand::thread_rng();
        // Box-Muller transform for normal distribution
        let u1: f64 = rng.gen::<f64>();
        let u2: f64 = rng.gen::<f64>();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        z * sigma
    }

    /// Get a privacy explanation string for the current configuration.
    pub fn explain(&self) -> PrivacyExplain {
        let noise_scale = match self.config.mechanism {
            NoiseMechanism::Laplace => self.config.sensitivity / self.config.epsilon,
            NoiseMechanism::Gaussian => {
                self.config.sensitivity
                    * (2.0 * (1.25 / self.config.delta).ln()).sqrt()
                    / self.config.epsilon
            }
        };

        PrivacyExplain {
            mechanism: format!("{:?}", self.config.mechanism),
            epsilon: self.config.epsilon,
            delta: self.config.delta,
            sensitivity: self.config.sensitivity,
            noise_scale,
            budget_tracking: self.config.budget_tracking,
            max_budget: self.config.max_budget_per_session,
        }
    }

    /// Get the config.
    pub fn config(&self) -> &PrivacyConfig {
        &self.config
    }
}

/// Explanation of privacy parameters for a query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyExplain {
    /// Noise mechanism used
    pub mechanism: String,
    /// Epsilon parameter
    pub epsilon: f64,
    /// Delta parameter
    pub delta: f64,
    /// Sensitivity
    pub sensitivity: f64,
    /// Effective noise scale
    pub noise_scale: f64,
    /// Whether budget tracking is enabled
    pub budget_tracking: bool,
    /// Maximum budget per session
    pub max_budget: f64,
}

/// Composition theorem for privacy budget accounting across repeated queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompositionTheorem {
    /// Basic composition: total ε = sum of individual ε values
    Basic,
    /// Advanced composition: total ε ≈ √(2k·ln(1/δ'))·ε + k·ε·(e^ε - 1)
    Advanced,
    /// Rényi Differential Privacy composition (tighter bounds)
    Renyi,
}

impl Default for CompositionTheorem {
    fn default() -> Self {
        Self::Basic
    }
}

/// Per-collection privacy policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionPrivacyPolicy {
    /// Collection name this policy applies to
    pub collection_name: String,
    /// Privacy configuration for this collection
    pub config: PrivacyConfig,
    /// Composition theorem to use for budget accounting
    pub composition: CompositionTheorem,
    /// Whether privacy is enforced (vs advisory)
    pub enforced: bool,
    /// Optional per-field sensitivity overrides
    pub field_sensitivities: HashMap<String, f64>,
}

impl CollectionPrivacyPolicy {
    /// Create a new privacy policy for a collection.
    pub fn new(collection_name: impl Into<String>, config: PrivacyConfig) -> Self {
        Self {
            collection_name: collection_name.into(),
            config,
            composition: CompositionTheorem::default(),
            enforced: true,
            field_sensitivities: HashMap::new(),
        }
    }

    /// Set the composition theorem.
    #[must_use]
    pub fn with_composition(mut self, composition: CompositionTheorem) -> Self {
        self.composition = composition;
        self
    }

    /// Set whether the policy is enforced.
    #[must_use]
    pub fn with_enforced(mut self, enforced: bool) -> Self {
        self.enforced = enforced;
        self
    }

    /// Add a per-field sensitivity override.
    #[must_use]
    pub fn with_field_sensitivity(mut self, field: impl Into<String>, sensitivity: f64) -> Self {
        self.field_sensitivities.insert(field.into(), sensitivity);
        self
    }
}

/// Registry of per-collection privacy policies.
#[derive(Debug, Default)]
pub struct PrivacyPolicyRegistry {
    policies: HashMap<String, CollectionPrivacyPolicy>,
    budget: PrivacyBudget,
    /// Tracks which sessions have queried each collection (for compliance reports).
    collection_sessions: HashMap<String, std::collections::HashSet<String>>,
}

impl PrivacyPolicyRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a privacy policy for a collection.
    pub fn register(&mut self, policy: CollectionPrivacyPolicy) {
        self.policies.insert(policy.collection_name.clone(), policy);
    }

    /// Remove a privacy policy for a collection.
    pub fn unregister(&mut self, collection_name: &str) -> Option<CollectionPrivacyPolicy> {
        self.policies.remove(collection_name)
    }

    /// Get the policy for a collection.
    pub fn get_policy(&self, collection_name: &str) -> Option<&CollectionPrivacyPolicy> {
        self.policies.get(collection_name)
    }

    /// List all registered policies.
    pub fn list_policies(&self) -> Vec<&CollectionPrivacyPolicy> {
        self.policies.values().collect()
    }

    /// Compute the composed epsilon for k queries using the specified theorem.
    pub fn composed_epsilon(
        &self,
        single_epsilon: f64,
        delta: f64,
        k: usize,
        theorem: CompositionTheorem,
    ) -> f64 {
        let k_f64 = k as f64;
        match theorem {
            CompositionTheorem::Basic => single_epsilon * k_f64,
            CompositionTheorem::Advanced => {
                // Advanced composition: ε_total = √(2k·ln(1/δ'))·ε + k·ε·(e^ε - 1)
                let delta_prime = delta / 2.0;
                let term1 = (2.0 * k_f64 * (1.0 / delta_prime).ln()).sqrt() * single_epsilon;
                let term2 = k_f64 * single_epsilon * (single_epsilon.exp() - 1.0);
                term1 + term2
            }
            CompositionTheorem::Renyi => {
                // Simplified Rényi composition: tighter than advanced for small ε
                let alpha = 1.0 + 1.0 / (single_epsilon + 1e-10);
                let rdp_epsilon = k_f64 * single_epsilon.powi(2) / (2.0 * (alpha - 1.0));
                rdp_epsilon + (1.0 / delta).ln() / (alpha - 1.0)
            }
        }
    }

    /// Apply privacy to search results for a collection. Returns modified results
    /// with perturbed distances, or an error if the privacy budget is exhausted.
    pub fn apply_privacy(
        &mut self,
        collection_name: &str,
        session_id: &str,
        results: Vec<SearchResult>,
    ) -> Result<Vec<SearchResult>> {
        let policy = match self.policies.get(collection_name) {
            Some(p) => p.clone(),
            None => return Ok(results),
        };

        if policy.enforced
            && !self.budget.has_budget(
                session_id,
                policy.config.epsilon,
                policy.config.max_budget_per_session,
            )
        {
            return Err(NeedleError::QuotaExceeded(format!(
                "Privacy budget exhausted for session '{}' on collection '{}'",
                session_id, collection_name
            )));
        }

        self.budget.consume(
            session_id,
            policy.config.epsilon,
            policy.config.max_budget_per_session,
        );

        // Track which sessions queried which collections
        self.collection_sessions
            .entry(collection_name.to_string())
            .or_default()
            .insert(session_id.to_string());

        let mechanism = PrivacyMechanism::new(policy.config.clone());
        let mut perturbed_results = results;
        for result in &mut perturbed_results {
            result.distance = mechanism.perturb_distance(result.distance, policy.config.sensitivity);
        }
        perturbed_results.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(perturbed_results)
    }

    /// Get the budget tracker.
    pub fn budget(&self) -> &PrivacyBudget {
        &self.budget
    }

    /// Get a mutable reference to the budget tracker.
    pub fn budget_mut(&mut self) -> &mut PrivacyBudget {
        &mut self.budget
    }

    /// Generate a compliance report for all registered policies.
    /// Only sessions that actually queried each collection are included.
    pub fn compliance_report(&self) -> PrivacyComplianceReport {
        let mut policy_summaries = Vec::new();
        for (name, policy) in &self.policies {
            let relevant_sessions = self.collection_sessions.get(name);
            let all_sessions = self.budget.summary();
            let collection_sessions: Vec<_> = all_sessions
                .iter()
                .filter(|(s, _)| {
                    relevant_sessions.map_or(false, |rs| rs.contains(s))
                })
                .map(|(s, e)| SessionBudgetSummary {
                    session_id: s.clone(),
                    epsilon_consumed: *e,
                    remaining: self.budget.remaining(s, policy.config.max_budget_per_session),
                })
                .collect();

            policy_summaries.push(PolicyComplianceSummary {
                collection_name: name.clone(),
                epsilon: policy.config.epsilon,
                delta: policy.config.delta,
                mechanism: format!("{:?}", policy.config.mechanism),
                sensitivity: policy.config.sensitivity,
                max_budget: policy.config.max_budget_per_session,
                enforced: policy.enforced,
                composition: format!("{:?}", policy.composition),
                sessions: collection_sessions,
            });
        }

        PrivacyComplianceReport {
            generated_at: chrono::Utc::now().to_rfc3339(),
            total_queries: self.budget.total_queries(),
            total_policies: self.policies.len(),
            policies: policy_summaries,
        }
    }
}

/// Privacy compliance report for audit purposes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyComplianceReport {
    /// ISO 8601 timestamp of report generation.
    pub generated_at: String,
    /// Total queries processed across all sessions.
    pub total_queries: u64,
    /// Number of registered privacy policies.
    pub total_policies: usize,
    /// Per-policy compliance summaries.
    pub policies: Vec<PolicyComplianceSummary>,
}

impl PrivacyComplianceReport {
    /// Export report as JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }

    /// Export report as human-readable text.
    pub fn to_text(&self) -> String {
        let mut s = String::new();
        s.push_str("═══ Privacy Compliance Report ═══\n");
        s.push_str(&format!("Generated: {}\n", self.generated_at));
        s.push_str(&format!("Total queries: {}\n", self.total_queries));
        s.push_str(&format!("Policies: {}\n\n", self.total_policies));

        for p in &self.policies {
            s.push_str(&format!("Collection: {}\n", p.collection_name));
            s.push_str(&format!("  ε={}, δ={}, mechanism={}\n", p.epsilon, p.delta, p.mechanism));
            s.push_str(&format!("  sensitivity={}, max_budget={}, enforced={}\n",
                               p.sensitivity, p.max_budget, p.enforced));
            s.push_str(&format!("  composition={}\n", p.composition));
            if !p.sessions.is_empty() {
                s.push_str("  Sessions:\n");
                for sess in &p.sessions {
                    s.push_str(&format!("    {}: consumed={:.2}, remaining={:.2}\n",
                                       sess.session_id, sess.epsilon_consumed, sess.remaining));
                }
            }
            s.push('\n');
        }
        s
    }
}

/// Per-policy compliance summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyComplianceSummary {
    /// Collection this policy applies to.
    pub collection_name: String,
    /// Epsilon parameter.
    pub epsilon: f64,
    /// Delta parameter.
    pub delta: f64,
    /// Noise mechanism used.
    pub mechanism: String,
    /// Distance sensitivity.
    pub sensitivity: f64,
    /// Maximum budget per session.
    pub max_budget: f64,
    /// Whether enforcement is active.
    pub enforced: bool,
    /// Composition theorem used.
    pub composition: String,
    /// Per-session budget usage.
    pub sessions: Vec<SessionBudgetSummary>,
}

/// Budget summary for a single session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionBudgetSummary {
    /// Session identifier.
    pub session_id: String,
    /// Total epsilon consumed.
    pub epsilon_consumed: f64,
    /// Remaining budget.
    pub remaining: f64,
}

/// Audit log entry for privacy-related operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyAuditEntry {
    /// Timestamp of the operation.
    pub timestamp: String,
    /// Session ID.
    pub session_id: String,
    /// Collection name.
    pub collection_name: String,
    /// Operation type.
    pub operation: String,
    /// Epsilon consumed.
    pub epsilon_consumed: f64,
    /// Remaining budget after operation.
    pub remaining_budget: f64,
    /// Number of results returned.
    pub result_count: usize,
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_laplace_perturbation() {
        let config = PrivacyConfig::new(1.0, 1e-5);
        let mechanism = PrivacyMechanism::new(config);

        let original = 0.5f32;
        let perturbed = mechanism.perturb_distance(original, 1.0);
        // Should be different (with very high probability)
        // But within reasonable range
        assert!(perturbed >= 0.0);
    }

    #[test]
    fn test_gaussian_perturbation() {
        let config = PrivacyConfig::new(1.0, 1e-5).with_mechanism(NoiseMechanism::Gaussian);
        let mechanism = PrivacyMechanism::new(config);

        let original = 0.5f32;
        let perturbed = mechanism.perturb_distance(original, 1.0);
        assert!(perturbed >= 0.0);
    }

    #[test]
    fn test_privacy_budget() {
        let mut budget = PrivacyBudget::new();
        let max = 5.0;

        assert!(budget.has_budget("session1", 1.0, max));
        assert!(budget.consume("session1", 2.0, max));
        assert!(budget.consume("session1", 2.0, max));
        assert!(!budget.consume("session1", 2.0, max)); // would exceed 5.0

        assert_eq!(budget.remaining("session1", max), 1.0);

        budget.reset("session1");
        assert_eq!(budget.remaining("session1", max), 5.0);
    }

    #[test]
    fn test_privacy_explain() {
        let config = PrivacyConfig::new(1.0, 1e-5);
        let mechanism = PrivacyMechanism::new(config);
        let explain = mechanism.explain();

        assert_eq!(explain.epsilon, 1.0);
        assert_eq!(explain.mechanism, "Laplace");
        assert!(explain.noise_scale > 0.0);
    }

    #[test]
    fn test_batch_perturbation() {
        let config = PrivacyConfig::new(1.0, 1e-5);
        let mechanism = PrivacyMechanism::new(config);

        let mut distances = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        mechanism.perturb_distances(&mut distances, 1.0);
        // All should be non-negative
        for d in &distances {
            assert!(*d >= 0.0);
        }
    }

    #[test]
    fn test_composition_basic() {
        let registry = PrivacyPolicyRegistry::new();
        let eps = registry.composed_epsilon(1.0, 1e-5, 10, CompositionTheorem::Basic);
        assert!((eps - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_composition_advanced() {
        let registry = PrivacyPolicyRegistry::new();
        // Use small ε and large k where advanced composition is provably tighter
        let eps = registry.composed_epsilon(0.1, 1e-5, 100, CompositionTheorem::Advanced);
        let basic = registry.composed_epsilon(0.1, 1e-5, 100, CompositionTheorem::Basic);
        assert!(eps < basic, "advanced ({eps}) should be < basic ({basic}) for small ε, large k");
    }

    #[test]
    fn test_policy_registry() {
        let mut registry = PrivacyPolicyRegistry::new();
        let policy = CollectionPrivacyPolicy::new(
            "my_collection",
            PrivacyConfig::new(0.5, 1e-5),
        )
        .with_composition(CompositionTheorem::Advanced)
        .with_enforced(true);

        registry.register(policy);
        assert!(registry.get_policy("my_collection").is_some());
        assert!(registry.get_policy("other").is_none());
        assert_eq!(registry.list_policies().len(), 1);
    }

    #[test]
    fn test_apply_privacy_no_policy() {
        let mut registry = PrivacyPolicyRegistry::new();
        let results = vec![SearchResult {
            id: "v1".to_string(),
            distance: 0.5,
            metadata: None,
        }];
        let applied = registry.apply_privacy("unregistered", "s1", results.clone()).unwrap();
        assert_eq!(applied[0].distance, results[0].distance);
    }

    #[test]
    fn test_apply_privacy_budget_exhaustion() {
        let mut registry = PrivacyPolicyRegistry::new();
        let policy = CollectionPrivacyPolicy::new(
            "coll",
            PrivacyConfig::new(5.0, 1e-5).with_max_budget(6.0),
        );
        registry.register(policy);

        let results = vec![SearchResult {
            id: "v1".to_string(),
            distance: 0.5,
            metadata: None,
        }];
        // First query: consumes 5.0 of 6.0 budget
        assert!(registry.apply_privacy("coll", "s1", results.clone()).is_ok());
        // Second query: would exceed budget (5.0 + 5.0 > 6.0)
        assert!(registry.apply_privacy("coll", "s1", results).is_err());
    }

    #[test]
    fn test_compliance_report_generation() {
        let mut registry = PrivacyPolicyRegistry::new();
        let policy_docs = CollectionPrivacyPolicy::new(
            "docs",
            PrivacyConfig::new(1.0, 1e-5).with_max_budget(10.0),
        );
        let policy_images = CollectionPrivacyPolicy::new(
            "images",
            PrivacyConfig::new(2.0, 1e-5).with_max_budget(20.0),
        );
        registry.register(policy_docs);
        registry.register(policy_images);

        let results = vec![SearchResult {
            id: "v1".to_string(),
            distance: 0.5,
            metadata: None,
        }];
        // user-1 queries docs, user-2 queries images
        registry.apply_privacy("docs", "user-1", results.clone()).unwrap();
        registry.apply_privacy("docs", "user-1", results.clone()).unwrap();
        registry.apply_privacy("images", "user-2", results).unwrap();

        let report = registry.compliance_report();
        assert_eq!(report.total_queries, 3);
        assert_eq!(report.total_policies, 2);

        // Find each policy's report
        let docs_report = report.policies.iter().find(|p| p.collection_name == "docs").unwrap();
        let images_report = report.policies.iter().find(|p| p.collection_name == "images").unwrap();

        // docs should only show user-1 (not user-2)
        assert_eq!(docs_report.sessions.len(), 1);
        assert_eq!(docs_report.sessions[0].session_id, "user-1");

        // images should only show user-2 (not user-1)
        assert_eq!(images_report.sessions.len(), 1);
        assert_eq!(images_report.sessions[0].session_id, "user-2");

        let text = report.to_text();
        assert!(text.contains("Privacy Compliance Report"));
        let json = report.to_json();
        assert!(json.contains("total_queries"));
    }

    #[test]
    fn test_privacy_audit_entry() {
        let entry = PrivacyAuditEntry {
            timestamp: "2025-01-01T00:00:00Z".to_string(),
            session_id: "s1".to_string(),
            collection_name: "docs".to_string(),
            operation: "search".to_string(),
            epsilon_consumed: 1.0,
            remaining_budget: 9.0,
            result_count: 5,
        };
        let json = serde_json::to_string(&entry).unwrap();
        assert!(json.contains("s1"));
        assert!(json.contains("search"));
    }
}
