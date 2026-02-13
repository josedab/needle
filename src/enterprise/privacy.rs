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

#[cfg(test)]
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
}
