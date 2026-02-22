//! Embedded Drift Monitor
//!
//! Continuous distribution monitoring of vector embeddings with statistical
//! tests for drift detection, alerting, and re-index recommendations.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::services::drift_monitor::{
//!     DriftMonitor, DriftConfig, DriftAlert,
//! };
//!
//! let mut monitor = DriftMonitor::new(DriftConfig::new(4));
//!
//! // Establish baseline from existing vectors
//! monitor.add_baseline(&[1.0, 0.0, 0.0, 0.0]);
//! monitor.add_baseline(&[0.0, 1.0, 0.0, 0.0]);
//! monitor.seal_baseline();
//!
//! // Monitor new vectors
//! monitor.observe(&[1.0, 0.0, 0.0, 0.0]); // similar to baseline
//! monitor.observe(&[0.0, 0.0, 0.0, 1.0]); // different direction
//!
//! let report = monitor.check_drift();
//! println!("Drift score: {:.4}", report.drift_score);
//! ```

use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::error::{NeedleError, Result};

// ── Configuration ────────────────────────────────────────────────────────────

/// Drift monitor configuration.
#[derive(Debug, Clone)]
pub struct DriftConfig {
    /// Vector dimensions.
    pub dimensions: usize,
    /// Alert threshold (KL-divergence approximation).
    pub alert_threshold: f32,
    /// Minimum observations before checking drift.
    pub min_observations: usize,
    /// Window size for rolling statistics.
    pub window_size: usize,
}

impl DriftConfig {
    /// Create config for given dimensions.
    pub fn new(dimensions: usize) -> Self {
        Self {
            dimensions,
            alert_threshold: 0.5,
            min_observations: 100,
            window_size: 1000,
        }
    }

    /// Set alert threshold.
    #[must_use]
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.alert_threshold = threshold;
        self
    }

    /// Set minimum observations.
    #[must_use]
    pub fn with_min_observations(mut self, min: usize) -> Self {
        self.min_observations = min;
        self
    }
}

// ── Distribution Statistics ──────────────────────────────────────────────────

/// Running statistics for a dimension.
#[derive(Debug, Clone, Default)]
struct DimStats {
    count: u64,
    mean: f64,
    m2: f64, // for Welford's online variance
    min: f64,
    max: f64,
}

impl DimStats {
    fn update(&mut self, value: f64) {
        self.count += 1;
        if self.count == 1 {
            self.min = value;
            self.max = value;
        }
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
        self.min = self.min.min(value);
        self.max = self.max.max(value);
    }

    fn variance(&self) -> f64 {
        if self.count < 2 { 0.0 } else { self.m2 / (self.count - 1) as f64 }
    }

    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
}

// ── Drift Report ─────────────────────────────────────────────────────────────

/// Report from a drift check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftReport {
    /// Overall drift score (0 = no drift, higher = more drift).
    pub drift_score: f32,
    /// Whether drift exceeds the alert threshold.
    pub alert: bool,
    /// Per-dimension drift scores.
    pub dimension_scores: Vec<f32>,
    /// Number of baseline observations.
    pub baseline_count: u64,
    /// Number of current observations.
    pub current_count: u64,
    /// Recommendation.
    pub recommendation: DriftRecommendation,
    /// Timestamp of check.
    pub timestamp: u64,
}

/// Drift-based recommendation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DriftRecommendation {
    /// No action needed.
    NoAction,
    /// Consider monitoring more closely.
    Monitor,
    /// Re-index recommended.
    ReIndex,
    /// Model update recommended.
    UpdateModel,
}

// ── Drift Alert ──────────────────────────────────────────────────────────────

/// An alert triggered by drift detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftAlert {
    /// Drift score that triggered the alert.
    pub score: f32,
    /// Dimensions with highest drift.
    pub top_dimensions: Vec<(usize, f32)>,
    /// Timestamp.
    pub timestamp: u64,
}

// ── Drift Monitor ────────────────────────────────────────────────────────────

/// Embedded drift monitor with running statistics.
pub struct DriftMonitor {
    config: DriftConfig,
    baseline: Vec<DimStats>,
    current: Vec<DimStats>,
    baseline_sealed: bool,
    alerts: Vec<DriftAlert>,
}

impl DriftMonitor {
    /// Create a new drift monitor.
    pub fn new(config: DriftConfig) -> Self {
        let dim = config.dimensions;
        Self {
            config,
            baseline: vec![DimStats::default(); dim],
            current: vec![DimStats::default(); dim],
            baseline_sealed: false,
            alerts: Vec::new(),
        }
    }

    /// Add a vector to the baseline distribution.
    pub fn add_baseline(&mut self, vector: &[f32]) {
        if self.baseline_sealed { return; }
        for (i, &v) in vector.iter().enumerate().take(self.config.dimensions) {
            self.baseline[i].update(v as f64);
        }
    }

    /// Seal the baseline (no more baseline additions).
    pub fn seal_baseline(&mut self) {
        self.baseline_sealed = true;
    }

    /// Observe a new vector (adds to current distribution).
    pub fn observe(&mut self, vector: &[f32]) {
        for (i, &v) in vector.iter().enumerate().take(self.config.dimensions) {
            self.current[i].update(v as f64);
        }
    }

    /// Check for drift between baseline and current distributions.
    pub fn check_drift(&mut self) -> DriftReport {
        let baseline_count = self.baseline[0].count;
        let current_count = self.current[0].count;

        let mut dim_scores = Vec::with_capacity(self.config.dimensions);
        let mut total_drift: f64 = 0.0;

        for i in 0..self.config.dimensions {
            let b = &self.baseline[i];
            let c = &self.current[i];

            // Approximate KL-divergence using mean/variance comparison
            let score = if b.std_dev() > f64::EPSILON && c.std_dev() > f64::EPSILON {
                let mean_diff = (c.mean - b.mean).powi(2);
                let var_ratio = (c.variance() / b.variance()).ln().abs();
                (mean_diff / b.variance().max(f64::EPSILON) + var_ratio) as f32
            } else if b.count == 0 || c.count == 0 {
                0.0
            } else {
                (c.mean - b.mean).abs() as f32
            };
            dim_scores.push(score);
            total_drift += score as f64;
        }

        let drift_score = (total_drift / self.config.dimensions.max(1) as f64) as f32;
        let alert = drift_score > self.config.alert_threshold
            && current_count >= self.config.min_observations as u64;

        let recommendation = if drift_score < 0.1 {
            DriftRecommendation::NoAction
        } else if drift_score < self.config.alert_threshold {
            DriftRecommendation::Monitor
        } else if drift_score < self.config.alert_threshold * 2.0 {
            DriftRecommendation::ReIndex
        } else {
            DriftRecommendation::UpdateModel
        };

        if alert {
            let mut top_dims: Vec<(usize, f32)> = dim_scores.iter().enumerate()
                .map(|(i, &s)| (i, s)).collect();
            top_dims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            top_dims.truncate(5);

            self.alerts.push(DriftAlert {
                score: drift_score,
                top_dimensions: top_dims,
                timestamp: now_secs(),
            });
        }

        DriftReport {
            drift_score,
            alert,
            dimension_scores: dim_scores,
            baseline_count,
            current_count,
            recommendation,
            timestamp: now_secs(),
        }
    }

    /// Get historical alerts.
    pub fn alerts(&self) -> &[DriftAlert] {
        &self.alerts
    }

    /// Reset current observations (keep baseline).
    pub fn reset_current(&mut self) {
        self.current = vec![DimStats::default(); self.config.dimensions];
    }

    /// Reset everything.
    pub fn reset_all(&mut self) {
        self.baseline = vec![DimStats::default(); self.config.dimensions];
        self.current = vec![DimStats::default(); self.config.dimensions];
        self.baseline_sealed = false;
        self.alerts.clear();
    }

    /// Baseline observation count.
    pub fn baseline_count(&self) -> u64 {
        self.baseline.first().map_or(0, |d| d.count)
    }

    /// Current observation count.
    pub fn current_count(&self) -> u64 {
        self.current.first().map_or(0, |d| d.count)
    }
}

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_drift() {
        let mut mon = DriftMonitor::new(DriftConfig::new(4).with_min_observations(2));
        for _ in 0..10 {
            mon.add_baseline(&[1.0, 0.0, 0.0, 0.0]);
        }
        mon.seal_baseline();
        for _ in 0..10 {
            mon.observe(&[1.0, 0.0, 0.0, 0.0]);
        }
        let report = mon.check_drift();
        assert!(report.drift_score < 0.1);
        assert!(!report.alert);
    }

    #[test]
    fn test_drift_detected() {
        let mut mon = DriftMonitor::new(DriftConfig::new(4).with_threshold(0.1).with_min_observations(5));
        for _ in 0..20 {
            mon.add_baseline(&[1.0, 0.0, 0.0, 0.0]);
        }
        mon.seal_baseline();
        for _ in 0..20 {
            mon.observe(&[0.0, 0.0, 0.0, 1.0]); // completely different
        }
        let report = mon.check_drift();
        assert!(report.drift_score > 0.1);
        assert!(report.alert);
    }

    #[test]
    fn test_recommendation_levels() {
        let mut mon = DriftMonitor::new(DriftConfig::new(2).with_threshold(0.5).with_min_observations(1));
        mon.add_baseline(&[0.0, 0.0]);
        mon.seal_baseline();
        mon.observe(&[0.0, 0.0]);

        let report = mon.check_drift();
        assert!(matches!(report.recommendation, DriftRecommendation::NoAction));
    }

    #[test]
    fn test_reset() {
        let mut mon = DriftMonitor::new(DriftConfig::new(4));
        mon.add_baseline(&[1.0; 4]);
        mon.seal_baseline();
        mon.observe(&[2.0; 4]);

        assert_eq!(mon.current_count(), 1);
        mon.reset_current();
        assert_eq!(mon.current_count(), 0);
        assert_eq!(mon.baseline_count(), 1);

        mon.reset_all();
        assert_eq!(mon.baseline_count(), 0);
    }

    #[test]
    fn test_alert_history() {
        let mut mon = DriftMonitor::new(DriftConfig::new(2).with_threshold(0.01).with_min_observations(1));
        for _ in 0..5 { mon.add_baseline(&[1.0, 0.0]); }
        mon.seal_baseline();
        for _ in 0..5 { mon.observe(&[-1.0, 0.0]); }
        mon.check_drift();
        assert!(!mon.alerts().is_empty());
    }

    #[test]
    fn test_dimension_scores() {
        let mut mon = DriftMonitor::new(DriftConfig::new(4));
        for _ in 0..10 { mon.add_baseline(&[1.0, 0.0, 0.0, 0.0]); }
        mon.seal_baseline();
        for _ in 0..10 { mon.observe(&[1.0, 0.0, 0.0, 1.0]); } // only dim 3 changes
        let report = mon.check_drift();
        assert_eq!(report.dimension_scores.len(), 4);
    }
}
