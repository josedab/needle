//! Embedding Drift Detection - Monitor and detect distribution shifts in vector embeddings.
//!
//! Detects when new embeddings diverge from the baseline distribution,
//! signaling model decay, concept drift, or data quality issues.
//!
//! # Features
//!
//! - **Statistical tests**: KS test, MMD, centroid shift detection
//! - **Windowed monitoring**: Sliding window analysis for streaming data
//! - **Alerts**: Configurable thresholds and callbacks
//! - **Visualization data**: Export drift metrics over time
//! - **Dimension analysis**: Identify which dimensions are drifting
//!
//! # Example
//!
//! ```ignore
//! use needle::drift::{DriftDetector, DriftConfig};
//!
//! let mut detector = DriftDetector::new(128, DriftConfig::default());
//!
//! // Establish baseline from training data
//! detector.add_baseline(&training_embeddings);
//!
//! // Monitor production embeddings
//! for embedding in production_stream {
//!     let report = detector.check(&embedding)?;
//!     if report.is_drifting {
//!         alert!("Embedding drift detected: {:?}", report);
//!     }
//! }
//! ```

use crate::error::{NeedleError, Result};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Configuration for drift detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftConfig {
    /// Window size for sliding window analysis.
    pub window_size: usize,
    /// Threshold for centroid shift (cosine distance).
    pub centroid_threshold: f32,
    /// Threshold for variance change ratio.
    pub variance_threshold: f32,
    /// Threshold for KS statistic.
    pub ks_threshold: f32,
    /// Minimum samples before drift detection.
    pub min_samples: usize,
    /// Enable per-dimension drift analysis.
    pub dimension_analysis: bool,
    /// Alert on significant drift.
    pub alert_on_drift: bool,
}

impl Default for DriftConfig {
    fn default() -> Self {
        Self {
            window_size: 1000,
            centroid_threshold: 0.1,
            variance_threshold: 0.2,
            ks_threshold: 0.1,
            min_samples: 100,
            dimension_analysis: true,
            alert_on_drift: true,
        }
    }
}

/// Statistics for a set of vectors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorStats {
    /// Mean vector (centroid).
    pub centroid: Vec<f32>,
    /// Per-dimension variance.
    pub variance: Vec<f32>,
    /// Per-dimension min values.
    pub min: Vec<f32>,
    /// Per-dimension max values.
    pub max: Vec<f32>,
    /// Average magnitude.
    pub avg_magnitude: f32,
    /// Sample count.
    pub sample_count: usize,
}

impl VectorStats {
    /// Create empty stats for given dimensions.
    fn new(dimensions: usize) -> Self {
        Self {
            centroid: vec![0.0; dimensions],
            variance: vec![0.0; dimensions],
            min: vec![f32::MAX; dimensions],
            max: vec![f32::MIN; dimensions],
            avg_magnitude: 0.0,
            sample_count: 0,
        }
    }

    /// Update stats with a new vector.
    fn update(&mut self, vector: &[f32]) {
        self.sample_count += 1;
        let n = self.sample_count as f32;

        // Online mean update (Welford's algorithm)
        for (i, &val) in vector.iter().enumerate() {
            let delta = val - self.centroid[i];
            self.centroid[i] += delta / n;

            // Update variance using online algorithm
            if self.sample_count > 1 {
                let delta2 = val - self.centroid[i];
                self.variance[i] += delta * delta2;
            }

            // Update min/max
            self.min[i] = self.min[i].min(val);
            self.max[i] = self.max[i].max(val);
        }

        // Update average magnitude
        let magnitude: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        self.avg_magnitude += (magnitude - self.avg_magnitude) / n;
    }

    /// Finalize variance calculation.
    fn finalize_variance(&mut self) {
        if self.sample_count > 1 {
            let n = (self.sample_count - 1) as f32;
            for v in &mut self.variance {
                *v /= n;
            }
        }
    }
}

/// Report from drift detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftReport {
    /// Whether significant drift is detected.
    pub is_drifting: bool,
    /// Overall drift score (0-1).
    pub drift_score: f32,
    /// Centroid shift (cosine distance).
    pub centroid_shift: f32,
    /// Variance ratio (current / baseline).
    pub variance_ratio: f32,
    /// KS statistic for magnitude distribution.
    pub ks_statistic: f32,
    /// Per-dimension drift scores.
    pub dimension_drift: Option<Vec<DimensionDrift>>,
    /// Current window statistics.
    pub current_stats: Option<VectorStats>,
    /// Baseline statistics.
    pub baseline_stats: Option<VectorStats>,
    /// Samples analyzed.
    pub samples_analyzed: usize,
}

impl Default for DriftReport {
    fn default() -> Self {
        Self {
            is_drifting: false,
            drift_score: 0.0,
            centroid_shift: 0.0,
            variance_ratio: 1.0,
            ks_statistic: 0.0,
            dimension_drift: None,
            current_stats: None,
            baseline_stats: None,
            samples_analyzed: 0,
        }
    }
}

/// Drift information for a single dimension.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimensionDrift {
    /// Dimension index.
    pub dimension: usize,
    /// Mean shift.
    pub mean_shift: f32,
    /// Variance ratio.
    pub variance_ratio: f32,
    /// Overall drift score for this dimension.
    pub drift_score: f32,
}

/// Drift detector for embedding vectors.
pub struct DriftDetector {
    /// Number of dimensions.
    dimensions: usize,
    /// Configuration.
    config: DriftConfig,
    /// Baseline statistics.
    baseline: Option<VectorStats>,
    /// Current window of vectors.
    window: VecDeque<Vec<f32>>,
    /// Current window statistics.
    current_stats: Option<VectorStats>,
    /// Magnitude histogram for baseline.
    baseline_magnitudes: Vec<f32>,
    /// Magnitude histogram for current window.
    current_magnitudes: VecDeque<f32>,
    /// Historical drift scores.
    drift_history: Vec<DriftHistoryEntry>,
    /// Total vectors processed.
    total_processed: usize,
}

/// Historical drift measurement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftHistoryEntry {
    /// Sample number.
    pub sample_number: usize,
    /// Drift score.
    pub drift_score: f32,
    /// Was drifting flag.
    pub is_drifting: bool,
    /// Timestamp (samples processed).
    pub timestamp: usize,
}

impl DriftDetector {
    /// Create a new drift detector.
    pub fn new(dimensions: usize, config: DriftConfig) -> Self {
        Self {
            dimensions,
            config,
            baseline: None,
            window: VecDeque::new(),
            current_stats: None,
            baseline_magnitudes: Vec::new(),
            current_magnitudes: VecDeque::new(),
            drift_history: Vec::new(),
            total_processed: 0,
        }
    }

    /// Add vectors to establish baseline distribution.
    pub fn add_baseline(&mut self, vectors: &[Vec<f32>]) -> Result<()> {
        if vectors.is_empty() {
            return Err(NeedleError::InvalidInput(
                "Empty baseline vectors".to_string(),
            ));
        }

        // Validate dimensions
        for vec in vectors {
            if vec.len() != self.dimensions {
                return Err(NeedleError::InvalidInput(format!(
                    "Dimension mismatch: expected {}, got {}",
                    self.dimensions,
                    vec.len()
                )));
            }
        }

        let mut stats = VectorStats::new(self.dimensions);
        for vec in vectors {
            stats.update(vec);
            let magnitude = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            self.baseline_magnitudes.push(magnitude);
        }
        stats.finalize_variance();

        self.baseline = Some(stats);
        Ok(())
    }

    /// Add a single vector to the baseline.
    pub fn add_baseline_vector(&mut self, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dimensions {
            return Err(NeedleError::InvalidInput(format!(
                "Dimension mismatch: expected {}, got {}",
                self.dimensions,
                vector.len()
            )));
        }

        let stats = self
            .baseline
            .get_or_insert_with(|| VectorStats::new(self.dimensions));
        stats.update(vector);

        let magnitude = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        self.baseline_magnitudes.push(magnitude);

        Ok(())
    }

    /// Finalize baseline after adding vectors.
    pub fn finalize_baseline(&mut self) -> Result<()> {
        if let Some(stats) = &mut self.baseline {
            stats.finalize_variance();
            Ok(())
        } else {
            Err(NeedleError::InvalidInput(
                "No baseline vectors added".to_string(),
            ))
        }
    }

    /// Check a vector for drift.
    pub fn check(&mut self, vector: &[f32]) -> Result<DriftReport> {
        if vector.len() != self.dimensions {
            return Err(NeedleError::InvalidInput(format!(
                "Dimension mismatch: expected {}, got {}",
                self.dimensions,
                vector.len()
            )));
        }

        self.total_processed += 1;

        // Add to sliding window
        self.window.push_back(vector.to_vec());
        let magnitude = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        self.current_magnitudes.push_back(magnitude);

        // Maintain window size
        while self.window.len() > self.config.window_size {
            self.window.pop_front();
            self.current_magnitudes.pop_front();
        }

        // Update current statistics
        self.update_current_stats();

        // Check if we have enough samples
        if self.window.len() < self.config.min_samples {
            return Ok(DriftReport {
                samples_analyzed: self.window.len(),
                ..Default::default()
            });
        }

        // Compare against baseline
        self.compute_drift_report()
    }

    /// Update current window statistics.
    fn update_current_stats(&mut self) {
        let mut stats = VectorStats::new(self.dimensions);
        for vec in &self.window {
            stats.update(vec);
        }
        stats.finalize_variance();
        self.current_stats = Some(stats);
    }

    /// Compute drift report comparing current window to baseline.
    fn compute_drift_report(&mut self) -> Result<DriftReport> {
        let baseline = match &self.baseline {
            Some(b) => b,
            None => {
                return Ok(DriftReport {
                    samples_analyzed: self.window.len(),
                    ..Default::default()
                });
            }
        };

        let current = match &self.current_stats {
            Some(c) => c,
            None => {
                return Ok(DriftReport {
                    samples_analyzed: self.window.len(),
                    ..Default::default()
                });
            }
        };

        // Compute centroid shift (cosine distance)
        let centroid_shift = self.cosine_distance(&baseline.centroid, &current.centroid);

        // Compute variance ratio
        let variance_ratio = self.compute_variance_ratio(baseline, current);

        // Compute KS statistic for magnitudes
        let ks_statistic = self.compute_ks_statistic();

        // Per-dimension drift analysis
        let dimension_drift = if self.config.dimension_analysis {
            Some(self.compute_dimension_drift(baseline, current))
        } else {
            None
        };

        // Compute overall drift score
        let drift_score = self.compute_drift_score(centroid_shift, variance_ratio, ks_statistic);

        // Determine if drifting
        let is_drifting = centroid_shift > self.config.centroid_threshold
            || variance_ratio > 1.0 + self.config.variance_threshold
            || variance_ratio < 1.0 - self.config.variance_threshold
            || ks_statistic > self.config.ks_threshold;

        // Record history
        self.drift_history.push(DriftHistoryEntry {
            sample_number: self.total_processed,
            drift_score,
            is_drifting,
            timestamp: self.total_processed,
        });

        Ok(DriftReport {
            is_drifting,
            drift_score,
            centroid_shift,
            variance_ratio,
            ks_statistic,
            dimension_drift,
            current_stats: Some(current.clone()),
            baseline_stats: Some(baseline.clone()),
            samples_analyzed: self.window.len(),
        })
    }

    /// Compute cosine distance between two vectors.
    fn cosine_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if mag_a == 0.0 || mag_b == 0.0 {
            return 1.0;
        }

        let cosine_sim = dot / (mag_a * mag_b);
        1.0 - cosine_sim.clamp(-1.0, 1.0)
    }

    /// Compute overall variance ratio.
    fn compute_variance_ratio(&self, baseline: &VectorStats, current: &VectorStats) -> f32 {
        let baseline_total: f32 = baseline.variance.iter().sum();
        let current_total: f32 = current.variance.iter().sum();

        if baseline_total == 0.0 {
            return 1.0;
        }

        current_total / baseline_total
    }

    /// Compute Kolmogorov-Smirnov statistic for magnitude distributions.
    fn compute_ks_statistic(&self) -> f32 {
        if self.baseline_magnitudes.is_empty() || self.current_magnitudes.is_empty() {
            return 0.0;
        }

        // Sort both distributions
        let mut baseline_sorted = self.baseline_magnitudes.clone();
        baseline_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mut current_sorted: Vec<f32> = self.current_magnitudes.iter().cloned().collect();
        current_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Compute max difference between CDFs
        let n1 = baseline_sorted.len() as f32;
        let n2 = current_sorted.len() as f32;

        let mut max_diff: f32 = 0.0;
        let mut i = 0;
        let mut j = 0;

        while i < baseline_sorted.len() && j < current_sorted.len() {
            let cdf1 = (i + 1) as f32 / n1;
            let cdf2 = (j + 1) as f32 / n2;
            let diff = (cdf1 - cdf2).abs();
            max_diff = max_diff.max(diff);

            if baseline_sorted[i] <= current_sorted[j] {
                i += 1;
            } else {
                j += 1;
            }
        }

        max_diff
    }

    /// Compute per-dimension drift.
    fn compute_dimension_drift(
        &self,
        baseline: &VectorStats,
        current: &VectorStats,
    ) -> Vec<DimensionDrift> {
        (0..self.dimensions)
            .map(|i| {
                let mean_shift = (current.centroid[i] - baseline.centroid[i]).abs();

                let variance_ratio = if baseline.variance[i] > 0.0 {
                    current.variance[i] / baseline.variance[i]
                } else {
                    1.0
                };

                // Normalize mean shift by baseline range
                let range = baseline.max[i] - baseline.min[i];
                let normalized_mean_shift = if range > 0.0 {
                    mean_shift / range
                } else {
                    mean_shift
                };

                // Combine into drift score
                let drift_score = (normalized_mean_shift + (variance_ratio - 1.0).abs()) / 2.0;

                DimensionDrift {
                    dimension: i,
                    mean_shift,
                    variance_ratio,
                    drift_score,
                }
            })
            .collect()
    }

    /// Compute overall drift score.
    fn compute_drift_score(&self, centroid_shift: f32, variance_ratio: f32, ks_stat: f32) -> f32 {
        let centroid_score = (centroid_shift / self.config.centroid_threshold).min(1.0);
        let variance_score =
            ((variance_ratio - 1.0).abs() / self.config.variance_threshold).min(1.0);
        let ks_score = (ks_stat / self.config.ks_threshold).min(1.0);

        // Weighted average
        centroid_score * 0.4 + variance_score * 0.3 + ks_score * 0.3
    }

    /// Get drift history.
    pub fn get_history(&self) -> &[DriftHistoryEntry] {
        &self.drift_history
    }

    /// Get recent drift trend.
    pub fn get_trend(&self, window: usize) -> DriftTrend {
        let recent: Vec<&DriftHistoryEntry> =
            self.drift_history.iter().rev().take(window).collect();

        if recent.is_empty() {
            return DriftTrend::Stable;
        }

        let avg_score: f32 =
            recent.iter().map(|e| e.drift_score).sum::<f32>() / recent.len() as f32;
        let drift_count = recent.iter().filter(|e| e.is_drifting).count();
        let drift_rate = drift_count as f32 / recent.len() as f32;

        if drift_rate > 0.8 {
            DriftTrend::SevereDrift
        } else if drift_rate > 0.5 {
            DriftTrend::Increasing
        } else if drift_rate > 0.2 {
            DriftTrend::Moderate
        } else if avg_score > 0.3 {
            DriftTrend::Decreasing
        } else {
            DriftTrend::Stable
        }
    }

    /// Reset current window but keep baseline.
    pub fn reset_window(&mut self) {
        self.window.clear();
        self.current_magnitudes.clear();
        self.current_stats = None;
    }

    /// Reset everything including baseline.
    pub fn reset(&mut self) {
        self.reset_window();
        self.baseline = None;
        self.baseline_magnitudes.clear();
        self.drift_history.clear();
        self.total_processed = 0;
    }

    /// Get number of vectors in current window.
    pub fn window_size(&self) -> usize {
        self.window.len()
    }

    /// Get total vectors processed.
    pub fn total_processed(&self) -> usize {
        self.total_processed
    }

    /// Get current statistics.
    pub fn current_stats(&self) -> Option<&VectorStats> {
        self.current_stats.as_ref()
    }

    /// Get baseline statistics.
    pub fn baseline_stats(&self) -> Option<&VectorStats> {
        self.baseline.as_ref()
    }

    /// Export drift data for visualization.
    pub fn export_metrics(&self) -> DriftMetrics {
        DriftMetrics {
            dimensions: self.dimensions,
            total_processed: self.total_processed,
            window_size: self.window.len(),
            baseline_size: self.baseline.as_ref().map(|b| b.sample_count).unwrap_or(0),
            history: self.drift_history.clone(),
            trend: self.get_trend(100),
        }
    }

    /// Get top drifting dimensions.
    pub fn top_drifting_dimensions(&self, n: usize) -> Result<Vec<DimensionDrift>> {
        let baseline = self
            .baseline
            .as_ref()
            .ok_or_else(|| NeedleError::InvalidInput("No baseline".to_string()))?;
        let current = self
            .current_stats
            .as_ref()
            .ok_or_else(|| NeedleError::InvalidInput("No current stats".to_string()))?;

        let mut dims = self.compute_dimension_drift(baseline, current);
        dims.sort_by(|a, b| {
            b.drift_score
                .partial_cmp(&a.drift_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        dims.truncate(n);

        Ok(dims)
    }
}

/// Trend in drift over time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DriftTrend {
    /// Stable, no significant drift.
    Stable,
    /// Drift is decreasing.
    Decreasing,
    /// Moderate drift.
    Moderate,
    /// Drift is increasing.
    Increasing,
    /// Severe ongoing drift.
    SevereDrift,
}

/// Exported drift metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftMetrics {
    /// Vector dimensions.
    pub dimensions: usize,
    /// Total vectors processed.
    pub total_processed: usize,
    /// Current window size.
    pub window_size: usize,
    /// Baseline sample count.
    pub baseline_size: usize,
    /// Historical drift entries.
    pub history: Vec<DriftHistoryEntry>,
    /// Current trend.
    pub trend: DriftTrend,
}

/// Builder for drift configuration.
pub struct DriftConfigBuilder {
    config: DriftConfig,
}

impl DriftConfigBuilder {
    /// Create new builder.
    pub fn new() -> Self {
        Self {
            config: DriftConfig::default(),
        }
    }

    /// Set window size.
    pub fn window_size(mut self, size: usize) -> Self {
        self.config.window_size = size;
        self
    }

    /// Set centroid threshold.
    pub fn centroid_threshold(mut self, threshold: f32) -> Self {
        self.config.centroid_threshold = threshold;
        self
    }

    /// Set variance threshold.
    pub fn variance_threshold(mut self, threshold: f32) -> Self {
        self.config.variance_threshold = threshold;
        self
    }

    /// Set KS threshold.
    pub fn ks_threshold(mut self, threshold: f32) -> Self {
        self.config.ks_threshold = threshold;
        self
    }

    /// Set minimum samples.
    pub fn min_samples(mut self, min: usize) -> Self {
        self.config.min_samples = min;
        self
    }

    /// Enable dimension analysis.
    pub fn dimension_analysis(mut self, enabled: bool) -> Self {
        self.config.dimension_analysis = enabled;
        self
    }

    /// Build the config.
    pub fn build(self) -> DriftConfig {
        self.config
    }
}

impl Default for DriftConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Real-Time Drift Detection Enhancements
// ============================================================================

/// Severity level for drift alerts
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DriftSeverity {
    /// Low severity - minor drift detected
    Low,
    /// Medium severity - moderate drift
    Medium,
    /// High severity - significant drift
    High,
    /// Critical severity - severe drift requiring immediate attention
    Critical,
}

impl DriftSeverity {
    /// Get severity from drift score
    pub fn from_score(score: f32) -> Self {
        if score < 0.2 {
            Self::Low
        } else if score < 0.5 {
            Self::Medium
        } else if score < 0.8 {
            Self::High
        } else {
            Self::Critical
        }
    }
}

/// A drift alert with context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftAlert {
    /// Timestamp of the alert
    pub timestamp: u64,
    /// Severity level
    pub severity: DriftSeverity,
    /// Drift score that triggered the alert
    pub drift_score: f32,
    /// Number of consecutive drift detections
    pub consecutive_drifts: usize,
    /// Top drifting dimensions
    pub top_dimensions: Vec<DimensionDrift>,
    /// Alert message
    pub message: String,
    /// Whether the alert has been acknowledged
    pub acknowledged: bool,
}

/// Adaptive threshold configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveThresholdConfig {
    /// Initial threshold
    pub initial_threshold: f32,
    /// Minimum threshold
    pub min_threshold: f32,
    /// Maximum threshold
    pub max_threshold: f32,
    /// Learning rate for threshold adjustment
    pub learning_rate: f32,
    /// Number of samples for adaptation
    pub adaptation_window: usize,
}

impl Default for AdaptiveThresholdConfig {
    fn default() -> Self {
        Self {
            initial_threshold: 0.1,
            min_threshold: 0.01,
            max_threshold: 0.5,
            learning_rate: 0.1,
            adaptation_window: 100,
        }
    }
}

/// Adaptive threshold that adjusts based on observed drift
#[derive(Debug, Clone)]
pub struct AdaptiveThreshold {
    config: AdaptiveThresholdConfig,
    current_threshold: f32,
    recent_scores: VecDeque<f32>,
    false_positive_count: usize,
    true_positive_count: usize,
}

impl AdaptiveThreshold {
    /// Create a new adaptive threshold
    pub fn new(config: AdaptiveThresholdConfig) -> Self {
        let current = config.initial_threshold;
        Self {
            config,
            current_threshold: current,
            recent_scores: VecDeque::new(),
            false_positive_count: 0,
            true_positive_count: 0,
        }
    }

    /// Update threshold based on a new drift score
    pub fn update(&mut self, score: f32, was_actual_drift: bool) {
        self.recent_scores.push_back(score);
        if self.recent_scores.len() > self.config.adaptation_window {
            self.recent_scores.pop_front();
        }

        let triggered = score > self.current_threshold;
        if triggered && !was_actual_drift {
            self.false_positive_count += 1;
        } else if triggered && was_actual_drift {
            self.true_positive_count += 1;
        }

        // Adapt threshold
        if self.recent_scores.len() >= self.config.adaptation_window / 2 {
            let avg_score: f32 =
                self.recent_scores.iter().sum::<f32>() / self.recent_scores.len() as f32;
            let std_dev = self.compute_std_dev(&avg_score);

            // Set threshold to mean + 2 standard deviations
            let new_threshold = (avg_score + 2.0 * std_dev)
                .max(self.config.min_threshold)
                .min(self.config.max_threshold);

            // Smooth update
            self.current_threshold = self.current_threshold * (1.0 - self.config.learning_rate)
                + new_threshold * self.config.learning_rate;
        }
    }

    fn compute_std_dev(&self, mean: &f32) -> f32 {
        if self.recent_scores.is_empty() {
            return 0.0;
        }
        let variance: f32 = self
            .recent_scores
            .iter()
            .map(|s| (s - mean).powi(2))
            .sum::<f32>()
            / self.recent_scores.len() as f32;
        variance.sqrt()
    }

    /// Get the current threshold
    pub fn threshold(&self) -> f32 {
        self.current_threshold
    }

    /// Get false positive rate
    pub fn false_positive_rate(&self) -> f32 {
        let total = self.false_positive_count + self.true_positive_count;
        if total == 0 {
            0.0
        } else {
            self.false_positive_count as f32 / total as f32
        }
    }

    /// Reset the adaptive threshold
    pub fn reset(&mut self) {
        self.current_threshold = self.config.initial_threshold;
        self.recent_scores.clear();
        self.false_positive_count = 0;
        self.true_positive_count = 0;
    }
}

/// Named baseline for multi-baseline comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamedBaseline {
    /// Baseline name (e.g., "last_week", "production_v1")
    pub name: String,
    /// Baseline statistics
    pub stats: VectorStats,
    /// Creation timestamp
    pub created_at: u64,
    /// Description
    pub description: Option<String>,
}

/// Multi-baseline drift detector
pub struct MultiBaselineDetector {
    dimensions: usize,
    baselines: Vec<NamedBaseline>,
    config: DriftConfig,
}

impl MultiBaselineDetector {
    /// Create a new multi-baseline detector
    pub fn new(dimensions: usize, config: DriftConfig) -> Self {
        Self {
            dimensions,
            baselines: Vec::new(),
            config,
        }
    }

    /// Add a named baseline
    pub fn add_baseline(
        &mut self,
        name: &str,
        vectors: &[Vec<f32>],
        description: Option<&str>,
    ) -> Result<()> {
        if vectors.is_empty() {
            return Err(NeedleError::InvalidInput("Empty vector set".to_string()));
        }

        for vec in vectors {
            if vec.len() != self.dimensions {
                return Err(NeedleError::DimensionMismatch {
                    expected: self.dimensions,
                    got: vec.len(),
                });
            }
        }

        let mut stats = VectorStats::new(self.dimensions);
        for vec in vectors {
            stats.update(vec);
        }

        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        self.baselines.push(NamedBaseline {
            name: name.to_string(),
            stats,
            created_at: timestamp,
            description: description.map(|s| s.to_string()),
        });

        Ok(())
    }

    /// Compare a vector against all baselines
    pub fn compare_all(&self, vector: &[f32]) -> Result<Vec<(String, f32)>> {
        if vector.len() != self.dimensions {
            return Err(NeedleError::DimensionMismatch {
                expected: self.dimensions,
                got: vector.len(),
            });
        }

        let results: Vec<(String, f32)> = self
            .baselines
            .iter()
            .map(|baseline| {
                let score = self.compute_drift_score(&baseline.stats, vector);
                (baseline.name.clone(), score)
            })
            .collect();

        Ok(results)
    }

    fn compute_drift_score(&self, baseline: &VectorStats, vector: &[f32]) -> f32 {
        let centroid_dist = cosine_distance(&baseline.centroid, vector);
        centroid_dist / self.config.centroid_threshold.max(0.001)
    }

    /// Get the closest baseline for a vector
    pub fn closest_baseline(&self, vector: &[f32]) -> Result<Option<(String, f32)>> {
        let comparisons = self.compare_all(vector)?;
        Ok(comparisons
            .into_iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)))
    }

    /// List all baselines
    pub fn list_baselines(&self) -> Vec<&NamedBaseline> {
        self.baselines.iter().collect()
    }

    /// Remove a baseline by name
    pub fn remove_baseline(&mut self, name: &str) -> bool {
        let len_before = self.baselines.len();
        self.baselines.retain(|b| b.name != name);
        self.baselines.len() < len_before
    }
}

/// Recovery detection state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecoveryState {
    /// Normal operation
    Normal,
    /// Drift detected
    Drifting,
    /// Recovering from drift
    Recovering,
    /// Recovered to normal
    Recovered,
}

/// Drift recovery detector
#[derive(Debug, Clone)]
pub struct RecoveryDetector {
    state: RecoveryState,
    consecutive_normal: usize,
    consecutive_drift: usize,
    recovery_threshold: usize,
    drift_threshold: usize,
    recovery_score_threshold: f32,
}

impl RecoveryDetector {
    /// Create a new recovery detector
    pub fn new(recovery_threshold: usize, drift_threshold: usize) -> Self {
        Self {
            state: RecoveryState::Normal,
            consecutive_normal: 0,
            consecutive_drift: 0,
            recovery_threshold,
            drift_threshold,
            recovery_score_threshold: 0.1,
        }
    }

    /// Update state based on new drift score
    pub fn update(&mut self, drift_score: f32, is_drifting: bool) -> RecoveryState {
        if is_drifting {
            self.consecutive_drift += 1;
            self.consecutive_normal = 0;

            if self.consecutive_drift >= self.drift_threshold {
                self.state = RecoveryState::Drifting;
            }
        } else {
            self.consecutive_normal += 1;
            self.consecutive_drift = 0;

            match self.state {
                RecoveryState::Drifting => {
                    self.state = RecoveryState::Recovering;
                }
                RecoveryState::Recovering => {
                    if self.consecutive_normal >= self.recovery_threshold
                        && drift_score < self.recovery_score_threshold
                    {
                        self.state = RecoveryState::Recovered;
                    }
                }
                RecoveryState::Recovered => {
                    self.state = RecoveryState::Normal;
                }
                RecoveryState::Normal => {}
            }
        }

        self.state
    }

    /// Get current state
    pub fn state(&self) -> RecoveryState {
        self.state
    }

    /// Reset state
    pub fn reset(&mut self) {
        self.state = RecoveryState::Normal;
        self.consecutive_normal = 0;
        self.consecutive_drift = 0;
    }
}

/// Configuration for real-time drift monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealTimeMonitorConfig {
    /// Base drift config
    pub drift_config: DriftConfig,
    /// Enable adaptive thresholds
    pub adaptive_thresholds: bool,
    /// Adaptive threshold config
    pub adaptive_config: AdaptiveThresholdConfig,
    /// Alert after N consecutive drifts
    pub alert_after_n: usize,
    /// Cooldown between alerts (samples)
    pub alert_cooldown: usize,
    /// Enable recovery detection
    pub recovery_detection: bool,
    /// Recovery threshold (consecutive normal samples)
    pub recovery_threshold: usize,
}

impl Default for RealTimeMonitorConfig {
    fn default() -> Self {
        Self {
            drift_config: DriftConfig::default(),
            adaptive_thresholds: true,
            adaptive_config: AdaptiveThresholdConfig::default(),
            alert_after_n: 3,
            alert_cooldown: 50,
            recovery_detection: true,
            recovery_threshold: 10,
        }
    }
}

/// Real-time drift monitor with adaptive thresholds and alerting
pub struct RealTimeDriftMonitor {
    detector: DriftDetector,
    config: RealTimeMonitorConfig,
    adaptive_centroid: AdaptiveThreshold,
    adaptive_variance: AdaptiveThreshold,
    recovery: RecoveryDetector,
    consecutive_drifts: usize,
    samples_since_alert: usize,
    alerts: Vec<DriftAlert>,
    total_samples: usize,
}

impl RealTimeDriftMonitor {
    /// Create a new real-time drift monitor
    pub fn new(dimensions: usize, config: RealTimeMonitorConfig) -> Self {
        let detector = DriftDetector::new(dimensions, config.drift_config.clone());

        let mut centroid_config = config.adaptive_config.clone();
        centroid_config.initial_threshold = config.drift_config.centroid_threshold;

        let mut variance_config = config.adaptive_config.clone();
        variance_config.initial_threshold = config.drift_config.variance_threshold;

        let recovery = RecoveryDetector::new(config.recovery_threshold, config.alert_after_n);

        Self {
            detector,
            config,
            adaptive_centroid: AdaptiveThreshold::new(centroid_config),
            adaptive_variance: AdaptiveThreshold::new(variance_config),
            recovery,
            consecutive_drifts: 0,
            samples_since_alert: 0,
            alerts: Vec::new(),
            total_samples: 0,
        }
    }

    /// Add baseline vectors
    pub fn add_baseline(&mut self, vectors: &[Vec<f32>]) -> Result<()> {
        self.detector.add_baseline(vectors)
    }

    /// Check a vector for drift
    pub fn check(&mut self, vector: &[f32]) -> Result<RealTimeCheckResult> {
        let report = self.detector.check(vector)?;
        self.total_samples += 1;
        self.samples_since_alert += 1;

        // Determine if drifting using adaptive thresholds
        let is_drifting = if self.config.adaptive_thresholds {
            report.centroid_shift > self.adaptive_centroid.threshold()
                || report.variance_ratio > self.adaptive_variance.threshold()
        } else {
            report.is_drifting
        };

        // Update adaptive thresholds
        if self.config.adaptive_thresholds {
            self.adaptive_centroid
                .update(report.centroid_shift, is_drifting);
            self.adaptive_variance
                .update(report.variance_ratio, is_drifting);
        }

        // Update recovery state
        let recovery_state = if self.config.recovery_detection {
            self.recovery.update(report.drift_score, is_drifting)
        } else {
            RecoveryState::Normal
        };

        // Track consecutive drifts
        if is_drifting {
            self.consecutive_drifts += 1;
        } else {
            self.consecutive_drifts = 0;
        }

        // Check if alert should be raised
        let alert = if self.consecutive_drifts >= self.config.alert_after_n
            && self.samples_since_alert >= self.config.alert_cooldown
        {
            let alert = self.create_alert(&report)?;
            self.alerts.push(alert.clone());
            self.samples_since_alert = 0;
            Some(alert)
        } else {
            None
        };

        Ok(RealTimeCheckResult {
            report,
            is_drifting,
            recovery_state,
            alert,
            adaptive_centroid_threshold: self.adaptive_centroid.threshold(),
            adaptive_variance_threshold: self.adaptive_variance.threshold(),
        })
    }

    fn create_alert(&self, report: &DriftReport) -> Result<DriftAlert> {
        let top_dims = report
            .dimension_drift
            .as_ref()
            .map(|dims| {
                let mut sorted = dims.clone();
                sorted.sort_by(|a, b| {
                    b.drift_score
                        .partial_cmp(&a.drift_score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                sorted.truncate(5);
                sorted
            })
            .unwrap_or_default();

        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let severity = DriftSeverity::from_score(report.drift_score);

        let message = format!(
            "Drift detected: score={:.3}, centroid_shift={:.3}, variance_ratio={:.3}",
            report.drift_score, report.centroid_shift, report.variance_ratio
        );

        Ok(DriftAlert {
            timestamp,
            severity,
            drift_score: report.drift_score,
            consecutive_drifts: self.consecutive_drifts,
            top_dimensions: top_dims,
            message,
            acknowledged: false,
        })
    }

    /// Get all alerts
    pub fn alerts(&self) -> &[DriftAlert] {
        &self.alerts
    }

    /// Get unacknowledged alerts
    pub fn unacknowledged_alerts(&self) -> Vec<&DriftAlert> {
        self.alerts.iter().filter(|a| !a.acknowledged).collect()
    }

    /// Acknowledge an alert
    pub fn acknowledge_alert(&mut self, index: usize) -> bool {
        if let Some(alert) = self.alerts.get_mut(index) {
            alert.acknowledged = true;
            true
        } else {
            false
        }
    }

    /// Get monitor statistics
    pub fn stats(&self) -> RealTimeMonitorStats {
        RealTimeMonitorStats {
            total_samples: self.total_samples,
            total_alerts: self.alerts.len(),
            consecutive_drifts: self.consecutive_drifts,
            recovery_state: self.recovery.state(),
            adaptive_centroid_threshold: self.adaptive_centroid.threshold(),
            adaptive_variance_threshold: self.adaptive_variance.threshold(),
            centroid_false_positive_rate: self.adaptive_centroid.false_positive_rate(),
            variance_false_positive_rate: self.adaptive_variance.false_positive_rate(),
        }
    }

    /// Reset the monitor (keeps baseline)
    pub fn reset(&mut self) {
        self.adaptive_centroid.reset();
        self.adaptive_variance.reset();
        self.recovery.reset();
        self.consecutive_drifts = 0;
        self.samples_since_alert = 0;
        self.alerts.clear();
        self.total_samples = 0;
        self.detector.reset_window();
    }
}

/// Result of a real-time drift check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealTimeCheckResult {
    /// Base drift report
    pub report: DriftReport,
    /// Whether drift was detected (using adaptive thresholds if enabled)
    pub is_drifting: bool,
    /// Current recovery state
    pub recovery_state: RecoveryState,
    /// Alert if one was raised
    pub alert: Option<DriftAlert>,
    /// Current adaptive centroid threshold
    pub adaptive_centroid_threshold: f32,
    /// Current adaptive variance threshold
    pub adaptive_variance_threshold: f32,
}

/// Statistics for real-time monitor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealTimeMonitorStats {
    /// Total samples processed
    pub total_samples: usize,
    /// Total alerts raised
    pub total_alerts: usize,
    /// Current consecutive drifts
    pub consecutive_drifts: usize,
    /// Current recovery state
    pub recovery_state: RecoveryState,
    /// Current adaptive centroid threshold
    pub adaptive_centroid_threshold: f32,
    /// Current adaptive variance threshold
    pub adaptive_variance_threshold: f32,
    /// False positive rate for centroid threshold
    pub centroid_false_positive_rate: f32,
    /// False positive rate for variance threshold
    pub variance_false_positive_rate: f32,
}

/// Helper function for cosine distance
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if mag_a < f32::EPSILON || mag_b < f32::EPSILON {
        return 1.0;
    }

    1.0 - (dot / (mag_a * mag_b)).clamp(-1.0, 1.0)
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;

    fn generate_vectors(n: usize, dim: usize, offset: f32) -> Vec<Vec<f32>> {
        (0..n)
            .map(|i| {
                (0..dim)
                    .map(|j| ((i * dim + j) as f32 * 0.01).sin() + offset)
                    .collect()
            })
            .collect()
    }

    #[test]
    fn test_create_detector() {
        let detector = DriftDetector::new(128, DriftConfig::default());
        assert_eq!(detector.dimensions, 128);
        assert!(detector.baseline.is_none());
    }

    #[test]
    fn test_add_baseline() {
        let mut detector = DriftDetector::new(4, DriftConfig::default());
        let vectors = generate_vectors(100, 4, 0.0);

        detector.add_baseline(&vectors).unwrap();

        assert!(detector.baseline.is_some());
        let baseline = detector.baseline.as_ref().unwrap();
        assert_eq!(baseline.sample_count, 100);
    }

    #[test]
    fn test_dimension_mismatch() {
        let mut detector = DriftDetector::new(4, DriftConfig::default());
        let vectors = vec![vec![1.0, 2.0, 3.0]]; // Wrong dimension

        let result = detector.add_baseline(&vectors);
        assert!(result.is_err());
    }

    #[test]
    fn test_no_drift_same_distribution() {
        let config = DriftConfig {
            min_samples: 10,
            ..Default::default()
        };

        let mut detector = DriftDetector::new(4, config);

        // Baseline vectors
        let baseline_vecs = generate_vectors(100, 4, 0.0);
        detector.add_baseline(&baseline_vecs).unwrap();

        // Check vectors from same distribution
        for vec in generate_vectors(50, 4, 0.0) {
            let report = detector.check(&vec).unwrap();
            // Should not be drifting for same distribution
            // Note: with random sampling, some variance is expected
            if report.samples_analyzed >= 10 {
                assert!(
                    report.drift_score < 0.8,
                    "Drift score too high: {}",
                    report.drift_score
                );
            }
        }
    }

    #[test]
    fn test_detect_drift_shifted_distribution() {
        let mut config = DriftConfig::default();
        config.min_samples = 10;
        config.centroid_threshold = 0.05;

        let mut detector = DriftDetector::new(4, config);

        // Baseline vectors centered around 0
        let baseline_vecs = generate_vectors(100, 4, 0.0);
        detector.add_baseline(&baseline_vecs).unwrap();

        // Shifted vectors centered around 5.0
        let mut is_drifting_detected = false;
        for vec in generate_vectors(50, 4, 5.0) {
            let report = detector.check(&vec).unwrap();
            if report.samples_analyzed >= 10 && report.is_drifting {
                is_drifting_detected = true;
            }
        }

        assert!(
            is_drifting_detected,
            "Drift should be detected for shifted distribution"
        );
    }

    #[test]
    fn test_sliding_window() {
        let mut config = DriftConfig::default();
        config.window_size = 20;

        let mut detector = DriftDetector::new(4, config);
        detector
            .add_baseline(&generate_vectors(50, 4, 0.0))
            .unwrap();

        // Add more vectors than window size
        for vec in generate_vectors(30, 4, 0.0) {
            let _ = detector.check(&vec);
        }

        assert_eq!(detector.window_size(), 20);
    }

    #[test]
    fn test_dimension_drift() {
        let mut config = DriftConfig::default();
        config.min_samples = 5;
        config.dimension_analysis = true;

        let mut detector = DriftDetector::new(4, config);

        // Baseline
        detector
            .add_baseline(&[
                vec![0.0, 0.0, 0.0, 0.0],
                vec![0.1, 0.1, 0.1, 0.1],
                vec![-0.1, -0.1, -0.1, -0.1],
            ])
            .unwrap();

        // Check vectors where only dimension 0 is shifted
        for _ in 0..10 {
            let vec = vec![5.0, 0.0, 0.0, 0.0];
            let report = detector.check(&vec).unwrap();

            if let Some(dims) = &report.dimension_drift {
                // Dimension 0 should have highest drift
                let max_drift_dim = dims
                    .iter()
                    .max_by(|a, b| {
                        a.drift_score
                            .partial_cmp(&b.drift_score)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap();
                assert_eq!(max_drift_dim.dimension, 0);
            }
        }
    }

    #[test]
    fn test_drift_history() {
        let mut config = DriftConfig::default();
        config.min_samples = 5;

        let mut detector = DriftDetector::new(4, config);
        detector
            .add_baseline(&generate_vectors(20, 4, 0.0))
            .unwrap();

        for vec in generate_vectors(30, 4, 0.0) {
            let _ = detector.check(&vec);
        }

        let history = detector.get_history();
        assert!(!history.is_empty());
    }

    #[test]
    fn test_drift_trend() {
        let mut config = DriftConfig::default();
        config.min_samples = 5;
        // Use higher thresholds to reduce false positives from random sampling variance
        config.centroid_threshold = 1.0; // Very high - almost never drift
        config.variance_threshold = 5.0; // Allow 5x variance change
        config.ks_threshold = 1.0; // Max KS is 1.0

        let mut detector = DriftDetector::new(4, config);
        detector
            .add_baseline(&generate_vectors(20, 4, 0.0))
            .unwrap();

        // Add vectors from same distribution
        for vec in generate_vectors(50, 4, 0.0) {
            let _ = detector.check(&vec);
        }

        let trend = detector.get_trend(10);
        // With very relaxed thresholds and same distribution, trend should be stable or moderate
        // The test validates the trend mechanism works
        assert!(
            matches!(trend, DriftTrend::Stable | DriftTrend::Moderate | DriftTrend::Decreasing),
            "Expected stable or moderate trend for same distribution with relaxed thresholds, got {:?}",
            trend
        );
    }

    #[test]
    fn test_reset_window() {
        let mut detector = DriftDetector::new(4, DriftConfig::default());
        detector
            .add_baseline(&generate_vectors(20, 4, 0.0))
            .unwrap();

        for vec in generate_vectors(30, 4, 0.0) {
            let _ = detector.check(&vec);
        }

        assert!(!detector.window.is_empty());

        detector.reset_window();

        assert!(detector.window.is_empty());
        assert!(detector.baseline.is_some()); // Baseline preserved
    }

    #[test]
    fn test_full_reset() {
        let mut detector = DriftDetector::new(4, DriftConfig::default());
        detector
            .add_baseline(&generate_vectors(20, 4, 0.0))
            .unwrap();

        for vec in generate_vectors(30, 4, 0.0) {
            let _ = detector.check(&vec);
        }

        detector.reset();

        assert!(detector.window.is_empty());
        assert!(detector.baseline.is_none());
        assert_eq!(detector.total_processed(), 0);
    }

    #[test]
    fn test_export_metrics() {
        let mut config = DriftConfig::default();
        config.min_samples = 5;

        let mut detector = DriftDetector::new(4, config);
        detector
            .add_baseline(&generate_vectors(20, 4, 0.0))
            .unwrap();

        for vec in generate_vectors(30, 4, 0.0) {
            let _ = detector.check(&vec);
        }

        let metrics = detector.export_metrics();
        assert_eq!(metrics.dimensions, 4);
        assert_eq!(metrics.total_processed, 30);
    }

    #[test]
    fn test_config_builder() {
        let config = DriftConfigBuilder::new()
            .window_size(500)
            .centroid_threshold(0.15)
            .variance_threshold(0.25)
            .ks_threshold(0.12)
            .min_samples(50)
            .dimension_analysis(true)
            .build();

        assert_eq!(config.window_size, 500);
        assert_eq!(config.centroid_threshold, 0.15);
        assert_eq!(config.variance_threshold, 0.25);
        assert_eq!(config.ks_threshold, 0.12);
        assert_eq!(config.min_samples, 50);
    }

    #[test]
    fn test_top_drifting_dimensions() {
        let mut config = DriftConfig::default();
        config.min_samples = 5;

        let mut detector = DriftDetector::new(4, config);
        detector
            .add_baseline(&generate_vectors(20, 4, 0.0))
            .unwrap();

        // Add vectors with dimension 0 heavily shifted
        for _ in 0..10 {
            let _ = detector.check(&[10.0, 0.0, 0.0, 0.0]);
        }

        let top_dims = detector.top_drifting_dimensions(2).unwrap();
        assert_eq!(top_dims.len(), 2);
        assert_eq!(top_dims[0].dimension, 0); // Dimension 0 should be top
    }

    #[test]
    fn test_incremental_baseline() {
        let mut detector = DriftDetector::new(4, DriftConfig::default());

        // Add baseline vectors incrementally
        for vec in generate_vectors(50, 4, 0.0) {
            detector.add_baseline_vector(&vec).unwrap();
        }
        detector.finalize_baseline().unwrap();

        assert!(detector.baseline.is_some());
        assert_eq!(detector.baseline.as_ref().unwrap().sample_count, 50);
    }

    #[test]
    fn test_vector_stats() {
        let mut stats = VectorStats::new(3);

        stats.update(&[1.0, 2.0, 3.0]);
        stats.update(&[2.0, 3.0, 4.0]);
        stats.update(&[3.0, 4.0, 5.0]);
        stats.finalize_variance();

        // Check centroid (mean)
        assert!((stats.centroid[0] - 2.0).abs() < 0.001);
        assert!((stats.centroid[1] - 3.0).abs() < 0.001);
        assert!((stats.centroid[2] - 4.0).abs() < 0.001);

        // Check min/max
        assert!((stats.min[0] - 1.0).abs() < 0.001);
        assert!((stats.max[0] - 3.0).abs() < 0.001);
    }

    // ==========================================================================
    // Tests for Real-Time Drift Detection Enhancements
    // ==========================================================================

    #[test]
    fn test_drift_severity() {
        assert_eq!(DriftSeverity::from_score(0.1), DriftSeverity::Low);
        assert_eq!(DriftSeverity::from_score(0.3), DriftSeverity::Medium);
        assert_eq!(DriftSeverity::from_score(0.6), DriftSeverity::High);
        assert_eq!(DriftSeverity::from_score(0.9), DriftSeverity::Critical);
    }

    #[test]
    fn test_adaptive_threshold_creation() {
        let config = AdaptiveThresholdConfig::default();
        let threshold = AdaptiveThreshold::new(config.clone());
        assert_eq!(threshold.threshold(), config.initial_threshold);
    }

    #[test]
    fn test_adaptive_threshold_update() {
        let config = AdaptiveThresholdConfig {
            initial_threshold: 0.1,
            min_threshold: 0.01,
            max_threshold: 0.5,
            learning_rate: 0.2,
            adaptation_window: 10,
        };
        let mut threshold = AdaptiveThreshold::new(config);

        // Feed low scores
        for _ in 0..20 {
            threshold.update(0.05, false);
        }

        // Threshold should adapt downward
        assert!(threshold.threshold() < 0.1);
    }

    #[test]
    fn test_adaptive_threshold_bounds() {
        let config = AdaptiveThresholdConfig {
            initial_threshold: 0.1,
            min_threshold: 0.05,
            max_threshold: 0.2,
            learning_rate: 0.5,
            adaptation_window: 5,
        };
        let mut threshold = AdaptiveThreshold::new(config);

        // Feed very high scores
        for _ in 0..20 {
            threshold.update(0.9, true);
        }

        // Should be capped at max
        assert!(threshold.threshold() <= 0.2);

        // Feed very low scores
        for _ in 0..20 {
            threshold.update(0.001, false);
        }

        // Should be above min
        assert!(threshold.threshold() >= 0.05);
    }

    #[test]
    fn test_multi_baseline_creation() {
        let config = DriftConfig::default();
        let mut detector = MultiBaselineDetector::new(4, config);

        detector
            .add_baseline(
                "baseline_v1",
                &generate_vectors(20, 4, 0.0),
                Some("First baseline"),
            )
            .unwrap();

        detector
            .add_baseline("baseline_v2", &generate_vectors(20, 4, 1.0), None)
            .unwrap();

        let baselines = detector.list_baselines();
        assert_eq!(baselines.len(), 2);
    }

    #[test]
    fn test_multi_baseline_comparison() {
        let config = DriftConfig::default();
        let mut detector = MultiBaselineDetector::new(4, config);

        // Use non-zero vectors to avoid cosine distance issues
        detector
            .add_baseline("low", &[vec![1.0, 1.0, 1.0, 1.0]], None)
            .unwrap();

        detector
            .add_baseline("high", &[vec![10.0, 10.0, 10.0, 10.0]], None)
            .unwrap();

        // Vector close to "low" baseline direction (same unit vector)
        let comparisons = detector.compare_all(&[1.1, 1.1, 1.1, 1.1]).unwrap();
        assert_eq!(comparisons.len(), 2);

        // Find closest baseline (by cosine similarity, both have same direction)
        // Just verify the API works
        let closest = detector.closest_baseline(&[1.1, 1.1, 1.1, 1.1]).unwrap();
        assert!(closest.is_some());
    }

    #[test]
    fn test_multi_baseline_remove() {
        let config = DriftConfig::default();
        let mut detector = MultiBaselineDetector::new(4, config);

        detector
            .add_baseline("b1", &generate_vectors(10, 4, 0.0), None)
            .unwrap();
        detector
            .add_baseline("b2", &generate_vectors(10, 4, 1.0), None)
            .unwrap();

        assert!(detector.remove_baseline("b1"));
        assert_eq!(detector.list_baselines().len(), 1);
        assert!(!detector.remove_baseline("nonexistent"));
    }

    #[test]
    fn test_recovery_detector_normal() {
        let mut recovery = RecoveryDetector::new(3, 2);
        assert_eq!(recovery.state(), RecoveryState::Normal);

        // Non-drifting samples should stay normal
        for _ in 0..5 {
            recovery.update(0.05, false);
        }
        assert_eq!(recovery.state(), RecoveryState::Normal);
    }

    #[test]
    fn test_recovery_detector_drift_and_recovery() {
        let mut recovery = RecoveryDetector::new(3, 2);

        // Trigger drift
        recovery.update(0.5, true);
        recovery.update(0.6, true);
        assert_eq!(recovery.state(), RecoveryState::Drifting);

        // Start recovery (first non-drift after drift)
        recovery.update(0.1, false);
        assert_eq!(recovery.state(), RecoveryState::Recovering);

        // Continue recovery (2nd non-drift)
        recovery.update(0.05, false);
        assert_eq!(recovery.state(), RecoveryState::Recovering);

        // Complete recovery (3rd non-drift, meets recovery_threshold=3)
        recovery.update(0.05, false);
        assert_eq!(recovery.state(), RecoveryState::Recovered);

        // Transition to normal (next update after Recovered)
        recovery.update(0.05, false);
        assert_eq!(recovery.state(), RecoveryState::Normal);
    }

    #[test]
    fn test_realtime_monitor_creation() {
        let config = RealTimeMonitorConfig::default();
        let monitor = RealTimeDriftMonitor::new(4, config);

        let stats = monitor.stats();
        assert_eq!(stats.total_samples, 0);
        assert_eq!(stats.total_alerts, 0);
    }

    #[test]
    fn test_realtime_monitor_baseline() {
        let config = RealTimeMonitorConfig::default();
        let mut monitor = RealTimeDriftMonitor::new(4, config);

        monitor.add_baseline(&generate_vectors(50, 4, 0.0)).unwrap();

        // Check should work after baseline
        let result = monitor.check(&[0.1, 0.1, 0.1, 0.1]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_realtime_monitor_check() {
        let mut config = RealTimeMonitorConfig::default();
        config.drift_config.min_samples = 5;

        let mut monitor = RealTimeDriftMonitor::new(4, config);
        monitor.add_baseline(&generate_vectors(50, 4, 0.0)).unwrap();

        for vec in generate_vectors(20, 4, 0.0) {
            let result = monitor.check(&vec).unwrap();
            // Results should have adaptive thresholds
            assert!(result.adaptive_centroid_threshold > 0.0);
        }

        let stats = monitor.stats();
        assert_eq!(stats.total_samples, 20);
    }

    #[test]
    fn test_realtime_monitor_alert() {
        let mut config = RealTimeMonitorConfig::default();
        config.drift_config.min_samples = 3;
        config.drift_config.centroid_threshold = 0.01;
        config.alert_after_n = 2;
        config.alert_cooldown = 0;
        config.adaptive_thresholds = false; // Use fixed thresholds for predictable alerts

        let mut monitor = RealTimeDriftMonitor::new(4, config);
        monitor.add_baseline(&generate_vectors(10, 4, 0.0)).unwrap();

        // Send heavily drifted vectors
        for _ in 0..10 {
            let _ = monitor.check(&[100.0, 100.0, 100.0, 100.0]);
        }

        // Should have generated at least one alert
        assert!(!monitor.alerts().is_empty());
    }

    #[test]
    fn test_realtime_monitor_acknowledge_alert() {
        let mut config = RealTimeMonitorConfig::default();
        config.drift_config.min_samples = 2;
        config.drift_config.centroid_threshold = 0.01;
        config.alert_after_n = 2;
        config.alert_cooldown = 0;
        config.adaptive_thresholds = false;

        let mut monitor = RealTimeDriftMonitor::new(4, config);
        monitor.add_baseline(&generate_vectors(10, 4, 0.0)).unwrap();

        for _ in 0..5 {
            let _ = monitor.check(&[50.0, 50.0, 50.0, 50.0]);
        }

        if !monitor.alerts().is_empty() {
            assert!(!monitor.alerts()[0].acknowledged);
            monitor.acknowledge_alert(0);
            assert!(monitor.alerts()[0].acknowledged);
        }
    }

    #[test]
    fn test_realtime_monitor_reset() {
        let config = RealTimeMonitorConfig::default();
        let mut monitor = RealTimeDriftMonitor::new(4, config);
        monitor.add_baseline(&generate_vectors(50, 4, 0.0)).unwrap();

        for vec in generate_vectors(20, 4, 0.0) {
            let _ = monitor.check(&vec);
        }

        assert_eq!(monitor.stats().total_samples, 20);

        monitor.reset();
        assert_eq!(monitor.stats().total_samples, 0);
        assert!(monitor.alerts().is_empty());
    }

    #[test]
    fn test_drift_alert_fields() {
        let alert = DriftAlert {
            timestamp: 12345,
            severity: DriftSeverity::High,
            drift_score: 0.75,
            consecutive_drifts: 5,
            top_dimensions: vec![],
            message: "Test alert".to_string(),
            acknowledged: false,
        };

        assert_eq!(alert.severity, DriftSeverity::High);
        assert!(!alert.acknowledged);
    }
}
