//! ⚠️ **Experimental**: This module is under active development. APIs may change without notice.
//!
//! # Learned Index Tuning
//!
//! This module provides reinforcement learning and online optimization for
//! dynamic index parameter tuning based on query patterns and performance feedback.
//!
//! ## Features
//!
//! - **Online Learning**: Continuously learns from query feedback
//! - **Multi-Armed Bandit**: Explores parameter configurations efficiently
//! - **Workload Analysis**: Adapts to changing query patterns
//! - **Performance Prediction**: ML-based latency/recall prediction
//! - **Automatic Adjustment**: Dynamic ef_search tuning during queries
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use needle::learned_tuning::{LearnedTuner, TunerConfig, QueryFeedback};
//!
//! // Create a learned tuner
//! let config = TunerConfig::default();
//! let tuner = LearnedTuner::new(config);
//!
//! // Get recommended parameters for a query
//! let params = tuner.recommend_params(100, 0.95)?; // k=100, target_recall=0.95
//!
//! // Provide feedback after query execution
//! tuner.record_feedback(QueryFeedback {
//!     ef_search: params.ef_search,
//!     k: 100,
//!     latency_ms: 2.5,
//!     estimated_recall: 0.97,
//!     ..Default::default()
//! })?;
//! ```

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

use crate::error::Result;

/// Configuration for the learned tuner
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TunerConfig {
    /// Learning rate for parameter updates (0.0-1.0)
    pub learning_rate: f32,
    /// Exploration rate for multi-armed bandit (0.0-1.0)
    pub exploration_rate: f32,
    /// Minimum samples before making predictions
    pub min_samples: usize,
    /// Maximum history size for feedback
    pub max_history: usize,
    /// Enable workload-aware tuning
    pub workload_aware: bool,
    /// Decay factor for older samples (0.0-1.0)
    pub decay_factor: f32,
    /// Target latency in milliseconds
    pub target_latency_ms: Option<f32>,
    /// Target recall (0.0-1.0)
    pub target_recall: Option<f32>,
}

impl Default for TunerConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.1,
            exploration_rate: 0.15,
            min_samples: 50,
            max_history: 10_000,
            workload_aware: true,
            decay_factor: 0.99,
            target_latency_ms: None,
            target_recall: Some(0.95),
        }
    }
}

impl TunerConfig {
    /// Create with specific learning rate
    #[must_use]
    pub fn with_learning_rate(mut self, rate: f32) -> Self {
        self.learning_rate = rate.clamp(0.0, 1.0);
        self
    }

    /// Set exploration rate
    #[must_use]
    pub fn with_exploration_rate(mut self, rate: f32) -> Self {
        self.exploration_rate = rate.clamp(0.0, 1.0);
        self
    }

    /// Set target latency
    #[must_use]
    pub fn with_target_latency(mut self, ms: f32) -> Self {
        self.target_latency_ms = Some(ms);
        self
    }

    /// Set target recall
    #[must_use]
    pub fn with_target_recall(mut self, recall: f32) -> Self {
        self.target_recall = Some(recall.clamp(0.0, 1.0));
        self
    }

    /// Set minimum samples before making predictions
    #[must_use]
    pub fn with_min_samples(mut self, samples: usize) -> Self {
        self.min_samples = samples;
        self
    }
}

/// Query feedback for learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryFeedback {
    /// ef_search value used
    pub ef_search: usize,
    /// Number of results requested (k)
    pub k: usize,
    /// Actual latency in milliseconds
    pub latency_ms: f32,
    /// Estimated recall (if available)
    pub estimated_recall: f32,
    /// Whether the query was satisfied (user-defined)
    pub satisfied: bool,
    /// Query vector norm (for workload analysis)
    pub query_norm: f32,
    /// Filter complexity (number of filter conditions)
    pub filter_complexity: usize,
    /// Timestamp
    pub timestamp: u64,
}

impl Default for QueryFeedback {
    fn default() -> Self {
        Self {
            ef_search: 50,
            k: 10,
            latency_ms: 0.0,
            estimated_recall: 0.0,
            satisfied: true,
            query_norm: 1.0,
            filter_complexity: 0,
            timestamp: current_timestamp(),
        }
    }
}

/// Recommended parameters for a query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecommendedParams {
    /// Recommended ef_search value
    pub ef_search: usize,
    /// Confidence in the recommendation (0.0-1.0)
    pub confidence: f32,
    /// Predicted latency in milliseconds
    pub predicted_latency_ms: f32,
    /// Predicted recall
    pub predicted_recall: f32,
    /// Whether this is an exploratory recommendation
    pub is_exploration: bool,
    /// Reasoning for the recommendation
    pub reasoning: String,
}

/// Arm in the multi-armed bandit for ef_search values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BanditArm {
    /// ef_search value
    pub ef_search: usize,
    /// Total reward accumulated
    pub total_reward: f64,
    /// Number of times selected
    pub pull_count: u64,
    /// Average reward
    pub avg_reward: f64,
    /// Upper confidence bound
    pub ucb: f64,
}

impl BanditArm {
    fn new(ef_search: usize) -> Self {
        Self {
            ef_search,
            total_reward: 0.0,
            pull_count: 0,
            avg_reward: 0.0,
            ucb: f64::MAX, // Initialize high to encourage exploration
        }
    }

    fn update(&mut self, reward: f64, total_pulls: u64) {
        self.pull_count += 1;
        self.total_reward += reward;
        self.avg_reward = self.total_reward / self.pull_count as f64;

        // UCB1 formula
        if total_pulls > 0 && self.pull_count > 0 {
            let exploration_term = (2.0 * (total_pulls as f64).ln() / self.pull_count as f64).sqrt();
            self.ucb = self.avg_reward + exploration_term;
        }
    }
}

/// Workload characteristics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WorkloadProfile {
    /// Average k value
    pub avg_k: f32,
    /// Average filter complexity
    pub avg_filter_complexity: f32,
    /// Average query norm
    pub avg_query_norm: f32,
    /// Query rate (queries per second)
    pub query_rate: f32,
    /// Recall sensitivity (how much users care about recall)
    pub recall_sensitivity: f32,
    /// Latency sensitivity (how much users care about latency)
    pub latency_sensitivity: f32,
}

/// Online learning model for parameter prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnlineModel {
    /// Weights for latency prediction: [bias, ef_search, k, filter_complexity, query_norm]
    pub latency_weights: Vec<f64>,
    /// Weights for recall prediction: [bias, ef_search, k, log_ef_search]
    pub recall_weights: Vec<f64>,
    /// Learning rate
    pub learning_rate: f64,
    /// Number of training samples
    pub sample_count: u64,
}

impl OnlineModel {
    fn new(learning_rate: f64) -> Self {
        Self {
            // Initialize with reasonable defaults
            latency_weights: vec![0.5, 0.02, 0.001, 0.1, 0.01],
            recall_weights: vec![0.5, 0.005, -0.001, 0.1],
            learning_rate,
            sample_count: 0,
        }
    }

    fn predict_latency(&self, ef_search: usize, k: usize, filter_complexity: usize, query_norm: f32) -> f64 {
        let features = vec![
            1.0,                        // bias
            ef_search as f64,           // ef_search
            k as f64,                   // k
            filter_complexity as f64,   // filter complexity
            query_norm as f64,          // query norm
        ];

        features.iter()
            .zip(self.latency_weights.iter())
            .map(|(f, w)| f * w)
            .sum()
    }

    fn predict_recall(&self, ef_search: usize, k: usize) -> f64 {
        let log_ef = (ef_search as f64).ln();
        let features = vec![
            1.0,                // bias
            ef_search as f64,   // ef_search
            k as f64,           // k
            log_ef,             // log(ef_search)
        ];

        let raw: f64 = features.iter()
            .zip(self.recall_weights.iter())
            .map(|(f, w)| f * w)
            .sum();

        // Sigmoid to bound recall between 0 and 1
        1.0 / (1.0 + (-raw).exp())
    }

    fn update(&mut self, feedback: &QueryFeedback) {
        self.sample_count += 1;

        // Update latency model
        let predicted_latency = self.predict_latency(
            feedback.ef_search,
            feedback.k,
            feedback.filter_complexity,
            feedback.query_norm,
        );
        let latency_error = feedback.latency_ms as f64 - predicted_latency;

        let features = vec![
            1.0,
            feedback.ef_search as f64,
            feedback.k as f64,
            feedback.filter_complexity as f64,
            feedback.query_norm as f64,
        ];

        for (i, feature) in features.iter().enumerate() {
            self.latency_weights[i] += self.learning_rate * latency_error * feature;
        }

        // Update recall model (if recall is provided)
        if feedback.estimated_recall > 0.0 {
            let predicted_recall = self.predict_recall(feedback.ef_search, feedback.k);
            let recall_error = feedback.estimated_recall as f64 - predicted_recall;

            let log_ef = (feedback.ef_search as f64).ln();
            let recall_features = vec![
                1.0,
                feedback.ef_search as f64,
                feedback.k as f64,
                log_ef,
            ];

            for (i, feature) in recall_features.iter().enumerate() {
                // Use derivative of sigmoid for logistic regression update
                let gradient = recall_error * predicted_recall * (1.0 - predicted_recall);
                self.recall_weights[i] += self.learning_rate * gradient * feature;
            }
        }
    }
}

/// Statistics from the learned tuner
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedTunerStats {
    /// Total feedback samples collected
    pub total_samples: u64,
    /// Total explorations performed
    pub explorations: u64,
    /// Total exploitations performed
    pub exploitations: u64,
    /// Average observed latency
    pub avg_latency_ms: f32,
    /// Average observed recall
    pub avg_recall: f32,
    /// Current best ef_search value
    pub best_ef_search: usize,
    /// Workload profile
    pub workload: WorkloadProfile,
    /// Model accuracy (RMSE of latency predictions)
    pub latency_rmse: f32,
}

/// Learned index tuner using reinforcement learning
pub struct LearnedTuner {
    config: TunerConfig,
    /// Multi-armed bandit arms for ef_search values
    arms: RwLock<HashMap<usize, BanditArm>>,
    /// Total arm pulls
    total_pulls: AtomicU64,
    /// Feedback history
    history: RwLock<VecDeque<QueryFeedback>>,
    /// Online learning model
    model: RwLock<OnlineModel>,
    /// Workload profile
    workload: RwLock<WorkloadProfile>,
    /// Exploration count
    exploration_count: AtomicU64,
    /// Exploitation count
    exploitation_count: AtomicU64,
    /// Running latency sum for average
    latency_sum: RwLock<f64>,
    /// Running recall sum for average
    recall_sum: RwLock<f64>,
    /// Prediction error sum for RMSE
    prediction_error_sum: RwLock<f64>,
}

impl LearnedTuner {
    /// Create a new learned tuner with the given configuration
    pub fn new(config: TunerConfig) -> Self {
        let mut arms = HashMap::new();

        // Initialize arms for common ef_search values
        for &ef in &[10, 20, 30, 40, 50, 75, 100, 150, 200, 300, 500] {
            arms.insert(ef, BanditArm::new(ef));
        }

        Self {
            model: RwLock::new(OnlineModel::new(config.learning_rate as f64)),
            arms: RwLock::new(arms),
            total_pulls: AtomicU64::new(0),
            history: RwLock::new(VecDeque::with_capacity(config.max_history)),
            workload: RwLock::new(WorkloadProfile::default()),
            exploration_count: AtomicU64::new(0),
            exploitation_count: AtomicU64::new(0),
            latency_sum: RwLock::new(0.0),
            recall_sum: RwLock::new(0.0),
            prediction_error_sum: RwLock::new(0.0),
            config,
        }
    }

    /// Create with default configuration
    pub fn default_tuner() -> Self {
        Self::new(TunerConfig::default())
    }

    /// Recommend parameters for a query
    pub fn recommend_params(&self, k: usize, target_recall: f32) -> Result<RecommendedParams> {
        let total_samples = self.history.read().len();
        let total_pulls = self.total_pulls.load(Ordering::Relaxed);

        // If not enough samples, return default with exploration
        if total_samples < self.config.min_samples {
            return Ok(RecommendedParams {
                ef_search: 50,
                confidence: 0.0,
                predicted_latency_ms: 5.0,
                predicted_recall: 0.9,
                is_exploration: true,
                reasoning: format!(
                    "Not enough samples ({}/{}), using default with exploration",
                    total_samples, self.config.min_samples
                ),
            });
        }

        // Decide whether to explore or exploit
        let explore = rand_float() < self.config.exploration_rate;

        let (ef_search, is_exploration, reasoning) = if explore {
            self.exploration_count.fetch_add(1, Ordering::Relaxed);

            // UCB1 selection for exploration
            let arms = self.arms.read();
            let best_arm = arms.values()
                .max_by(|a, b| a.ucb.partial_cmp(&b.ucb).unwrap_or(std::cmp::Ordering::Equal))
                .map(|a| a.ef_search)
                .unwrap_or(50);

            (best_arm, true, format!("UCB1 exploration, selected ef_search={}", best_arm))
        } else {
            self.exploitation_count.fetch_add(1, Ordering::Relaxed);

            // Use model to find optimal ef_search for target recall
            let model = self.model.read();
            let workload = self.workload.read();

            // Binary search for ef_search that achieves target recall
            let mut best_ef = 50;
            let mut best_score = f64::MIN;

            for &ef_candidate in &[10, 20, 30, 40, 50, 75, 100, 150, 200, 300, 500] {
                let predicted_recall = model.predict_recall(ef_candidate, k);
                let predicted_latency = model.predict_latency(
                    ef_candidate,
                    k,
                    workload.avg_filter_complexity as usize,
                    workload.avg_query_norm,
                );

                // Score: maximize recall while penalizing latency
                let recall_diff = predicted_recall - target_recall as f64;
                let latency_penalty = if let Some(target) = self.config.target_latency_ms {
                    (predicted_latency - target as f64).max(0.0) * 0.1
                } else {
                    predicted_latency * 0.01
                };

                let score = if recall_diff >= 0.0 {
                    // Met recall target, optimize for latency
                    recall_diff - latency_penalty
                } else {
                    // Below recall target, heavily penalize
                    recall_diff * 10.0 - latency_penalty
                };

                if score > best_score {
                    best_score = score;
                    best_ef = ef_candidate;
                }
            }

            (best_ef, false, format!(
                "Model exploitation: predicted optimal ef_search={} for recall={:.2}",
                best_ef, target_recall
            ))
        };

        // Make predictions
        let model = self.model.read();
        let workload = self.workload.read();

        let predicted_latency = model.predict_latency(
            ef_search,
            k,
            workload.avg_filter_complexity as usize,
            workload.avg_query_norm,
        ).max(0.1) as f32;

        let predicted_recall = model.predict_recall(ef_search, k).clamp(0.0, 1.0) as f32;

        // Calculate confidence based on sample count and arm pull count
        let arms = self.arms.read();
        let arm_pulls = arms.get(&ef_search).map(|a| a.pull_count).unwrap_or(0);
        let confidence = if total_pulls > 0 {
            ((arm_pulls as f64 / total_pulls as f64) * (total_samples as f64 / self.config.min_samples as f64).min(1.0))
                .min(1.0) as f32
        } else {
            0.0
        };

        Ok(RecommendedParams {
            ef_search,
            confidence,
            predicted_latency_ms: predicted_latency,
            predicted_recall,
            is_exploration,
            reasoning,
        })
    }

    /// Record feedback from a query execution
    pub fn record_feedback(&self, feedback: QueryFeedback) -> Result<()> {
        // Update model
        {
            let mut model = self.model.write();
            model.update(&feedback);
        }

        // Update bandit arm
        {
            let mut arms = self.arms.write();
            let total_pulls = self.total_pulls.fetch_add(1, Ordering::Relaxed) + 1;

            // Calculate reward: higher for good recall and low latency
            let recall_reward = feedback.estimated_recall as f64;
            let latency_reward = 1.0 / (1.0 + feedback.latency_ms as f64 * 0.1);
            let satisfaction_bonus = if feedback.satisfied { 0.2 } else { 0.0 };
            let reward = recall_reward * 0.6 + latency_reward * 0.3 + satisfaction_bonus;

            // Get or create arm
            let arm = arms.entry(feedback.ef_search).or_insert_with(|| BanditArm::new(feedback.ef_search));
            arm.update(reward, total_pulls);
        }

        // Update workload profile
        {
            let mut workload = self.workload.write();
            let history_len = self.history.read().len() as f32;
            let decay = self.config.decay_factor;

            workload.avg_k = workload.avg_k * decay + feedback.k as f32 * (1.0 - decay);
            workload.avg_filter_complexity = workload.avg_filter_complexity * decay
                + feedback.filter_complexity as f32 * (1.0 - decay);
            workload.avg_query_norm = workload.avg_query_norm * decay
                + feedback.query_norm * (1.0 - decay);

            // Infer sensitivities from satisfaction patterns
            if !feedback.satisfied {
                if feedback.estimated_recall < 0.9 {
                    workload.recall_sensitivity = (workload.recall_sensitivity * 0.9 + 0.1).min(1.0);
                }
                if feedback.latency_ms > 10.0 {
                    workload.latency_sensitivity = (workload.latency_sensitivity * 0.9 + 0.1).min(1.0);
                }
            }

            // Update query rate
            if history_len > 0.0 {
                let oldest = self.history.read().front().map(|f| f.timestamp).unwrap_or(0);
                let newest = feedback.timestamp;
                let duration_secs = ((newest - oldest) as f32 / 1000.0).max(1.0);
                workload.query_rate = history_len / duration_secs;
            }
        }

        // Track prediction error
        {
            let model = self.model.read();
            let predicted = model.predict_latency(
                feedback.ef_search,
                feedback.k,
                feedback.filter_complexity,
                feedback.query_norm,
            );
            let error = (predicted - feedback.latency_ms as f64).powi(2);
            *self.prediction_error_sum.write() += error;
        }

        // Update running averages
        *self.latency_sum.write() += feedback.latency_ms as f64;
        *self.recall_sum.write() += feedback.estimated_recall as f64;

        // Add to history
        {
            let mut history = self.history.write();
            if history.len() >= self.config.max_history {
                history.pop_front();
            }
            history.push_back(feedback);
        }

        debug!(
            samples = self.history.read().len(),
            "Recorded feedback"
        );

        Ok(())
    }

    /// Get current statistics
    pub fn stats(&self) -> LearnedTunerStats {
        let history_len = self.history.read().len() as u64;
        let _total_pulls = self.total_pulls.load(Ordering::Relaxed);

        let avg_latency = if history_len > 0 {
            (*self.latency_sum.read() / history_len as f64) as f32
        } else {
            0.0
        };

        let avg_recall = if history_len > 0 {
            (*self.recall_sum.read() / history_len as f64) as f32
        } else {
            0.0
        };

        let latency_rmse = if history_len > 0 {
            ((*self.prediction_error_sum.read() / history_len as f64).sqrt()) as f32
        } else {
            0.0
        };

        // Find best ef_search by average reward
        let best_ef_search = {
            let arms = self.arms.read();
            arms.values()
                .filter(|a| a.pull_count > 0)
                .max_by(|a, b| a.avg_reward.partial_cmp(&b.avg_reward).unwrap_or(std::cmp::Ordering::Equal))
                .map(|a| a.ef_search)
                .unwrap_or(50)
        };

        LearnedTunerStats {
            total_samples: history_len,
            explorations: self.exploration_count.load(Ordering::Relaxed),
            exploitations: self.exploitation_count.load(Ordering::Relaxed),
            avg_latency_ms: avg_latency,
            avg_recall,
            best_ef_search,
            workload: self.workload.read().clone(),
            latency_rmse,
        }
    }

    /// Reset the tuner, clearing all learned state
    pub fn reset(&self) {
        *self.arms.write() = {
            let mut arms = HashMap::new();
            for &ef in &[10, 20, 30, 40, 50, 75, 100, 150, 200, 300, 500] {
                arms.insert(ef, BanditArm::new(ef));
            }
            arms
        };
        self.total_pulls.store(0, Ordering::Relaxed);
        self.history.write().clear();
        *self.model.write() = OnlineModel::new(self.config.learning_rate as f64);
        *self.workload.write() = WorkloadProfile::default();
        self.exploration_count.store(0, Ordering::Relaxed);
        self.exploitation_count.store(0, Ordering::Relaxed);
        *self.latency_sum.write() = 0.0;
        *self.recall_sum.write() = 0.0;
        *self.prediction_error_sum.write() = 0.0;

        info!("Learned tuner reset");
    }

    /// Export the learned model state for persistence
    pub fn export_state(&self) -> LearnedTunerState {
        LearnedTunerState {
            config: self.config.clone(),
            arms: self.arms.read().clone(),
            total_pulls: self.total_pulls.load(Ordering::Relaxed),
            model: self.model.read().clone(),
            workload: self.workload.read().clone(),
        }
    }

    /// Import a previously exported state
    pub fn import_state(&self, state: LearnedTunerState) {
        *self.arms.write() = state.arms;
        self.total_pulls.store(state.total_pulls, Ordering::Relaxed);
        *self.model.write() = state.model;
        *self.workload.write() = state.workload;

        info!(samples = state.total_pulls, "Imported learned tuner state");
    }
}

/// Serializable state for the learned tuner
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedTunerState {
    /// Configuration
    pub config: TunerConfig,
    /// Bandit arms state
    pub arms: HashMap<usize, BanditArm>,
    /// Total pulls
    pub total_pulls: u64,
    /// Online model state
    pub model: OnlineModel,
    /// Workload profile
    pub workload: WorkloadProfile,
}

/// Adaptive query executor that uses learned tuning
pub struct AdaptiveExecutor {
    tuner: LearnedTuner,
    /// Minimum ef_search to use
    min_ef_search: usize,
    /// Maximum ef_search to use
    max_ef_search: usize,
    /// Whether to automatically record feedback
    auto_feedback: bool,
}

impl AdaptiveExecutor {
    /// Create a new adaptive executor
    pub fn new(tuner: LearnedTuner) -> Self {
        Self {
            tuner,
            min_ef_search: 10,
            max_ef_search: 500,
            auto_feedback: true,
        }
    }

    /// Set ef_search bounds
    #[must_use]
    pub fn with_bounds(mut self, min: usize, max: usize) -> Self {
        self.min_ef_search = min;
        self.max_ef_search = max;
        self
    }

    /// Get recommended ef_search for a query
    pub fn get_ef_search(&self, k: usize, target_recall: f32) -> usize {
        match self.tuner.recommend_params(k, target_recall) {
            Ok(params) => params.ef_search.clamp(self.min_ef_search, self.max_ef_search),
            Err(_) => 50, // Fallback
        }
    }

    /// Record query execution feedback
    pub fn record_execution(
        &self,
        ef_search: usize,
        k: usize,
        latency_ms: f32,
        estimated_recall: f32,
        satisfied: bool,
    ) {
        if self.auto_feedback {
            let _ = self.tuner.record_feedback(QueryFeedback {
                ef_search,
                k,
                latency_ms,
                estimated_recall,
                satisfied,
                ..Default::default()
            });
        }
    }

    /// Get the underlying tuner
    pub fn tuner(&self) -> &LearnedTuner {
        &self.tuner
    }
}

/// Get current timestamp in milliseconds
fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Generate a random float between 0 and 1
fn rand_float() -> f32 {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hash, Hasher};

    let mut hasher = RandomState::new().build_hasher();
    hasher.write_u64(current_timestamp());
    // Hash the thread ID rather than trying to convert it to u64
    std::thread::current().id().hash(&mut hasher);
    hasher.finish() as f32 / u64::MAX as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tuner_creation() {
        let config = TunerConfig::default()
            .with_learning_rate(0.05)
            .with_exploration_rate(0.2);

        let tuner = LearnedTuner::new(config);
        let stats = tuner.stats();

        assert_eq!(stats.total_samples, 0);
        assert_eq!(stats.explorations, 0);
    }

    #[test]
    fn test_recommend_params_cold_start() {
        let tuner = LearnedTuner::default_tuner();

        let params = tuner.recommend_params(10, 0.95).unwrap();

        // Should be exploration during cold start
        assert!(params.is_exploration);
        assert!(params.confidence < 0.5);
    }

    #[test]
    fn test_feedback_recording() {
        let tuner = LearnedTuner::default_tuner();

        for i in 0..100 {
            tuner.record_feedback(QueryFeedback {
                ef_search: 50,
                k: 10,
                latency_ms: 2.0 + (i as f32 * 0.01),
                estimated_recall: 0.95,
                satisfied: true,
                ..Default::default()
            }).unwrap();
        }

        let stats = tuner.stats();
        assert_eq!(stats.total_samples, 100);
        assert!(stats.avg_latency_ms > 2.0);
        assert!(stats.avg_recall > 0.9);
    }

    #[test]
    fn test_learning_effect() {
        let config = TunerConfig::default()
            .with_min_samples(10)
            .with_exploration_rate(0.0); // Pure exploitation

        let tuner = LearnedTuner::new(config);

        // Train on ef_search=100 being better
        for _ in 0..50 {
            tuner.record_feedback(QueryFeedback {
                ef_search: 100,
                k: 10,
                latency_ms: 3.0,
                estimated_recall: 0.98,
                satisfied: true,
                ..Default::default()
            }).unwrap();

            tuner.record_feedback(QueryFeedback {
                ef_search: 50,
                k: 10,
                latency_ms: 2.0,
                estimated_recall: 0.85,
                satisfied: false,
                ..Default::default()
            }).unwrap();
        }

        // Should now prefer ef_search=100
        let stats = tuner.stats();
        assert_eq!(stats.best_ef_search, 100);
    }

    #[test]
    fn test_state_export_import() {
        let tuner = LearnedTuner::default_tuner();

        // Record some feedback
        for _ in 0..20 {
            tuner.record_feedback(QueryFeedback {
                ef_search: 75,
                k: 20,
                latency_ms: 4.0,
                estimated_recall: 0.93,
                satisfied: true,
                ..Default::default()
            }).unwrap();
        }

        // Export state
        let state = tuner.export_state();

        // Verify state contains expected data
        assert_eq!(state.total_pulls, 20);
        assert!(state.arms.contains_key(&75)); // ef_search=75 should have an arm

        // Create new tuner and import
        let tuner2 = LearnedTuner::default_tuner();
        tuner2.import_state(state.clone());

        // After import, workload profile and model should be restored
        // (history is not exported, so total_samples will be 0)
        let state2 = tuner2.export_state();
        assert_eq!(state2.total_pulls, state.total_pulls);
    }

    #[test]
    fn test_adaptive_executor() {
        let tuner = LearnedTuner::default_tuner();
        let executor = AdaptiveExecutor::new(tuner)
            .with_bounds(20, 200);

        let ef = executor.get_ef_search(10, 0.95);
        assert!(ef >= 20);
        assert!(ef <= 200);

        executor.record_execution(ef, 10, 2.5, 0.96, true);

        let stats = executor.tuner().stats();
        assert_eq!(stats.total_samples, 1);
    }

    #[test]
    fn test_online_model_predictions() {
        let model = OnlineModel::new(0.1);

        let latency = model.predict_latency(50, 10, 0, 1.0);
        assert!(latency > 0.0);

        let recall = model.predict_recall(50, 10);
        assert!(recall > 0.0);
        assert!(recall < 1.0);
    }

    #[test]
    fn test_workload_profile_update() {
        let config = TunerConfig::default()
            .with_learning_rate(0.1);

        let tuner = LearnedTuner::new(config);

        // Simulate workload with large k values
        // More iterations needed due to exponential moving average decay
        for _ in 0..200 {
            tuner.record_feedback(QueryFeedback {
                ef_search: 100,
                k: 100,
                latency_ms: 5.0,
                estimated_recall: 0.95,
                filter_complexity: 3,
                satisfied: true,
                ..Default::default()
            }).unwrap();
        }

        let stats = tuner.stats();
        // With decay_factor 0.99, need many samples to converge
        // avg_k should be moving toward 100
        assert!(stats.workload.avg_k > 10.0, "avg_k should be increasing, got {}", stats.workload.avg_k);
        assert!(stats.workload.avg_filter_complexity > 0.5, "avg_filter_complexity should be increasing, got {}", stats.workload.avg_filter_complexity);
    }
}
